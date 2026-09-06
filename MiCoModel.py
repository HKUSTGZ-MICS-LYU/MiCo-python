import copy
import random

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from typing import List, Tuple

from tqdm import tqdm
from MiCoUtils import (
    list_quantize_layers, 
    list_quantize_attn_layers,
    set_to_qforward, 
    unset_qforward,
    replace_quantize_layers,
    replace_quantize_attn_layers,
    replace_quantize_layers_torchao)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MiCoFunc:
    def __init__(self, name, input_names=[], params=[]):
        self.name = name
        self.input_names = input_names
        self.params = params


def _optimizer_param_groups(model, weight_decay):
    """Split parameters so norms, biases and learned tokens are not decayed."""
    decay, no_decay = [], []
    no_decay_terms = ("norm", "bn", "positional_emb", "pos_embedding",
                      "cls_token", "class_emb")
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        name_lower = name.lower()
        if (parameter.ndim <= 1 or name_lower.endswith(".bias")
                or any(term in name_lower for term in no_decay_terms)):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _sample_beta(alpha):
    return float(torch.distributions.Beta(alpha, alpha).sample().item())


def _mix_batch(x, y, mixup_alpha=0.0, cutmix_alpha=0.0):
    """Return a mixed batch and its two target sets, or the original batch."""
    use_cutmix = x.ndim == 4 and cutmix_alpha > 0
    use_mixup = mixup_alpha > 0
    if not use_cutmix and not use_mixup:
        return x, y, None, 1.0

    if use_cutmix and use_mixup:
        use_cutmix = bool(torch.rand((), device=x.device) < 0.5)
    index = torch.randperm(x.size(0), device=x.device)

    if use_cutmix:
        lam = _sample_beta(cutmix_alpha)
        height, width = x.shape[-2:]
        cut_ratio = (1.0 - lam) ** 0.5
        cut_h = max(1, int(height * cut_ratio))
        cut_w = max(1, int(width * cut_ratio))
        cy = int(torch.randint(height, (1,), device=x.device).item())
        cx = int(torch.randint(width, (1,), device=x.device).item())
        y1, y2 = max(0, cy - cut_h // 2), min(height, cy + cut_h // 2)
        x1, x2 = max(0, cx - cut_w // 2), min(width, cx + cut_w // 2)
        x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
        lam = 1.0 - ((y2 - y1) * (x2 - x1) / float(height * width))
    else:
        lam = _sample_beta(mixup_alpha)
        x = x * lam + x[index] * (1.0 - lam)
    return x, y, y[index], lam


class _ShilabsLabelSmoothingCrossEntropy(nn.Module):
    """The label-smoothed CE used by the original CCT example."""

    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = float(smoothing)
        self.confidence = 1.0 - self.smoothing

    def forward(self, logits, target):
        log_prob = F.log_softmax(logits, dim=-1)
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_prob.mean(dim=-1)
        return (self.confidence * nll_loss
                + self.smoothing * smooth_loss).mean()

class MiCoModel(nn.Module):
    n_layers: int   
    default_dataset: str
    def __init__(self):
        super(MiCoModel, self).__init__()

    def get_qlayers(self):
        return list_quantize_layers(self)

    def get_attn_qlayers(self):
        return list_quantize_attn_layers(self)

    @property
    def n_attn_layers(self):
        return len(self.get_attn_qlayers())

    def set_qscheme(self, qscheme, qat=False, device=None, group_size = 1, use_bias = True, use_norm = False):
        if device is None:
            device = self.current_device()
        self.to(device)
        replace_quantize_layers(self, qscheme[0], qscheme[1], 
                                quant_aware=qat, group_size=group_size,
                                device=device, use_bias=use_bias, use_norm=use_norm)
        if not qat:
            set_to_qforward(self)
        return
    
    def unset_qscheme(self):
        unset_qforward(self)
        return

    def set_attn_qscheme(self, attn_qscheme, qat=False, **kwargs):
        replace_quantize_attn_layers(self, attn_qscheme, quant_aware=qat, **kwargs)
        if not qat:
            set_to_qforward(self)
        return

    def set_qscheme_torchao(self, qscheme, device=None):
        if device is None:
            device = self.current_device()
        self.to(device)
        unset_qforward(self)
        replace_quantize_layers_torchao(self, qscheme[0], qscheme[1], device=device)
        return
    
    def torchao_autoquant(self, example_input: torch.Tensor):
        raise NotImplementedError("autoquant was removed in torchao 0.17.0. Use set_qscheme_torchao() instead.")

    def current_device(self):
        for tensor in list(self.parameters()) + list(self.buffers()):
            return tensor.device
        return device

    def test(self, test_loader):
        self.eval()
        run_device = self.current_device()
        criterion = torch.nn.CrossEntropyLoss()
        test_loss = []
        test_total, test_correct = 0, 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                x, y = images.to(run_device), labels.to(run_device)
                output = self.forward(x)
                _, predicted = torch.max(output.data, 1)
                loss = criterion(output, y)
                test_loss.append(loss.item())
                test_total += y.size(0)
                test_correct += (predicted == y).sum().item()
        test_loss_mean = np.mean(test_loss)
        test_acc = test_correct / test_total
        return {"TestLoss": test_loss_mean, "TestAcc": test_acc}

    def train_loop(self, n_epoch, train_loader, test_loader, verbose = False,
                   lr = 0.001, scheduler = "none", early_stopping = True,
                   warmup_epochs = 0, warmup_lr = 1e-6,
                   min_lr = 0.0,
                   optimizer_name = "adamw", weight_decay = 1e-4,
                   grad_clip = None, best_checkpoint_path = None,
                   seed = None, label_smoothing = 0.0,
                   mixup_alpha = 0.0, cutmix_alpha = 0.0,
                   ema_decay = None, warmup_mode = "linear",
                   weight_decay_mode = "filtered",
                   label_smoothing_impl = "torch"):
        """Train a classification model with reproducible, resumable settings.

        ``optimizer_name`` and ``weight_decay`` are intentionally optional to
        preserve the old API.  The default AdamW path is better behaved for
        transformer-style FP32/QAT fine-tuning, while ``adam`` remains
        available for old experiments.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        optimizer_name = (optimizer_name or "adamw").lower()
        weight_decay_mode = (weight_decay_mode or "filtered").lower()
        if weight_decay_mode == "all":
            optimizer_params = self.parameters()
        elif weight_decay_mode == "filtered":
            optimizer_params = _optimizer_param_groups(self, weight_decay)
        else:
            raise ValueError("weight_decay_mode must be 'filtered' or 'all'")
        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                optimizer_params, lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                optimizer_params, lr=lr, weight_decay=weight_decay
            )
        else:
            raise ValueError("optimizer_name must be 'adamw' or 'adam'")
        label_smoothing = max(0.0, min(float(label_smoothing), 1.0))
        mixup_alpha = max(0.0, float(mixup_alpha))
        cutmix_alpha = max(0.0, float(cutmix_alpha))
        label_smoothing_impl = (label_smoothing_impl or "torch").lower()
        if label_smoothing_impl in {"shilabs", "official"}:
            criterion = _ShilabsLabelSmoothingCrossEntropy(label_smoothing)
        elif label_smoothing_impl == "torch":
            criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            raise ValueError(
                "label_smoothing_impl must be 'torch' or 'shilabs'"
            )
        run_device = self.current_device()
        n_epoch = max(1, int(n_epoch))
        warmup_epochs = max(0, int(warmup_epochs))
        warmup_epochs = min(warmup_epochs, max(0, n_epoch - 1))
        min_lr = max(0.0, float(min_lr))
        warmup_mode = (warmup_mode or "linear").lower()
        if warmup_mode not in {"linear", "shilabs", "official"}:
            raise ValueError("warmup_mode must be 'linear' or 'shilabs'")
        use_warmup = scheduler == "cosine" and warmup_epochs > 0

        if scheduler == "none":
            scheduler = None
        elif scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, max(1, n_epoch - warmup_epochs), eta_min=min_lr
            )
        elif scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                        step_size=max(1, n_epoch // 10),
                                                        gamma = 0.75)
        elif scheduler == "cifar100-step":
            milestones = sorted({
                max(1, int(n_epoch * ratio))
                for ratio in (0.5, 0.75, 0.9)
                if int(n_epoch * ratio) < n_epoch
            })
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, gamma=0.2)
        else:
            raise ValueError(f"Scheduler {scheduler} not recognized.")

        last_loss = np.inf
        best_test_acc = -np.inf
        best_epoch = 0
        res = None
        ema_decay = (None if ema_decay is None else float(ema_decay))
        if ema_decay is not None and not 0.0 < ema_decay < 1.0:
            raise ValueError("ema_decay must be between 0 and 1")
        ema_state = ({name: value.detach().clone()
                      for name, value in self.state_dict().items()}
                     if ema_decay is not None else None)
        for epoch in range(n_epoch):
            if use_warmup:
                if epoch < warmup_epochs:
                    if warmup_mode in {"shilabs", "official"}:
                        current_lr = lr / float(max(1, warmup_epochs - epoch))
                    else:
                        progress = float(epoch + 1) / float(max(1, warmup_epochs))
                        current_lr = warmup_lr + (lr - warmup_lr) * progress
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = current_lr
                elif epoch == warmup_epochs:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr

            train_loss = []
            train_total, train_correct = 0, 0
            loss = torch.tensor(np.inf, device=run_device)
            # Training
            self.train()
            loop = tqdm(enumerate(train_loader), total=len(train_loader), 
                        disable=not verbose)
            
            for i, (images, labels) in loop:
                x, y = images.to(run_device), labels.to(run_device)
                targets_a, targets_b, mix_lam = y, None, 1.0
                if self.training and (mixup_alpha > 0 or cutmix_alpha > 0):
                    x, targets_a, targets_b, mix_lam = _mix_batch(
                        x, y, mixup_alpha=mixup_alpha,
                        cutmix_alpha=cutmix_alpha)
                optimizer.zero_grad()
                output = self(x)
                _, predicted = torch.max(output.data, 1)
                if targets_b is None:
                    loss = criterion(output, targets_a)
                else:
                    loss = (mix_lam * criterion(output, targets_a)
                            + (1.0 - mix_lam) * criterion(output, targets_b))
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                train_loss.append(loss.item())
                train_total += y.size(0)
                train_correct += (predicted == y).sum().item()
                optimizer.step()
                if ema_state is not None:
                    with torch.no_grad():
                        for name, value in self.state_dict().items():
                            if torch.is_floating_point(value):
                                ema_state[name].mul_(ema_decay).add_(
                                    value.detach(), alpha=1.0 - ema_decay)
                            else:
                                ema_state[name].copy_(value)
                loop.set_description(
                    f"[Epoch {epoch+1}/{n_epoch} Loss: {loss.item():.3f}]")
            train_acc = train_correct / train_total

            if scheduler is not None and (not use_warmup or epoch >= warmup_epochs):
                scheduler.step()
            # Testing. EMA is used only for validation/checkpoint selection.
            self.eval()
            raw_state = None
            if ema_state is not None:
                raw_state = copy.deepcopy(self.state_dict())
                self.load_state_dict(ema_state)
            test_loss = []
            test_total, test_correct = 0, 0
            with torch.no_grad():
                for i, (images, labels) in enumerate(test_loader):
                    x, y = images.to(run_device), labels.to(run_device)
                    output = self.forward(x)
                    _, predicted = torch.max(output.data, 1)
                    loss = criterion(output, y)
                    test_loss.append(loss.item())
                    test_total += y.size(0)
                    test_correct += (predicted == y).sum().item()
            test_acc = test_correct / test_total
            evaluated_state = copy.deepcopy(self.state_dict())
            if raw_state is not None:
                self.load_state_dict(raw_state)

            train_loss_mean = float(np.mean(train_loss))
            test_loss_mean = float(np.mean(test_loss))

            log = [ 
                    f"Epoch [{epoch+1}/{n_epoch}]",
                    f"LR: {optimizer.param_groups[0]['lr']:.2e}",
                    f"Train Loss: {train_loss_mean:.3f}",
                    f"Test Loss: {test_loss_mean:.3f}",
                    f'Accuracy: {train_acc*100:.1f}%, {test_acc*100:.1f}%'
                ]
            
            should_stop = last_loss < train_loss_mean and early_stopping
            last_loss = train_loss_mean
            
            res = {"TrainLoss": train_loss_mean,
                    "TestLoss": test_loss_mean,
                    "TrainAcc": float(train_acc),
                    "TestAcc": float(test_acc),
                    "BestTestAcc": max(best_test_acc, test_acc),
                    "BestEpoch": best_epoch}

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch + 1
                res["BestTestAcc"] = best_test_acc
                res["BestEpoch"] = best_epoch
                if best_checkpoint_path is not None:
                    torch.save({
                        "model": evaluated_state,
                        "epoch": best_epoch,
                        "metrics": dict(res),
                    }, best_checkpoint_path)
            
            if verbose: 
                print(" ".join(log))
            if should_stop:
                break
        
        return res
    
def from_torch(model: torch.nn.Module, n_layers: int = -1):
    if isinstance(model, MiCoModel):
        model.n_layers = n_layers
        return model

    class MiCoWrapped(model.__class__, MiCoModel):
        pass
    
    model.__class__ = MiCoWrapped
    model.n_layers = len(model.get_qlayers())

    print(f"Model with {model.n_layers} layers wrapped to MiCoModel.")

    return model
