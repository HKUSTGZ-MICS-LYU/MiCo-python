import copy
import random

import torch
from torch import nn
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
                   optimizer_name = "adamw", weight_decay = 1e-4,
                   grad_clip = None, best_checkpoint_path = None,
                   seed = None):
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
        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            raise ValueError("optimizer_name must be 'adamw' or 'adam'")
        criterion = torch.nn.CrossEntropyLoss()
        run_device = self.current_device()
        n_epoch = max(1, int(n_epoch))
        warmup_epochs = max(0, int(warmup_epochs))
        warmup_epochs = min(warmup_epochs, max(0, n_epoch - 1))
        use_warmup = scheduler == "cosine" and warmup_epochs > 0

        if scheduler == "none":
            scheduler = None
        elif scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, max(1, n_epoch - warmup_epochs)
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
        for epoch in range(n_epoch):
            if use_warmup:
                if epoch < warmup_epochs:
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
                optimizer.zero_grad()
                output = self(x)
                _, predicted = torch.max(output.data, 1)
                loss = criterion(output, y)
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                train_loss.append(loss.item())
                train_total += y.size(0)
                train_correct += (predicted == y).sum().item()
                optimizer.step()
                loop.set_description(
                    f"[Epoch {epoch+1}/{n_epoch} Loss: {loss.item():.3f}]")
            train_acc = train_correct / train_total

            if scheduler is not None and (not use_warmup or epoch >= warmup_epochs):
                scheduler.step()
            # Testing
            self.eval()
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

            log = [ 
                    f"Epoch [{epoch+1}/{n_epoch}]",
                    f"LR: {optimizer.param_groups[0]['lr']:.2e}",
                    f"Train Loss: {np.mean(train_loss):.3f}", 
                    f"Test Loss: {np.mean(test_loss):.3f}",
                    f'Accuracy: {train_acc*100:.1f}%, {test_acc*100:.1f}%'
                ]
            
            if last_loss < np.mean(train_loss) and early_stopping:
                break
            last_loss = np.mean(train_loss)
            
            res = {"TrainLoss": np.mean(train_loss),
                    "TestLoss": np.mean(test_loss),
                    "TrainAcc": train_acc,
                    "TestAcc": test_acc,
                    "BestTestAcc": max(best_test_acc, test_acc),
                    "BestEpoch": best_epoch}

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch + 1
                res["BestTestAcc"] = best_test_acc
                res["BestEpoch"] = best_epoch
                if best_checkpoint_path is not None:
                    torch.save({
                        "model": copy.deepcopy(self.state_dict()),
                        "epoch": best_epoch,
                        "metrics": dict(res),
                    }, best_checkpoint_path)
            
            if verbose: 
                print(" ".join(log))
        
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
