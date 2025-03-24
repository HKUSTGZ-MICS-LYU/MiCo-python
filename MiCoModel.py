import torch
from torchao import autoquant
from torch import nn
import numpy as np

from tqdm import tqdm
from MiCoUtils import (
    list_quantize_layers, 
    set_to_qforward, 
    replace_quantize_layers,
    replace_quantize_layers_torchao)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MiCoModel(nn.Module):
    n_layers: int
    def __init__(self):
        super(MiCoModel, self).__init__()

    def get_qlayers(self):
        return list_quantize_layers(self)

    def set_qscheme(self, qscheme, qat=False, device=device, use_bias = True):
        replace_quantize_layers(self, qscheme[0], qscheme[1], 
                                quant_aware=qat, 
                                device=device, use_bias=use_bias)
        if not qat:
            set_to_qforward(self)
        return
    
    def set_qscheme_torchao(self, qscheme,device=device):
        replace_quantize_layers_torchao(self, qscheme[0], qscheme[1], device=device)
        return
    
    def torchao_autoquant(self, example_input: torch.Tensor):
        autoquant(self)
        self(example_input)
        return

    def test(self, test_loader):
        self.eval()
        criterion = torch.nn.CrossEntropyLoss()
        test_loss = []
        test_total, test_correct = 0, 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                x, y = images.to(device), labels.to(device)
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
                   lr = 0.001, scheduler = "none", early_stopping = True):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        if scheduler == "none":
            scheduler = None
        elif scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epoch)
        elif scheduler == "cifar100-step":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[60, 120, 160, 200], gamma=0.2)

        last_loss = np.inf
        for epoch in range(n_epoch):
            train_loss = []
            train_total, train_correct = 0, 0
            loss = torch.tensor(np.inf)
            # Training
            self.train()
            for i, (images, labels) in tqdm(enumerate(train_loader), 
                                            total=len(train_loader),
                                            desc=f"[Epoch {epoch+1}/{n_epoch}]",
                                            disable=not verbose):
                x, y = images.to(device), labels.to(device)
                optimizer.zero_grad()
                output = self(x)
                _, predicted = torch.max(output.data, 1)
                loss = criterion(output, y)
                loss.backward()
                train_loss.append(loss.item())
                train_total += y.size(0)
                train_correct += (predicted == y).sum().item()
                optimizer.step()
            train_acc = train_correct / train_total
            if scheduler is not None:
                scheduler.step()
            # Testing
            self.eval()
            test_loss = []
            test_total, test_correct = 0, 0
            with torch.no_grad():
                for i, (images, labels) in enumerate(test_loader):
                    x, y = images.to(device), labels.to(device)
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
                    "TestAcc": test_acc}
            
            if verbose: 
                print(" ".join(log))
        
        return res