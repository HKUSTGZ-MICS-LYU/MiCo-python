import torch
import numpy as np

import sys

from models import model_zoo

model_name = sys.argv[1]
epoches = sys.argv[2]

if __name__ == "__main__":

    model, train_loader, test_loader = model_zoo.from_zoo(
        model_name, shuffle=True, batch_size=128)

    res = model.train_loop(n_epoch=int(epoches),
                        train_loader=train_loader, 
                        test_loader=test_loader,
                        lr = 0.1, 
                        scheduler = "cifar100-step",
                        early_stopping = False,
                        verbose=True)
    
    torch.save(model.state_dict(), f"output/ckpt/{model_name}.pth")

    print("Model Train Results: ", res)