import torch
import numpy as np

import sys
import argparse

from models import model_zoo

argsparse = argparse.ArgumentParser()
argsparse.add_argument("model_name", type=str)
argsparse.add_argument("epoches", type=int, default=40)
argsparse.add_argument("--batch-size", type=int, default=32)
argsparse.add_argument("--lr", type=float, default=0.001)
argsparse.add_argument("--scheduler", type=str, default="none")

args = argsparse.parse_args()
batch_size = args.batch_size
model_name = args.model_name
epoches = args.epoches
lr = args.lr
scheduler = args.scheduler

if __name__ == "__main__":

    model, train_loader, test_loader = model_zoo.from_zoo(
        model_name, shuffle=True, batch_size=batch_size)

    res = model.train_loop(n_epoch=int(epoches),
                        train_loader=train_loader, 
                        test_loader=test_loader,
                        lr = lr, 
                        scheduler = scheduler,
                        early_stopping = False,
                        verbose=True)
    
    torch.save(model.state_dict(), f"output/ckpt/{model_name}.pth")

    print("Model Train Results: ", res)