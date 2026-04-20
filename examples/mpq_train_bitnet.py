import torch
import numpy as np

import sys
import argparse

from models import model_zoo

argsparse = argparse.ArgumentParser()
argsparse.add_argument("model_name", type=str)
argsparse.add_argument("epoches", type=int, default=10000)
argsparse.add_argument("--batch-size", type=int, default=32)
argsparse.add_argument("--lr", type=float, default=0.005)
argsparse.add_argument("-q", "--weight_quant", type=float, choices=[1,1.5,2], default=1)
argsparse.add_argument("-aq", "--act_quant", type=int, choices=[4,8], default=8)
argsparse.add_argument("--keep-last", action="store_true", default=False)
argsparse.add_argument("--keep-first", action="store_true", default=False)
argsparse.add_argument("--scheduler", type=str, default="none")

args = argsparse.parse_args()
batch_size = args.batch_size
model_name = args.model_name
epoches = args.epoches
lr = args.lr
scheduler = args.scheduler
weight_quant = args.weight_quant
act_quant = args.act_quant
keep_last = args.keep_last
keep_first = args.keep_first

if __name__ == "__main__":

    model, train_loader, test_loader = model_zoo.from_zoo(
        model_name, shuffle=True, batch_size=batch_size)

    qscheme = [
        [weight_quant] * model.n_layers, # weight qscheme
        [act_quant] * model.n_layers, # activation qscheme
    ]
    
    if keep_first:
        qscheme[0][0] = 8
        qscheme[1][0] = 8

    if keep_last:
        # Retain Last Layer in W8A8
        qscheme[0][-1] = 8
        qscheme[1][-1] = 8

    model.set_qscheme(qscheme, qat=True)

    if "llama" in model_name:
        res = model.train_loop(n_iter=int(epoches),
                train_loader=train_loader, 
                test_loader=test_loader,
                lr = lr, 
                eval_interval = epoches // 2,
                verbose=True)
    else:
        res = model.train_loop(n_epoch=int(epoches),
                        train_loader=train_loader, 
                        test_loader=test_loader,
                        lr = lr, 
                        scheduler = scheduler,
                        early_stopping = False,
                        verbose=True)
        
    torch.save(model.state_dict(), f"output/ckpt/{model_name}_bitnet.pth")
    print("Model Train Results: ", res)

    model.set_qscheme(qscheme)

    res = model.test(test_loader)

    print("Model Test Results: ", res)