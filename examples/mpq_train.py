import torch
import numpy as np

import sys

from models import model_zoo

model_name = sys.argv[1]
epoches = sys.argv[2]

if __name__ == "__main__":

    model, train_loader, test_loader = model_zoo.from_zoo(
        model_name, shuffle=True)

    res = model.train_loop(train_loader=train_loader,
                     test_loader=test_loader,
                     epoches=int(epoches))
    
    torch.save(model.state_dict(), f"output/ckpt/{model_name}.pth")

    print("Model Train Results: ", res)

