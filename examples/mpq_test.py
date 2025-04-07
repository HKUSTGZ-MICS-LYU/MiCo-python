import torch
import numpy as np

import sys

from models import model_zoo

model_name = sys.argv[1]

if __name__ == "__main__":

    model, train_loader, test_loader = model_zoo.from_zoo(model_name)
    
    ckpt = torch.load(f"output/ckpt/{model_name}.pth")
    model.load_state_dict(ckpt)

    res = model.test(test_loader)

    print("Model Test Results: ", res)

    model.set_qscheme(
        [[8] * model.n_layers, [8] * model.n_layers]
    )

    res = model.test(test_loader)
    print("Model (INT8) Test Results: ", res)