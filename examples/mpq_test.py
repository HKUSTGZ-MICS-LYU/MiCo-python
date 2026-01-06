import torch
import numpy as np

import sys

from models import model_zoo
from MiCoEval import MiCoEval


model_name = sys.argv[1]

if __name__ == "__main__":

    model, train_loader, test_loader = model_zoo.from_zoo(model_name)
    evaluator = MiCoEval(model, 1, train_loader, test_loader, 
                         f"output/ckpt/{model_name}.pth")
    