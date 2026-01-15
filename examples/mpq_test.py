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
    
    model_info = evaluator.get_layer_info()
    layer_counts = {}
    for info in model_info:
        print(info)
        # Count Layers Information
        key = str(info["Layer Features"])
        if key not in layer_counts:
            layer_counts[key] = 1
        else:
            layer_counts[key] += 1

    print("\nLayer Counts Summary:")
    for layer_type, count in layer_counts.items():
        print(f"{layer_type}: {count}") 