import json
import torch
import numpy as np

from models import MLP
from MiCoUtils import (
    list_quantize_layers, 
    replace_quantize_layers,
    set_to_qforward,
    export_layer_weights
)
from MiCoDatasets import mnist

from tqdm import tqdm

num_epoch = 1
batch_size = 64

torch.manual_seed(0)
torch.cuda.manual_seed(0)

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device", device)

    config = {
        "Layers": [64, 64, 64, 10],
    }

    model = MLP(in_features=256, config=config).to(device)
    n_layers = len(list_quantize_layers(model))
    print("Number of Quantizable Layers: ", n_layers)
    weight_q = [8, 8, 8, 8]
    activation_q = [4, 4, 4, 4]
    model.set_qscheme([weight_q, activation_q], qat=True)

    train_loader, test_loader = mnist(batch_size=batch_size, num_works=0, resize=16)
    res = model.train_loop(num_epoch, train_loader, test_loader, verbose=True)
    torch.save(model.state_dict(), 'output/ckpt/mlp_mnist_mp.pth')

    print("Model Results: ", res)
    res = model.test(test_loader)
    print("Model Test Results: ", res)

    set_to_qforward(model)
    res = model.test(test_loader)
    print("Model Quantized Test Results: ", res)