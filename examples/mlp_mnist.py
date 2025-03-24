import time
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
from datasets import mnist

from tqdm import tqdm

num_epoch = 5
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

    train_loader, test_loader = mnist(batch_size=batch_size, num_works=0, resize=16)
    # res = model.train_loop(num_epoch, train_loader, test_loader, verbose=True)
    # torch.save(model.state_dict(), 'output/ckpt/mlp_mnist.pth')
    # print("Model Results: ", res)

    # Load
    ckpt = torch.load('output/ckpt/mlp_mnist.pth')
    model.load_state_dict(ckpt)

    start_time = time.time()
    res = model.test(test_loader)
    end_time = time.time()
    print("Model Test Results: ", res)
    print("Model Test Time: ", end_time - start_time)

    model.set_qscheme_torchao([[4] * n_layers, [4] * n_layers], device=device)
    start_time = time.time()
    res = model.test(test_loader)
    end_time = time.time()
    print("Model Torch AO Test Results: ", res)
    print("Model Torch AO Test Time: ", end_time - start_time)

    # Need to reload if torchao is used
    model = MLP(in_features=256, config=config).to(device)
    ckpt = torch.load('output/ckpt/mlp_mnist.pth')
    model.load_state_dict(ckpt)

    weight_q = [8] * n_layers
    activation_q = [8] * n_layers
    model.set_qscheme(
        [weight_q, activation_q], 
        qat=False, device=device, use_bias=True)

    res = model.test(test_loader)
    print("Model Quantized Test Results: ", res)

    data = export_layer_weights(model)
    with open('output/json/mlp_mnist.json', 'w') as f:
        json.dump(data, f)