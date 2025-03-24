import json
import time
import torch
import numpy as np

from models import VGG
from MiCoUtils import (
    list_quantize_layers, 
    replace_quantize_layers,
    set_to_qforward,
    export_layer_weights,
    fuse_model
)
from datasets import cifar10

from tqdm import tqdm

num_epoch = 5
batch_size = 64

model_name = "vgg_cifar10"

torch.manual_seed(0)
torch.cuda.manual_seed(0)

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device", device)

    model = VGG(in_channels=3, num_class=10).to(device)
    n_layers = len(list_quantize_layers(model))
    print("Number of Quantizable Layers: ", n_layers)

    train_loader, test_loader = cifar10(batch_size=batch_size, num_works=0)
    # res = model.train_loop(num_epoch, train_loader, test_loader, verbose=True)
    # torch.save(model.state_dict(), f'output/ckpt/{model_name}.pth')
    # print("Model Results: ", res)

    # Load Test
    ckpt = torch.load(f'output/ckpt/{model_name}.pth')
    model.load_state_dict(ckpt)

    start_time = time.time()
    res = model.test(test_loader)
    end_time = time.time()
    print("Model Test Results: ", res)
    print("Model Test Time: ", end_time - start_time)

    model.set_qscheme_torchao([[8] * n_layers, [32] * n_layers], device=device)
    start_time = time.time()
    res = model.test(test_loader)
    end_time = time.time()
    print("Model Torch AO Test Results: ", res)
    print("Model Torch AO Test Time: ", end_time - start_time)

    # Need to reload if torchao is used
    model = VGG(in_channels=3, num_class=10).to(device)
    ckpt = torch.load(f'output/ckpt/{model_name}.pth')
    model.load_state_dict(ckpt)

    # Fuse Model
    # model = fuse_model(model)
    # res = model.test(test_loader)
    # print("Model Fused Test Results: ", res)

    weight_q = [8] * n_layers
    activation_q = [8] * n_layers
    replace_quantize_layers(model, weight_q, activation_q, 
        quant_aware=False, device=device, use_bias=True)

    set_to_qforward(model)
    res = model.test(test_loader)
    print("Model Quantized Test Results: ", res)

    # data = export_layer_weights(model)
    # with open(f'output/json/{model_name}.json', 'w') as f:
    #     json.dump(data, f)