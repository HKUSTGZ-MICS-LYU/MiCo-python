import json
import torch
import numpy as np

import torch.nn.utils.fusion as fusion

from models import LeNetBN
from MiCoUtils import (
    list_quantize_layers, 
    replace_quantize_layers,
    fuse_bitconv_bn,
    set_to_qforward,
    export_layer_weights,
)
from MiCoDatasets import mnist

from tqdm import tqdm

num_epoch = 5
batch_size = 64

model_name = "lenetbn_mnist"

torch.manual_seed(0)
torch.cuda.manual_seed(0)

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device", device)

    model = LeNetBN(in_channels=1).to(device)
    ckpt = torch.load(f'output/ckpt/{model_name}.pth')
    model.load_state_dict(ckpt)

    n_layers = len(list_quantize_layers(model))
    print("Number of Quantizable Layers: ", n_layers)
    weight_q = [8] * n_layers
    activation_q = [8] * n_layers
    replace_quantize_layers(model, weight_q, activation_q, 
        quant_aware=False, device=device)

    train_loader, test_loader = mnist(batch_size=batch_size, num_works=0, resize=28)

    res = model.test(test_loader)
    print("Model Test Results: ", res)


    fuse_bitconv_bn(model.layers[3], model.layers[4])
    model.layers[4] = torch.nn.Identity()
    print("Model after Fusion: ", model)
    
    res = model.test(test_loader)
    print("Model Test Results after Fusion: ", res)
