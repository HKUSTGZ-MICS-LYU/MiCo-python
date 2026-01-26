import json
import torch
import numpy as np

from models import LeNetBN, LeNet
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

model_name = "lenet_mnist"

torch.manual_seed(0)
torch.cuda.manual_seed(0)

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device", device)

    model = LeNet(in_channels=1).to(device)
    n_layers = len(list_quantize_layers(model))
    print("Number of Quantizable Layers: ", n_layers)
    weight_q = [8] * n_layers
    activation_q = [8] * n_layers
    replace_quantize_layers(model, weight_q, activation_q, 
        quant_aware=False, device=device, use_bias=True)

    train_loader, test_loader = mnist(batch_size=batch_size, num_works=0, resize=28)

    # Training
    res = model.train_loop(num_epoch, train_loader, test_loader, verbose=True)
    torch.save(model.state_dict(), f'output/ckpt/{model_name}.pth')
    print("Model Results: ", res)

    # Load Test
    # ckpt = torch.load(f'output/ckpt/{model_name}.pth')
    # model.load_state_dict(ckpt)

    res = model.test(test_loader)
    print("Model Test Results: ", res)

    set_to_qforward(model)
    res = model.test(test_loader)
    print("Model Quantized Test Results: ", res)

    data = export_layer_weights(model)
    with open(f'output/json/{model_name}.json', 'w') as f:
        json.dump(data, f)