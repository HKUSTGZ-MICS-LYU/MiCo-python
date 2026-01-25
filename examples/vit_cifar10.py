import json
import time
import torch
import numpy as np

from models import ViT1M_cifar10
from MiCoUtils import (
    list_quantize_layers, 
    get_model_params,
    replace_quantize_layers,
    set_to_qforward,
    export_layer_weights,
    fuse_model
)
from MiCoDatasets import cifar10

from tqdm import tqdm

num_epoch = 1
batch_size = 64

model_name = "vit_cifar10"

torch.manual_seed(0)
torch.cuda.manual_seed(0)

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device", device)

    model = ViT1M_cifar10().to(device)
    n_layers = len(list_quantize_layers(model))
    print("Number of Quantizable Layers: ", n_layers)
    print("Model Parameters: ", get_model_params(model))

    train_loader, test_loader = cifar10(batch_size=batch_size, num_works=0)
    res = model.train_loop(num_epoch, train_loader, test_loader, verbose=True)

    torch.save(model.state_dict(), f'output/ckpt/{model_name}.pth')
    print("Model Results: ", res)

    # Load Test
    # ckpt = torch.load(f'output/ckpt/{model_name}.pth')
    # model.load_state_dict(ckpt)

    start_time = time.time()
    res = model.test(test_loader)
    end_time = time.time()
    print("Model Test Results: ", res)
    print("Model Test Time: ", end_time - start_time)

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