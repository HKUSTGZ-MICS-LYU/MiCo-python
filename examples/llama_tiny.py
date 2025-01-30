import json
import torch
import numpy as np

from models import TinyLLaMa1M
from MiCoUtils import (
    list_quantize_layers, 
    replace_quantize_layers,
    set_to_qforward,
    export_layer_weights
)
from TinyStories.tokenizer import Tokenizer
from datasets import tinystories

from tqdm import tqdm

epochs = 10000
batch_size = 128

model_name = "llama_tiny"

torch.manual_seed(0)
torch.cuda.manual_seed(0)

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device", device)

    model = TinyLLaMa1M().to(device)
    n_layers = len(list_quantize_layers(model))
    print("Number of Quantizable Layers: ", n_layers)
    weight_q = ["8b"] * n_layers
    activation_q = [8] * n_layers
    replace_quantize_layers(model, weight_q, activation_q, 
        quant_aware=False, use_norm=False, device=device)

    print("Model Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # model = torch.compile(model)

    train_loader, test_loader = tinystories(
        max_seq_len=model.params.max_seq_len, 
        vocab_size=model.params.vocab_size, 
        device=device, 
        batch_size=batch_size, 
        num_works=8)
    
    res = model.train_loop(n_iter=epochs, 
                     train_loader=train_loader, 
                     test_loader=test_loader, 
                     eval_interval=1000,
                     verbose=True)
    
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

    # data = export_layer_weights(model)
    # with open(f'output/json/{model_name}.json', 'w') as f:
    #     json.dump(data, f)

    # Test Model Generation
    TOKENIZER_PATH = "data/tinystories/tok4096.model"
    enc = Tokenizer(tokenizer_model=TOKENIZER_PATH)
    start = ""
    start_ids = enc.encode(start, bos=True, eos=False)
    temperature = 0.1
    top_k = 300
    max_new_tokens = 100
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    num_samples = 1

    print("Generation With FP Model:")
    with torch.no_grad():
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(enc.decode(y[0].tolist()))
            print('---------------')

    set_to_qforward(model)
    res = model.estimate_loss()
    print("Quantized Model Result: ", res)
    print("Generation With Quantized Model:")
    with torch.no_grad():
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(enc.decode(y[0].tolist()))
            print('---------------')