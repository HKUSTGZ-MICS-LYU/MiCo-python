import json
import torch
import numpy as np

from models import MLP
from MiCoUtils import (
    get_model_macs,
)
from datasets import mnist
from MiCoSearch import MiCoSearch
from searchers import MiCoBOSearcher


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

    wq_types = [4,5,6,7,8]
    aq_types = [4,5,6,7,8]
    train_loader, test_loader = mnist(batch_size=batch_size, num_works=0, resize=16)

    search = MiCoSearch(model, num_epoch, 
                   train_loader, test_loader, seed=0,
                   wq_types=wq_types, aq_types=aq_types,
                   pretrained_model="output/ckpt/mlp_mnist.pth")
    
    searcher = MiCoBOSearcher(search)

    max_qscheme = [8] * search.n_layers

    baseline_res = search.eval_scheme([max_qscheme, max_qscheme], 
                                      verbose=True, ptq=True)

    constr_bops = baseline_res["MaxBOPs"] * 0.5

    best_res, trace = searcher.search(
        search_budget=10, 
        data_path="output/json/mlp_mnist_search.json",
        constr_bops=constr_bops,
        n_init=5, ptq=True,
        use_max_q=True,
    )

    print("Baseline (8b) Results:", baseline_res)
    print("Best Results:", best_res)