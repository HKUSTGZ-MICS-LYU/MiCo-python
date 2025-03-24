import sys
import json
import torch
import numpy as np

from models import resnet_alt_8
from MiCoUtils import (
    get_model_macs,
)
from datasets import cifar10
from MiCoSearch import MiCoSearch
from searchers import MiCoBOSearcher, NLPSearcher, HAQSearcher


from tqdm import tqdm

num_epoch = 5
batch_size = 64

BUDGET = 20
INIT   = 10

seed = 0
if len(sys.argv) == 2:
    seed = int(sys.argv[1])

torch.manual_seed(0)
torch.cuda.manual_seed(0)

model_name = "resnet8_cifar10"

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device", device)

    model = resnet_alt_8(n_class=10).to(device)

    wq_types = [4,5,6,7,8]
    aq_types = [4,5,6,7,8]
    train_loader, test_loader = cifar10(batch_size=batch_size, num_works=4)

    search = MiCoSearch(model, num_epoch, 
                   train_loader, test_loader, seed=seed,
                   wq_types=wq_types, aq_types=aq_types,
                   pretrained_model=f"output/ckpt/{model_name}.pth")

    print("--------------------------------------")
    print("Bayesian Optimization MPQ Search")
    searcher = MiCoBOSearcher(search)

    max_qscheme = [8] * search.n_layers

    baseline_res = search.eval_scheme([max_qscheme, max_qscheme], 
                                      verbose=True, ptq=True)

    constr_bops = baseline_res["MaxBOPs"] * 0.5

    bo_best_res, trace = searcher.search(
        search_budget=BUDGET, 
        data_path=f"output/json/{model_name}_search.json",
        constr_bops=constr_bops,
        n_init=INIT, ptq=True,
        use_max_q=True,
        init_method='rand', roi=1.0
    )

    print("Best Results (BO):", bo_best_res)