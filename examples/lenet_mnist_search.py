import json
import torch
import numpy as np

from models import LeNet
from MiCoUtils import (
    get_model_macs,
)
from datasets import mnist
from MiCoSearch import MiCoSearch
from searchers import MiCoBOSearcher, NLPSearcher, HAQSearcher


from tqdm import tqdm

num_epoch = 5
batch_size = 64

BUDGET = 32
INIT   = 16

USE_CBOPS = False

torch.manual_seed(0)
torch.cuda.manual_seed(0)

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device", device)
    model = LeNet(1).to(device)

    wq_types = [4,5,6,7,8]
    aq_types = [4,5,6,7,8]
    train_loader, test_loader = mnist(batch_size=batch_size, num_works=0)

    search = MiCoSearch(model, num_epoch, 
                   train_loader, test_loader, seed=0,
                   wq_types=wq_types, aq_types=aq_types,
                   pretrained_model="output/ckpt/lenet_mnist.pth")
    
    print("--------------------------------------")
    print("Bayesian Optimization MPQ Search")
    searcher = MiCoBOSearcher(search)

    max_qscheme = [8] * search.n_layers

    baseline_res = search.eval_scheme([max_qscheme, max_qscheme], 
                                      verbose=True, ptq=True)

    constr_bops = baseline_res["MaxBOPs"] * 0.5

    bo_best_res, trace = searcher.search(
        search_budget=BUDGET, 
        data_path="output/json/lenet_mnist_search.json",
        constr_bops=constr_bops,
        n_init=INIT, ptq=True,
        use_max_q=USE_CBOPS,
    )

    print("--------------------------------------")
    print("w-based NLP MPQ Search")

    searcher = NLPSearcher(search, qbits=wq_types)

    nlp_best_res = searcher.search(
        constr_bops=constr_bops,
        ptq=True,
        use_max_q=USE_CBOPS
    )

    print("--------------------------------------")
    print("HAQ MPQ Search")

    searcher = HAQSearcher(search, qbits=wq_types, 
                           seed=0, org_acc=baseline_res["Accuracy"])
    

    haq_best_res, _ = searcher.search(
        search_budget=BUDGET,
        constr_bops=constr_bops,
        n_init=INIT,
        ptq=True,
        use_max_q=USE_CBOPS
    )

    print("Baseline (8b) Results:", baseline_res)
    print("Best Results (BO) :", bo_best_res)
    print("Best Results (NLP):", nlp_best_res)
    print("Best Results (HAQ):", haq_best_res)