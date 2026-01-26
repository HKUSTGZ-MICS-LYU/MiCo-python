import sys
import json
import torch
import numpy as np

from models import VGG
from MiCoUtils import (
    get_model_macs,
)
from MiCoDatasets import cifar10
from MiCoSearch import MiCoSearch
from searchers.legacy import NLPSearcher, HAQSearcher


from tqdm import tqdm

num_epoch = 5
batch_size = 64

BUDGET = 32
INIT   = 16

seed = 0
if len(sys.argv) == 1:
    seed = int(sys.argv[1])

torch.manual_seed(0)
torch.cuda.manual_seed(0)


if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device", device)

    model = VGG(in_channels=3, num_class=10).to(device)

    wq_types = [4,5,6,7,8]
    aq_types = [4,5,6,7,8]
    train_loader, test_loader = cifar10(batch_size=batch_size, num_works=0)

    search = MiCoSearch(model, num_epoch, 
                   train_loader, test_loader, seed=seed,
                   wq_types=wq_types, aq_types=aq_types,
                   pretrained_model="output/ckpt/vgg_cifar10.pth")
    
    print("--------------------------------------")
    print("Bayesian Optimization MPQ Search")
    searcher = NLPSearcher(search, wq_types)

    max_qscheme = [8] * search.n_layers

    baseline_res = search.eval_scheme([max_qscheme, max_qscheme], 
                                      verbose=True, ptq=True)

    constr_bops = baseline_res["BOPs"] * 0.8

    nlp_res = searcher.search(constr_bops=constr_bops, ptq=True, use_max_q=False)

    print("Best Results (NLP):", nlp_res)