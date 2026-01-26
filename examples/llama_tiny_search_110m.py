import json
import torch
import numpy as np

from models.LLaMa import TinyLLaMa2c110M
from MiCoUtils import (
    get_model_macs,
)
from MiCoDatasets import tinystories

from MiCoEval import MiCoEval
from searchers import (
    RegressionSearcher, BayesSearcher, 
    NLPSearcher, HAQSearcher, MiCoSearcher
)

from tqdm import tqdm

num_epoch = 5
batch_size = 64

BUDGET = 16
INIT   = 16

torch.manual_seed(0)
torch.cuda.manual_seed(0)

if __name__ == "__main__":
    
    device = torch.device("cpu")
    print("Using Device", device)

    model = TinyLLaMa2c110M().to(device)

    wq_types = [4, 8]
    aq_types = [4, 8]
    train_loader, test_loader = tinystories(
        max_seq_len=model.params.max_seq_len,
        vocab_size=model.params.vocab_size,
        device=device,
        batch_size=batch_size, num_works=8)

    evaluator = MiCoEval(model, num_epoch, train_loader, test_loader, 
                         "output/ckpt/stories110M.pt")

    max_bops = evaluator.eval_bops([8] * evaluator.n_layers * 2)
    print("INT8 BOPs:", max_bops)
    ptq_acc = evaluator.eval_ptq([8] * evaluator.n_layers * 2)
    print("INT8 PTQ Acc:", ptq_acc)