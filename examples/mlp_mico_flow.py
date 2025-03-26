import torch
import random
import numpy as np

from matplotlib import pyplot as plt

from MiCoEval import MiCoEval
from MiCoProxy import get_mico_proxy

from models import MLP
from datasets import mnist

from tqdm import tqdm

from searchers import (
    RegressionSearcher, BayesSearcher, 
    NLPSearcher, HAQSearcher, MiCoSearcher
)

random.seed(0)
np.random.seed(0)

if __name__ == "__main__":
    config = {
        "Layers": [64, 64, 64, 10],
    }
    model = MLP(256, config=config)
    train_loader, test_loader = mnist(shuffle=False, resize=16)
    evaluator = MiCoEval(model, 10, train_loader, test_loader, 
                         "output/ckpt/mlp_mnist.pth")
    proxy = get_mico_proxy()
    evaluator.set_proxy(proxy)

    dim = model.n_layers * 2
    bitwidths = [2, 4, 8]
    max_latency = evaluator.eval_pred_latency([8] * dim)
    print("INT8 Predicted Latency:", max_latency)
    min_latency = evaluator.eval_pred_latency([2] * dim)
    print("INT2 Predicted Latency:", min_latency)

    for model in ["mico"]:
        random.seed(0)
        np.random.seed(0)
        print("Model Type:", model)

        if model == "bo":
            searcher = BayesSearcher(
                evaluator, n_inits=10, qtypes=bitwidths)
        elif model == "nlp":
            searcher = NLPSearcher(
                evaluator, n_inits=10, qtypes=bitwidths
            )
        elif model == "mico":
            searcher = MiCoSearcher(
                evaluator, n_inits=10, qtypes=bitwidths
            )
        elif model == "haq":
            searcher = HAQSearcher(
                evaluator, n_inits=10, qtypes=bitwidths
                )
        else:
            searcher = RegressionSearcher(
                evaluator, n_inits=10, qtypes=bitwidths,
                model_type=model)

        res_x, res_y = searcher.search(
            20, 'ptq_acc', 'latency_mico_proxy', max_latency*0.9)
        
        print(f"Best Scheme: {res_x}")
        print(f"Best Accuracy: {res_y}")
        print(f"Deploying Model to MiCo....")
        res = evaluator.eval_latency(res_x, target="mico")
        print(f"MPQ Real Latency: {res}")
        res_int8 = evaluator.eval_latency([8] * dim, target="mico")
        print(f"INT8 Real Latency: {res_int8}")