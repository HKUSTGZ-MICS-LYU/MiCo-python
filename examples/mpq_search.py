import torch
import random
import numpy as np

import sys

from matplotlib import pyplot as plt

from MiCoEval import MiCoEval
from MiCoProxy import get_mico_proxy, get_bitfusion_proxy

from models import model_zoo

from tqdm import tqdm

from searchers import (
    RegressionSearcher, BayesSearcher, 
    NLPSearcher, HAQSearcher, MiCoSearcher
)

N_SEARCH = 40
CONSTR_RATIO = 0.5

model_name = sys.argv[1]

if __name__ == "__main__":

    model, train_loader, test_loader = model_zoo.from_zoo(model_name)

    evaluator = MiCoEval(model, 10, train_loader, test_loader, 
                         f"output/ckpt/{model_name}.pth",
                         output_json=f"output/json/{model_name}_search.json")

    dim = model.n_layers * 2
    bitwidths = [4, 5, 6, 7, 8]

    max_bops = evaluator.eval_bops([8] * dim)
    print("INT8 Predicted Latency:", max_bops)
    min_bops = evaluator.eval_bops([1] * dim)
    print("INT1 Predicted Latency:", min_bops)

    res_data = {}

    for seed in [0,1,2,3,4]:
        
        for model in ["bo", "mico", "haq", "nlp"]:
            
            if model not in res_data:
                res_data[model] = []

            random.seed(seed)
            np.random.seed(seed)

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
                N_SEARCH, 'ptq_acc', 'bops', max_bops*CONSTR_RATIO)

            print(f"Best Scheme: {res_x}")
            print(f"Best Accuracy: {res_y}")
            res = evaluator.eval_bops(res_x)
            print(f"MPQ Real Latency: {res}")

            res_data[model].append(searcher.best_trace)

    # Plot Average Results
    for model in res_data.keys():
        avg_trace = np.mean(res_data[model], axis=0)
        plt.plot(avg_trace, label=model)

    plt.legend()
    plt.savefig(f"output/figs/{model_name}_search.pdf")
    plt.show()