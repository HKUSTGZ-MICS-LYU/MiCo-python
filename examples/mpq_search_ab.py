import torch
import argparse
import random
import numpy as np

from matplotlib import pyplot as plt

from MiCoEval import MiCoEval

from models import model_zoo

from tqdm import tqdm

from searchers import (
    RegressionSearcher, BayesSearcher, 
    NLPSearcher, HAQSearcher, MiCoSearcher
)

argsparse = argparse.ArgumentParser()
argsparse.add_argument("model_name", type=str)
argsparse.add_argument("--init", type=int , default=16)
argsparse.add_argument("-n", "--n-search", type=int, default=16)
argsparse.add_argument("-c", "--constraint-factor", type=float, default=0.5)
argsparse.add_argument("-ctype", "--constraint", type=str, default="bops")
argsparse.add_argument("-t", "--trails", type=int, default=5)
argsparse.add_argument("-m", "--mode", type=str, default="ptq_acc")

args = argsparse.parse_args()

model_name = args.model_name
N_INIT = args.init
N_SEARCH = args.n_search
CONSTR_RATIO = args.constraint_factor
CONSTR_TYPE = args.constraint
TRAILS = args.trails
MODE = args.mode

if __name__ == "__main__":

    model, train_loader, test_loader = model_zoo.from_zoo(model_name)

    evaluator = MiCoEval(model, 1, train_loader, test_loader, 
                         f"output/ckpt/{model_name}.pth",
                         output_json=f"output/json/{model_name}_search.json")

    dim = evaluator.n_layers * 2

    if MODE == "ptq_acc":
        bitwidths = [4, 5, 6, 7, 8]
    elif MODE == "qat_acc":
        bitwidths = [1, 2, 4, 8]

    max_bops = evaluator.eval_bops([8] * dim)
    print("INT8 Predicted Latency:", max_bops)
    min_bops = evaluator.eval_bops([1] * dim)
    print("INT1 Predicted Latency:", min_bops)

    res_data = {}

    for seed in range(TRAILS):
        
        for method in ["mico", "mico+init", "mico+near"]:
            
            if method not in res_data:
                res_data[method] = []

            random.seed(seed)
            np.random.seed(seed)

            print("Model Type:", method)

            if method == "mico":
                searcher = MiCoSearcher(
                    evaluator, n_inits=N_INIT, qtypes=bitwidths,
                    initial_method="random", sample_method="random"
                )
            elif method == "mico+init":
                searcher = MiCoSearcher(
                    evaluator, n_inits=N_INIT, qtypes=bitwidths,
                    initial_method="orth", sample_method="random"
                )
            elif method == "mico+near":
                searcher = MiCoSearcher(
                    evaluator, n_inits=N_INIT, qtypes=bitwidths,
                    initial_method="orth", sample_method="near-constr"
                )

            res_x, res_y = searcher.search(
                N_SEARCH, MODE, CONSTR_TYPE, max_bops*CONSTR_RATIO)

            print(f"Best Scheme: {res_x}")
            print(f"Best Accuracy: {res_y}")
            res = evaluator.eval_bops(res_x)
            print(f"MPQ Real Latency: {res} ({res / max_bops:.2%})")

            res_data[method].append(searcher.best_trace)

    final_res = {}
    # Plot Average Results
    for method in res_data.keys():
        avg_trace = np.mean(res_data[method], axis=0)
        final_res[method] = avg_trace[-1]
        plt.plot(avg_trace, label=method)

    plt.legend()
    plt.savefig(f"output/figs/{model_name}_search_{CONSTR_RATIO}_{MODE}.pdf")

    with open(f"output/txt/{model_name}_search_{CONSTR_RATIO}_{MODE}.txt", "w") as f:
        for method in final_res.keys():
            f.write(f"{method}: {final_res[method]}\n")

    # plt.show()