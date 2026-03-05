import json
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
argsparse.add_argument("--methods", type=str, default="nlp,haq,bo,mico")
argsparse.add_argument("-n", "--n-search", type=int, default=16)
argsparse.add_argument("-c", "--constraint-factor", type=float, default=0.5)
argsparse.add_argument("-ctype", "--constraint", type=str, default="bops")
argsparse.add_argument("-t", "--trails", type=int, default=5)
argsparse.add_argument("-m", "--mode", type=str, default="ptq_acc")
argsparse.add_argument("-e", "--epochs", type=int, default=1)

args = argsparse.parse_args()

model_name = args.model_name
N_INIT = args.init
N_SEARCH = args.n_search
CONSTR_RATIO = args.constraint_factor
CONSTR_TYPE = args.constraint
METHODS = args.methods.split(",")
TRAILS = args.trails
MODE = args.mode
EPOCHS = args.epochs # required for QAT search, ignored for PTQ search

if __name__ == "__main__":

    model, train_loader, test_loader = model_zoo.from_zoo(model_name)

    evaluator = MiCoEval(model, EPOCHS, train_loader, test_loader, 
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

    methods = METHODS

    for seed in range(TRAILS):
        
        for method in methods:
            
            if method not in res_data:
                res_data[method] = []

            random.seed(seed)
            np.random.seed(seed)

            print("Model Type:", method)

            if method == "bo":
                searcher = BayesSearcher(
                    evaluator, n_inits=N_INIT, qtypes=bitwidths)
            elif method == "nlp":
                searcher = NLPSearcher(
                    evaluator, n_inits=N_INIT, qtypes=bitwidths, use_sos=(MODE=="qat_acc")
                )
            elif method == "mico":
                searcher = MiCoSearcher(
                    evaluator, n_inits=N_INIT, qtypes=bitwidths
                )
            elif method == "haq":
                searcher = HAQSearcher(
                    evaluator, n_inits=N_INIT, qtypes=bitwidths
                    )
            else:
                searcher = RegressionSearcher(
                    evaluator, n_inits=N_INIT, qtypes=bitwidths,
                    model_type=method)

            res_x, res_y = searcher.search(
                N_SEARCH, MODE, CONSTR_TYPE, max_bops*CONSTR_RATIO)

            print(f"Best Scheme: {res_x}")
            print(f"Best Accuracy: {res_y}")
            res = evaluator.eval_bops(res_x)
            print(f"MPQ Real Latency: {res} ({res / max_bops:.2%})")

            if MODE == "qat_acc":
                # Final Fine-tuning and evaluation
                print("Starting QAT fine-tuning...")
                final_acc = evaluator.eval_qat(res_x, EPOCHS*2)
                print(f"Final QAT Accuracy: {final_acc}")
                searcher.best_trace.append(final_acc)
            res_data[method].append(searcher.best_trace)

    final_res = {}
    final_trace = {}
    # Plot Average Results
    for method in res_data.keys():
        avg_trace = np.mean(res_data[method], axis=0)
        final_res[method] = avg_trace[-1]
        final_trace[method] = avg_trace.tolist()
        plt.plot(avg_trace, label=method)

    plt.legend()
    plt.savefig(f"output/figs/{model_name}_search_{CONSTR_RATIO}_{MODE}.pdf")

    with open(f"output/txt/{model_name}_search_{CONSTR_RATIO}_{MODE}.txt", "w") as f:
        for method in final_res.keys():
            f.write(f"{method}: {final_res[method]}\n")

    with open(f"output/json/{model_name}_search_{CONSTR_RATIO}_{MODE}_trace.json", "w") as f:
        json.dump(final_trace, f)

    # plt.show()