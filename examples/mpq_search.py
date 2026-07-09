import json
import torch
import argparse
import random
import numpy as np
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matplotlib import pyplot as plt

from MiCoEval import MiCoEval

from models import model_zoo

from tqdm import tqdm

from searchers import (
    RegressionSearcher, BayesSearcher, 
    NLPSearcher, HAQSearcher, MiCoSearcher,
    BayesianBitsSearcher,
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
argsparse.add_argument("--output-json", type=str, default=None)
argsparse.add_argument("--qtypes", type=str, default=None,
                       help="Comma-separated bitwidth search space, e.g. 1,2,4,8 or 4,5,6,7,8")
argsparse.add_argument("--bb-mu", type=float, default=None,
                       help="Single BayesianBits regularization strength. Overrides --bb-mu-grid when set.")
argsparse.add_argument("--bb-mu-grid", type=str, default="0.01,0.03,0.05,0.07,0.1,0.2")
argsparse.add_argument("--bb-lr", type=float, default=1e-3)
argsparse.add_argument("--bb-gamma-init", type=str, default="6.0",
                       help="Scalar or comma-separated gate initialization, e.g. 6.0 or 0,0,-1,-1")
argsparse.add_argument("--bb-objective", type=str, default="constrained",
                       choices=["constrained", "pareto"],
                       help="constrained uses a target-BOP violation penalty; pareto uses original BB CE + mu * gate regularizer")
argsparse.add_argument("--bb-constraint-lambda", type=float, default=10.0)
argsparse.add_argument("--bb-constraint-margin", type=float, default=1.0,
                       help="Multiplier on target constraint for constrained BB; <1 is stricter")
argsparse.add_argument("--bb-decode", type=str, default="hard",
                       choices=["hard", "expected_floor", "expected_nearest"],
                       help="Decode learned BayesianBits gates into a discrete MiCo scheme")
argsparse.add_argument("--bb-max-batches", type=int, default=None)
argsparse.add_argument("--bb-train-weights", type=str, default="auto",
                       choices=["auto", "yes", "no"])
argsparse.add_argument("--batch-size", type=int, default=32)
argsparse.add_argument("--num-workers", type=int, default=None,
                       help="Override model_zoo.NUM_WORKERS for local smoke tests or constrained environments")

args = argsparse.parse_args()


def parse_bb_gamma_init(raw_value: str, bitwidths: list):
    gate_transitions = list(zip(bitwidths[:-1], bitwidths[1:]))
    n_gates = len(gate_transitions)
    if "," not in raw_value:
        return float(raw_value)
    values = [float(x) for x in raw_value.split(",") if x]
    if len(values) != n_gates:
        raise ValueError(
            f"--bb-gamma-init has {len(values)} values, but qtypes={bitwidths} "
            f"requires {n_gates} gates for transitions {gate_transitions}"
        )
    return values

model_name = args.model_name
N_INIT = args.init
N_SEARCH = args.n_search
CONSTR_RATIO = args.constraint_factor
CONSTR_TYPE = args.constraint
METHODS = args.methods.split(",")
TRAILS = args.trails
MODE = args.mode
EPOCHS = args.epochs # required for QAT search, ignored for PTQ search
OUTPUT_JSON = args.output_json or f"output/json/{model_name}_search.json"
QTYPES = None if args.qtypes is None else [int(x) for x in args.qtypes.split(",") if x]
if args.bb_mu is not None:
    BB_MU_GRID = [args.bb_mu]
elif args.bb_objective == "constrained":
    BB_MU_GRID = [0.0]
else:
    BB_MU_GRID = [float(x) for x in args.bb_mu_grid.split(",") if x]

if __name__ == "__main__":

    if QTYPES is not None:
        bitwidths = QTYPES
    elif MODE == "ptq_acc":
        bitwidths = [4, 5, 6, 7, 8]
    elif MODE == "qat_acc":
        bitwidths = [1, 2, 4, 8]
    else:
        bitwidths = [4, 5, 6, 7, 8]
    BB_GAMMA_INIT = parse_bb_gamma_init(args.bb_gamma_init, bitwidths)
    print("BB Gate Transitions:", list(zip(bitwidths[:-1], bitwidths[1:])))

    if args.num_workers is not None:
        model_zoo.NUM_WORKERS = args.num_workers
    model, train_loader, test_loader = model_zoo.from_zoo(model_name, batch_size=args.batch_size)

    evaluator = MiCoEval(model, EPOCHS, train_loader, test_loader, 
                         f"output/ckpt/{model_name}.pth",
                         output_json=OUTPUT_JSON)

    dim = evaluator.n_layers * 2

    evaluator.set_constraint(CONSTR_TYPE)

    max_c = evaluator.constr([max(bitwidths)] * dim)
    print("Maximum Constraints:", max_c)
    min_c = evaluator.constr([min(bitwidths)] * dim)
    print("Minimum Constraints:", min_c)

    print(f"Target Constraints: {max_c*CONSTR_RATIO} ({CONSTR_RATIO:.2%} of max)")
    res_data = {}

    methods = METHODS

    for seed in range(TRAILS):
        
        for method in methods:
            
            if method not in res_data:
                res_data[method] = []

            random.seed(seed)
            np.random.seed(seed)
            torch.random.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.random.manual_seed(seed)

            print("Model Type:", method)
            search_start = time.time()
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
            elif method == "bb":
                searcher = BayesianBitsSearcher(
                    evaluator, n_inits=N_INIT, qtypes=bitwidths,
                    mu_grid=BB_MU_GRID, lr=args.bb_lr,
                    gamma_init=BB_GAMMA_INIT,
                    max_batches=args.bb_max_batches,
                    train_weights=args.bb_train_weights,
                    objective=args.bb_objective,
                    constraint_lambda=args.bb_constraint_lambda,
                    constraint_margin=args.bb_constraint_margin,
                    decode_mode=args.bb_decode,
                )
            else:
                searcher = RegressionSearcher(
                    evaluator, n_inits=N_INIT, qtypes=bitwidths,
                    model_type=method)

            if (method == "nlp") and (len(res_data[method]) > 0):
                # NLP gives fixed solution
                continue

            res_x, res_y = searcher.search(
                N_SEARCH, MODE, CONSTR_TYPE, max_c*CONSTR_RATIO)

            print(f"Best Scheme: {res_x}")
            print(f"Best Accuracy: {res_y}")
            print(f"Accuracy Drop: {evaluator.baseline_acc - res_y:.4f}" )
            print(f"Search Time:", time.time() - search_start)
            res = evaluator.constr(res_x)
            print(f"MPQ Real Constraint: {res} ({res / max_c:.2%})")
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
    os.makedirs("output/figs", exist_ok=True)
    os.makedirs("output/txt", exist_ok=True)
    os.makedirs("output/json", exist_ok=True)
    plt.savefig(f"output/figs/{model_name}_search_{CONSTR_RATIO}_{MODE}.pdf")

    with open(f"output/txt/{model_name}_search_{CONSTR_RATIO}_{MODE}.txt", "w") as f:
        for method in final_res.keys():
            f.write(f"{method}: {final_res[method]}\n")

    with open(f"output/json/{model_name}_search_{CONSTR_RATIO}_{MODE}_trace.json", "w") as f:
        json.dump(final_trace, f)

    # plt.show()
