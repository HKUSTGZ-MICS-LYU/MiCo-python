import random
import argparse
import numpy as np
import os
import json

from matplotlib import pyplot as plt

from MiCoEval import MiCoEval
from MiCoProxy import get_mico_matmul_proxy, get_mico_conv2d_proxy, get_mico_misc_kernel_proxy

from models import model_zoo

from tqdm import tqdm

from searchers import (
    RegressionSearcher, BayesSearcher, 
    NLPSearcher, HAQSearcher, MiCoSearcher
)

random.seed(0)
np.random.seed(0)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("-c","--target_ratio", type=float, default=0.8)
    parser.add_argument("-s","--seed", type=int, default=0)
    parser.add_argument("-n","--n_inits", type=int, default=10)
    parser.add_argument("-i","--n_iter", type=int, default=10)
    parser.add_argument("--qat", type=int, default=-1)
    parser.add_argument("--mode", type=str, default="proxy", choices=["proxy", "bops"])
    parser.add_argument("--mico_target", type=str, default="high", choices=["high", "small", "cacheless"])
    args = parser.parse_args()

    model, train_loader, test_loader = model_zoo.from_zoo(args.model)
    evaluator = MiCoEval(model, args.qat, train_loader, test_loader, 
                         f"output/ckpt/{args.model}.pth",
                         output_json=f"output/json/{args.model}_search.json",)

    mico_target = args.mico_target
    matmul_proxy = get_mico_matmul_proxy(mico_target)
    conv2d_proxy = get_mico_conv2d_proxy(mico_target)
    evaluator.set_proxy(matmul_proxy, conv2d_proxy)
    evaluator.set_misc_proxy(get_mico_misc_kernel_proxy)
    evaluator.set_mico_target(mico_target)
    evaluator.get_misc_latency()

    dim = model.n_layers * 2
    bitwidths = [1, 2, 4, 8]
    target_ratio = args.target_ratio

    # Calibration 
    if args.mode == "proxy":
        min_real, max_real = evaluator.calibrate_proxy(min_q=1, max_q=8, target="mico")
    elif args.mode == "bops":
        max_real = evaluator.eval_bops([8] * dim)
        min_real = evaluator.eval_bops([1] * dim)
    print("Maximum Speedup: ", min_real/max_real)
    assert max_real*target_ratio >= min_real, "Target latency is too low!"

    random.seed(args.seed)
    np.random.seed(args.seed)
    searcher = MiCoSearcher(
        evaluator, n_inits=args.n_inits, qtypes=bitwidths
    )
    mode = 'latency_proxy' if args.mode == "proxy" else 'bops'

    method = "qat_acc" if args.qat > 0 else "ptq_acc"

    res_x, res_y = searcher.search(
        args.n_iter, method, mode, max_real*target_ratio)

    history = []
    for idx, best_acc in enumerate(searcher.best_trace):
        scheme = searcher.best_scheme_trace[idx] if idx < len(searcher.best_scheme_trace) else None
        constr_val = evaluator.eval_dict()[mode](scheme) if scheme is not None else None
        history.append({
            "iter": idx + 1,
            "accuracy": float(best_acc) if best_acc is not None else None,
            "constraint": float(constr_val) if constr_val is not None else None,
            "scheme": scheme
        })

    os.makedirs("output/json", exist_ok=True)
    dashboard_json = f"output/json/{args.model}_deploy_mico_dashboard.json"
    with open(dashboard_json, "w") as f:
        json.dump({
            "runs": [{
                "method": "mico",
                "seed": args.seed,
                "objective": method,
                "constraint_name": mode,
                "constraint_limit": float(max_real * target_ratio),
                "history": history
            }]
        }, f, indent=2)
    print(f"Dashboard history JSON saved to {dashboard_json}")
        
    print(f"Best Scheme: {res_x}")
    print(f"Best Accuracy: {res_y}")
    if args.mode == "bops":
        max_real = evaluator.eval_latency([8] * dim, target="mico")
        print("INT8 Real Latency: ", max_real)
    print(f"Deploying Model to VexiiMiCo....")
    res = evaluator.eval_latency(res_x, target="mico")
    print(f"MPQ Real Latency: {res}")
    print(f"Real Speedup: {res/max_real}")
