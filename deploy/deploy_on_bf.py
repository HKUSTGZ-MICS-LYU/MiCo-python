import random
import argparse
import numpy as np

from matplotlib import pyplot as plt

from MiCoEval import MiCoEval
from MiCoProxy import get_bitfusion_matmul_proxy, get_bitfusion_conv2d_proxy

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
    parser.add_argument("-c","--target_ratio", type=float, default=0.6)
    parser.add_argument("-s","--seed", type=int, default=0)
    parser.add_argument("-n","--n_inits", type=int, default=10)
    parser.add_argument("-i","--n_iter", type=int, default=10)
    parser.add_argument("--mode", type=str, default="proxy", choices=["proxy", "bops"])
    args = parser.parse_args()

    model, train_loader, test_loader = model_zoo.from_zoo(args.model)
    evaluator = MiCoEval(model, 1, train_loader, test_loader, 
                         f"output/ckpt/{args.model}.pth",
                         output_json=f"output/json/{args.model}_search.json",)
    

    matmul_proxy = get_bitfusion_matmul_proxy('raw', regressor="LogXGBRegressor")
    conv2d_proxy = get_bitfusion_conv2d_proxy('raw', regressor="LogXGBRegressor")
    evaluator.set_proxy(matmul_proxy, conv2d_proxy)

    dim = model.n_layers * 2
    bitwidths = [4, 5, 6, 7, 8]
    target_ratio = args.target_ratio

    # Calibration 
    if args.mode == "proxy":
        min_real, max_real = evaluator.calibrate_proxy(min_q=4, max_q=8, target="bitfusion")
    elif args.mode == "bops":
        max_real = evaluator.eval_bops([8] * dim)
        min_real = evaluator.eval_bops([4] * dim)
    print("Maximum Speedup: ", min_real/max_real)
    assert max_real*target_ratio >= min_real, "Target latency is too low!"

    random.seed(args.seed)
    np.random.seed(args.seed)
    searcher = MiCoSearcher(
        evaluator, n_inits=args.n_inits, qtypes=bitwidths
    )
    mode = 'latency_proxy' if args.mode == "proxy" else 'bops'

    res_x, res_y = searcher.search(
        args.n_iter, 'ptq_acc', mode, max_real*target_ratio)
        
    print(f"Best Scheme: {res_x}")
    print(f"Best Accuracy: {res_y}")
    if args.mode == "bops":
        max_real = evaluator.eval_latency([8] * dim, target="bitfusion")
    print(f"Deploying Model to BitFusion....")
    res = evaluator.eval_latency(res_x, target="bitfusion")
    print(f"MPQ Real Latency: {res}")
    print(f"Real Speedup: {res/max_real}")