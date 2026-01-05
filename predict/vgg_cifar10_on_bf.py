import torch
import random
import numpy as np

from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, r2_score

from MiCoEval import MiCoEval
from MiCoProxy import get_bitfusion_matmul_proxy, get_bitfusion_conv2d_proxy

from models import VGG
from datasets import cifar10

from tqdm import tqdm

from searchers import (
    RegressionSearcher, BayesSearcher, 
    NLPSearcher, HAQSearcher, MiCoSearcher
)

N = 64

random.seed(0)
np.random.seed(0)

if __name__ == "__main__":

    model = VGG(3, 10)
    train_loader, test_loader = cifar10(shuffle=False, num_works=8)
    evaluator = MiCoEval(model, 1, train_loader, test_loader, 
                         "output/ckpt/vgg_cifar10.pth",
                         model_name="vgg",
                         output_json="output/json/vgg_cifar10_latency_bitfusion2.json",)
    

    matmul_proxy = get_bitfusion_matmul_proxy()
    conv2d_proxy = get_bitfusion_conv2d_proxy()
    evaluator.set_proxy(matmul_proxy, conv2d_proxy)

    dim = evaluator.n_layers * 2
    bitwidths = [2, 3, 4, 5, 6, 7, 8]

    evaluator.set_eval('latency_bitfusion')
    X = []
    X_bops = []
    Y = []
    rand_scheme = [random.choices(bitwidths, k=dim) for _ in range(N)]
    min_scheme = [[min(bitwidths) for _ in range(dim)]]
    max_scheme = [[max(bitwidths) for _ in range(dim)]]
    rand_scheme = min_scheme + max_scheme + rand_scheme
    for scheme in tqdm(rand_scheme):
        res = evaluator.eval(scheme)

        res_pred = evaluator.eval_pred_latency(scheme)
        X.append(res)
        X_bops.append(evaluator.eval_bops(scheme))
        Y.append(res_pred)
        
        print(scheme)
        print(res)
        print(res_pred)

    X = np.array(X)
    # X = np.array(X_bops)
    Y = np.array(Y)

    plt.figure(figsize=(6,6))
    plt.scatter(Y, X, alpha=0.5)
    
    # Plot Fit Line
    r2 = r2_score(X, Y)
    mape = mean_absolute_percentage_error(X, Y)
    print("R2:", r2)
    print("MAPE:", mape)

    # Two Shot Correction
    X_tune = X[:2]
    Y_tune = Y[:2]

    plt.plot(X, X, color='red', label='Ideal Prediction')
    plt.xlabel("Actual Latency (Cycles)")
    plt.ylabel("Predicted Latency (Cycles)")
    plt.legend()
    plt.grid(True)
    plt.savefig("output/figs/vgg_cifar10_bitfusion_latency_pred.pdf")
