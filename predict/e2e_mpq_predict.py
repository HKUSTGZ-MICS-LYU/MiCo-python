import random
import numpy as np

from matplotlib import pyplot as plt

from sklearn.metrics import mean_absolute_percentage_error, r2_score, root_mean_squared_error
from scipy.stats import pearsonr, spearmanr

from MiCoEval import MiCoEval
from MiCoProxy import (
    get_bitfusion_matmul_proxy, get_bitfusion_conv2d_proxy,
    get_mico_matmul_proxy, get_mico_conv2d_proxy, get_mico_misc_kernel_proxy
)

from models import model_zoo

from tqdm import tqdm

import argparse

random.seed(0)
np.random.seed(0)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('model', type=str, help='Model name from model_zoo')
    args.add_argument('-n', '--num-samples', type=int, default=64, help='Number of random samples to evaluate')
    args.add_argument('--target', type=str, default='latency_bitfusion', help='Target metric to evaluate')
    args.add_argument('--plot', type=str, default='norm', help='Plot Actual vs Predicted (norm/abs)')
    args.add_argument('--calib', action='store_true', help='Whether to perform one-shot scaling calibration')
    parsed_args = args.parse_args()

    model_name = parsed_args.model
    target = parsed_args.target
    N = parsed_args.num_samples

    model, train_loader, test_loader = model_zoo.from_zoo(model_name)
    evaluator = MiCoEval(model, 1, train_loader, test_loader, 
                     f"output/ckpt/{model_name}.pth",
                     model_name=model_name,
                     output_json=f"output/json/{model_name}_{target}_predict.json")
    
    if target.startswith("latency_mico"):
        mico_target = target.split("_")[-1]
        evaluator.set_mico_target(mico_target)
        matmul_proxy = get_mico_matmul_proxy(mico_type=mico_target)
        conv2d_proxy = get_mico_conv2d_proxy(mico_type=mico_target)
        evaluator.set_proxy(matmul_proxy, conv2d_proxy)
        evaluator.set_misc_proxy(get_mico_misc_kernel_proxy)

    elif target == "latency_bitfusion":
        matmul_proxy = get_bitfusion_matmul_proxy()
        conv2d_proxy = get_bitfusion_conv2d_proxy()
        evaluator.set_proxy(matmul_proxy, conv2d_proxy)

    dim = evaluator.n_layers * 2
    bitwidths = []
    if target == "latency_bitfusion":
        bitwidths = [2, 3, 4, 5, 6, 7, 8]
        evaluator.set_eval("latency_bitfusion")
    elif target.startswith("latency_mico"):
        bitwidths = [1, 2, 4, 8]
        evaluator.set_eval("latency_mico")

    X = []
    Y = []
    rand_scheme = [random.choices(bitwidths, k=dim) for _ in range(N)]
    min_scheme = [[min(bitwidths) for _ in range(dim)]]
    max_scheme = [[max(bitwidths) for _ in range(dim)]]
    rand_scheme = min_scheme + max_scheme + rand_scheme
    for scheme in tqdm(rand_scheme):
        res = evaluator.eval(scheme)

        res_pred = evaluator.eval_pred_latency(scheme)
        X.append(res)
        Y.append(res_pred)
        # print(scheme)
        # print(f"Actual  Latency:{res:>10}")
        # print(f"Predict Latency:{int(res_pred):>10}")

    X = np.array(X)
    Y = np.array(Y)

    # Use Baselines to correct the prediction bias
    X_min = X[0]
    Y_min = Y[0]

    X_max = X[1]
    Y_max = Y[1]

    # One-shot Scaling
    # bias = X_max / Y_max
    # Y = Y * bias

    # Two-shot Scaling
    if parsed_args.calib:
        slope = (X_max - X_min) / (Y_max - Y_min)
        intercept = X_min - slope * Y_min
        Y = slope * Y + intercept

    ratio_X = X / np.max(X)
    ratio_Y = Y / np.max(Y)

    plt.figure(figsize=(6,6))

    if parsed_args.plot == 'abs':

        plt.scatter(X, Y, alpha=0.5)
        plt.plot(X, X, color='red', label='Ideal Prediction')
        plt.xlabel("Actual Latency")
        plt.ylabel("Predicted Latency")

    elif parsed_args.plot == 'norm':

        plt.scatter(ratio_X, ratio_Y, alpha=0.5)
        plt.plot(ratio_X, ratio_X, color='red', label='Ideal Prediction')
        plt.xlabel("Normalized Actual Latency")
        plt.ylabel("Normalized Predicted Latency")

    
    plt.legend()
    plt.grid(True)
    plt.savefig(f"output/figs/{model_name}_{target}_pred.pdf")

    # Ignore first two data
    X = X[2:]
    Y = Y[2:]

    r2 = r2_score(X, Y)
    norm_r2 = r2_score(X / np.max(X), Y / np.max(Y))
    mape = mean_absolute_percentage_error(X, Y)
    rmse = root_mean_squared_error(X, Y)

    print("R2:", r2)
    print("Ratio R2:", norm_r2)
    print("MAPE:", mape)
    print("RMSE:", rmse)

    correlation_matrix = np.corrcoef(X, Y)
    correlation_xy = correlation_matrix[0,1]
    print("Correlation Coefficient (R):", correlation_xy)
    correlation_matrix = np.corrcoef(X / np.max(X), Y / np.max(Y))
    correlation_xy = correlation_matrix[0,1]
    print("Normalized Correlation Coefficient (R):", correlation_xy)

    spears_corr, _ = spearmanr(X, Y)
    print("Spearman's Rank Correlation Coefficient:", spears_corr)

    # Get Maximum Error
    errors = (Y - X) / X
    max_error = np.max(np.abs(errors))
    print("Max Error:", max_error)

    # Count Positive/Negative Errors
    pos_count = np.sum(errors > 0)
    neg_count = np.sum(errors < 0)
    print("Positive Errors:", pos_count)
    print("Negative Errors:", neg_count)

    error_upper = np.max(errors)
    error_lower = np.min(errors)

    print("Error Upper Bound:", error_upper)
    print("Error Lower Bound:", error_lower)

