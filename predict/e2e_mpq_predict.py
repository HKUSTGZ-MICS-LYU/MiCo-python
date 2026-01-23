import random
import numpy as np

from matplotlib import pyplot as plt

from sklearn.metrics import mean_absolute_percentage_error, r2_score, root_mean_squared_error

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
    parsed_args = args.parse_args()

    model_name = parsed_args.model
    target = parsed_args.target
    N = parsed_args.num_samples

    model, train_loader, test_loader = model_zoo.from_zoo(model_name)
    evaluator = MiCoEval(model, 1, train_loader, test_loader, 
                     f"output/ckpt/{model_name}.pth",
                     model_name=model_name,
                     output_json=f"output/json/{model_name}_{target}_predict.json")
    
    if target == "latency_mico":
        evaluator.set_mico_target('small')
        matmul_proxy = get_mico_matmul_proxy(mico_type='small')
        conv2d_proxy = get_mico_conv2d_proxy(mico_type='small')
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
    elif target == "latency_mico":
        bitwidths = [1, 2, 4, 8]

    evaluator.set_eval(target)

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
        
        print(scheme)
        print(f"Actual  Latency:{res:>10}")
        print(f"Predict Latency:{int(res_pred):>10}")

    X = np.array(X)
    Y = np.array(Y)

    ratio_X = X / np.max(X)
    ratio_Y = Y / np.max(Y)

    plt.figure(figsize=(6,6))
    plt.scatter(ratio_X, ratio_Y, alpha=0.5)

    plt.plot(ratio_X, ratio_X, color='red', label='Ideal Prediction')
    plt.xlabel("Normalized Actual Latency")
    plt.ylabel("Normalized Predicted Latency")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"output/figs/{model_name}_{target}_pred.pdf")

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

