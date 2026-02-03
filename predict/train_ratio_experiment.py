"""
Experiment script to analyze the impact of train_ratio on end-to-end MPQ prediction accuracy.

This script varies the train_ratio parameter in MiCoProxy and measures how it affects
the prediction quality (R2, MAPE, Spearman correlation) for different models and targets.

Usage:
    python predict/train_ratio_experiment.py lenet_mnist --target latency_mico_small -n 32
    python predict/train_ratio_experiment.py resnet8_cifar10 --target latency_bitfusion -n 64
"""

import random
import numpy as np
from matplotlib import pyplot as plt

from sklearn.metrics import mean_absolute_percentage_error, r2_score, root_mean_squared_error
from scipy.stats import spearmanr

from MiCoEval import MiCoEval
from MiCoProxy import (
    get_bitfusion_matmul_proxy, get_bitfusion_conv2d_proxy,
    get_mico_matmul_proxy, get_mico_conv2d_proxy, get_mico_misc_kernel_proxy
)

from models import model_zoo

from tqdm import tqdm
import argparse


def run_prediction_experiment(evaluator, rand_scheme, target):
    """Run prediction experiment and return actual vs predicted latencies."""
    X = []
    Y = []
    for scheme in tqdm(rand_scheme, desc="Evaluating schemes"):
        res = evaluator.eval(scheme)
        res_pred = evaluator.eval_pred_latency(scheme)
        X.append(res)
        Y.append(res_pred)

    X, Y = np.array(X), np.array(Y)

    # Min/Max Correction
    min_X, max_X = X[0], X[1]
    min_Y, max_Y = Y[0], Y[1]
    
    slope = (max_X - min_X) / (max_Y - min_Y)
    intercept = min_X - slope * min_Y
    Y = slope * Y + intercept
    return X, Y


def compute_metrics(X, Y):
    """Compute prediction quality metrics."""
    r2 = r2_score(X, Y)
    norm_r2 = r2_score(X / np.max(X), Y / np.max(Y))
    mape = mean_absolute_percentage_error(X, Y)
    rmse = root_mean_squared_error(X, Y)
    spears_corr, _ = spearmanr(X, Y)
    
    correlation_matrix = np.corrcoef(X, Y)
    pearson_r = correlation_matrix[0, 1]
    
    return {
        'r2': r2,
        'norm_r2': norm_r2,
        'mape': mape,
        'rmse': rmse,
        'spearman': spears_corr,
        'pearson': pearson_r
    }


def setup_evaluator_with_train_ratio(model_name, target, train_ratio, seed=42):
    """Setup evaluator and proxies with specified train_ratio."""
    model, train_loader, test_loader = model_zoo.from_zoo(model_name)
    evaluator = MiCoEval(model, 1, train_loader, test_loader, 
                         f"output/ckpt/{model_name}.pth",
                         model_name=model_name,
                         output_json=f"output/json/{model_name}_{target}_predict.json")
    
    if target.startswith("latency_mico"):
        mico_target = target.split("_")[-1]
        evaluator.set_mico_target(mico_target)
        # Get proxies with specified train_ratio
        matmul_proxy = get_mico_matmul_proxy(mico_type=mico_target)
        conv2d_proxy = get_mico_conv2d_proxy(mico_type=mico_target)
        # Re-fit proxies with the specified train_ratio
        matmul_proxy.set_train_ratio(train_ratio)
        matmul_proxy.seed = seed
        matmul_proxy.fit()
        conv2d_proxy.set_train_ratio(train_ratio)
        conv2d_proxy.seed = seed
        conv2d_proxy.fit()
        evaluator.set_proxy(matmul_proxy, conv2d_proxy)
        evaluator.set_misc_proxy(get_mico_misc_kernel_proxy)
        bitwidths = [1, 2, 4, 8]
        evaluator.set_eval("latency_mico")

    elif target == "latency_bitfusion":
        matmul_proxy = get_bitfusion_matmul_proxy()
        conv2d_proxy = get_bitfusion_conv2d_proxy()
        # Re-fit proxies with the specified train_ratio
        matmul_proxy.set_train_ratio(train_ratio)
        matmul_proxy.seed = seed
        matmul_proxy.fit()
        conv2d_proxy.set_train_ratio(train_ratio)
        conv2d_proxy.seed = seed
        conv2d_proxy.fit()
        evaluator.set_proxy(matmul_proxy, conv2d_proxy)
        bitwidths = [2, 3, 4, 5, 6, 7, 8]
        evaluator.set_eval("latency_bitfusion")
    else:
        raise ValueError(f"Unsupported target: {target}")
    
    return evaluator, bitwidths


def main():
    parser = argparse.ArgumentParser(description='Train ratio impact experiment for E2E MPQ prediction')
    parser.add_argument('model', type=str, help='Model name from model_zoo')
    parser.add_argument('-n', '--num-samples', type=int, default=64, 
                        help='Number of random samples to evaluate')
    parser.add_argument('--target', type=str, default='latency_bitfusion', 
                        help='Target metric (latency_bitfusion, latency_mico_small, latency_mico_high)')
    parser.add_argument('--ratios', type=str, default='0.2,0.4,0.6,0.8,1.0',
                        help='Comma-separated list of train_ratios to test')
    parser.add_argument('--seeds', type=int, default=5, 
                        help='Number of random seeds for each train_ratio')
    parser.add_argument('--save-fig', action='store_true', 
                        help='Save figure to output/figs/')
    args = parser.parse_args()

    model_name = args.model
    target = args.target
    N = args.num_samples
    train_ratios = [float(r) for r in args.ratios.split(',')]
    num_seeds = args.seeds

    print(f"\n{'='*80}")
    print(f"Train Ratio Impact Experiment")
    print(f"Model: {model_name}, Target: {target}")
    print(f"Train Ratios: {train_ratios}")
    print(f"Num Seeds: {num_seeds}, Num Samples: {N}")
    print(f"{'='*80}\n")

    # Generate fixed random schemes for fair comparison
    random.seed(0)
    np.random.seed(0)
    
    # Get model dimension from a dummy evaluator
    model, train_loader, test_loader = model_zoo.from_zoo(model_name)
    dim = model.n_layers * 2
    
    if target == "latency_bitfusion":
        bitwidths = [2, 3, 4, 5, 6, 7, 8]
    elif target.startswith("latency_mico"):
        bitwidths = [1, 2, 4, 8]
    else:
        raise ValueError(f"Unsupported target: {target}")
    
    rand_scheme = [random.choices(bitwidths, k=dim) for _ in range(N)]
    min_scheme = [[min(bitwidths) for _ in range(dim)]]
    max_scheme = [[max(bitwidths) for _ in range(dim)]]
    all_schemes = min_scheme + max_scheme + rand_scheme

    # Store results for each train_ratio
    results = {ratio: {'r2': [], 'norm_r2': [], 'mape': [], 'rmse': [], 
                       'spearman': [], 'pearson': []} for ratio in train_ratios}

    for train_ratio in train_ratios:
        print(f"\n--- Train Ratio: {train_ratio} ---")
        
        for seed in range(num_seeds):
            print(f"  Seed {seed}...")
            evaluator, _ = setup_evaluator_with_train_ratio(
                model_name, target, train_ratio, seed=seed
            )
            
            X, Y = run_prediction_experiment(evaluator, all_schemes, target)
            
            # Skip first two (min/max) for metrics
            X_eval = X[2:]
            Y_eval = Y[2:]
            
            metrics = compute_metrics(X_eval, Y_eval)
            
            for key in metrics:
                results[train_ratio][key].append(metrics[key])
            
            print(f"    R2: {metrics['r2']:.4f}, MAPE: {metrics['mape']*100:.2f}%, "
                  f"Spearman: {metrics['spearman']:.4f}")
            if train_ratio >= 1.0:
                break # No need to repeat for full ratio
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Train Ratio':<12} {'R2 (mean±std)':<20} {'MAPE% (mean±std)':<20} {'Spearman (mean±std)':<20}")
    print("-" * 80)
    
    summary_data = []
    for train_ratio in train_ratios:
        r2_mean = np.mean(results[train_ratio]['norm_r2'])
        r2_std = np.std(results[train_ratio]['norm_r2'])
        mape_mean = np.mean(results[train_ratio]['mape']) * 100
        mape_std = np.std(results[train_ratio]['mape']) * 100
        spearman_mean = np.mean(results[train_ratio]['spearman'])
        spearman_std = np.std(results[train_ratio]['spearman'])
        
        print(f"{train_ratio:<12.2f} {r2_mean:>7.4f}±{r2_std:<8.4f} "
              f"{mape_mean:>7.2f}±{mape_std:<8.2f} "
              f"{spearman_mean:>7.4f}±{spearman_std:<8.4f}")
        
        summary_data.append({
            'train_ratio': train_ratio,
            'r2_mean': r2_mean, 'r2_std': r2_std,
            'mape_mean': mape_mean, 'mape_std': mape_std,
            'spearman_mean': spearman_mean, 'spearman_std': spearman_std
        })

    # Plot results
    if args.save_fig:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        x = train_ratios
        
        # R2 plot
        r2_means = [d['r2_mean'] for d in summary_data]
        r2_stds = [d['r2_std'] for d in summary_data]
        axes[0].errorbar(x, r2_means, yerr=r2_stds, marker='o', capsize=5)
        axes[0].set_xlabel('Train Ratio')
        axes[0].set_ylabel('R²')
        axes[0].set_title('R² vs Train Ratio')
        axes[0].grid(True)
        
        # MAPE plot
        mape_means = [d['mape_mean'] for d in summary_data]
        mape_stds = [d['mape_std'] for d in summary_data]
        axes[1].errorbar(x, mape_means, yerr=mape_stds, marker='o', capsize=5, color='orange')
        axes[1].set_xlabel('Train Ratio')
        axes[1].set_ylabel('MAPE (%)')
        axes[1].set_title('MAPE vs Train Ratio')
        axes[1].grid(True)
        
        # Spearman plot
        spearman_means = [d['spearman_mean'] for d in summary_data]
        spearman_stds = [d['spearman_std'] for d in summary_data]
        axes[2].errorbar(x, spearman_means, yerr=spearman_stds, marker='o', capsize=5, color='green')
        axes[2].set_xlabel('Train Ratio')
        axes[2].set_ylabel("Spearman's Correlation")
        axes[2].set_title("Spearman's Correlation vs Train Ratio")
        axes[2].grid(True)
        
        plt.tight_layout()
        fig_path = f"output/figs/{model_name}_{target}_train_ratio_exp.pdf"
        plt.savefig(fig_path)
        print(f"\nFigure saved to: {fig_path}")

    print(f"\n{'='*80}")
    print("Experiment completed!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
