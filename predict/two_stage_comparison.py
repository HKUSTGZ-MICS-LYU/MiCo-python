"""
Comparison script to evaluate two-stage proxy predictor against original proxy.

This script compares:
1. Original single-stage proxy (all precisions trained together)
2. Two-stage proxy (base INT8 + speedup predictor)

Tests on: BitFusion, MiCo Small, MiCo High datasets
Evaluates: MatMul and Conv2D kernels
"""

import sys
import os
import csv
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import proxy functions
from MiCoProxy import (
    get_proxy, get_two_stage_proxy,
    get_mico_matmul_proxy, get_mico_conv2d_proxy,
    get_mico_matmul_two_stage_proxy, get_mico_conv2d_two_stage_proxy,
    get_bitfusion_matmul_proxy, get_bitfusion_conv2d_proxy,
    get_bitfusion_matmul_two_stage_proxy, get_bitfusion_conv2d_two_stage_proxy
)


def load_dataset(profile_dataset: str, kernel_type: str = 'matmul'):
    """Load and preprocess dataset."""
    with open(profile_dataset, 'r') as f:
        csv_data = csv.reader(f)
        data = []
        next(csv_data)  # skip header
        for row in csv_data:
            data.append(list(map(int, row)))
    data = np.array(data)
    
    if kernel_type == 'matmul':
        N = data[:, 0]
        M = data[:, 1]
        K = data[:, 2]
        QA = data[:, 3]
        QW = data[:, 4]
        latency = data[:, -1]
        MACS = N * M * K
        if 'bitfusion' in profile_dataset:
            MACS = MACS / 16
        RAW = (MACS, M, K, QA, QW)
    elif kernel_type == 'conv2d':
        H, W, C, K, Ks, S = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5]
        QA = data[:, 6]
        QW = data[:, 7]
        latency = data[:, -1]
        H_out = (H - Ks) / S + 1
        W_out = (W - Ks) / S + 1
        MACS = H_out * W_out * C * K * Ks * Ks
        RAW = (MACS, H, W, C, K, Ks, S, QA, QW)
    
    y = latency
    X = np.column_stack(RAW)
    return X, y


def compare_proxies(dataset_name: str, profile_dataset: str, kernel_type: str):
    """
    Compare original vs two-stage proxy on a specific dataset.
    
    Returns:
        dict: Comparison metrics for both approaches
    """
    print(f"\n{'='*80}")
    print(f"Comparing proxies on: {dataset_name} ({kernel_type})")
    print(f"{'='*80}\n")
    
    # Load dataset
    X, y = load_dataset(profile_dataset, kernel_type)
    
    print(f"Dataset stats:")
    print(f"  Total samples: {len(X)}")
    int8_count = np.sum((X[:, -2] == 8) & (X[:, -1] == 8))
    print(f"  INT8 samples (QA=8, QW=8): {int8_count}")
    print(f"  Other precision samples: {len(X) - int8_count}")
    print(f"  Unique QA values: {sorted(set(X[:, -2]))}")
    print(f"  Unique QW values: {sorted(set(X[:, -1]))}")
    
    # Test both approaches with cross-validation
    print("\n" + "-"*80)
    print("ORIGINAL PROXY (Single-stage)")
    print("-"*80)
    original_proxy = get_proxy(profile_dataset, kernel_type)
    
    print("\n" + "-"*80)
    print("TWO-STAGE PROXY")
    print("-"*80)
    two_stage_proxy = get_two_stage_proxy(profile_dataset, kernel_type)
    
    # Full dataset evaluation
    print("\n" + "-"*80)
    print("Full Dataset Evaluation")
    print("-"*80)
    
    # Original proxy
    y_pred_original = original_proxy.predict(X)
    original_metrics = {
        'mape': mean_absolute_percentage_error(y, y_pred_original),
        'r2': r2_score(y, y_pred_original),
        'mae': mean_absolute_error(y, y_pred_original)
    }
    
    # Two-stage proxy
    y_pred_two_stage = two_stage_proxy.predict(X)
    two_stage_metrics = {
        'mape': mean_absolute_percentage_error(y, y_pred_two_stage),
        'r2': r2_score(y, y_pred_two_stage),
        'mae': mean_absolute_error(y, y_pred_two_stage)
    }
    
    print(f"\nOriginal Proxy:   MAPE={original_metrics['mape']*100:6.2f}%, "
          f"R2={original_metrics['r2']:7.4f}, MAE={original_metrics['mae']:.2f}")
    print(f"Two-Stage Proxy:  MAPE={two_stage_metrics['mape']*100:6.2f}%, "
          f"R2={two_stage_metrics['r2']:7.4f}, MAE={two_stage_metrics['mae']:.2f}")
    
    # Calculate improvement
    mape_improvement = (original_metrics['mape'] - two_stage_metrics['mape']) / original_metrics['mape'] * 100
    r2_improvement = (two_stage_metrics['r2'] - original_metrics['r2']) / abs(original_metrics['r2']) * 100
    
    print(f"\nImprovement:")
    print(f"  MAPE: {mape_improvement:+.2f}% ({'better' if mape_improvement > 0 else 'worse'})")
    print(f"  R2:   {r2_improvement:+.2f}% ({'better' if r2_improvement > 0 else 'worse'})")
    
    # Separate evaluation on INT8 vs Other precisions
    print("\n" + "-"*80)
    print("Breakdown by Precision")
    print("-"*80)
    
    int8_mask = (X[:, -2] == 8) & (X[:, -1] == 8)
    X_int8, y_int8 = X[int8_mask], y[int8_mask]
    X_other, y_other = X[~int8_mask], y[~int8_mask]
    
    if len(X_int8) > 0:
        print(f"\nINT8 samples (n={len(X_int8)}):")
        y_pred_orig_int8 = original_proxy.predict(X_int8)
        y_pred_two_int8 = two_stage_proxy.predict(X_int8)
        
        print(f"  Original:   MAPE={mean_absolute_percentage_error(y_int8, y_pred_orig_int8)*100:6.2f}%, "
              f"R2={r2_score(y_int8, y_pred_orig_int8):7.4f}")
        print(f"  Two-Stage:  MAPE={mean_absolute_percentage_error(y_int8, y_pred_two_int8)*100:6.2f}%, "
              f"R2={r2_score(y_int8, y_pred_two_int8):7.4f}")
    
    if len(X_other) > 0:
        print(f"\nOther precision samples (n={len(X_other)}):")
        y_pred_orig_other = original_proxy.predict(X_other)
        y_pred_two_other = two_stage_proxy.predict(X_other)
        
        print(f"  Original:   MAPE={mean_absolute_percentage_error(y_other, y_pred_orig_other)*100:6.2f}%, "
              f"R2={r2_score(y_other, y_pred_orig_other):7.4f}")
        print(f"  Two-Stage:  MAPE={mean_absolute_percentage_error(y_other, y_pred_two_other)*100:6.2f}%, "
              f"R2={r2_score(y_other, y_pred_two_other):7.4f}")
    
    return {
        'dataset': dataset_name,
        'kernel': kernel_type,
        'original': original_metrics,
        'two_stage': two_stage_metrics,
        'mape_improvement': mape_improvement,
        'r2_improvement': r2_improvement
    }


def main():
    """Run comparison on all datasets."""
    datasets = [
        ('BitFusion MatMul', 'benchmark_results/bitfusion_matmul_zoo.csv', 'matmul'),
        ('BitFusion Conv2D', 'benchmark_results/bitfusion_conv2d_zoo.csv', 'conv2d'),
        ('MiCo Small MatMul', 'benchmark_results/mico_small_matmul_zoo.csv', 'matmul'),
        ('MiCo Small Conv2D', 'benchmark_results/mico_small_conv2d_zoo.csv', 'conv2d'),
        ('MiCo High MatMul', 'benchmark_results/mico_high_matmul_zoo.csv', 'matmul'),
        ('MiCo High Conv2D', 'benchmark_results/mico_high_conv2d_zoo.csv', 'conv2d'),
    ]
    
    results = []
    
    for dataset_name, profile_dataset, kernel_type in datasets:
        try:
            result = compare_proxies(dataset_name, profile_dataset, kernel_type)
            results.append(result)
        except Exception as e:
            print(f"\nError processing {dataset_name}: {e}")
            continue
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n{'Dataset':<30} {'Kernel':<10} {'Original MAPE':<15} {'Two-Stage MAPE':<15} {'Improvement':<12}")
    print("-"*80)
    
    for result in results:
        print(f"{result['dataset']:<30} {result['kernel']:<10} "
              f"{result['original']['mape']*100:6.2f}%{'':<8} "
              f"{result['two_stage']['mape']*100:6.2f}%{'':<8} "
              f"{result['mape_improvement']:+6.2f}%")
    
    # Overall statistics
    avg_improvement = np.mean([r['mape_improvement'] for r in results])
    print("-"*80)
    print(f"Average MAPE Improvement: {avg_improvement:+.2f}%")
    
    better_count = sum(1 for r in results if r['mape_improvement'] > 0)
    print(f"Two-stage better in {better_count}/{len(results)} cases")


if __name__ == "__main__":
    main()
