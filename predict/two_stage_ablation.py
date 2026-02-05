"""
Ablation study for two-stage proxy predictor.

Experiments:
1. Train data ratio: 0.4, 0.6, 0.8, 1.0
2. Feature selection: with/without CBOPs features

This provides insights into:
- Data efficiency of two-stage approach
- Impact of CBOPs features on prediction accuracy
"""

import sys
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MiCoProxy import (
    TwoStageProxy, LogXGBRegressor, LogRandomForestRegressor
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


def train_ratio_ablation(dataset_name, profile_dataset, kernel_type):
    """
    Ablation study on train data ratio.
    Tests: 0.4, 0.6, 0.8, 1.0
    """
    print(f"\n{'='*80}")
    print(f"Train Ratio Ablation: {dataset_name} ({kernel_type})")
    print(f"{'='*80}\n")
    
    X, y = load_dataset(profile_dataset, kernel_type)
    
    # Test different train ratios
    train_ratios = [0.4, 0.6, 0.8, 1.0]
    results = []
    
    print(f"Dataset size: {len(X)} samples\n")
    print(f"{'Train Ratio':<15} {'Train Size':<12} {'MAPE (%)':<12} {'R²':<10} {'MAE':<12}")
    print("-"*80)
    
    for train_ratio in train_ratios:
        # Create two-stage proxy with train ratio
        proxy = TwoStageProxy(
            LogXGBRegressor(random_state=42),
            LogXGBRegressor(random_state=42),
            base_preprocess='raw',
            speedup_preprocess='cbops+',
            train_ratio=train_ratio,
            seed=42
        )
        
        # Train
        proxy.fit(X, y)
        
        # Evaluate on full dataset
        y_pred = proxy.predict(X)
        
        mape = mean_absolute_percentage_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        train_size = int(len(X) * train_ratio)
        
        print(f"{train_ratio:<15.1f} {train_size:<12} {mape*100:<12.2f} {r2:<10.4f} {mae:<12.2f}")
        
        results.append({
            'train_ratio': train_ratio,
            'train_size': train_size,
            'mape': mape,
            'r2': r2,
            'mae': mae
        })
    
    return results


def feature_ablation(dataset_name, profile_dataset, kernel_type):
    """
    Ablation study on feature selection.
    Tests: raw, bops, bops+, cbops, cbops+
    """
    print(f"\n{'='*80}")
    print(f"Feature Selection Ablation: {dataset_name} ({kernel_type})")
    print(f"{'='*80}\n")
    
    X, y = load_dataset(profile_dataset, kernel_type)
    
    # Test different speedup feature sets
    feature_sets = ['raw', 'bops', 'bops+', 'cbops', 'cbops+']
    results = []
    
    print(f"Base features: raw (all features except QA, QW)")
    print(f"Speedup features: Testing different feature sets\n")
    
    print(f"{'Speedup Features':<20} {'MAPE (%)':<12} {'R²':<10} {'MAE':<12} {'vs cbops+':<12}")
    print("-"*80)
    
    cbops_plus_mape = None
    
    for features in feature_sets:
        # Create two-stage proxy with different features
        proxy = TwoStageProxy(
            LogXGBRegressor(random_state=42),
            LogXGBRegressor(random_state=42),
            base_preprocess='raw',
            speedup_preprocess=features,
            train_ratio=1.0,
            seed=42
        )
        
        # Train
        proxy.fit(X, y)
        
        # Evaluate
        y_pred = proxy.predict(X)
        
        mape = mean_absolute_percentage_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        if features == 'cbops+':
            cbops_plus_mape = mape
        
        # Calculate difference vs cbops+
        if cbops_plus_mape is not None:
            diff = ((mape - cbops_plus_mape) / cbops_plus_mape) * 100
            diff_str = f"{diff:+.1f}%"
        else:
            diff_str = "-"
        
        print(f"{features:<20} {mape*100:<12.2f} {r2:<10.4f} {mae:<12.2f} {diff_str:<12}")
        
        results.append({
            'features': features,
            'mape': mape,
            'r2': r2,
            'mae': mae
        })
    
    return results


def combined_ablation(dataset_name, profile_dataset, kernel_type):
    """
    Combined ablation: train ratio × feature selection.
    Focus on key combinations.
    """
    print(f"\n{'='*80}")
    print(f"Combined Ablation: {dataset_name} ({kernel_type})")
    print(f"{'='*80}\n")
    
    X, y = load_dataset(profile_dataset, kernel_type)
    
    # Test combinations
    train_ratios = [0.4, 0.6, 0.8, 1.0]
    feature_sets = ['raw', 'bops+', 'cbops+']  # Focus on key features
    
    print(f"{'Train Ratio':<15} {'Features':<15} {'MAPE (%)':<12} {'R²':<10}")
    print("-"*80)
    
    results = {}
    
    for features in feature_sets:
        results[features] = []
        for train_ratio in train_ratios:
            proxy = TwoStageProxy(
                LogXGBRegressor(random_state=42),
                LogXGBRegressor(random_state=42),
                base_preprocess='raw',
                speedup_preprocess=features,
                train_ratio=train_ratio,
                seed=42
            )
            
            proxy.fit(X, y)
            y_pred = proxy.predict(X)
            
            mape = mean_absolute_percentage_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            print(f"{train_ratio:<15.1f} {features:<15} {mape*100:<12.2f} {r2:<10.4f}")
            
            results[features].append({
                'train_ratio': train_ratio,
                'mape': mape,
                'r2': r2
            })
    
    return results


def plot_train_ratio_results(all_results, output_dir='output/figs'):
    """Plot train ratio ablation results."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: MAPE vs Train Ratio
    ax = axes[0]
    for dataset_name, results in all_results.items():
        train_ratios = [r['train_ratio'] for r in results]
        mapes = [r['mape'] * 100 for r in results]
        ax.plot(train_ratios, mapes, marker='o', linewidth=2, label=dataset_name)
    
    ax.set_xlabel('Train Data Ratio', fontsize=12)
    ax.set_ylabel('MAPE (%)', fontsize=12)
    ax.set_title('Effect of Training Data Ratio on Accuracy', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0])
    
    # Plot 2: R² vs Train Ratio
    ax = axes[1]
    for dataset_name, results in all_results.items():
        train_ratios = [r['train_ratio'] for r in results]
        r2s = [r['r2'] for r in results]
        ax.plot(train_ratios, r2s, marker='o', linewidth=2, label=dataset_name)
    
    ax.set_xlabel('Train Data Ratio', fontsize=12)
    ax.set_ylabel('R²', fontsize=12)
    ax.set_title('Effect of Training Data Ratio on R²', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0])
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'train_ratio_ablation.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\nSaved train ratio ablation plot to: {filepath}")
    plt.close()


def plot_feature_results(all_results, output_dir='output/figs'):
    """Plot feature selection ablation results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    datasets = list(all_results.keys())
    feature_sets = ['raw', 'bops', 'bops+', 'cbops', 'cbops+']
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(datasets))
    width = 0.15
    
    for i, features in enumerate(feature_sets):
        mapes = []
        for dataset in datasets:
            # Find MAPE for this feature set
            result = next((r for r in all_results[dataset] if r['features'] == features), None)
            if result:
                mapes.append(result['mape'] * 100)
            else:
                mapes.append(0)
        
        ax.bar(x + i * width, mapes, width, label=features, alpha=0.8)
    
    ax.set_ylabel('MAPE (%)', fontsize=12, fontweight='bold')
    ax.set_title('Feature Selection Ablation: Impact on Accuracy', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(datasets, rotation=15, ha='right', fontsize=10)
    ax.legend(title='Speedup Features', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'feature_ablation.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved feature ablation plot to: {filepath}")
    plt.close()


def main():
    """Run all ablation studies."""
    datasets = [
        ('BitFusion MatMul', 'benchmark_results/bitfusion_matmul_zoo.csv', 'matmul'),
        ('MiCo Small MatMul', 'benchmark_results/mico_small_matmul_zoo.csv', 'matmul'),
        ('MiCo High Conv2D', 'benchmark_results/mico_high_conv2d_zoo.csv', 'conv2d'),
    ]
    
    print("\n" + "="*80)
    print(" TWO-STAGE PROXY ABLATION STUDY")
    print("="*80)
    
    # 1. Train Ratio Ablation
    print("\n" + "="*80)
    print(" EXPERIMENT 1: TRAIN DATA RATIO ABLATION")
    print("="*80)
    
    train_ratio_results = {}
    for dataset_name, profile_dataset, kernel_type in datasets:
        results = train_ratio_ablation(dataset_name, profile_dataset, kernel_type)
        train_ratio_results[dataset_name] = results
    
    # 2. Feature Selection Ablation
    print("\n" + "="*80)
    print(" EXPERIMENT 2: FEATURE SELECTION ABLATION")
    print("="*80)
    
    feature_results = {}
    for dataset_name, profile_dataset, kernel_type in datasets:
        results = feature_ablation(dataset_name, profile_dataset, kernel_type)
        feature_results[dataset_name] = results
    
    # 3. Combined Ablation (optional, for key combinations)
    print("\n" + "="*80)
    print(" EXPERIMENT 3: COMBINED ABLATION (Key Combinations)")
    print("="*80)
    
    for dataset_name, profile_dataset, kernel_type in datasets:
        combined_ablation(dataset_name, profile_dataset, kernel_type)
    
    # Generate plots
    print("\n" + "="*80)
    print(" GENERATING VISUALIZATIONS")
    print("="*80)
    
    plot_train_ratio_results(train_ratio_results)
    plot_feature_results(feature_results)
    
    # Summary
    print("\n" + "="*80)
    print(" SUMMARY OF KEY FINDINGS")
    print("="*80)
    
    print("\n1. TRAIN DATA RATIO:")
    print("   - Effect of reducing training data from 100% to 40%")
    for dataset_name, results in train_ratio_results.items():
        mape_100 = results[-1]['mape'] * 100  # 1.0 ratio
        mape_40 = results[0]['mape'] * 100   # 0.4 ratio
        degradation = ((mape_40 - mape_100) / mape_100) * 100
        print(f"   - {dataset_name}: {mape_100:.2f}% → {mape_40:.2f}% ({degradation:+.1f}%)")
    
    print("\n2. FEATURE SELECTION:")
    print("   - Effect of different feature sets on accuracy")
    for dataset_name, results in feature_results.items():
        cbops_plus = next(r for r in results if r['features'] == 'cbops+')
        raw = next(r for r in results if r['features'] == 'raw')
        improvement = ((raw['mape'] - cbops_plus['mape']) / raw['mape']) * 100
        print(f"   - {dataset_name}: cbops+ vs raw: {improvement:+.1f}% better")
    
    print("\n" + "="*80)
    print(" ABLATION STUDY COMPLETE")
    print("="*80)
    print("\nResults saved to output/figs/")
    print("- train_ratio_ablation.png")
    print("- feature_ablation.png")


if __name__ == "__main__":
    main()
