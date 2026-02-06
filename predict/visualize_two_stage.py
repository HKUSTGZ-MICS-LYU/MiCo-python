"""
Visualization script for two-stage proxy comparison results.

Creates plots comparing original vs two-stage proxy performance.
"""

import sys
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MiCoProxy import get_proxy, get_two_stage_proxy


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


def plot_predictions(dataset_name, profile_dataset, kernel_type, output_dir='output/figs'):
    """
    Create prediction comparison plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    X, y = load_dataset(profile_dataset, kernel_type)
    
    # Train proxies (suppress output)
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    original_proxy = get_proxy(profile_dataset, kernel_type)
    two_stage_proxy = get_two_stage_proxy(profile_dataset, kernel_type)
    
    sys.stdout = old_stdout
    
    # Get predictions
    y_pred_original = original_proxy.predict(X)
    y_pred_two_stage = two_stage_proxy.predict(X)
    
    # Separate INT8 vs others
    int8_mask = (X[:, -2] == 8) & (X[:, -1] == 8)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Original proxy
    ax = axes[0]
    ax.scatter(y[~int8_mask], y_pred_original[~int8_mask], 
               alpha=0.6, s=30, c='blue', label='Other precisions')
    ax.scatter(y[int8_mask], y_pred_original[int8_mask], 
               alpha=0.8, s=50, c='red', marker='s', label='INT8')
    
    # Perfect prediction line
    min_val = min(y.min(), y_pred_original.min())
    max_val = max(y.max(), y_pred_original.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2)
    
    ax.set_xlabel('Actual Latency (μs)', fontsize=12)
    ax.set_ylabel('Predicted Latency (μs)', fontsize=12)
    ax.set_title(f'Original Proxy\n{dataset_name}', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Calculate and display metrics
    from sklearn.metrics import mean_absolute_percentage_error, r2_score
    mape = mean_absolute_percentage_error(y, y_pred_original) * 100
    r2 = r2_score(y, y_pred_original)
    ax.text(0.05, 0.95, f'MAPE: {mape:.2f}%\nR²: {r2:.4f}', 
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Two-stage proxy
    ax = axes[1]
    ax.scatter(y[~int8_mask], y_pred_two_stage[~int8_mask], 
               alpha=0.6, s=30, c='green', label='Other precisions')
    ax.scatter(y[int8_mask], y_pred_two_stage[int8_mask], 
               alpha=0.8, s=50, c='orange', marker='s', label='INT8')
    
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2)
    
    ax.set_xlabel('Actual Latency (μs)', fontsize=12)
    ax.set_ylabel('Predicted Latency (μs)', fontsize=12)
    ax.set_title(f'Two-Stage Proxy\n{dataset_name}', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    mape = mean_absolute_percentage_error(y, y_pred_two_stage) * 100
    r2 = r2_score(y, y_pred_two_stage)
    ax.text(0.05, 0.95, f'MAPE: {mape:.2f}%\nR²: {r2:.4f}', 
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    filename = f"{dataset_name.replace(' ', '_')}_{kernel_type}_comparison.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {filepath}")
    plt.close()


def plot_summary_bar_chart(output_dir='output/figs'):
    """Create summary bar chart of MAPE improvements."""
    datasets = [
        ('BitFusion\nMatMul', 2.29, 0.29),
        ('BitFusion\nConv2D', 3.10, 1.96),
        ('MiCo Small\nMatMul', 2.91, 0.91),
        ('MiCo Small\nConv2D', 2.89, 1.28),
        ('MiCo High\nMatMul', 2.92, 1.20),
        ('MiCo High\nConv2D', 2.52, 0.99),
    ]
    
    names = [d[0] for d in datasets]
    original_mape = [d[1] for d in datasets]
    two_stage_mape = [d[2] for d in datasets]
    
    x = np.arange(len(names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, original_mape, width, label='Original Proxy', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, two_stage_mape, width, label='Two-Stage Proxy', color='forestgreen', alpha=0.8)
    
    ax.set_ylabel('MAPE (%)', fontsize=13, fontweight='bold')
    ax.set_title('Prediction Accuracy Comparison: Original vs Two-Stage Proxy', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.legend(fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom', fontsize=9)
    
    # Add improvement percentages
    for i, (name, orig, two) in enumerate(datasets):
        improvement = (orig - two) / orig * 100
        ax.text(i, max(orig, two) + 0.3, f'+{improvement:.0f}%', 
                ha='center', fontsize=10, fontweight='bold', color='darkgreen')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'summary_mape_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved summary plot to: {filepath}")
    plt.close()


def main():
    """Generate all visualizations."""
    datasets = [
        ('BitFusion MatMul', 'benchmark_results/bitfusion_matmul_zoo.csv', 'matmul'),
        ('BitFusion Conv2D', 'benchmark_results/bitfusion_conv2d_zoo.csv', 'conv2d'),
        ('MiCo Small MatMul', 'benchmark_results/mico_small_matmul_zoo.csv', 'matmul'),
        ('MiCo Small Conv2D', 'benchmark_results/mico_small_conv2d_zoo.csv', 'conv2d'),
        ('MiCo High MatMul', 'benchmark_results/mico_high_matmul_zoo.csv', 'matmul'),
        ('MiCo High Conv2D', 'benchmark_results/mico_high_conv2d_zoo.csv', 'conv2d'),
    ]
    
    print("Generating visualizations...")
    print("="*80)
    
    for dataset_name, profile_dataset, kernel_type in datasets:
        print(f"\nProcessing {dataset_name} ({kernel_type})...")
        try:
            plot_predictions(dataset_name, profile_dataset, kernel_type)
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    print("\nGenerating summary bar chart...")
    plot_summary_bar_chart()
    
    print("\n" + "="*80)
    print("All visualizations generated successfully!")
    print("Check output/figs/ directory for the plots.")


if __name__ == "__main__":
    main()
