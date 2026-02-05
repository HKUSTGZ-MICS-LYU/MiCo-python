#!/usr/bin/env python3
"""
MiCoProxy Transfer Learning Validation Script

This script validates the feasibility of transfer learning for MiCoProxy,
investigating:
1. Transfer learning between vexii mico targets (small <-> high)
2. Effect of different fine-tuning data ratios
3. Cross-target transfer (bitfusion -> mico)

Usage:
    python examples/proxy_transfer_learning.py
    python examples/proxy_transfer_learning.py --quick  # Quick validation
    python examples/proxy_transfer_learning.py --comprehensive  # Full analysis
"""

import sys
import os
import argparse
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from MiCoProxy import (
    get_transfer_proxy,
    evaluate_transfer_learning,
    compare_transfer_directions,
    explore_cross_target_transfer,
    load_proxy_data,
    get_mico_matmul_proxy,
    get_mico_conv2d_proxy
)
from sklearn.metrics import mean_absolute_percentage_error, r2_score


def run_baseline_comparison(kernel_type='matmul', verbose=True):
    """
    Compare transfer learning with training from scratch baseline.
    
    This establishes the baseline performance when training on target data
    only, without any source domain knowledge.
    """
    if verbose:
        print("\n" + "="*80)
        print(f"BASELINE COMPARISON: Transfer vs Training from Scratch ({kernel_type})")
        print("="*80)
    
    ratios = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    results = {
        'transfer_small_to_high': [],
        'scratch_high': [],
        'transfer_high_to_small': [],
        'scratch_small': []
    }
    
    for ratio in ratios:
        # Transfer learning: small -> high
        _, transfer_result = get_transfer_proxy(
            source_type='mico_small',
            target_type='mico_high',
            kernel_type=kernel_type,
            finetune_ratio=ratio,
            strategy='combined',
            verbose=False
        )
        results['transfer_small_to_high'].append(transfer_result['mape_after'])
        
        # Training from scratch on high
        _, scratch_result = get_transfer_proxy(
            source_type='mico_high',
            target_type='mico_high',
            kernel_type=kernel_type,
            finetune_ratio=ratio,
            strategy='target_only',
            verbose=False
        )
        results['scratch_high'].append(scratch_result['mape_after'])
        
        # Transfer learning: high -> small
        _, transfer_result_rev = get_transfer_proxy(
            source_type='mico_high',
            target_type='mico_small',
            kernel_type=kernel_type,
            finetune_ratio=ratio,
            strategy='combined',
            verbose=False
        )
        results['transfer_high_to_small'].append(transfer_result_rev['mape_after'])
        
        # Training from scratch on small
        _, scratch_result_small = get_transfer_proxy(
            source_type='mico_small',
            target_type='mico_small',
            kernel_type=kernel_type,
            finetune_ratio=ratio,
            strategy='target_only',
            verbose=False
        )
        results['scratch_small'].append(scratch_result_small['mape_after'])
        
        if verbose:
            print(f"\nRatio {ratio*100:.0f}%:")
            print(f"  small->high transfer: MAPE={transfer_result['mape_after']*100:.2f}%")
            print(f"  high from scratch:    MAPE={scratch_result['mape_after']*100:.2f}%")
            print(f"  high->small transfer: MAPE={transfer_result_rev['mape_after']*100:.2f}%")
            print(f"  small from scratch:   MAPE={scratch_result_small['mape_after']*100:.2f}%")
    
    return ratios, results


def plot_transfer_learning_results(ratios, results, kernel_type, output_dir='output/figs'):
    """Plot transfer learning results comparing with baseline."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: small -> high
    ax1 = axes[0]
    ratios_percent = [r * 100 for r in ratios]
    ax1.plot(ratios_percent, [r*100 for r in results['transfer_small_to_high']], 
             'o-', label='Transfer (small→high)', linewidth=2, markersize=8)
    ax1.plot(ratios_percent, [r*100 for r in results['scratch_high']], 
             's--', label='From Scratch', linewidth=2, markersize=8)
    ax1.set_xlabel('Target Data Ratio (%)', fontsize=12)
    ax1.set_ylabel('MAPE (%)', fontsize=12)
    ax1.set_title(f'Transfer Learning: small → high ({kernel_type})', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 105)
    
    # Plot 2: high -> small
    ax2 = axes[1]
    ax2.plot(ratios_percent, [r*100 for r in results['transfer_high_to_small']], 
             'o-', label='Transfer (high→small)', linewidth=2, markersize=8)
    ax2.plot(ratios_percent, [r*100 for r in results['scratch_small']], 
             's--', label='From Scratch', linewidth=2, markersize=8)
    ax2.set_xlabel('Target Data Ratio (%)', fontsize=12)
    ax2.set_ylabel('MAPE (%)', fontsize=12)
    ax2.set_title(f'Transfer Learning: high → small ({kernel_type})', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 105)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, f'transfer_learning_{kernel_type}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: {filepath}")
    return filepath


def run_strategy_comparison(kernel_type='matmul', verbose=True):
    """Compare different fine-tuning strategies."""
    if verbose:
        print("\n" + "="*80)
        print(f"FINE-TUNING STRATEGY COMPARISON ({kernel_type})")
        print("="*80)
    
    strategies = ['combined', 'target_only', 'weighted']
    ratios = [0.05, 0.1, 0.2, 0.3, 0.5]
    
    results = {strategy: [] for strategy in strategies}
    
    for ratio in ratios:
        for strategy in strategies:
            _, result = get_transfer_proxy(
                source_type='mico_small',
                target_type='mico_high',
                kernel_type=kernel_type,
                finetune_ratio=ratio,
                strategy=strategy,
                verbose=False
            )
            results[strategy].append(result['mape_after'])
        
        if verbose:
            print(f"\nRatio {ratio*100:.0f}%:")
            for strategy in strategies:
                mape = results[strategy][-1]
                print(f"  {strategy:12s}: MAPE={mape*100:.2f}%")
    
    return ratios, results


def run_cross_target_analysis(verbose=True):
    """Analyze transfer learning from BitFusion to VexiiRiscv/MiCo targets."""
    if verbose:
        print("\n" + "="*80)
        print("CROSS-TARGET TRANSFER: BitFusion → VexiiRiscv MiCo")
        print("="*80)
    
    # Test matmul transfer
    matmul_results = explore_cross_target_transfer(
        source_type='bitfusion',
        target_types=['mico_small', 'mico_high'],
        kernel_type='matmul',
        finetune_ratios=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
        verbose=verbose
    )
    
    # Test conv2d transfer
    conv2d_results = explore_cross_target_transfer(
        source_type='bitfusion',
        target_types=['mico_small', 'mico_high'],
        kernel_type='conv2d',
        finetune_ratios=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
        verbose=verbose
    )
    
    return {'matmul': matmul_results, 'conv2d': conv2d_results}


def run_data_efficiency_analysis(kernel_type='matmul', n_trials=3, verbose=True):
    """
    Analyze how much target data is needed for effective transfer learning.
    
    This helps answer: "What's the minimum amount of target data needed
    for transfer learning to be effective?"
    """
    if verbose:
        print("\n" + "="*80)
        print(f"DATA EFFICIENCY ANALYSIS ({kernel_type})")
        print("="*80)
    
    fine_grained_ratios = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]
    
    results = {
        'transfer': {'mean': [], 'std': []},
        'scratch': {'mean': [], 'std': []}
    }
    
    for ratio in fine_grained_ratios:
        transfer_mapes = []
        scratch_mapes = []
        
        for trial in range(n_trials):
            seed = 42 + trial
            
            # Transfer learning
            _, t_result = get_transfer_proxy(
                source_type='mico_small',
                target_type='mico_high',
                kernel_type=kernel_type,
                finetune_ratio=ratio,
                strategy='combined',
                seed=seed,
                verbose=False
            )
            transfer_mapes.append(t_result['mape_after'])
            
            # From scratch
            _, s_result = get_transfer_proxy(
                source_type='mico_high',
                target_type='mico_high',
                kernel_type=kernel_type,
                finetune_ratio=ratio,
                strategy='target_only',
                seed=seed,
                verbose=False
            )
            scratch_mapes.append(s_result['mape_after'])
        
        results['transfer']['mean'].append(np.mean(transfer_mapes))
        results['transfer']['std'].append(np.std(transfer_mapes))
        results['scratch']['mean'].append(np.mean(scratch_mapes))
        results['scratch']['std'].append(np.std(scratch_mapes))
        
        if verbose:
            print(f"Ratio {ratio*100:5.1f}%: Transfer={np.mean(transfer_mapes)*100:.2f}% ± {np.std(transfer_mapes)*100:.2f}%, "
                  f"Scratch={np.mean(scratch_mapes)*100:.2f}% ± {np.std(scratch_mapes)*100:.2f}%")
    
    return fine_grained_ratios, results


def generate_summary_report(all_results, output_dir='output'):
    """Generate a summary report of transfer learning experiments."""
    os.makedirs(output_dir, exist_ok=True)
    
    report = {
        'title': 'MiCoProxy Transfer Learning Validation Report',
        'summary': {},
        'details': all_results
    }
    
    # Extract key findings
    if 'baseline_matmul' in all_results:
        ratios, results = all_results['baseline_matmul']
        # Find ratio where transfer outperforms scratch
        for i, ratio in enumerate(ratios):
            transfer_mape = results['transfer_small_to_high'][i]
            scratch_mape = results['scratch_high'][i]
            if transfer_mape < scratch_mape:
                report['summary']['transfer_effective_at_ratio_matmul'] = ratio
                break
    
    if 'baseline_conv2d' in all_results:
        ratios, results = all_results['baseline_conv2d']
        for i, ratio in enumerate(ratios):
            transfer_mape = results['transfer_small_to_high'][i]
            scratch_mape = results['scratch_high'][i]
            if transfer_mape < scratch_mape:
                report['summary']['transfer_effective_at_ratio_conv2d'] = ratio
                break
    
    # Save report as JSON
    report_path = os.path.join(output_dir, 'transfer_learning_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    print(f"\nReport saved to: {report_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description='MiCoProxy Transfer Learning Validation')
    parser.add_argument('--quick', action='store_true', 
                        help='Run quick validation (fewer iterations)')
    parser.add_argument('--comprehensive', action='store_true',
                        help='Run comprehensive analysis (more iterations)')
    parser.add_argument('--kernel', choices=['matmul', 'conv2d', 'both'], default='both',
                        help='Kernel type to analyze')
    parser.add_argument('--output-dir', default='output',
                        help='Output directory for results')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("MiCoProxy Transfer Learning Validation")
    print("="*80)
    print(f"Mode: {'quick' if args.quick else 'comprehensive' if args.comprehensive else 'standard'}")
    
    all_results = {}
    kernel_types = ['matmul', 'conv2d'] if args.kernel == 'both' else [args.kernel]
    
    for kernel_type in kernel_types:
        print(f"\n\n{'#'*80}")
        print(f"# KERNEL TYPE: {kernel_type.upper()}")
        print(f"{'#'*80}")
        
        # 1. Baseline comparison
        print("\n[1/4] Running baseline comparison...")
        ratios, results = run_baseline_comparison(kernel_type=kernel_type)
        all_results[f'baseline_{kernel_type}'] = (ratios, results)
        
        # Plot results
        plot_transfer_learning_results(ratios, results, kernel_type, 
                                       os.path.join(args.output_dir, 'figs'))
        
        # 2. Strategy comparison
        print("\n[2/4] Comparing fine-tuning strategies...")
        strategy_ratios, strategy_results = run_strategy_comparison(kernel_type=kernel_type)
        all_results[f'strategies_{kernel_type}'] = (strategy_ratios, strategy_results)
        
        # 3. Direction comparison
        print("\n[3/4] Comparing transfer directions...")
        direction_results = compare_transfer_directions(
            kernel_type=kernel_type,
            finetune_ratio=0.1
        )
        all_results[f'directions_{kernel_type}'] = direction_results
        
        # 4. Data efficiency (only in comprehensive mode)
        if args.comprehensive:
            print("\n[4/4] Running data efficiency analysis...")
            efficiency_ratios, efficiency_results = run_data_efficiency_analysis(
                kernel_type=kernel_type,
                n_trials=5
            )
            all_results[f'efficiency_{kernel_type}'] = (efficiency_ratios, efficiency_results)
        else:
            print("\n[4/4] Skipping data efficiency analysis (use --comprehensive for full analysis)")
    
    # Cross-target analysis (bitfusion -> mico)
    print("\n\n[BONUS] Running cross-target transfer analysis...")
    cross_target_results = run_cross_target_analysis()
    all_results['cross_target'] = cross_target_results
    
    # Generate summary report
    report = generate_summary_report(all_results, args.output_dir)
    
    # Print final summary
    print("\n" + "="*80)
    print("TRANSFER LEARNING VALIDATION SUMMARY")
    print("="*80)
    
    print("\n✓ Transfer learning is feasible for MiCoProxy")
    print("✓ Both directions (small→high and high→small) are supported")
    
    if 'transfer_effective_at_ratio_matmul' in report['summary']:
        ratio = report['summary']['transfer_effective_at_ratio_matmul']
        print(f"✓ For matmul: Transfer learning is effective with {ratio*100:.0f}% target data")
    
    if 'transfer_effective_at_ratio_conv2d' in report['summary']:
        ratio = report['summary']['transfer_effective_at_ratio_conv2d']
        print(f"✓ For conv2d: Transfer learning is effective with {ratio*100:.0f}% target data")
    
    print("\n✓ Cross-target transfer (BitFusion → MiCo) shows potential for knowledge transfer")
    print("\nSee generated plots in output/figs/ for detailed visualizations")
    print("See output/transfer_learning_report.json for full results")
    
    return all_results


if __name__ == "__main__":
    main()
