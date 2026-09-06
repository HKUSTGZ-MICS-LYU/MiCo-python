"""
Transfer Learning Experiment: Test transfer learning quality between mico small and mico high.

Following the same experiment structure as batch_e2e_experiment.py:
- Uses raw features + LogRandomForest as basic setup
- Training data ratio: 0.8 src + tgt_ratio or 0.2 new (target)
- Tests transfer learning from mico_small -> mico_high and mico_high -> mico_small
- Tests both rational and residual transfer modes
- Compares against direct training baseline
"""
import random
import numpy as np
import sys, os
import io, contextlib
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sklearn.metrics import mean_absolute_percentage_error, r2_score
from scipy.stats import spearmanr

from MiCoEval import MiCoEval
from MiCoProxy import (
    get_mico_matmul_proxy, get_mico_conv2d_proxy,
    get_transfer_mico_pair,
)
from models import model_zoo


tgt_ratio = 0.05

# Experiment configurations
EXPERIMENTS = [
    {'name': 'CMSIS-Tiny',  'model': 'cmsiscnn_cifar10', 'target': 'latency_mico_cacheless',
     'n': 32, 'bitwidths': [1,2,4,8]},
    {'name': 'CMSIS-Small', 'model': 'cmsiscnn_cifar10', 'target': 'latency_mico_small',
     'n': 32, 'bitwidths': [1,2,4,8]},
    {'name': 'CMSIS-High',  'model': 'cmsiscnn_cifar10', 'target': 'latency_mico_high',
     'n': 32, 'bitwidths': [1,2,4,8]},
]

# Transfer learning configurations (both rational and residual modes)
TRANSFER_CONFIGS = [
    {'src': 'small', 'tgt': 'high', 'src_ratio': 0.8, 'tgt_ratio': tgt_ratio, 'mode': 'rational'},
    {'src': 'small', 'tgt': 'high', 'src_ratio': 0.8, 'tgt_ratio': tgt_ratio, 'mode': 'residual'},
    {'src': 'high', 'tgt': 'small', 'src_ratio': 0.8, 'tgt_ratio': tgt_ratio, 'mode': 'rational'},
    {'src': 'high', 'tgt': 'small', 'src_ratio': 0.8, 'tgt_ratio': tgt_ratio, 'mode': 'residual'},
    {'src': 'small', 'tgt': 'cacheless', 'src_ratio': 0.8, 'tgt_ratio': tgt_ratio, 'mode': 'rational'},
    {'src': 'small', 'tgt': 'cacheless', 'src_ratio': 0.8, 'tgt_ratio': tgt_ratio, 'mode': 'residual'},
    {'src': 'high', 'tgt': 'cacheless', 'src_ratio': 0.8, 'tgt_ratio': tgt_ratio, 'mode': 'rational'},
    {'src': 'high', 'tgt': 'cacheless', 'src_ratio': 0.8, 'tgt_ratio': tgt_ratio, 'mode': 'residual'},
    {'src': 'cacheless', 'tgt': 'small', 'src_ratio': 0.8, 'tgt_ratio': tgt_ratio, 'mode': 'rational'},
    {'src': 'cacheless', 'tgt': 'small', 'src_ratio': 0.8, 'tgt_ratio': tgt_ratio, 'mode': 'residual'},
    {'src': 'cacheless', 'tgt': 'high', 'src_ratio': 0.8, 'tgt_ratio': tgt_ratio, 'mode': 'rational'},
    {'src': 'cacheless', 'tgt': 'high', 'src_ratio': 0.8, 'tgt_ratio': tgt_ratio, 'mode': 'residual'},
]

# Baseline configs (no transfer learning, direct training on target)
BASELINE_CONFIGS = [
    {'tgt': 'cacheless', 'ratio': tgt_ratio},
    {'tgt': 'small', 'ratio': tgt_ratio},
    {'tgt': 'high', 'ratio': tgt_ratio},
]

NUM_SEEDS = 5


def build_transfer_proxies(tgt_mico_type, src_mico_type, src_ratio, tgt_ratio, mode):
    """Build transfer learning proxy pair using get_transfer_mico_pair."""
    with contextlib.redirect_stdout(io.StringIO()):
        mp, cp = get_transfer_mico_pair(
            src_mico_type=src_mico_type,
            tgt_mico_type=tgt_mico_type,
            src_ratio=src_ratio,
            tgt_ratio=tgt_ratio,
            preprocess='raw',
            regressor='LogRandomForest',
            mode=mode
        )
    return mp, cp


def build_baseline_proxies(tgt_mico_type, train_ratio):
    """Build baseline proxy (no transfer learning, direct training)."""
    with contextlib.redirect_stdout(io.StringIO()):
        mp = get_mico_matmul_proxy(mico_type=tgt_mico_type, preprocess='raw', regressor='LogRandomForest')
        cp = get_mico_conv2d_proxy(mico_type=tgt_mico_type, preprocess='raw', regressor='LogRandomForest')
        mp.set_train_ratio(train_ratio)
        mp.fit()
        cp.set_train_ratio(train_ratio)
        cp.fit()
    return mp, cp


def run_transfer_experiment(exp, position=0):
    """Run transfer learning experiment (designed to run in a subprocess)."""
    model_name = exp['model']
    target = exp['target']
    N = exp['n']
    bitwidths = exp['bitwidths']
    tgt_mico_type = target.split('_')[-1]

    # Load evaluator
    with contextlib.redirect_stdout(io.StringIO()):
        model_obj, train_loader, test_loader = model_zoo.from_zoo(model_name)
        evaluator = MiCoEval(model_obj, 1, train_loader, test_loader,
                             f"output/ckpt/{model_name}.pth",
                             model_name=model_name,
                             output_json=f"output/json/{model_name}_{target}_predict.json")
        evaluator.set_mico_target(tgt_mico_type)
        evaluator.set_misc_proxy(lambda *args, **kwargs: 0)  # Dummy misc proxy
        evaluator.set_eval("latency_mico")

    dim = evaluator.n_layers * 2

    # Generate fixed random schemes
    random.seed(0)
    np.random.seed(0)
    rand_schemes = [random.choices(bitwidths, k=dim) for _ in range(N)]
    min_scheme = [min(bitwidths)] * dim
    max_scheme = [max(bitwidths)] * dim
    all_schemes = [min_scheme, max_scheme] + rand_schemes

    # Get ground truth (cached)
    X_true = np.array([evaluator.eval(s, offline=True) for s in all_schemes])

    # Filter transfer configs for this target
    transfer_configs_for_tgt = [c for c in TRANSFER_CONFIGS if c['tgt'] == tgt_mico_type]

    results = {}
    total_steps = len(transfer_configs_for_tgt) * NUM_SEEDS
    pbar = tqdm(total=total_steps, desc=f"{exp['name']} Transfer", position=position, leave=True)

    for config in transfer_configs_for_tgt:
        config_key = f"tl_{config['src']}->{config['tgt']}_{config['mode']}_{config['src_ratio']:.1f}_{config['tgt_ratio']:.1f}"
        results[config_key] = {
            'mape': [],
            'spearman': [],
            'r2': [],
            'rrse': [],
            'preds': [],
            'x_eval': None
        }

        for seed in range(NUM_SEEDS):
            # Build transfer proxy
            mp, cp = build_transfer_proxies(
                tgt_mico_type=config['tgt'],
                src_mico_type=config['src'],
                src_ratio=config['src_ratio'],
                tgt_ratio=config['tgt_ratio'],
                mode=config['mode']
            )
            # Override seed
            mp.seed = seed
            cp.seed = seed

            evaluator.set_proxy(mp, cp)

            Y_pred = np.array([evaluator.eval_pred_latency(s) for s in all_schemes])

            # Min/Max correction
            min_X, max_X = X_true[0], X_true[1]
            min_Y, max_Y = Y_pred[0], Y_pred[1]
            if max_Y != min_Y:
                slope = (max_X - min_X) / (max_Y - min_Y)
                intercept = min_X - slope * min_Y
                Y_pred = slope * Y_pred + intercept

            X_eval = X_true[2:]
            Y_eval = Y_pred[2:]
            mape = mean_absolute_percentage_error(X_eval, Y_eval)
            spear, _ = spearmanr(X_eval, Y_eval)
            r2 = r2_score(X_eval / np.max(X_eval), Y_eval / np.max(Y_eval))
            rrse = np.sqrt(np.sum((X_eval - Y_eval) ** 2) / np.sum((X_eval - np.mean(X_eval)) ** 2))

            results[config_key]['mape'].append(mape)
            results[config_key]['spearman'].append(spear)
            results[config_key]['r2'].append(r2)
            results[config_key]['rrse'].append(rrse)
            results[config_key]['preds'].append(Y_eval)
            if results[config_key]['x_eval'] is None:
                results[config_key]['x_eval'] = X_eval
            pbar.update(1)

    pbar.close()
    return exp, results


def run_baseline_experiment(exp, position=0):
    """Run baseline experiment (no transfer learning, designed to run in a subprocess)."""
    model_name = exp['model']
    target = exp['target']
    N = exp['n']
    bitwidths = exp['bitwidths']
    tgt_mico_type = target.split('_')[-1]

    # Load evaluator
    with contextlib.redirect_stdout(io.StringIO()):
        model_obj, train_loader, test_loader = model_zoo.from_zoo(model_name)
        evaluator = MiCoEval(model_obj, 1, train_loader, test_loader,
                             f"output/ckpt/{model_name}.pth",
                             model_name=model_name,
                             output_json=f"output/json/{model_name}_{target}_predict.json")
        evaluator.set_mico_target(tgt_mico_type)
        evaluator.set_misc_proxy(lambda *args, **kwargs: 0)
        evaluator.set_eval("latency_mico")

    dim = evaluator.n_layers * 2

    # Generate fixed random schemes
    random.seed(0)
    np.random.seed(0)
    rand_schemes = [random.choices(bitwidths, k=dim) for _ in range(N)]
    min_scheme = [min(bitwidths)] * dim
    max_scheme = [max(bitwidths)] * dim
    all_schemes = [min_scheme, max_scheme] + rand_schemes

    # Get ground truth
    X_true = np.array([evaluator.eval(s, offline=True) for s in all_schemes])

    # Filter baseline configs for this target
    baseline_configs_for_tgt = [c for c in BASELINE_CONFIGS if c['tgt'] == tgt_mico_type]

    results = {}
    total_steps = len(baseline_configs_for_tgt) * NUM_SEEDS
    pbar = tqdm(total=total_steps, desc=f"{exp['name']} Baseline", position=position, leave=True)

    for config in baseline_configs_for_tgt:
        config_key = f"baseline_{config['tgt']}_{config['ratio']:.1f}"
        results[config_key] = {
            'mape': [],
            'spearman': [],
            'r2': [],
            'rrse': [],
            'preds': [],
            'x_eval': None
        }

        for seed in range(NUM_SEEDS):
            mp, cp = build_baseline_proxies(tgt_mico_type=config['tgt'], train_ratio=config['ratio'])
            mp.seed = seed
            cp.seed = seed
            evaluator.set_proxy(mp, cp)

            Y_pred = np.array([evaluator.eval_pred_latency(s) for s in all_schemes])

            # Min/Max correction
            min_X, max_X = X_true[0], X_true[1]
            min_Y, max_Y = Y_pred[0], Y_pred[1]
            if max_Y != min_Y:
                slope = (max_X - min_X) / (max_Y - min_Y)
                intercept = min_X - slope * min_Y
                Y_pred = slope * Y_pred + intercept

            X_eval = X_true[2:]
            Y_eval = Y_pred[2:]
            mape = mean_absolute_percentage_error(X_eval, Y_eval)
            spear, _ = spearmanr(X_eval, Y_eval)
            r2 = r2_score(X_eval / np.max(X_eval), Y_eval / np.max(Y_eval))
            rrse = np.sqrt(np.sum((X_eval - Y_eval) ** 2) / np.sum((X_eval - np.mean(X_eval)) ** 2))

            results[config_key]['mape'].append(mape)
            results[config_key]['spearman'].append(spear)
            results[config_key]['r2'].append(r2)
            results[config_key]['rrse'].append(rrse)
            results[config_key]['preds'].append(Y_eval)
            if results[config_key]['x_eval'] is None:
                results[config_key]['x_eval'] = X_eval
            pbar.update(1)

    pbar.close()
    return exp, results


def format_transfer_results(all_results):
    """Format transfer learning results as markdown."""
    lines = []

    lines.append("\n# Transfer Learning Experiment Results\n")
    lines.append("## Configuration: raw features + LogRandomForest")
    lines.append("## Training: 0.8 source + tgt_ratio/0.2 target (rational & residual modes)\n")

    for exp, results in all_results:
        lines.append(f"\n### {exp['name']} (`{exp['model']}`, `{exp['target']}`)\n")

        header = "| Config | MAPE% (mean±std) | R² (mean±std) | RRSE (mean±std) | Spearman (mean±std) |"
        sep = "|---|---|---|---|---|"

        lines.append(header)
        lines.append(sep)

        for config_key, config_results in sorted(results.items()):
            mape = np.array(config_results['mape']) * 100
            r2 = np.array(config_results['r2'])
            rrse = np.array(config_results['rrse'])
            spear = np.array(config_results['spearman'])

            row = f"| {config_key} | {mape.mean():.2f}±{mape.std():.2f} | {r2.mean():.4f}±{r2.std():.4f} | {rrse.mean():.4f}±{rrse.std():.4f} | {spear.mean():.4f}±{spear.std():.4f} |"
            lines.append(row)

    return "\n".join(lines)


def main():
    multiprocessing.set_start_method('spawn', force=True)
    n_workers = min(len(EXPERIMENTS), os.cpu_count() or 1)
    print(f"Running transfer learning experiments with {n_workers} workers...")

    all_results = [None] * (len(EXPERIMENTS) * 2)
    exp_index = {}

    # Assign indices for both transfer and baseline experiments
    for idx, exp in enumerate(EXPERIMENTS):
        exp_index[('transfer', exp['name'])] = idx
        exp_index[('baseline', exp['name'])] = len(EXPERIMENTS) + idx

    # Run all experiments in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {}

        # Submit transfer learning experiments
        for idx, exp in enumerate(EXPERIMENTS):
            future = executor.submit(run_transfer_experiment, exp, idx)
            futures[future] = ('transfer', exp)

        # Submit baseline experiments
        for idx, exp in enumerate(EXPERIMENTS):
            future = executor.submit(run_baseline_experiment, exp, len(EXPERIMENTS) + idx)
            futures[future] = ('baseline', exp)

        # Collect results as they complete
        for future in as_completed(futures):
            exp_type, exp = futures[future]
            try:
                exp_result = future.result()
                idx = exp_index[(exp_type, exp_result[0]['name'])]
                all_results[idx] = (exp_type, exp_result)
                print(f"  {'✓' if exp_type == 'transfer' else '✓'} {exp_result[0]['name']} {exp_type.capitalize()} done")
            except Exception as e:
                print(f"  ✗ {exp['name']} {exp_type} failed: {e}")

    # Format and save results
    md = format_transfer_results([r[1] for r in all_results if r is not None])
    print("\n\n" + "=" * 80)
    print("RESULTS (Markdown)")
    print("=" * 80)
    print(md)

    with open("predict/transfer_learning_results.md", "w") as f:
        f.write(md)
    print("\nResults saved to predict/transfer_learning_results.md")


if __name__ == "__main__":
    main()
