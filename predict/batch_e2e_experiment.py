"""
Batch end-to-end experiment: sweep feature sets and regressors across
train ratios for all experiments listed in ExpRecord.md.

Loads each evaluator once and swaps proxy configurations for efficiency.
Experiments run in parallel via multiprocessing.
"""
import random
import numpy as np
import sys, os
import io, contextlib
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sklearn.metrics import mean_absolute_percentage_error, r2_score
from scipy.stats import spearmanr

from MiCoEval import MiCoEval
from MiCoProxy import (
    get_bitfusion_matmul_proxy, get_bitfusion_conv2d_proxy,
    get_mico_matmul_proxy, get_mico_conv2d_proxy, get_mico_misc_kernel_proxy,
    REGRESSOR_FACTORIES,
)
from models import model_zoo


EXPERIMENTS = [
    {'name': 'VGG-BF',     'model': 'vgg_cifar10',      'target': 'latency_bitfusion',
     'n': 64, 'bitwidths': [2,3,4,5,6,7,8]},
    {'name': 'ResNet18-BF', 'model': 'resnet18_cifar100', 'target': 'latency_bitfusion',
     'n': 64, 'bitwidths': [2,3,4,5,6,7,8]},
    {'name': 'CMSIS-Small', 'model': 'cmsiscnn_cifar10', 'target': 'latency_mico_small',
     'n': 32, 'bitwidths': [1,2,4,8]},
    {'name': 'CMSIS-High',  'model': 'cmsiscnn_cifar10', 'target': 'latency_mico_high',
     'n': 32, 'bitwidths': [1,2,4,8]},
]

FEATURE_SETS = ['raw', 'bops', 'cbops', 'cbops+']
REGRESSORS = ['LogRandomForest']
TRAIN_RATIOS = [0.2, 0.4, 0.6, 0.8, 1.0]
NUM_SEEDS = 5


def build_proxies(target, features, regressor):
    """Build proxy pair for given config (suppressing CV output)."""
    with contextlib.redirect_stdout(io.StringIO()):
        if target.startswith("latency_mico"):
            mico_type = target.split("_")[-1]
            mp = get_mico_matmul_proxy(mico_type=mico_type, preprocess=features, regressor=regressor)
            cp = get_mico_conv2d_proxy(mico_type=mico_type, preprocess=features, regressor=regressor)
        else:
            mp = get_bitfusion_matmul_proxy(preprocess=features, regressor=regressor)
            cp = get_bitfusion_conv2d_proxy(preprocess=features, regressor=regressor)
    return mp, cp


def run_experiment(exp):
    """Run one experiment across all configs. Designed to run in a subprocess."""
    model_name = exp['model']
    target = exp['target']
    N = exp['n']
    bitwidths = exp['bitwidths']

    # Load evaluator
    with contextlib.redirect_stdout(io.StringIO()):
        model_obj, train_loader, test_loader = model_zoo.from_zoo(model_name)
        evaluator = MiCoEval(model_obj, 1, train_loader, test_loader,
                             f"output/ckpt/{model_name}.pth",
                             model_name=model_name,
                             output_json=f"output/json/{model_name}_{target}_predict.json")
        if target.startswith("latency_mico"):
            mico_type = target.split("_")[-1]
            evaluator.set_mico_target(mico_type)
            evaluator.set_misc_proxy(get_mico_misc_kernel_proxy)
            evaluator.set_eval("latency_mico")
        else:
            evaluator.set_eval("latency_bitfusion")

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

    results = {}
    for features in FEATURE_SETS:
        for regressor in REGRESSORS:
            config_key = f"{features}+{regressor}"
            results[config_key] = {r: {'mape': [], 'spearman': [], 'r2': []} for r in TRAIN_RATIOS}

            for train_ratio in TRAIN_RATIOS:
                n_seeds = 1 if train_ratio >= 1.0 else NUM_SEEDS
                for seed in range(n_seeds):
                    mp, cp = build_proxies(target, features, regressor)
                    mp.set_train_ratio(train_ratio)
                    mp.seed = seed
                    mp.fit()
                    cp.set_train_ratio(train_ratio)
                    cp.seed = seed
                    cp.fit()
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

                    results[config_key][train_ratio]['mape'].append(mape)
                    results[config_key][train_ratio]['spearman'].append(spear)
                    results[config_key][train_ratio]['r2'].append(r2)

    return exp, results


def format_results(all_results):
    """Format results as markdown tables."""
    lines = []

    for exp, results in all_results:
        lines.append(f"\n### {exp['name']} (`{exp['model']}`, `{exp['target']}`)\n")

        header = "| Features | " + " | ".join(f"{r:.0%}" for r in TRAIN_RATIOS) + " |"
        sep = "|" + "---|" * (len(TRAIN_RATIOS) + 1)

        for metric_name, metric_key, fmt, scale in [
            ("MAPE% (mean±std, lower is better)", 'mape', "{:.2f}", 100),
            ("R² (mean±std, higher is better)", 'r2', "{:.4f}", 1),
            ("Spearman Correlation (mean±std, higher is better)", 'spearman', "{:.4f}", 1),
        ]:
            lines.append(f"\n**{metric_name}**\n")
            lines.append(header)
            lines.append(sep)

            for config_key in results:
                row = f"| {config_key} |"
                for ratio in TRAIN_RATIOS:
                    vals = np.array(results[config_key][ratio][metric_key]) * scale
                    if len(vals) == 1:
                        row += f" {fmt.format(vals[0])} |"
                    else:
                        row += f" {fmt.format(vals.mean())}±{fmt.format(vals.std())} |"
                lines.append(row)

    return "\n".join(lines)


def main():
    multiprocessing.set_start_method('spawn', force=True)
    n_workers = min(len(EXPERIMENTS), os.cpu_count() or 1)
    print(f"Running {len(EXPERIMENTS)} experiments with {n_workers} workers...")

    all_results = [None] * len(EXPERIMENTS)
    exp_index = {exp['name']: i for i, exp in enumerate(EXPERIMENTS)}

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(run_experiment, exp): exp for exp in EXPERIMENTS}
        for future in as_completed(futures):
            exp, results = future.result()
            idx = exp_index[exp['name']]
            all_results[idx] = (exp, results)
            print(f"  ✓ {exp['name']} done")

    md = format_results(all_results)
    print("\n\n" + "=" * 80)
    print("RESULTS (Markdown)")
    print("=" * 80)
    print(md)

    with open("predict/e2e_results.md", "w") as f:
        f.write(md)
    print("\nResults saved to predict/e2e_results.md")


if __name__ == "__main__":
    main()
