"""
Batch end-to-end experiment: sweep feature sets and regressors across
train ratios for all experiments listed in ExpRecord.md.

Loads each evaluator once and swaps proxy configurations for efficiency.
Experiments run in parallel via multiprocessing.
"""
import os

# Keep library-level thread pools from multiplying with ProcessPoolExecutor.
# These defaults must be set before importing numpy/sklearn/xgboost/torch users.
_THREAD_ENV_DEFAULT = os.environ.get("MICO_PROXY_MODEL_THREADS", "1")
for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_env, _THREAD_ENV_DEFAULT)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mico_matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import random
import time
import numpy as np
import sys
import io, contextlib
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from matplotlib import pyplot as plt
from tqdm import tqdm

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
    {'name': 'CMSIS-Tiny',  'model': 'cmsiscnn_cifar10', 'target': 'latency_mico_cacheless',
     'n': 32, 'bitwidths': [1,2,4,8]},
    {'name': 'CMSIS-Small', 'model': 'cmsiscnn_cifar10', 'target': 'latency_mico_small',
     'n': 32, 'bitwidths': [1,2,4,8]},
    {'name': 'CMSIS-High',  'model': 'cmsiscnn_cifar10', 'target': 'latency_mico_high',
     'n': 32, 'bitwidths': [1,2,4,8]},
]

FEATURE_SETS = ["cbops","sumq", "maxq", "bops"]
REGRESSORS = ["LinearRegression"]
# Optional explicit proxy configs. When this is not None, the script runs only
# these (label, feature, regressor) entries instead of FEATURE_SETS x REGRESSORS.
#
# Example:
# PROXY_CONFIGS = [
#     ("CBOPs_Linear", "cbops", "LinearRegression"),
#     ("Raw+_LogRF", "raw+", "LogRandomForest"),
# ]
PROXY_CONFIGS = None
TRAIN_RATIOS = [0.8]
NUM_SEEDS = 5
PROXY_MODEL_THREADS = max(1, int(os.environ.get("MICO_PROXY_MODEL_THREADS", "1")))
BATCH_E2E_WORKERS = int(os.environ.get("MICO_BATCH_E2E_WORKERS", "0"))
BATCH_E2E_DATALOADER_WORKERS = int(os.environ.get("MICO_BATCH_E2E_DATALOADER_WORKERS", "0"))

remap_labels = {
    "bops": "BOPs",
    "cbops": "CBOPs",
    "raw+": "Full"
}


def get_proxy_configs():
    """Return explicit proxy configs or the legacy feature x regressor sweep."""
    if PROXY_CONFIGS is None:
        return [
            (f"{features}+{regressor}", features, regressor)
            for features in FEATURE_SETS
            for regressor in REGRESSORS
        ]

    configs = []
    for entry in PROXY_CONFIGS:
        if isinstance(entry, dict):
            features = entry.get("features", entry.get("feature"))
            regressor = entry["regressor"]
            label = entry.get("label", f"{features}+{regressor}")
        elif isinstance(entry, str):
            features, regressor = parse_proxy_config_string(entry)
            label = f"{features}+{regressor}"
        elif len(entry) == 2:
            features, regressor = entry
            label = f"{features}+{regressor}"
        elif len(entry) == 3:
            label, features, regressor = entry
        else:
            raise ValueError(
                "Proxy config entries must be strings, dicts, "
                "(features, regressor), or (label, features, regressor)."
            )

        if features not in remap_labels:
            remap_labels[features] = features
        if regressor not in REGRESSOR_FACTORIES:
            raise ValueError(f"Unsupported regressor in proxy config '{label}': {regressor}")
        configs.append((label, features, regressor))
    return configs


def parse_proxy_config_string(entry):
    if "::" in entry:
        features, regressor = entry.split("::", 1)
        return features, regressor

    for regressor in sorted(REGRESSOR_FACTORIES, key=len, reverse=True):
        suffix = f"+{regressor}"
        if entry.endswith(suffix):
            return entry[:-len(suffix)], regressor

    raise ValueError(
        f"Invalid proxy config string '{entry}'. Use 'feature::regressor' "
        "or 'feature+RegressorName'."
    )


def proxy_label(label, features, regressor):
    if PROXY_CONFIGS is not None:
        return label
    return remap_labels.get(features, features)


def configure_regressor_threads():
    """Patch proxy factories so per-process model fits do not oversubscribe CPUs."""
    from MiCoProxy import LogRandomForestRegressor, LogXGBRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from xgboost import XGBRegressor

    REGRESSOR_FACTORIES.update({
        'LogRandomForest': lambda: LogRandomForestRegressor(
            random_state=42, n_jobs=PROXY_MODEL_THREADS),
        'LogXGBRegressor': lambda: LogXGBRegressor(
            random_state=42, n_jobs=PROXY_MODEL_THREADS),
        'XGBRegressor': lambda: XGBRegressor(
            random_state=42, n_jobs=PROXY_MODEL_THREADS),
        'RandomForest': lambda: RandomForestRegressor(
            random_state=42, n_jobs=PROXY_MODEL_THREADS),
        'LinearRegression': lambda: LinearRegression(),
    })


def configure_worker_threads():
    """Apply lightweight thread caps inside spawned workers as well."""
    configure_regressor_threads()
    try:
        import torch
        torch.set_num_threads(PROXY_MODEL_THREADS)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def get_worker_count():
    if BATCH_E2E_WORKERS > 0:
        return max(1, min(len(EXPERIMENTS), BATCH_E2E_WORKERS))

    cpu_count = os.cpu_count() or 1
    workers_by_threads = max(1, cpu_count // PROXY_MODEL_THREADS)
    return max(1, min(len(EXPERIMENTS), workers_by_threads))


configure_regressor_threads()


def output_avg_regression_scatter(exp, results):
    """Output averaged regression scatter charts per train ratio (x: actual, y: predicted)."""
    os.makedirs("output/figs", exist_ok=True)
    proxy_configs = get_proxy_configs()
    fig_paths = []
    for train_ratio in TRAIN_RATIOS:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        all_x = []
        all_y = []
        for label, features, regressor in proxy_configs:
            pred_runs = results[label][train_ratio]['preds']
            if not pred_runs:
                continue

            x_eval = results[label][train_ratio]['x_eval'] / 1e6
            y_avg = np.mean(np.vstack(pred_runs), axis=0) / 1e6
            ax.scatter(
                x_eval,
                y_avg,
                alpha=0.6,
                s=48,
                label=proxy_label(label, features, regressor),
            )
            all_x.append(x_eval)
            all_y.append(y_avg)

        if all_x and all_y:
            x_all = np.concatenate(all_x)
            y_all = np.concatenate(all_y)
            lo = min(np.min(x_all), np.min(y_all))
            hi = max(np.max(x_all), np.max(y_all))
            ax.plot([lo, hi], [lo, hi], color='red', linestyle='--', linewidth=1, label='Ideal')

        # Set all font as times new roman
        # plt.rcParams["font.family"] = "Times New Roman"
        # ax.set_title(f"{exp['name']} ({train_ratio:.0%} train)")

        ax.set_xlabel("Actual Latency (M Cycles)", fontsize=24)
        ax.set_ylabel("Predicted Latency (M Cycles)", fontsize=24)
        # Set tick font size
        ax.tick_params(axis='both', which='both', labelsize=20)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=20)
        plt.tight_layout()
        fig_path = f"output/figs/{exp['model']}_{exp['target']}_avg_reg_scatter_{int(train_ratio*100)}.pdf"
        plt.savefig(fig_path)
        plt.close()
        fig_paths.append(fig_path)
    return fig_paths


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


def run_experiment(exp, position=0):
    """Run one experiment across all configs. Designed to run in a subprocess."""
    configure_worker_threads()
    model_zoo.NUM_WORKERS = BATCH_E2E_DATALOADER_WORKERS
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
    proxy_configs = get_proxy_configs()

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
    total_steps = len(proxy_configs) * sum(
        1 if train_ratio >= 1.0 else NUM_SEEDS for train_ratio in TRAIN_RATIOS
    )
    pbar = tqdm(total=total_steps, desc=exp['name'], position=position, leave=True)
    for config_key, features, regressor in proxy_configs:
        results[config_key] = {r: {
            'mape': [],
            'spearman': [],
            'r2': [],
            'rrse': [],
            'fit_ms': [],
            'predict_ms': [],
            'predict_us_per_scheme': [],
            'preds': [],
            'x_eval': None} for r in TRAIN_RATIOS}

        for train_ratio in TRAIN_RATIOS:
            n_seeds = 1 if train_ratio >= 1.0 else NUM_SEEDS
            for seed in range(n_seeds):
                fit_start = time.perf_counter()
                mp, cp = build_proxies(target, features, regressor)
                mp.set_train_ratio(train_ratio)
                mp.seed = seed
                mp.fit()
                cp.set_train_ratio(train_ratio)
                cp.seed = seed
                cp.fit()
                fit_ms = (time.perf_counter() - fit_start) * 1000.0
                evaluator.set_proxy(mp, cp)

                predict_start = time.perf_counter()
                Y_pred = np.array([evaluator.eval_pred_latency(s) for s in all_schemes])
                predict_ms = (time.perf_counter() - predict_start) * 1000.0
                predict_us_per_scheme = predict_ms * 1000.0 / len(all_schemes)

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
                # r2 = r2_score(X_eval, Y_eval)
                rrse = np.sqrt(np.sum((X_eval - Y_eval) ** 2) / np.sum((X_eval - np.mean(X_eval)) ** 2))

                results[config_key][train_ratio]['mape'].append(mape)
                results[config_key][train_ratio]['spearman'].append(spear)
                results[config_key][train_ratio]['r2'].append(r2)
                results[config_key][train_ratio]['rrse'].append(rrse)
                results[config_key][train_ratio]['fit_ms'].append(fit_ms)
                results[config_key][train_ratio]['predict_ms'].append(predict_ms)
                results[config_key][train_ratio]['predict_us_per_scheme'].append(predict_us_per_scheme)
                results[config_key][train_ratio]['preds'].append(Y_eval)
                if results[config_key][train_ratio]['x_eval'] is None:
                    results[config_key][train_ratio]['x_eval'] = X_eval
                pbar.update(1)

    pbar.close()
    return exp, results


def format_results(all_results):
    """Format results as markdown tables."""
    lines = []

    lines.append("## Average Across Tasks\n")
    header = "| Proxy | Metric | " + " | ".join(f"{r:.0%}" for r in TRAIN_RATIOS) + " |"
    sep = "|" + "---|" * (len(TRAIN_RATIOS) + 2)
    lines.append(header)
    lines.append(sep)

    config_keys = list(all_results[0][1].keys()) if all_results else []
    for config_key in config_keys:
        for metric_name, metric_key, fmt, scale in [
            ("MAPE%", "mape", "{:.2f}", 100),
            ("R²", "r2", "{:.4f}", 1),
        ]:
            row = f"| {config_key} | {metric_name} |"
            for ratio in TRAIN_RATIOS:
                task_vals = []
                for _, results in all_results:
                    if config_key not in results:
                        continue
                    vals = np.array(results[config_key][ratio][metric_key]) * scale
                    if len(vals) > 0:
                        task_vals.append(vals.mean())
                task_vals = np.array(task_vals)
                if len(task_vals) == 0:
                    row += " N/A |"
                elif len(task_vals) == 1:
                    row += f" {fmt.format(task_vals[0])} |"
                else:
                    row += f" {fmt.format(task_vals.mean())}±{fmt.format(task_vals.std())} |"
            lines.append(row)

    lines.append("\n## Average Runtime Across Tasks\n")
    lines.append(header)
    lines.append(sep)

    for config_key in config_keys:
        for metric_name, metric_key, fmt, scale in [
            ("Fit ms", "fit_ms", "{:.1f}", 1),
            ("Predict ms", "predict_ms", "{:.1f}", 1),
            ("Predict us/scheme", "predict_us_per_scheme", "{:.1f}", 1),
        ]:
            row = f"| {config_key} | {metric_name} |"
            for ratio in TRAIN_RATIOS:
                task_vals = []
                for _, results in all_results:
                    if config_key not in results:
                        continue
                    vals = np.array(results[config_key][ratio][metric_key]) * scale
                    if len(vals) > 0:
                        task_vals.append(vals.mean())
                task_vals = np.array(task_vals)
                if len(task_vals) == 0:
                    row += " N/A |"
                elif len(task_vals) == 1:
                    row += f" {fmt.format(task_vals[0])} |"
                else:
                    row += f" {fmt.format(task_vals.mean())}±{fmt.format(task_vals.std())} |"
            lines.append(row)

    for exp, results in all_results:
        lines.append(f"\n### {exp['name']} (`{exp['model']}`, `{exp['target']}`)\n")

        header = "| Proxy | " + " | ".join(f"{r:.0%}" for r in TRAIN_RATIOS) + " |"
        sep = "|" + "---|" * (len(TRAIN_RATIOS) + 1)

        for metric_name, metric_key, fmt, scale in [
            ("MAPE% (mean±std, lower is better)", 'mape', "{:.2f}", 100),
            ("R² (mean±std, higher is better)", 'r2', "{:.4f}", 1),
            ("RRSE (mean±std, lower is better)", 'rrse', "{:.4f}", 1),
            ("Spearman Correlation (mean±std, higher is better)", 'spearman', "{:.4f}", 1),
            ("Fit Time ms (mean±std, lower is better)", 'fit_ms', "{:.1f}", 1),
            ("Predict Time ms (mean±std, lower is better)", 'predict_ms', "{:.1f}", 1),
            ("Predict Time us/scheme (mean±std, lower is better)", 'predict_us_per_scheme', "{:.1f}", 1),
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
    configure_worker_threads()
    n_workers = get_worker_count()
    print(
        f"Running {len(EXPERIMENTS)} experiments with {n_workers} workers "
        f"(proxy_model_threads={PROXY_MODEL_THREADS}, "
        f"dataloader_workers={BATCH_E2E_DATALOADER_WORKERS}, "
        f"env MICO_BATCH_E2E_WORKERS={BATCH_E2E_WORKERS or 'auto'})..."
    )

    all_results = [None] * len(EXPERIMENTS)
    exp_index = {exp['name']: i for i, exp in enumerate(EXPERIMENTS)}

    if n_workers == 1:
        for idx, exp in enumerate(EXPERIMENTS):
            exp, results = run_experiment(exp, idx)
            all_results[idx] = (exp, results)
            print(f"  ✓ {exp['name']} done")
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(run_experiment, exp, idx): exp
                for idx, exp in enumerate(EXPERIMENTS)
            }
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

    print("\nGenerating averaged regression scatter charts...")
    for exp, results in all_results:
        fig_paths = output_avg_regression_scatter(exp, results)
        for fig_path in fig_paths:
            print(f"  ✓ saved {fig_path}")


if __name__ == "__main__":
    main()
