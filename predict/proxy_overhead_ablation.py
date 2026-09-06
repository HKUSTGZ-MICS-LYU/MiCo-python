"""
Measure CBOPs vs Raw+ proxy overhead on MiCo's real proxy feature path.

This script separates feature preprocessing, surrogate fitting, batched network
prediction, and near-constraint sampling overhead. It intentionally avoids live
hardware simulation; cached real latencies are used only when available.
"""

import argparse
import contextlib
import csv
import io
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from MiCoProxy import MiCoProxy, REGRESSOR_FACTORIES
from predict.proxy_speed_gap import (
    Calibration,
    build_min_max_calibrations,
    cache_path_for,
    default_bitwidths,
    extract_network_features,
    generate_schemes,
    load_actual_cache,
    parse_csv_list,
    predict_network_latency,
    predict_schemes,
    target_objective,
    write_csv,
    write_markdown,
)
from searchers.SearchUtils import near_constr_sample


@dataclass
class DatasetBundle:
    X: np.ndarray
    y: np.ndarray
    load_ms: float


@dataclass
class FastNetworkFeatures:
    macs: np.ndarray
    conv_idx: np.ndarray
    linear_idx: np.ndarray


def timed_ms(fn):
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1000.0


def summarize(values: Sequence[float]) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.median(arr)), float(np.std(arr))


def fmt_ms(value: float) -> str:
    return f"{value:.3f}"


def base_empty_metrics() -> Dict[str, object]:
    return {
        "load_ms": "",
        "preprocess_mean_ms": "",
        "preprocess_median_ms": "",
        "preprocess_std_ms": "",
        "fit_mean_ms": "",
        "fit_median_ms": "",
        "fit_std_ms": "",
        "predict_size": "",
        "predict_mean_ms": "",
        "predict_median_ms": "",
        "predict_std_ms": "",
        "near_samples": "",
        "near_returned_samples": "",
        "near_sampling_ms": "",
        "near_constraint_evals": "",
        "near_fast_path": "",
        "near_constraint_ms_per_eval": "",
    }


def profile_paths(target: str) -> Dict[str, Tuple[str, str]]:
    if target == "latency_bitfusion":
        return {
            "matmul": ("benchmark_results/bitfusion_matmul_zoo.csv", "matmul"),
            "conv2d": ("benchmark_results/bitfusion_conv2d_zoo.csv", "conv2d"),
        }
    if target.startswith("latency_mico"):
        mico_type = target.split("_")[-1]
        return {
            "matmul": (f"benchmark_results/mico_{mico_type}_matmul_zoo.csv", "matmul"),
            "conv2d": (f"benchmark_results/mico_{mico_type}_conv2d_zoo.csv", "conv2d"),
        }
    raise ValueError(f"Unsupported target: {target}")


def load_kernel_dataset(profile_dataset: str, kernel_type: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(profile_dataset, "r") as f:
        reader = csv.reader(f)
        data = []
        next(reader)
        for row in reader:
            data.append(list(map(int, row)))
    data = np.array(data)

    if kernel_type == "matmul":
        n = data[:, 0]
        m = data[:, 1]
        k = data[:, 2]
        qa = data[:, 3]
        qw = data[:, 4]
        latency = data[:, -1]
        macs = n * m * k
        if "bitfusion" in profile_dataset:
            macs = macs / 16
        raw = (macs, m, k, qa, qw)
    elif kernel_type == "conv2d":
        h, w, c, k, ks, s = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5]
        qa = data[:, 6]
        qw = data[:, 7]
        latency = data[:, -1]
        h_out = (h - ks) / s + 1
        w_out = (w - ks) / s + 1
        macs = h_out * w_out * c * k * ks * ks
        raw = (macs, h, w, c, k, ks, s, qa, qw)
    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")

    return np.column_stack(raw), latency


def load_datasets(target: str) -> Dict[str, DatasetBundle]:
    bundles = {}
    for kernel_name, (path, kernel_type) in profile_paths(target).items():
        (x_y, load_ms) = timed_ms(lambda p=path, k=kernel_type: load_kernel_dataset(p, k))
        x, y = x_y
        bundles[kernel_name] = DatasetBundle(x, y, load_ms)
    return bundles


def make_proxy(preprocess: str, regressor: str, bundle: DatasetBundle, train_ratio: float, seed: int) -> MiCoProxy:
    proxy = MiCoProxy(
        REGRESSOR_FACTORIES[regressor](),
        preprocess=preprocess,
        train_ratio=train_ratio,
        seed=seed,
        train_x=bundle.X,
        train_y=bundle.y,
    )
    return proxy


def fit_proxy(preprocess: str, regressor: str, bundle: DatasetBundle, train_ratio: float, seed: int) -> MiCoProxy:
    proxy = make_proxy(preprocess, regressor, bundle, train_ratio, seed)
    proxy.fit()
    return proxy


def sample_rows(x: np.ndarray, n_rows: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=n_rows, replace=n_rows > len(x))
    return x[idx]


def make_fast_network_features(net) -> FastNetworkFeatures:
    return FastNetworkFeatures(
        macs=np.asarray(net.layer_macs, dtype=float),
        conv_idx=np.asarray(
            [idx for idx, kind in enumerate(net.layer_kinds) if kind == "conv2d"],
            dtype=int,
        ),
        linear_idx=np.asarray(
            [idx for idx, kind in enumerate(net.layer_kinds) if kind == "linear"],
            dtype=int,
        ),
    )


def uses_compact_fast_path(
    enabled: bool,
    feature: str,
    matmul_proxy: MiCoProxy,
    conv2d_proxy: MiCoProxy,
) -> bool:
    # These preprocessors only need [MACS, QA, QW]. Keep the network-side
    # benchmark on the same compact descriptor path used by MiCoSearcher.
    compact_features = {"macs+", "bops", "sumq", "maxq", "cbops"}
    return (
        enabled
        and feature in compact_features
        and getattr(matmul_proxy, "preprocess_name", None) == feature
        and getattr(conv2d_proxy, "preprocess_name", None) == feature
    )


def compact_layer_features(
    fast_net: FastNetworkFeatures,
    scheme: Sequence[int],
    idx: np.ndarray,
) -> np.ndarray:
    if len(idx) == 0:
        return np.empty((0, 3), dtype=float)
    n_layers = len(fast_net.macs)
    wq = np.asarray(scheme[:n_layers], dtype=float)
    aq = np.asarray(scheme[n_layers:], dtype=float)
    macs = fast_net.macs[idx]
    return np.column_stack((macs, aq[idx], wq[idx]))


def predict_preprocessed(proxy: MiCoProxy, features: np.ndarray) -> np.ndarray:
    return proxy.model.predict(features)


def predict_compact_latency(
    fast_net: FastNetworkFeatures,
    scheme: Sequence[int],
    matmul_proxy: MiCoProxy,
    conv2d_proxy: MiCoProxy,
    calibration: Calibration,
) -> float:
    latency = 0.0
    conv2d_features = compact_layer_features(fast_net, scheme, fast_net.conv_idx)
    if len(conv2d_features):
        latency += float(np.sum(conv2d_proxy.model.predict(conv2d_proxy.preprocess(conv2d_features))))
    linear_features = compact_layer_features(fast_net, scheme, fast_net.linear_idx)
    if len(linear_features):
        latency += float(np.sum(matmul_proxy.model.predict(matmul_proxy.preprocess(linear_features))))
    if calibration is not None:
        latency = calibration.apply(latency)
    return latency


def build_batched_compact_features(
    fast_net: FastNetworkFeatures,
    schemes: Sequence[Tuple[str, Sequence[int]]],
    idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(idx) == 0:
        return np.empty((0, 3), dtype=float), np.empty((0,), dtype=int)
    scheme_array = np.asarray([scheme for _, scheme in schemes], dtype=float)
    n_layers = len(fast_net.macs)
    wq = scheme_array[:, :n_layers]
    aq = scheme_array[:, n_layers:]
    features = np.column_stack(
        (
            np.tile(fast_net.macs[idx], len(schemes)),
            aq[:, idx].reshape(-1),
            wq[:, idx].reshape(-1),
        )
    )
    scheme_idx = np.repeat(np.arange(len(schemes), dtype=int), len(idx))
    return features, scheme_idx


def predict_schemes_compact(
    fast_net: FastNetworkFeatures,
    schemes: Sequence[Tuple[str, Sequence[int]]],
    matmul_proxy: MiCoProxy,
    conv2d_proxy: MiCoProxy,
    calibration: Calibration,
) -> List[float]:
    pred = np.zeros(len(schemes), dtype=float)
    conv2d_features, conv2d_scheme_idx = build_batched_compact_features(
        fast_net, schemes, fast_net.conv_idx
    )
    if len(conv2d_features):
        conv2d_pred = conv2d_proxy.model.predict(conv2d_proxy.preprocess(conv2d_features))
        np.add.at(pred, conv2d_scheme_idx, conv2d_pred)
    linear_features, linear_scheme_idx = build_batched_compact_features(
        fast_net, schemes, fast_net.linear_idx
    )
    if len(linear_features):
        linear_pred = matmul_proxy.model.predict(matmul_proxy.preprocess(linear_features))
        np.add.at(pred, linear_scheme_idx, linear_pred)
    if calibration is not None:
        pred = calibration.slope * pred + calibration.intercept
    return pred.tolist()


def measure_kernel_overhead(
    target: str,
    features: Sequence[str],
    regressors: Sequence[str],
    repeat: int,
    train_ratio: float,
    seed: int,
    predict_sizes: Sequence[int],
) -> Tuple[List[Dict[str, object]], Dict[Tuple[str, str], Tuple[MiCoProxy, MiCoProxy]]]:
    rows = []
    bundles = load_datasets(target)
    fitted_pairs: Dict[Tuple[str, str], Tuple[MiCoProxy, MiCoProxy]] = {}

    for feature in features:
        for regressor in regressors:
            proxies_by_kernel = {}
            for kernel_name, bundle in bundles.items():
                preprocess_times = []
                fit_times = []
                last_proxy = None
                for rep in range(repeat):
                    proxy = make_proxy(feature, regressor, bundle, train_ratio, seed + rep)
                    _, preprocess_ms = timed_ms(lambda p=proxy, x=bundle.X: p.preprocess(x))
                    preprocess_times.append(preprocess_ms)
                    last_proxy, fit_ms = timed_ms(lambda p=proxy: (p.fit(), p)[1])
                    fit_times.append(fit_ms)

                proxies_by_kernel[kernel_name] = last_proxy
                proc_mean, proc_median, proc_std = summarize(preprocess_times)
                fit_mean, fit_median, fit_std = summarize(fit_times)
                rows.append({
                    "stage": "kernel_preprocess_fit",
                    "target": target,
                    "model": "",
                    "kernel": kernel_name,
                    "feature": feature,
                    "regressor": regressor,
                    "n_rows": len(bundle.X),
                    "raw_dim": bundle.X.shape[1],
                    "feature_dim": last_proxy.preprocess(bundle.X[:1]).shape[1],
                    "repeat": repeat,
                    "load_ms": bundle.load_ms,
                    "preprocess_mean_ms": proc_mean,
                    "preprocess_median_ms": proc_median,
                    "preprocess_std_ms": proc_std,
                    "fit_mean_ms": fit_mean,
                    "fit_median_ms": fit_median,
                    "fit_std_ms": fit_std,
                    "predict_size": "",
                    "predict_mean_ms": "",
                    "predict_median_ms": "",
                    "predict_std_ms": "",
                    "near_samples": "",
                    "near_returned_samples": "",
                    "near_sampling_ms": "",
                    "near_constraint_evals": "",
                    "near_fast_path": "",
                    "near_constraint_ms_per_eval": "",
                })

                for predict_size in predict_sizes:
                    pred_times = []
                    x_pred = sample_rows(bundle.X, predict_size, seed)
                    for _ in range(repeat):
                        _, pred_ms = timed_ms(lambda p=last_proxy, x=x_pred: p.predict(x))
                        pred_times.append(pred_ms)
                    pred_mean, pred_median, pred_std = summarize(pred_times)
                    rows.append({
                        "stage": "kernel_predict",
                        "target": target,
                        "model": "",
                        "kernel": kernel_name,
                        "feature": feature,
                        "regressor": regressor,
                        "n_rows": len(bundle.X),
                        "raw_dim": bundle.X.shape[1],
                        "feature_dim": last_proxy.preprocess(bundle.X[:1]).shape[1],
                        "repeat": repeat,
                        "load_ms": bundle.load_ms,
                        "preprocess_mean_ms": "",
                        "preprocess_median_ms": "",
                        "preprocess_std_ms": "",
                        "fit_mean_ms": "",
                        "fit_median_ms": "",
                        "fit_std_ms": "",
                        "predict_size": predict_size,
                        "predict_mean_ms": pred_mean,
                        "predict_median_ms": pred_median,
                        "predict_std_ms": pred_std,
                        "near_samples": "",
                        "near_returned_samples": "",
                        "near_sampling_ms": "",
                        "near_constraint_evals": "",
                        "near_fast_path": "",
                        "near_constraint_ms_per_eval": "",
                    })

            fitted_pairs[(feature, regressor)] = (
                proxies_by_kernel["matmul"],
                proxies_by_kernel["conv2d"],
            )

    return rows, fitted_pairs


def measure_network_extract(models: Sequence[str], repeat: int) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    rows = []
    networks = {}
    for model_name in models:
        times = []
        net = None
        for _ in range(repeat):
            net, ms = timed_ms(lambda m=model_name: extract_network_features(m))
            times.append(ms)
        mean_ms, median_ms, std_ms = summarize(times)
        networks[model_name] = net
        rows.append({
            "stage": "network_feature_extract",
            "target": "",
            "model": model_name,
            "kernel": "",
            "feature": "raw_layer_descriptor",
            "regressor": "",
            "n_rows": net.n_layers,
            "raw_dim": "",
            "feature_dim": "",
            "repeat": repeat,
            "load_ms": "",
            "preprocess_mean_ms": mean_ms,
            "preprocess_median_ms": median_ms,
            "preprocess_std_ms": std_ms,
            "fit_mean_ms": "",
            "fit_median_ms": "",
            "fit_std_ms": "",
            "predict_size": "",
            "predict_mean_ms": "",
            "predict_median_ms": "",
            "predict_std_ms": "",
            "near_samples": "",
            "near_returned_samples": "",
            "near_sampling_ms": "",
            "near_constraint_evals": "",
            "near_fast_path": "",
            "near_constraint_ms_per_eval": "",
        })
    return rows, networks


def measure_network_predict_and_near(
    target: str,
    models: Sequence[str],
    networks: Dict[str, object],
    fitted_pairs: Dict[Tuple[str, str], Tuple[MiCoProxy, MiCoProxy]],
    features: Sequence[str],
    regressors: Sequence[str],
    predict_sizes: Sequence[int],
    repeat: int,
    near_samples: int,
    constraint_factor: float,
    near_roi: float,
    seed: int,
    vectorized: bool,
    cbops_fast_path: bool,
    min_max_calibration: bool,
    cache_dir: str,
) -> List[Dict[str, object]]:
    rows = []
    bitwidths = default_bitwidths(target)
    objective = target_objective(target)

    for model_idx, model_name in enumerate(models):
        net = networks[model_name]
        fast_net = make_fast_network_features(net)
        label_meta = {
            f"{feature}::{regressor}": (feature, regressor)
            for feature in features
            for regressor in regressors
        }
        proxies = {
            label: fitted_pairs[(feature, regressor)]
            for label, (feature, regressor) in label_meta.items()
        }
        actual_cache = load_actual_cache(cache_path_for(cache_dir, model_name, target), objective)
        calibrations = build_min_max_calibrations(
            net=net,
            bitwidths=bitwidths,
            proxies=proxies,
            actual_cache=actual_cache,
            enabled=min_max_calibration,
        )

        for predict_size in predict_sizes:
            schemes = generate_schemes(
                net.n_layers,
                bitwidths,
                predict_size,
                seed + model_idx,
            )[2:]
            for label, proxy_pair in proxies.items():
                feature, regressor = label_meta[label]
                matmul_proxy, conv2d_proxy = proxy_pair
                use_fast_path = uses_compact_fast_path(
                    cbops_fast_path, feature, matmul_proxy, conv2d_proxy
                )
                pred_times = []
                for _ in range(repeat):
                    if use_fast_path:
                        _, pred_ms = timed_ms(
                            lambda p=proxy_pair: predict_schemes_compact(
                                fast_net,
                                schemes,
                                p[0],
                                p[1],
                                calibrations[label],
                            )
                        )
                    else:
                        _, pred_ms = timed_ms(
                            lambda l=label, p=proxy_pair: predict_schemes(
                                net,
                                schemes,
                                {l: p},
                                {l: calibrations[l]},
                                vectorized=vectorized,
                            )
                        )
                    pred_times.append(pred_ms)
                pred_mean, pred_median, pred_std = summarize(pred_times)
                rows.append({
                    "stage": "network_predict",
                    "target": target,
                    "model": model_name,
                    "kernel": "e2e",
                    "feature": feature,
                    "regressor": regressor,
                    "n_rows": len(schemes),
                    "raw_dim": "",
                    "feature_dim": "",
                    "repeat": repeat,
                    "load_ms": "",
                    "preprocess_mean_ms": "",
                    "preprocess_median_ms": "",
                    "preprocess_std_ms": "",
                    "fit_mean_ms": "",
                    "fit_median_ms": "",
                    "fit_std_ms": "",
                    "predict_size": predict_size,
                    "predict_mean_ms": pred_mean,
                    "predict_median_ms": pred_median,
                    "predict_std_ms": pred_std,
                    "near_samples": "",
                    "near_returned_samples": "",
                    "near_sampling_ms": "",
                    "near_constraint_evals": "",
                    "near_fast_path": "",
                    "near_constraint_ms_per_eval": "",
                })

        if near_samples <= 0:
            continue

        dims = net.n_layers * 2
        for label, proxy_pair in proxies.items():
            feature, regressor = label_meta[label]
            matmul_proxy, conv2d_proxy = proxy_pair
            calibration = calibrations[label]
            eval_count = {"n": 0}
            use_fast_path = uses_compact_fast_path(
                cbops_fast_path, feature, matmul_proxy, conv2d_proxy
            )
            near_fast_path = "compact_macs_qbits" if use_fast_path else "raw_descriptor"

            def constr_func(scheme):
                eval_count["n"] += 1
                if use_fast_path:
                    return predict_compact_latency(
                        fast_net,
                        scheme,
                        matmul_proxy,
                        conv2d_proxy,
                        calibration,
                    )
                return predict_network_latency(net, scheme, matmul_proxy, conv2d_proxy, calibration)

            max_scheme = [max(bitwidths)] * dims
            min_scheme = [min(bitwidths)] * dims
            max_latency = constr_func(max_scheme)
            min_latency = constr_func(min_scheme)
            constr_value = max(constraint_factor * max_latency, min_latency)
            if constr_value / max_latency <= near_roi:
                constr_value = (near_roi + 1e-6) * max_latency
            initial_pop = [
                scheme
                for _, scheme in generate_schemes(net.n_layers, bitwidths, min(64, near_samples), seed + model_idx)[2:]
            ]

            random.seed(seed + 10000 + model_idx)
            np.random.seed(seed + 10000 + model_idx)
            near_result, near_ms = timed_ms(
                lambda: near_constr_sample(
                    n_samples=near_samples,
                    qtypes=bitwidths,
                    dims=dims,
                    constr_func=constr_func,
                    constr_value=constr_value,
                    roi=near_roi,
                    initial_pop=initial_pop,
                    pbar=False,
                )
            )
            rows.append({
                "stage": "near_constraint_sampling",
                "target": target,
                "model": model_name,
                "kernel": "e2e",
                "feature": feature,
                "regressor": regressor,
                "n_rows": "",
                "raw_dim": "",
                "feature_dim": "",
                "repeat": 1,
                "load_ms": "",
                "preprocess_mean_ms": "",
                "preprocess_median_ms": "",
                "preprocess_std_ms": "",
                "fit_mean_ms": "",
                "fit_median_ms": "",
                "fit_std_ms": "",
                "predict_size": "",
                "predict_mean_ms": "",
                "predict_median_ms": "",
                "predict_std_ms": "",
                "near_samples": near_samples,
                "near_returned_samples": len(near_result),
                "near_sampling_ms": near_ms,
                "near_constraint_evals": eval_count["n"],
                "near_fast_path": near_fast_path,
                "near_constraint_ms_per_eval": near_ms / max(eval_count["n"], 1),
            })

    return rows


def make_markdown(rows: Iterable[Dict[str, object]], args) -> List[str]:
    rows = list(rows)
    lines = [
        "# Proxy Overhead Ablation",
        "",
        f"- Target: `{args.target}`",
        f"- Models: `{args.models}`",
        f"- Features: `{args.features}`",
        f"- Regressors: `{args.regressors}`",
        f"- Predict sizes: `{args.predict_sizes}`",
        f"- Repeat: `{args.repeat}`",
        f"- Vectorized network prediction: `{'enabled' if args.vectorized_predict else 'disabled'}`",
        f"- CBOPs fast path: `{'enabled' if args.cbops_fast_path else 'disabled'}`",
        "",
        "## Kernel Fit Summary",
        "",
        "| Feature | Regressor | Kernel | Dim | Fit ms median | Predict max-size ms median |",
        "|---|---|---:|---:|---:|---:|",
    ]

    max_predict_size = max(parse_csv_list(args.predict_sizes, int))
    fit_rows = [r for r in rows if r["stage"] == "kernel_preprocess_fit"]
    pred_rows = [r for r in rows if r["stage"] == "kernel_predict" and r["predict_size"] == max_predict_size]
    pred_lookup = {
        (r["feature"], r["regressor"], r["kernel"]): r
        for r in pred_rows
    }
    for row in fit_rows:
        pred_row = pred_lookup.get((row["feature"], row["regressor"], row["kernel"]))
        pred_ms = pred_row["predict_median_ms"] if pred_row else ""
        lines.append(
            f"| {row['feature']} | {row['regressor']} | {row['kernel']} | "
            f"{row['feature_dim']} | {fmt_ms(row['fit_median_ms'])} | "
            f"{fmt_ms(pred_ms) if pred_ms != '' else ''} |"
        )

    lines += [
        "",
        "## Network Prediction Summary",
        "",
        "| Model | Feature | Regressor | Predict size | Predict ms median |",
        "|---|---|---|---:|---:|",
    ]
    for row in rows:
        if row["stage"] != "network_predict" or row["predict_size"] != max_predict_size:
            continue
        lines.append(
            f"| {row['model']} | {row['feature']} | {row['regressor']} | "
            f"{row['predict_size']} | {fmt_ms(row['predict_median_ms'])} |"
        )

    if any(r["stage"] == "near_constraint_sampling" for r in rows):
        lines += [
            "",
            "## Near-Constraint Summary",
            "",
            "| Model | Feature | Regressor | Fast path | Samples | Returned | Sampling ms | Evals | ms/eval |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            if row["stage"] != "near_constraint_sampling":
                continue
            lines.append(
                f"| {row['model']} | {row['feature']} | {row['regressor']} | "
                f"{row['near_fast_path']} | "
                f"{row['near_samples']} | {row['near_returned_samples']} | "
                f"{fmt_ms(row['near_sampling_ms'])} | "
                f"{row['near_constraint_evals']} | "
                f"{fmt_ms(row['near_constraint_ms_per_eval'])} |"
            )

    return lines


def main():
    parser = argparse.ArgumentParser(description="Measure alternative proxy feature overheads.")
    parser.add_argument("--target", default="latency_mico_small")
    parser.add_argument("--models", default="lenet_mnist,vgg_cifar10,resnet18_cifar100")
    parser.add_argument(
        "--features",
        default="cbops,maxq,sumq,bops,raw+",
        help="Comma-separated features: cbops, maxq, sumq, bops, raw+, ...",
    )
    parser.add_argument("--regressors", default="LinearRegression,LogRandomForest,LogXGBRegressor")
    parser.add_argument("--predict-sizes", default="1000,10000,100000")
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--near-constraint-samples", type=int, default=128)
    parser.add_argument("--constraint-factor", type=float, default=0.8)
    parser.add_argument("--near-roi", type=float, default=0.2)
    parser.add_argument("--proxy-train-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", default="output/json")
    parser.add_argument("--output-csv", default="output/proxy_overhead_ablation.csv")
    parser.add_argument("--output-md", default="predict/proxy_overhead_ablation_results.md")
    parser.add_argument(
        "--no-vectorized-predict",
        dest="vectorized_predict",
        action="store_false",
        help="Use scalar network prediction in the overhead benchmark.",
    )
    parser.set_defaults(vectorized_predict=True)
    parser.add_argument(
        "--no-cbops-fast-path",
        dest="cbops_fast_path",
        action="store_false",
        help="Disable compact [MACS, QA, WQ] feature paths for near/predict timing.",
    )
    parser.set_defaults(cbops_fast_path=True)
    parser.add_argument(
        "--no-min-max-calibration",
        dest="min_max_calibration",
        action="store_false",
    )
    parser.set_defaults(min_max_calibration=True)
    args = parser.parse_args()

    features = parse_csv_list(args.features)
    regressors = parse_csv_list(args.regressors)
    predict_sizes = parse_csv_list(args.predict_sizes, int)
    models = parse_csv_list(args.models)

    invalid = [r for r in regressors if r not in REGRESSOR_FACTORIES]
    if invalid:
        raise ValueError(f"Unsupported regressors: {invalid}")

    rows, fitted_pairs = measure_kernel_overhead(
        target=args.target,
        features=features,
        regressors=regressors,
        repeat=args.repeat,
        train_ratio=args.proxy_train_ratio,
        seed=args.seed,
        predict_sizes=predict_sizes,
    )
    extract_rows, networks = measure_network_extract(models, repeat=max(1, min(args.repeat, 3)))
    rows.extend(extract_rows)
    rows.extend(
        measure_network_predict_and_near(
            target=args.target,
            models=models,
            networks=networks,
            fitted_pairs=fitted_pairs,
            features=features,
            regressors=regressors,
            predict_sizes=predict_sizes,
            repeat=args.repeat,
            near_samples=args.near_constraint_samples,
            constraint_factor=args.constraint_factor,
            near_roi=args.near_roi,
            seed=args.seed,
            vectorized=args.vectorized_predict,
            cbops_fast_path=args.cbops_fast_path,
            min_max_calibration=args.min_max_calibration,
            cache_dir=args.cache_dir,
        )
    )

    write_csv(args.output_csv, rows)
    md = make_markdown(rows, args)
    write_markdown(args.output_md, md)
    print("\n".join(md))
    print(f"\nCSV saved to: {args.output_csv}")
    print(f"Markdown saved to: {args.output_md}")


if __name__ == "__main__":
    main()
