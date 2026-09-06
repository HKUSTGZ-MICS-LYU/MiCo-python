"""
Compare end-to-end latency predictions from two MiCo proxy configurations.

Default comparison:
  - CBOPs_LogRF: cbops features + LogRandomForest
  - Raw+_LogRF: raw+ features + LogRandomForest

The script is intentionally lightweight: it extracts layer features with a dummy
forward pass and sums kernel-level proxy predictions. If a MiCoEval latency cache
exists, cached real latency is used for optional accuracy metrics. Live hardware
simulation is not started by default.

Example:
    python predict/proxy_speed_gap.py
    python predict/proxy_speed_gap.py --target latency_bitfusion -n 128
"""

import argparse
import csv
import io
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from contextlib import nullcontext, redirect_stdout
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import mean_absolute_percentage_error, r2_score

try:
    from scipy.stats import spearmanr
except ImportError:  # pragma: no cover - scipy is available in the normal env
    spearmanr = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from MiCoQLayers import BitConv2d, BitLinear
from MiCoProxy import (
    get_bitfusion_conv2d_proxy,
    get_bitfusion_matmul_proxy,
    get_mico_conv2d_proxy,
    get_mico_matmul_proxy,
)
from models import CmsisCNN, LeNet, VGG, resnet_alt_18
from searchers.SearchUtils import near_constr_sample


DEFAULT_MODELS = ["lenet_mnist", "vgg_cifar10", "resnet18_cifar100"]


@dataclass(frozen=True)
class ProxyConfig:
    label: str
    preprocess: str
    regressor: str


PROXY_CONFIGS = [
    ProxyConfig("CBOPs_LogRF", "cbops", "LogRandomForest"),
    ProxyConfig("MaxMACs_LogRF", "maxq", "LogRandomForest"),
    ProxyConfig("SumMACs_LogRF", "sumq", "LogRandomForest"),
    ProxyConfig("BOPs_LogRF", "bops", "LogRandomForest"),
    ProxyConfig("Raw+_LogRF", "raw+", "LogRandomForest"),
]


@dataclass
class NetworkFeatures:
    model_name: str
    n_layers: int
    layer_macs: List[float]
    layer_features: List[List[float]]
    layer_kinds: List[str]


@dataclass
class Calibration:
    slope: float = 1.0
    intercept: float = 0.0
    applied: bool = False
    reason: str = "identity"

    def apply(self, value: float) -> float:
        return self.slope * value + self.intercept


def parse_csv_list(text: str, cast=str) -> List:
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def canonical_scheme_key(scheme: Sequence[int]) -> str:
    return ",".join(str(int(x)) for x in scheme)


def legacy_scheme_key(scheme: Sequence[int]) -> str:
    return str([int(x) for x in scheme])


def default_bitwidths(target: str) -> List[int]:
    if target == "latency_bitfusion":
        return [2, 3, 4, 5, 6, 7, 8]
    if target.startswith("latency_mico"):
        return [1, 2, 4, 8]
    raise ValueError(f"Unsupported target: {target}")


def instantiate_model(model_name: str) -> Tuple[torch.nn.Module, Tuple[int, ...]]:
    """Instantiate supported CNNs without dataset/checkpoint side effects."""
    if model_name == "lenet_mnist":
        return LeNet(1).cpu(), (1, 1, 28, 28)
    if model_name == "vgg_cifar10":
        return VGG(3, 10).cpu(), (1, 3, 32, 32)
    if model_name == "vgg_cifar100":
        return VGG(3, 100).cpu(), (1, 3, 32, 32)
    if model_name == "resnet18_cifar10":
        return resnet_alt_18(10).cpu(), (1, 3, 32, 32)
    if model_name == "resnet18_cifar100":
        return resnet_alt_18(100).cpu(), (1, 3, 32, 32)
    if model_name == "cmsiscnn_cifar10":
        return CmsisCNN(3).cpu(), (1, 3, 32, 32)
    raise ValueError(
        f"Unsupported model '{model_name}'. Add it to instantiate_model() "
        "with its dummy input shape."
    )


def extract_network_features(model_name: str) -> NetworkFeatures:
    model, input_shape = instantiate_model(model_name)
    model.eval()
    n_layers = model.n_layers
    model.set_qscheme([[8] * n_layers, [8] * n_layers], device=torch.device("cpu"))

    with torch.no_grad():
        model(torch.randn(*input_shape))

    layer_macs: List[float] = []
    layer_features: List[List[float]] = []
    layer_kinds: List[str] = []
    for layer in model.get_qlayers():
        if isinstance(layer, BitConv2d):
            kind = "conv2d"
        elif isinstance(layer, BitLinear):
            kind = "linear"
        else:
            continue
        macs = float(layer.get_mac())
        if macs <= 0:
            raise ValueError(
                f"{model_name}: layer {layer} has zero MACs; dummy forward did "
                "not populate layer features."
            )
        layer_macs.append(macs)
        layer_features.append([float(x) for x in layer.layer_features])
        layer_kinds.append(kind)

    if len(layer_macs) != n_layers:
        raise ValueError(
            f"{model_name}: expected {n_layers} qlayers, got {len(layer_macs)}."
        )

    return NetworkFeatures(model_name, n_layers, layer_macs, layer_features, layer_kinds)


def build_proxy_pair(target: str, config: ProxyConfig, train_ratio: float, seed: int, verbose: bool):
    stdout_ctx = nullcontext() if verbose else redirect_stdout(io.StringIO())
    with stdout_ctx:
        if target == "latency_bitfusion":
            matmul_proxy = get_bitfusion_matmul_proxy(
                preprocess=config.preprocess, regressor=config.regressor
            )
            conv2d_proxy = get_bitfusion_conv2d_proxy(
                preprocess=config.preprocess, regressor=config.regressor
            )
        elif target.startswith("latency_mico"):
            mico_type = target.split("_")[-1]
            matmul_proxy = get_mico_matmul_proxy(
                mico_type=mico_type, preprocess=config.preprocess, regressor=config.regressor
            )
            conv2d_proxy = get_mico_conv2d_proxy(
                mico_type=mico_type, preprocess=config.preprocess, regressor=config.regressor
            )
        else:
            raise ValueError(f"Unsupported target: {target}")

        for proxy in (matmul_proxy, conv2d_proxy):
            proxy.set_train_ratio(train_ratio)
            proxy.seed = seed
            proxy.fit()
    return matmul_proxy, conv2d_proxy


def predict_network_latency(
    net: NetworkFeatures,
    scheme: Sequence[int],
    matmul_proxy,
    conv2d_proxy,
    calibration: Optional[Calibration] = None,
) -> float:
    wq = np.array(scheme[: net.n_layers], dtype=float)
    aq = np.array(scheme[net.n_layers :], dtype=float)

    conv2d_features = []
    linear_features = []
    for idx in range(net.n_layers):
        features = [net.layer_macs[idx]] + net.layer_features[idx] + [aq[idx], wq[idx]]
        if net.layer_kinds[idx] == "conv2d":
            conv2d_features.append(features)
        elif net.layer_kinds[idx] == "linear":
            linear_features.append(features)

    latency = 0.0
    if conv2d_features:
        latency += float(np.sum(conv2d_proxy.predict(np.array(conv2d_features))))
    if linear_features:
        latency += float(np.sum(matmul_proxy.predict(np.array(linear_features))))
    if calibration is not None:
        latency = calibration.apply(latency)
    return latency


def generate_schemes(n_layers: int, bitwidths: Sequence[int], n_random: int, seed: int):
    rng = random.Random(seed)
    dim = n_layers * 2
    schemes = [
        ("min", [min(bitwidths)] * dim),
        ("max", [max(bitwidths)] * dim),
    ]
    for idx in range(n_random):
        schemes.append((f"rand_{idx:04d}", [rng.choice(bitwidths) for _ in range(dim)]))
    return schemes


def load_actual_cache(path: str, objective: str) -> Dict[str, float]:
    if not os.path.exists(path):
        return {}

    if path.lower().endswith(".csv"):
        values: Dict[str, float] = {}
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("objective") == objective:
                    values[row["scheme"]] = float(row["value"])
        return values

    with open(path) as f:
        data = json.load(f)

    values = {}
    for scheme_key, objectives in data.items():
        if objective in objectives:
            values[scheme_key] = float(objectives[objective])
    return values


def lookup_actual(cache: Dict[str, float], scheme: Sequence[int]) -> Optional[float]:
    return cache.get(canonical_scheme_key(scheme), cache.get(legacy_scheme_key(scheme)))


def target_objective(target: str) -> str:
    if target == "latency_bitfusion":
        return "latency_bitfusion"
    if target.startswith("latency_mico"):
        return "latency_mico"
    raise ValueError(f"Unsupported target: {target}")


def cache_path_for(cache_dir: str, model_name: str, target: str) -> str:
    return os.path.join(cache_dir, f"{model_name}_{target}_predict.json")


def safe_r2(y_true: Sequence[float], y_pred: Sequence[float]) -> Optional[float]:
    if len(y_true) < 2 or np.allclose(y_true, y_true[0]):
        return None
    return float(r2_score(y_true, y_pred))


def safe_spearman(y_true: Sequence[float], y_pred: Sequence[float]) -> Optional[float]:
    if spearmanr is None or len(y_true) < 2:
        return None
    corr, _ = spearmanr(y_true, y_pred)
    if np.isnan(corr):
        return None
    return float(corr)


def fmt_optional(value: Optional[float], suffix: str = "") -> str:
    if value is None:
        return "NA"
    return f"{value:.4f}{suffix}"


def sanitize_label(label: str) -> str:
    return (
        label.replace("+", "plus")
        .replace("_", "-")
        .replace(" ", "-")
        .lower()
    )


def csv_label(label: str) -> str:
    """Stable field prefix for arbitrary proxy labels."""
    return (
        label.lower()
        .replace("+", "plus")
        .replace(" ", "_")
        .replace("-", "_")
    )


def parse_proxy_configs(text: str) -> List[ProxyConfig]:
    """Parse LABEL=FEATURE:REGRESSOR entries for focused comparisons."""
    configs = []
    for item in parse_csv_list(text):
        try:
            label, spec = item.split("=", 1)
            feature, regressor = spec.split(":", 1)
        except ValueError as exc:
            raise ValueError(
                f"Invalid proxy config '{item}', expected LABEL=FEATURE:REGRESSOR"
            ) from exc
        configs.append(ProxyConfig(label, feature, regressor))
    if not configs:
        raise ValueError("At least one proxy config is required")
    return configs


def write_csv(path: str, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: str, lines: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def predict_schemes(
    net: NetworkFeatures,
    schemes: Sequence[Tuple[str, Sequence[int]]],
    proxies: Dict[str, Tuple[object, object]],
    calibrations: Optional[Dict[str, Calibration]] = None,
    vectorized: bool = True,
) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
    if vectorized:
        return predict_schemes_vectorized(net, schemes, proxies, calibrations)
    return predict_schemes_scalar(net, schemes, proxies, calibrations)


def predict_schemes_scalar(
    net: NetworkFeatures,
    schemes: Sequence[Tuple[str, Sequence[int]]],
    proxies: Dict[str, Tuple[object, object]],
    calibrations: Optional[Dict[str, Calibration]] = None,
) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
    predictions: Dict[str, List[float]] = {label: [] for label in proxies}
    prediction_ms: Dict[str, float] = {}
    for label, (matmul_proxy, conv2d_proxy) in proxies.items():
        start = time.perf_counter()
        calibration = calibrations.get(label) if calibrations else None
        predictions[label] = [
            predict_network_latency(net, scheme, matmul_proxy, conv2d_proxy, calibration)
            for _, scheme in schemes
        ]
        prediction_ms[label] = (time.perf_counter() - start) * 1000.0
    return predictions, prediction_ms


def build_batched_layer_features(
    net: NetworkFeatures,
    schemes: Sequence[Tuple[str, Sequence[int]]],
):
    conv2d_features = []
    conv2d_scheme_idx = []
    linear_features = []
    linear_scheme_idx = []

    for scheme_idx, (_, scheme) in enumerate(schemes):
        wq = np.array(scheme[: net.n_layers], dtype=float)
        aq = np.array(scheme[net.n_layers :], dtype=float)
        for layer_idx in range(net.n_layers):
            features = (
                [net.layer_macs[layer_idx]]
                + net.layer_features[layer_idx]
                + [aq[layer_idx], wq[layer_idx]]
            )
            if net.layer_kinds[layer_idx] == "conv2d":
                conv2d_features.append(features)
                conv2d_scheme_idx.append(scheme_idx)
            elif net.layer_kinds[layer_idx] == "linear":
                linear_features.append(features)
                linear_scheme_idx.append(scheme_idx)

    return (
        np.array(conv2d_features, dtype=float),
        np.array(conv2d_scheme_idx, dtype=int),
        np.array(linear_features, dtype=float),
        np.array(linear_scheme_idx, dtype=int),
    )


def predict_schemes_vectorized(
    net: NetworkFeatures,
    schemes: Sequence[Tuple[str, Sequence[int]]],
    proxies: Dict[str, Tuple[object, object]],
    calibrations: Optional[Dict[str, Calibration]] = None,
) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
    predictions: Dict[str, List[float]] = {label: [] for label in proxies}
    prediction_ms: Dict[str, float] = {}
    (
        conv2d_features,
        conv2d_scheme_idx,
        linear_features,
        linear_scheme_idx,
    ) = build_batched_layer_features(net, schemes)

    for label, (matmul_proxy, conv2d_proxy) in proxies.items():
        start = time.perf_counter()
        pred = np.zeros(len(schemes), dtype=float)
        if len(conv2d_features):
            layer_pred = conv2d_proxy.predict(conv2d_features)
            np.add.at(pred, conv2d_scheme_idx, layer_pred)
        if len(linear_features):
            layer_pred = matmul_proxy.predict(linear_features)
            np.add.at(pred, linear_scheme_idx, layer_pred)
        calibration = calibrations.get(label) if calibrations else None
        if calibration is not None:
            pred = calibration.slope * pred + calibration.intercept
        predictions[label] = pred.tolist()
        prediction_ms[label] = (time.perf_counter() - start) * 1000.0

    return predictions, prediction_ms


def build_min_max_calibrations(
    net: NetworkFeatures,
    bitwidths: Sequence[int],
    proxies: Dict[str, Tuple[object, object]],
    actual_cache: Dict[str, float],
    enabled: bool,
) -> Dict[str, Calibration]:
    dims = net.n_layers * 2
    min_scheme = [min(bitwidths)] * dims
    max_scheme = [max(bitwidths)] * dims
    actual_min = lookup_actual(actual_cache, min_scheme)
    actual_max = lookup_actual(actual_cache, max_scheme)

    calibrations: Dict[str, Calibration] = {}
    for label, (matmul_proxy, conv2d_proxy) in proxies.items():
        if not enabled:
            calibrations[label] = Calibration(reason="disabled")
            continue
        if actual_min is None or actual_max is None:
            calibrations[label] = Calibration(reason="missing cached min/max actual latency")
            continue

        pred_min = predict_network_latency(net, min_scheme, matmul_proxy, conv2d_proxy)
        pred_max = predict_network_latency(net, max_scheme, matmul_proxy, conv2d_proxy)
        if pred_max == pred_min:
            calibrations[label] = Calibration(reason="degenerate predicted min/max")
            continue

        slope = (actual_max - actual_min) / (pred_max - pred_min)
        intercept = actual_min - slope * pred_min
        calibrations[label] = Calibration(
            slope=slope,
            intercept=intercept,
            applied=True,
            reason="cached min/max actual latency",
        )

    return calibrations


def append_scheme_rows(
    rows: List[Dict[str, object]],
    model_name: str,
    target: str,
    sample_set: str,
    sampling_proxy: str,
    schemes: Sequence[Tuple[str, Sequence[int]]],
    predictions: Dict[str, List[float]],
    pred_baseline: Dict[str, float],
    actuals: Sequence[Optional[float]],
    actual_baseline: Optional[float],
) -> None:
    for idx, (scheme_id, scheme) in enumerate(schemes):
        actual = actuals[idx]
        row = {
            "model": model_name,
            "target": target,
            "sample_set": sample_set,
            "sampling_proxy": sampling_proxy,
            "scheme_id": scheme_id,
            "scheme": canonical_scheme_key(scheme),
            "actual_latency": "" if actual is None else actual,
            "actual_speedup_vs_max": (
                "" if actual is None or actual_baseline is None else actual_baseline / actual
            ),
        }
        for label, values in predictions.items():
            field = csv_label(label)
            row[f"{field}_pred"] = values[idx]
            row[f"{field}_speedup_vs_max"] = pred_baseline[label] / values[idx]
        rows.append(row)


def build_near_constraint_schemes(
    net: NetworkFeatures,
    bitwidths: Sequence[int],
    proxy_label: str,
    proxy_pair,
    calibration: Calibration,
    constraint_factor: float,
    roi: float,
    n_samples: int,
    seed: int,
    initial_pop: Optional[List[List[int]]],
    pbar: bool,
) -> Tuple[List[Tuple[str, List[int]]], Dict[str, float]]:
    if constraint_factor <= roi:
        raise ValueError(
            f"constraint_factor ({constraint_factor}) must be larger than "
            f"near ROI ({roi}) because near_constr_sample asserts cur_ratio > roi."
        )

    matmul_proxy, conv2d_proxy = proxy_pair
    dims = net.n_layers * 2

    def constr_func(scheme):
        return predict_network_latency(net, scheme, matmul_proxy, conv2d_proxy, calibration)

    max_scheme = [max(bitwidths)] * dims
    min_scheme = [min(bitwidths)] * dims
    max_latency = constr_func(max_scheme)
    min_latency = constr_func(min_scheme)
    requested_constraint_value = constraint_factor * max_latency
    constraint_value = max(requested_constraint_value, min_latency)
    min_ratio = constraint_value / max_latency
    if min_ratio <= roi:
        constraint_value = (roi + 1e-6) * max_latency

    random.seed(seed)
    np.random.seed(seed)
    start = time.perf_counter()
    schemes = near_constr_sample(
        n_samples=n_samples,
        qtypes=list(bitwidths),
        dims=dims,
        constr_func=constr_func,
        constr_value=constraint_value,
        roi=roi,
        initial_pop=initial_pop,
        pbar=pbar,
    )
    sampling_ms = (time.perf_counter() - start) * 1000.0

    tagged = [
        (f"near_{sanitize_label(proxy_label)}_{idx:04d}", scheme)
        for idx, scheme in enumerate(schemes)
    ]
    stats = {
        "sampling_ms": sampling_ms,
        "constraint_value": constraint_value,
        "requested_constraint_value": requested_constraint_value,
        "max_latency": max_latency,
        "min_latency": min_latency,
        "n_returned": len(tagged),
    }
    return tagged, stats


def run(args):
    models = parse_csv_list(args.models)
    bitwidths = parse_csv_list(args.bitwidths, int) if args.bitwidths else default_bitwidths(args.target)
    objective = target_objective(args.target)
    configs = (
        parse_proxy_configs(args.proxy_configs)
        if getattr(args, "proxy_configs", None)
        else PROXY_CONFIGS
    )
    labels = [config.label for config in configs]
    reference_label = args.reference_proxy
    if reference_label not in labels:
        raise ValueError(
            f"Reference proxy '{reference_label}' is not present in configs: {labels}"
        )

    proxies = {
        cfg.label: build_proxy_pair(
            args.target, cfg, args.proxy_train_ratio, args.proxy_seed, args.verbose_proxy
        )
        for cfg in configs
    }

    all_rows = []
    md = [
        "# Proxy Speed Gap",
        "",
        f"- Target: `{args.target}`",
        f"- Bitwidths: `{bitwidths}`",
        f"- Random schemes per model: `{args.num_samples}`",
        f"- Near-constraint schemes per proxy: `{args.near_constraint_samples}`",
        f"- Near-constraint factor: `{args.constraint_factor}`",
        f"- Near-constraint ROI: `{args.near_roi}`",
        f"- Min/max calibration: `{'enabled' if args.min_max_calibration else 'disabled'}`",
        f"- Vectorized prediction: `{'enabled' if args.vectorized_predict else 'disabled'}`",
        f"- Proxy configs: `{', '.join(labels)}`",
        f"- Reference proxy: `{reference_label}`",
        f"- Proxy train ratio: `{args.proxy_train_ratio}`",
        "",
    ]

    for model_idx, model_name in enumerate(models):
        net = extract_network_features(model_name)
        schemes = generate_schemes(
            net.n_layers, bitwidths, args.num_samples, args.seed + model_idx
        )
        actual_cache = load_actual_cache(cache_path_for(args.cache_dir, model_name, args.target), objective)
        calibrations = build_min_max_calibrations(
            net=net,
            bitwidths=bitwidths,
            proxies=proxies,
            actual_cache=actual_cache,
            enabled=args.min_max_calibration,
        )

        predictions, prediction_ms = predict_schemes(
            net, schemes, proxies, calibrations, vectorized=args.vectorized_predict
        )

        baseline = {
            label: preds[1] for label, preds in predictions.items()
        }  # scheme[1] is all-max precision

        actuals = [lookup_actual(actual_cache, scheme) for _, scheme in schemes]
        actual_baseline = actuals[1]

        actual_indices = [idx for idx, value in enumerate(actuals[2:], start=2) if value is not None]
        actual_summary = {}
        if actual_indices:
            y_true = np.array([actuals[idx] for idx in actual_indices], dtype=float)
            for label in proxies:
                y_pred = np.array([predictions[label][idx] for idx in actual_indices], dtype=float)
                actual_summary[label] = {
                    "mape": float(mean_absolute_percentage_error(y_true, y_pred) * 100.0),
                    "r2": safe_r2(y_true, y_pred),
                    "spearman": safe_spearman(y_true, y_pred),
                }

        md.append(f"## {model_name}")
        md.append("")
        md.append(f"- Layers: `{net.n_layers}`, total MACs: `{int(np.sum(net.layer_macs))}`")
        for label, calibration in calibrations.items():
            if calibration.applied:
                md.append(
                    f"- {label} min/max calibration: slope `{calibration.slope:.6g}`, "
                    f"intercept `{calibration.intercept:.2f}`"
                )
            else:
                md.append(f"- {label} min/max calibration: not applied ({calibration.reason})")
        md.append(
            "- Prediction wall time: "
            + ", ".join(f"{label} `{prediction_ms[label]:.3f} ms`" for label in labels)
        )
        reference_pred = np.asarray(predictions[reference_label][2:], dtype=float)
        for label in labels:
            if label == reference_label:
                continue
            candidate_pred = np.asarray(predictions[label][2:], dtype=float)
            gap_pct = (
                np.abs(reference_pred - candidate_pred)
                / np.maximum(np.abs(candidate_pred), 1.0)
                * 100.0
            )
            md.append(
                f"- Predicted latency gap vs `{reference_label}` for `{label}`: "
                f"mean `{np.mean(gap_pct):.2f}%`, median `{np.median(gap_pct):.2f}%`, "
                f"max `{np.max(gap_pct):.2f}%`"
            )
        if actual_indices:
            md.append(f"- Cached actual latency found for `{len(actual_indices)}/{args.num_samples}` random schemes")
            for label, metrics in actual_summary.items():
                md.append(
                    f"- {label} vs actual: MAPE `{metrics['mape']:.2f}%`, "
                    f"R2 `{fmt_optional(metrics['r2'])}`, "
                    f"Spearman `{fmt_optional(metrics['spearman'])}`"
                )
        else:
            md.append("- Cached actual latency found for `0` random schemes; accuracy metrics skipped")
        md.append("")

        append_scheme_rows(
            all_rows,
            model_name,
            args.target,
            "random",
            "",
            schemes,
            predictions,
            baseline,
            actuals,
            actual_baseline,
        )

        if args.near_constraint_samples > 0:
            md.append("**Near-Constraint Samples**")
            md.append("")
            initial_pop = None
            if args.near_initial_pop == "random":
                initial_pop = [scheme for _, scheme in schemes[2:]]
            near_sets = {}
            for near_idx, (source_label, proxy_pair) in enumerate(proxies.items()):
                near_schemes, near_stats = build_near_constraint_schemes(
                    net=net,
                    bitwidths=bitwidths,
                    proxy_label=source_label,
                    proxy_pair=proxy_pair,
                    calibration=calibrations[source_label],
                    constraint_factor=args.constraint_factor,
                    roi=args.near_roi,
                    n_samples=args.near_constraint_samples,
                    seed=args.seed + args.near_seed_offset + model_idx * 97 + near_idx,
                    initial_pop=initial_pop,
                    pbar=args.near_pbar,
                )
                near_sets[source_label] = {canonical_scheme_key(s) for _, s in near_schemes}
                near_predictions, near_prediction_ms = predict_schemes(
                    net,
                    near_schemes,
                    proxies,
                    calibrations,
                    vectorized=args.vectorized_predict,
                )
                near_actuals = [lookup_actual(actual_cache, scheme) for _, scheme in near_schemes]
                source_pred = np.array(near_predictions[source_label])
                source_ratio = (
                    source_pred / near_stats["constraint_value"]
                    if len(source_pred) else np.array([])
                )
                md.append(
                    f"- Generated by `{source_label}`: `{near_stats['n_returned']}` schemes, "
                    f"sampling `{near_stats['sampling_ms']:.3f} ms`, "
                    f"constraint `{near_stats['constraint_value']:.2f}` "
                    f"({args.constraint_factor:.2f}x max-precision predicted latency)"
                )
                if near_stats["constraint_value"] > near_stats["requested_constraint_value"]:
                    md.append(
                        f"- `{source_label}` requested constraint was infeasible under all-min prediction; "
                        f"relaxed from `{near_stats['requested_constraint_value']:.2f}` "
                        f"to `{near_stats['constraint_value']:.2f}`"
                    )
                if len(source_ratio):
                    md.append(
                        f"- `{source_label}` candidate ratio to its constraint: "
                        f"mean `{np.mean(source_ratio):.4f}`, "
                        f"min `{np.min(source_ratio):.4f}`, max `{np.max(source_ratio):.4f}`"
                    )
                    md.append(
                        f"- Cross-proxy prediction time on `{source_label}` near set: "
                        + ", ".join(
                            f"{label} `{near_prediction_ms[label]:.3f} ms`"
                            for label in labels
                        )
                    )
                    for label in labels:
                        if label == reference_label:
                            continue
                        candidate_pred = np.asarray(near_predictions[label], dtype=float)
                        gap_pct = (
                            np.abs(np.asarray(near_predictions[reference_label]) - candidate_pred)
                            / np.maximum(np.abs(candidate_pred), 1.0)
                            * 100.0
                        )
                        md.append(
                            f"- Near-set prediction gap `{reference_label}` vs `{label}`: "
                            f"mean `{np.mean(gap_pct):.2f}%`, max `{np.max(gap_pct):.2f}%`"
                        )
                else:
                    md.append(f"- `{source_label}` near-constr sampler returned no schemes")

                append_scheme_rows(
                    all_rows,
                    model_name,
                    args.target,
                    f"near_{sanitize_label(source_label)}",
                    source_label,
                    near_schemes,
                    near_predictions,
                    baseline,
                    near_actuals,
                    actual_baseline,
                )

            if len(near_sets) >= 2:
                other_labels = [label for label in near_sets if label != reference_label]
                for other_label in other_labels:
                    overlap = len(near_sets[reference_label] & near_sets[other_label])
                    union = len(near_sets[reference_label] | near_sets[other_label])
                    jaccard = overlap / union if union else 0.0
                    md.append(
                        f"- Near-set overlap `{reference_label}` vs `{other_label}`: "
                        f"`{overlap}` shared, Jaccard `{jaccard:.4f}`"
                    )
            md.append("")

    write_csv(args.output_csv, all_rows)
    write_markdown(args.output_md, md)
    return md


def main():
    parser = argparse.ArgumentParser(
        description="Compare CBOPs_LogRF and Raw+_LogRF network-level proxy latency predictions."
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model names. Defaults to LeNet, VGG, ResNet18.",
    )
    parser.add_argument(
        "--target",
        default="latency_mico_small",
        help="latency_mico_small, latency_mico_high, latency_mico_cacheless, or latency_bitfusion.",
    )
    parser.add_argument("-n", "--num-samples", type=int, default=64, help="Random schemes per model.")
    parser.add_argument(
        "--near-constraint-samples",
        type=int,
        default=64,
        help="Near-constraint samples generated per proxy. Set to 0 to disable.",
    )
    parser.add_argument(
        "--constraint-factor",
        type=float,
        default=0.8,
        help="Constraint value as a fraction of each proxy's all-max predicted latency.",
    )
    parser.add_argument(
        "--near-roi",
        type=float,
        default=0.2,
        help="ROI width passed to near_constr_sample.",
    )
    parser.add_argument(
        "--near-initial-pop",
        choices=["none", "random"],
        default="random",
        help="Seed near_constr_sample with the random schemes from this script.",
    )
    parser.add_argument("--near-seed-offset", type=int, default=10000, help="Seed offset for near-constr sampling.")
    parser.add_argument("--near-pbar", action="store_true", help="Show near_constr_sample progress bars.")
    parser.add_argument(
        "--no-min-max-calibration",
        dest="min_max_calibration",
        action="store_false",
        help="Disable batch_e2e-style min/max affine calibration.",
    )
    parser.set_defaults(min_max_calibration=True)
    parser.add_argument(
        "--no-vectorized-predict",
        dest="vectorized_predict",
        action="store_false",
        help="Use the old per-scheme proxy prediction loop.",
    )
    parser.set_defaults(vectorized_predict=True)
    parser.add_argument("--seed", type=int, default=0, help="Random scheme seed.")
    parser.add_argument("--proxy-seed", type=int, default=42, help="Proxy train subset seed.")
    parser.add_argument("--proxy-train-ratio", type=float, default=1.0, help="Proxy training data ratio.")
    parser.add_argument(
        "--proxy-configs",
        default=None,
        help=(
            "Optional comma-separated LABEL=FEATURE:REGRESSOR entries. "
            "Default compares CBOPs, maxq, sumq, bops, and raw+."
        ),
    )
    parser.add_argument(
        "--reference-proxy",
        default="CBOPs_LogRF",
        help="Proxy used as the reference when reporting prediction gaps.",
    )
    parser.add_argument("--bitwidths", default=None, help="Comma-separated bitwidths; default depends on target.")
    parser.add_argument("--cache-dir", default="output/json", help="MiCoEval latency cache directory.")
    parser.add_argument("--output-csv", default="output/proxy_speed_gap.csv", help="Detailed CSV output.")
    parser.add_argument("--output-md", default="predict/proxy_speed_gap_results.md", help="Markdown summary output.")
    parser.add_argument("--verbose-proxy", action="store_true", help="Show proxy cross-validation logs.")
    args = parser.parse_args()

    summary = run(args)
    print("\n".join(summary))
    print(f"\nCSV saved to: {args.output_csv}")
    print(f"Markdown saved to: {args.output_md}")


if __name__ == "__main__":
    main()
