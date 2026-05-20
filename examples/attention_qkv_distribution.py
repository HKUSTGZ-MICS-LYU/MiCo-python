import argparse
import copy
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import cct_2, cct_7, tiny_waveformer
from models import model_zoo
from models.utils import AttentionScore, LinearAttentionScore


ATTENTION_SCORE_TYPES = (AttentionScore, LinearAttentionScore)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Q/K/V activation distributions for regular and BitNet-style "
            "attention models. Hooks AttentionScore and LinearAttentionScore inputs."
        )
    )
    parser.add_argument(
        "model_name",
        type=str,
        help=(
            "Model zoo name, e.g. cct2_cifar10, cct7_cifar100, waveformer. "
            "Use --synthetic to avoid loading datasets."
        ),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["regular", "bitnet1", "bitnet158"],
        choices=["regular", "bitnet1", "bitnet158"],
        help="Model variants to analyze.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--act-quant", type=float, default=8)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--use-norm", action="store_true")
    parser.add_argument(
        "--drop-bias",
        action="store_true",
        help="Create BitNet layers without bias. Use this if the BitNet checkpoint was trained that way.",
    )
    parser.add_argument(
        "--keep-first",
        action="store_true",
        help="Keep the first quantizable layer at W8A8 for BitNet variants.",
    )
    parser.add_argument(
        "--keep-last",
        action="store_true",
        help="Keep the last quantizable layer at W8A8 for BitNet variants.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="output/ckpt",
        help="Directory containing <model>.pth and <model>_bitnet.pth.",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Skip checkpoint loading and analyze randomly initialized models.",
    )
    parser.add_argument(
        "--strict-load",
        action="store_true",
        help="Require checkpoint state dicts to match exactly.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="test",
        help="Dataset split to sample when --synthetic is not set.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use random inputs instead of loading the model zoo dataset.",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=16000,
        help="Synthetic WaveFormer input length.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="output/attention_qkv")
    parser.add_argument(
        "--max-samples-per-call",
        type=int,
        default=20000,
        help="Maximum flattened values sampled from each Q/K/V tensor per forward call.",
    )
    parser.add_argument(
        "--max-samples-per-key",
        type=int,
        default=200000,
        help="Maximum stored values per variant/layer/QKV key.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=120,
        help="Histogram bins for generated plots.",
    )
    parser.add_argument(
        "--top-outliers",
        type=int,
        default=20,
        help="Number of largest absolute output values to save in JSON statistics.",
    )
    parser.add_argument(
        "--value-batch-index",
        type=int,
        default=0,
        help="Batch element used for direct Q/K/V per-neuron value plots.",
    )
    parser.add_argument(
        "--value-token-index",
        type=int,
        default=0,
        help="Token/time index used for direct Q/K/V per-neuron value plots.",
    )
    parser.add_argument(
        "--no-qkv-value-plots",
        action="store_true",
        help="Disable direct per-neuron Q/K/V value heatmaps.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Only write JSON statistics.",
    )
    return parser.parse_args()


def build_synthetic_model(model_name, device):
    name = model_name.lower()
    if name in ["cct2", "cct2_cifar10"]:
        return cct_2(10).to(device), (3, 32, 32)
    if name in ["cct7", "cct7_cifar10"]:
        return cct_7(10).to(device), (3, 32, 32)
    if name == "cct7_cifar100":
        return cct_7(100).to(device), (3, 32, 32)
    if name == "waveformer":
        return tiny_waveformer(n_classes=35).to(device), None
    raise ValueError(
        f"Synthetic input is not configured for {model_name}. "
        "Add its input shape to build_synthetic_model(), or run without --synthetic."
    )


def load_base_model_and_inputs(args, device):
    if args.synthetic:
        model, image_shape = build_synthetic_model(args.model_name, device)
        if args.model_name.lower() == "waveformer":
            input_shape = (1, args.input_len)
        else:
            input_shape = image_shape
        return model, synthetic_batches(args.batch_size, args.batches, input_shape, device)

    model, train_loader, test_loader = model_zoo.from_zoo(
        args.model_name, shuffle=False, batch_size=args.batch_size
    )
    loader = train_loader if args.split == "train" else test_loader
    return model.to(device), data_batches(loader, args.batches, device)


def synthetic_batches(batch_size, num_batches, input_shape, device):
    for _ in range(num_batches):
        yield torch.randn((batch_size, *input_shape), device=device)


def data_batches(loader, num_batches, device):
    for i, batch in enumerate(loader):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        yield x.to(device)
        if i + 1 >= num_batches:
            break


def checkpoint_path(args, variant):
    if variant == "regular":
        return os.path.join(args.checkpoint_dir, f"{args.model_name}.pth")
    return os.path.join(args.checkpoint_dir, f"{args.model_name}_bitnet.pth")


def maybe_load_checkpoint(model, args, variant, device):
    if args.no_checkpoint:
        return "checkpoint loading disabled"

    path = checkpoint_path(args, variant)
    if not os.path.exists(path):
        return f"checkpoint not found: {path}"

    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    try:
        model.load_state_dict(state, strict=args.strict_load)
        return f"loaded {path}"
    except RuntimeError as exc:
        if args.strict_load:
            raise
        missing, unexpected = model.load_state_dict(state, strict=False)
        return (
            f"loaded {path} with strict=False "
            f"(missing={len(missing)}, unexpected={len(unexpected)}); first error: {exc}"
        )


def bitnet_qscheme(model, weight_quant, act_quant, keep_first, keep_last):
    qscheme = [
        [weight_quant] * model.n_layers,
        [act_quant] * model.n_layers,
    ]
    if keep_first and model.n_layers > 0:
        qscheme[0][0] = 8
        qscheme[1][0] = 8
    if keep_last and model.n_layers > 0:
        qscheme[0][-1] = 8
        qscheme[1][-1] = 8
    return qscheme


def prepare_variant(base_model, args, variant, device):
    model = copy.deepcopy(base_model).to(device)
    if variant == "regular":
        status = maybe_load_checkpoint(model, args, variant, device)
        return model.eval(), status

    weight_quant = 1 if variant == "bitnet1" else 1.5
    if args.group_size != 1:
        raise ValueError(
            "--group-size is currently only suitable for regular integer-bit "
            "MiCo quantization; use --group-size 1 for bitnet1/bitnet158."
        )
    qscheme = bitnet_qscheme(
        model,
        weight_quant=weight_quant,
        act_quant=args.act_quant,
        keep_first=args.keep_first,
        keep_last=args.keep_last,
    )
    model.set_qscheme(
        qscheme,
        qat=True,
        device=device,
        group_size=args.group_size,
        use_bias=not args.drop_bias,
        use_norm=args.use_norm,
    )
    status = maybe_load_checkpoint(model, args, variant, device)
    return model.eval(), status


class QKVCapture:
    def __init__(self, max_samples_per_call, max_samples_per_key, value_batch_index, value_token_index):
        self.max_samples_per_call = max_samples_per_call
        self.max_samples_per_key = max_samples_per_key
        self.value_batch_index = value_batch_index
        self.value_token_index = value_token_index
        self.samples = defaultdict(list)
        self.counts = defaultdict(int)
        self.direct_values = {}

    def add(self, layer_name, qkv_name, tensor):
        self.add_direct_value(layer_name, qkv_name, tensor)

        flat = tensor.detach().float().reshape(-1)
        self.counts[(layer_name, qkv_name)] += int(flat.numel())
        if flat.numel() == 0:
            return

        if flat.numel() > self.max_samples_per_call:
            idx = torch.linspace(
                0,
                flat.numel() - 1,
                steps=self.max_samples_per_call,
                device=flat.device,
            ).long()
            flat = flat.index_select(0, idx)

        current = sum(arr.size for arr in self.samples[(layer_name, qkv_name)])
        remaining = self.max_samples_per_key - current
        if remaining <= 0:
            return
        flat = flat[:remaining]
        self.samples[(layer_name, qkv_name)].append(flat.cpu().numpy())

    def add_direct_value(self, layer_name, qkv_name, tensor):
        if qkv_name not in ["q", "k", "v"]:
            return

        key = (layer_name, qkv_name)
        if key in self.direct_values:
            return

        t = tensor.detach().float().cpu()
        if t.dim() < 3:
            return

        batch_index = normalize_index(self.value_batch_index, t.shape[0])
        if t.dim() == 4:
            # Attention tensors in this repo use [batch, heads, tokens, head_dim].
            token_index = normalize_index(self.value_token_index, t.shape[2])
            values = t[batch_index, :, token_index, :].numpy()
            axes = ["head", "head_dim"]
            selected_shape = list(values.shape)
        elif t.dim() == 3:
            # Fallback for [batch, tokens, channels] style attention tensors.
            token_index = normalize_index(self.value_token_index, t.shape[1])
            values = t[batch_index, token_index, :].unsqueeze(0).numpy()
            axes = ["head", "channel"]
            selected_shape = list(values.shape)
        else:
            return

        self.direct_values[key] = {
            "values": values,
            "source_shape": list(t.shape),
            "selected_batch_index": int(batch_index),
            "selected_token_index": int(token_index),
            "selected_shape": selected_shape,
            "axes": axes,
        }

    def arrays(self):
        result = {}
        for key, chunks in self.samples.items():
            if chunks:
                result[key] = np.concatenate(chunks)
            else:
                result[key] = np.array([], dtype=np.float32)
        return result


def normalize_index(index, size):
    if size <= 0:
        return 0
    if index < 0:
        index = size + index
    return max(0, min(index, size - 1))


def register_qkv_hooks(model, capture):
    handles = []

    def make_hook(layer_name):
        def hook(_module, inputs):
            if len(inputs) < 3:
                return
            for qkv_name, tensor in zip(["q", "k", "v"], inputs[:3]):
                capture.add(layer_name, qkv_name, tensor)

        return hook

    for name, module in model.named_modules():
        if isinstance(module, ATTENTION_SCORE_TYPES):
            handles.append(module.register_forward_pre_hook(make_hook(name)))

    return handles


def summarize_array(values, raw_count):
    if values.size == 0:
        return {"raw_count": raw_count, "sample_count": 0}

    percentiles = np.percentile(values, [1, 5, 25, 50, 75, 95, 99, 99.9])
    return {
        "raw_count": int(raw_count),
        "sample_count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p01": float(percentiles[0]),
        "p05": float(percentiles[1]),
        "p25": float(percentiles[2]),
        "p50": float(percentiles[3]),
        "p75": float(percentiles[4]),
        "p95": float(percentiles[5]),
        "p99": float(percentiles[6]),
        "p999": float(percentiles[7]),
        "max": float(np.max(values)),
        "abs_mean": float(np.mean(np.abs(values))),
        "rms": float(np.sqrt(np.mean(values * values))),
        "zero_frac": float(np.mean(values == 0)),
        "negative_frac": float(np.mean(values < 0)),
        "positive_frac": float(np.mean(values > 0)),
    }


def summarize_outliers(values, top_k):
    if values.size == 0 or top_k <= 0:
        return []

    k = min(top_k, values.size)
    top_idx = np.argpartition(np.abs(values), -k)[-k:]
    top_idx = top_idx[np.argsort(np.abs(values[top_idx]))[::-1]]
    return [
        {
            "sample_index": int(idx),
            "value": float(values[idx]),
            "abs_value": float(abs(values[idx])),
        }
        for idx in top_idx
    ]


def run_capture(model, batches, args):
    capture = QKVCapture(
        args.max_samples_per_call,
        args.max_samples_per_key,
        args.value_batch_index,
        args.value_token_index,
    )
    handles = register_qkv_hooks(model, capture)
    if not handles:
        raise RuntimeError(
            "No AttentionScore or LinearAttentionScore modules were found. "
            "This script captures Q/K/V at those modules."
        )

    with torch.no_grad():
        for x in batches:
            output = model(x)
            if isinstance(output, (list, tuple)):
                output = output[0]
            capture.add("model_output", "output", output)

    for handle in handles:
        handle.remove()

    arrays = capture.arrays()
    per_layer = {}
    aggregate_chunks = defaultdict(list)
    aggregate_counts = defaultdict(int)

    for (layer_name, qkv_name), values in arrays.items():
        per_layer.setdefault(layer_name, {})[qkv_name] = summarize_array(
            values, capture.counts[(layer_name, qkv_name)]
        )
        aggregate_chunks[qkv_name].append(values)
        aggregate_counts[qkv_name] += capture.counts[(layer_name, qkv_name)]

    aggregate = {}
    for qkv_name in ["q", "k", "v"]:
        chunks = aggregate_chunks.get(qkv_name, [])
        values = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
        aggregate[qkv_name] = summarize_array(values, aggregate_counts[qkv_name])

    output_values = arrays.get(("model_output", "output"), np.array([], dtype=np.float32))
    output_stats = summarize_array(
        output_values,
        capture.counts[("model_output", "output")],
    )
    output_stats["top_abs_values"] = summarize_outliers(output_values, args.top_outliers)

    direct_values = serialize_direct_values(capture.direct_values)

    return arrays, {
        "aggregate": aggregate,
        "output": output_stats,
        "direct_qkv_values": direct_values,
        "per_layer": per_layer,
    }


def serialize_direct_values(direct_values):
    result = {}
    for (layer_name, qkv_name), entry in direct_values.items():
        result.setdefault(layer_name, {})[qkv_name] = {
            "source_shape": entry["source_shape"],
            "selected_batch_index": entry["selected_batch_index"],
            "selected_token_index": entry["selected_token_index"],
            "selected_shape": entry["selected_shape"],
            "axes": entry["axes"],
            "values": entry["values"].tolist(),
        }
    return result


def safe_name(name):
    return name.replace("/", "_").replace(" ", "_")


def plot_variant_histograms(model_name, variant, arrays, output_dir, bins):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, qkv_name in zip(axes, ["q", "k", "v"]):
        chunks = [
            values
            for (layer_name, key), values in arrays.items()
            if key == qkv_name and values.size > 0
        ]
        if chunks:
            values = np.concatenate(chunks)
            ax.hist(values, bins=bins, density=True, alpha=0.8)
        ax.set_title(qkv_name.upper())
        ax.set_xlabel("value")
        ax.set_ylabel("density")
        ax.grid(alpha=0.25)

    fig.suptitle(f"{model_name} {variant} Q/K/V distributions")
    fig.tight_layout()
    path = os.path.join(output_dir, f"{safe_name(model_name)}_{variant}_qkv_hist.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_comparison(model_name, variant_arrays, output_dir, bins):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, qkv_name in zip(axes, ["q", "k", "v"]):
        for variant, arrays in variant_arrays.items():
            chunks = [
                values
                for (_layer_name, key), values in arrays.items()
                if key == qkv_name and values.size > 0
            ]
            if not chunks:
                continue
            values = np.concatenate(chunks)
            ax.hist(values, bins=bins, density=True, histtype="step", linewidth=1.5, label=variant)
        ax.set_title(qkv_name.upper())
        ax.set_xlabel("value")
        ax.set_ylabel("density")
        ax.grid(alpha=0.25)
        ax.legend()

    fig.suptitle(f"{model_name} Q/K/V distribution comparison")
    fig.tight_layout()
    path = os.path.join(output_dir, f"{safe_name(model_name)}_qkv_compare.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_output_outliers(model_name, variant, arrays, output_dir, bins):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = arrays.get(("model_output", "output"), np.array([], dtype=np.float32))
    if values.size == 0:
        return None

    abs_sorted = np.sort(np.abs(values))
    ranks = np.arange(abs_sorted.size)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(values, bins=bins, density=True, alpha=0.8)
    axes[0].axvline(np.percentile(values, 99), color="tab:red", linestyle="--", linewidth=1.2, label="p99")
    axes[0].axvline(np.percentile(values, 99.9), color="tab:purple", linestyle="--", linewidth=1.2, label="p99.9")
    axes[0].set_title("Output Distribution")
    axes[0].set_xlabel("value")
    axes[0].set_ylabel("density")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(ranks, abs_sorted, linewidth=1.3)
    axes[1].axhline(np.percentile(np.abs(values), 99), color="tab:red", linestyle="--", linewidth=1.2, label="abs p99")
    axes[1].axhline(np.percentile(np.abs(values), 99.9), color="tab:purple", linestyle="--", linewidth=1.2, label="abs p99.9")
    axes[1].set_title("Sorted Absolute Output")
    axes[1].set_xlabel("sample rank")
    axes[1].set_ylabel("abs(value)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.suptitle(f"{model_name} {variant} output outlier view")
    fig.tight_layout()
    path = os.path.join(output_dir, f"{safe_name(model_name)}_{variant}_output_outliers.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_output_comparison(model_name, variant_arrays, output_dir, bins):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    has_data = False
    for variant, arrays in variant_arrays.items():
        values = arrays.get(("model_output", "output"), np.array([], dtype=np.float32))
        if values.size == 0:
            continue
        has_data = True
        axes[0].hist(values, bins=bins, density=True, histtype="step", linewidth=1.5, label=variant)
        abs_sorted = np.sort(np.abs(values))
        axes[1].plot(np.arange(abs_sorted.size), abs_sorted, linewidth=1.3, label=variant)

    if not has_data:
        plt.close(fig)
        return None

    axes[0].set_title("Output Distribution")
    axes[0].set_xlabel("value")
    axes[0].set_ylabel("density")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].set_title("Sorted Absolute Output")
    axes[1].set_xlabel("sample rank")
    axes[1].set_ylabel("abs(value)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.suptitle(f"{model_name} output outlier comparison")
    fig.tight_layout()
    path = os.path.join(output_dir, f"{safe_name(model_name)}_output_outlier_compare.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_direct_qkv_values(model_name, variant, direct_values, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    for layer_name, qkv_entries in direct_values.items():
        if not all(name in qkv_entries for name in ["q", "k", "v"]):
            continue

        values = [np.array(qkv_entries[name]["values"], dtype=np.float32) for name in ["q", "k", "v"]]
        vmax = max(float(np.max(np.abs(value))) for value in values if value.size > 0)
        vmax = vmax if vmax > 0 else 1.0

        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        image = None
        for ax, qkv_name, value in zip(axes, ["q", "k", "v"], values):
            image = ax.imshow(value, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
            entry = qkv_entries[qkv_name]
            ax.set_title(qkv_name.upper())
            ax.set_xlabel(entry["axes"][1])
            ax.set_ylabel(entry["axes"][0])
            ax.set_xticks(np.arange(value.shape[1]))
            ax.set_yticks(np.arange(value.shape[0]))

        first_entry = qkv_entries["q"]
        fig.colorbar(image, ax=axes, shrink=0.85, label="activation value")
        fig.suptitle(
            f"{model_name} {variant} {layer_name} "
            f"batch={first_entry['selected_batch_index']} token={first_entry['selected_token_index']}"
        )
        path = os.path.join(
            output_dir,
            f"{safe_name(model_name)}_{variant}_{safe_name(layer_name)}_qkv_values.png",
        )
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    return paths


def write_direct_qkv_csv(model_name, variant, direct_values, output_dir):
    paths = []
    for layer_name, qkv_entries in direct_values.items():
        path = os.path.join(
            output_dir,
            f"{safe_name(model_name)}_{variant}_{safe_name(layer_name)}_qkv_values.csv",
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write("qkv,head,neuron,value\n")
            for qkv_name in ["q", "k", "v"]:
                if qkv_name not in qkv_entries:
                    continue
                values = np.array(qkv_entries[qkv_name]["values"], dtype=np.float32)
                for head in range(values.shape[0]):
                    for neuron in range(values.shape[1]):
                        f.write(f"{qkv_name},{head},{neuron},{float(values[head, neuron])}\n")
        paths.append(path)
    return paths


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    base_model, base_batches = load_base_model_and_inputs(args, device)
    base_batches = [x.detach().clone() for x in base_batches]

    all_stats = {
        "model_name": args.model_name,
        "device": str(device),
        "num_batches": len(base_batches),
        "batch_size": args.batch_size,
        "direct_qkv_value_selection": {
            "batch_index": args.value_batch_index,
            "token_index": args.value_token_index,
        },
        "variants": {},
    }
    all_arrays = {}

    for variant in args.variants:
        print(f"Analyzing {variant}...")
        model, load_status = prepare_variant(base_model, args, variant, device)
        arrays, stats = run_capture(model, base_batches, args)
        stats["checkpoint"] = load_status
        all_stats["variants"][variant] = stats
        all_arrays[variant] = arrays
        print(f"  {load_status}")
        for qkv_name, qkv_stats in stats["aggregate"].items():
            if qkv_stats["sample_count"] == 0:
                continue
            print(
                "  "
                f"{qkv_name}: mean={qkv_stats['mean']:.4g}, "
                f"std={qkv_stats['std']:.4g}, "
                f"p01={qkv_stats['p01']:.4g}, "
                f"p99={qkv_stats['p99']:.4g}, "
                f"samples={qkv_stats['sample_count']}"
            )
        output_stats = stats["output"]
        if output_stats["sample_count"] > 0:
            print(
                "  "
                f"output: mean={output_stats['mean']:.4g}, "
                f"std={output_stats['std']:.4g}, "
                f"p99={output_stats['p99']:.4g}, "
                f"p99.9={output_stats['p999']:.4g}, "
                f"max_abs={max(abs(output_stats['min']), abs(output_stats['max'])):.4g}, "
                f"samples={output_stats['sample_count']}"
            )
        direct_layers = len(stats["direct_qkv_values"])
        if direct_layers > 0:
            print(
                "  "
                f"direct qkv values captured for {direct_layers} attention layer(s) "
                f"at batch={args.value_batch_index}, token={args.value_token_index}"
            )

    stats_path = os.path.join(args.output_dir, f"{safe_name(args.model_name)}_qkv_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)

    csv_paths = []
    for variant in all_stats["variants"]:
        direct_values = all_stats["variants"][variant]["direct_qkv_values"]
        csv_paths.extend(write_direct_qkv_csv(args.model_name, variant, direct_values, args.output_dir))

    plot_paths = []
    if not args.no_plots:
        for variant, arrays in all_arrays.items():
            plot_paths.append(
                plot_variant_histograms(args.model_name, variant, arrays, args.output_dir, args.bins)
            )
            output_plot = plot_output_outliers(args.model_name, variant, arrays, args.output_dir, args.bins)
            if output_plot is not None:
                plot_paths.append(output_plot)
            direct_values = all_stats["variants"][variant]["direct_qkv_values"]
            if not args.no_qkv_value_plots:
                plot_paths.extend(
                    plot_direct_qkv_values(args.model_name, variant, direct_values, args.output_dir)
                )
        if len(all_arrays) > 1:
            plot_paths.append(plot_comparison(args.model_name, all_arrays, args.output_dir, args.bins))
            output_compare = plot_output_comparison(args.model_name, all_arrays, args.output_dir, args.bins)
            if output_compare is not None:
                plot_paths.append(output_compare)

    print(f"Wrote stats: {stats_path}")
    for path in csv_paths:
        print(f"Wrote CSV: {path}")
    for path in plot_paths:
        print(f"Wrote plot: {path}")


if __name__ == "__main__":
    main()
