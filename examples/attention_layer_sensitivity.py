import argparse
import copy
import csv
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

EXAMPLES_DIR = os.path.dirname(__file__)
if EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, EXAMPLES_DIR)

from attention_quant_test import (  # noqa: E402
    apply_smoothquant_if_enabled,
    load_checkpoint_if_available,
    load_model_and_loader,
    maybe_limit_loader,
    parse_int_dim,
    test_model,
)
from MiCoUtils import list_quantize_attn_layers, replace_quantize_attn_layers  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan per-attention-layer sensitivity by quantizing one attention layer at a time."
    )
    parser.add_argument("model_name", type=str)
    parser.add_argument(
        "--components",
        nargs="+",
        default=["k", "v"],
        choices=["q", "k", "v", "score"],
        help="Attention tensors to quantize in the selected layer.",
    )
    parser.add_argument("--quant", type=str, default="bitnet", help="Quantization mode for selected components.")
    parser.add_argument("--base-quant", type=str, default="none", help="Quantization mode for non-selected components.")
    parser.add_argument("--weight-q", type=int, default=8)
    parser.add_argument("--act-q", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batches", type=int, default=None, help="Limit test batches.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override model_zoo NUM_WORKERS.")
    parser.add_argument("--checkpoint-dir", type=str, default="output/ckpt")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--strict-load", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--fp8-dtype", type=str, default="e4m3fn", choices=["e4m3fn", "e5m2"])
    parser.add_argument("--bitnet-scale", type=str, default="max", choices=["max", "mean"])
    parser.add_argument("--bitnet-group-size", type=int, default=None)
    parser.add_argument("--bitnet-group-dim", type=int, default=-1)
    parser.add_argument("--bitnet-clip", type=float, default=None)
    parser.add_argument(
        "--llama-kv-quant-scope",
        type=str,
        default="all",
        choices=["all", "current_group", "history", "past", "cache", "kivi", "residual"],
        help="For LLaMa attention, quantize all KV or only closed KV groups.",
    )
    parser.add_argument("--llama-kv-group-size", type=int, default=32)
    parser.add_argument("--int-dim", type=str, default="none")
    parser.add_argument("--int-dim-q", type=str, default=None)
    parser.add_argument("--int-dim-k", type=str, default=None)
    parser.add_argument("--int-dim-v", type=str, default=None)
    parser.add_argument("--int-dim-score", type=str, default=None)
    parser.add_argument("--quantize-score", action="store_true", help="Enable score quantization when score is selected.")
    parser.add_argument("--smoothquant", action="store_true")
    parser.add_argument("--smoothquant-alpha", type=float, default=0.5)
    parser.add_argument("--smoothquant-batches", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-csv", type=str, default=None)
    return parser.parse_args()


def attention_layer_names(model):
    attn_layers = list_quantize_attn_layers(model)
    names = []
    for name, module in model.named_modules():
        if module in attn_layers:
            names.append(name)
    return names


def make_single_layer_qscheme(n_layers, layer_idx, components, quant, base_quant):
    qscheme = {
        "q": [base_quant] * n_layers,
        "k": [base_quant] * n_layers,
        "v": [base_quant] * n_layers,
        "score": [base_quant] * n_layers,
    }
    for component in components:
        qscheme[component][layer_idx] = quant
    return qscheme


def apply_linear_qscheme(model, weight_q, act_q):
    qscheme = [
        [weight_q] * model.n_layers,
        [act_q] * model.n_layers,
    ]
    if model.n_layers > 0:
        qscheme[0][-1] = 8
        qscheme[1][-1] = 8
    model.set_qscheme(qscheme)


def evaluate_with_attn_qscheme(base_model, test_loader, device, args, attn_qscheme=None):
    model = copy.deepcopy(base_model).to(device)
    apply_linear_qscheme(model, args.weight_q, args.act_q)
    if attn_qscheme is not None:
        replace_quantize_attn_layers(
            model,
            attn_qscheme,
            bitnet_scale=args.bitnet_scale,
            bitnet_group_size=args.bitnet_group_size,
            bitnet_group_dim=args.bitnet_group_dim,
            bitnet_clip=args.bitnet_clip,
            fp8_dtype=args.fp8_dtype,
            int_dim=parse_int_dim(args.int_dim),
            int_dim_q=parse_int_dim(args.int_dim_q) if args.int_dim_q is not None else None,
            int_dim_k=parse_int_dim(args.int_dim_k) if args.int_dim_k is not None else None,
            int_dim_v=parse_int_dim(args.int_dim_v) if args.int_dim_v is not None else None,
            int_dim_score=parse_int_dim(args.int_dim_score) if args.int_dim_score is not None else None,
            quantize_q="q" in args.components,
            quantize_k="k" in args.components,
            quantize_v="v" in args.components,
            llama_kv_quant_scope=args.llama_kv_quant_scope,
            llama_kv_group_size=args.llama_kv_group_size,
            quantize_score=args.quantize_score or "score" in args.components,
        )
    return test_model(model, test_loader, device)


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    base_model, calib_loader, test_loader = load_model_and_loader(args, device)
    test_loader = maybe_limit_loader(test_loader, args.batches)
    checkpoint_status = load_checkpoint_if_available(base_model, args, device)
    if args.smoothquant:
        smooth_status = apply_smoothquant_if_enabled(base_model, calib_loader, args, device)
    else:
        smooth_status = "disabled"

    attn_layers = list_quantize_attn_layers(base_model)
    names = attention_layer_names(base_model)
    n_attn = len(attn_layers)
    if n_attn == 0:
        raise RuntimeError("No AttentionScore/LLaMaAttention modules found.")

    print(f"Model: {args.model_name}")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_status}")
    print(f"SmoothQuant: {smooth_status}")
    print(f"Attention layers: {n_attn}")
    print(f"Components: {','.join(args.components)} -> {args.quant}")
    print(f"Base attention quant: {args.base_quant}")
    print(f"BitNet scale: {args.bitnet_scale}")
    print()

    baseline = evaluate_with_attn_qscheme(base_model, test_loader, device, args, attn_qscheme=None)
    print(f"baseline: {baseline}")

    rows = []
    for idx in range(n_attn):
        attn_qscheme = make_single_layer_qscheme(
            n_attn,
            idx,
            args.components,
            args.quant,
            args.base_quant,
        )
        result = evaluate_with_attn_qscheme(base_model, test_loader, device, args, attn_qscheme)
        row = {
            "Layer": idx,
            "Name": names[idx] if idx < len(names) else f"attn_{idx}",
            "Components": ",".join(args.components),
            "Quant": args.quant,
            "BaseQuant": args.base_quant,
            "BitNetScale": args.bitnet_scale,
            "BitNetGroupSize": args.bitnet_group_size,
            "BitNetClip": args.bitnet_clip,
            "IntDimK": args.int_dim_k if args.int_dim_k is not None else args.int_dim,
            "IntDimV": args.int_dim_v if args.int_dim_v is not None else args.int_dim,
            "TestLoss": result["TestLoss"],
            "TestAcc": result["TestAcc"],
            "LossIncrease": result["TestLoss"] - baseline["TestLoss"],
            "AccDrop": baseline["TestAcc"] - result["TestAcc"],
        }
        rows.append(row)
        print(
            f"[{idx:02d}] {row['Name']:<48} "
            f"loss={row['TestLoss']:.6f} acc={row['TestAcc']:.6f} "
            f"loss_inc={row['LossIncrease']:.6f} acc_drop={row['AccDrop']:.6f}"
        )

    rows.sort(key=lambda row: (row["AccDrop"], row["LossIncrease"]), reverse=True)

    print("\nMost sensitive layers")
    print(f"{'Layer':>5} {'AccDrop':>10} {'LossInc':>10} Name")
    for row in rows:
        print(f"{row['Layer']:>5} {row['AccDrop']:>10.6f} {row['LossIncrease']:>10.6f} {row['Name']}")

    if args.output_csv is not None:
        write_csv(args.output_csv, rows)
        print(f"\nWrote {args.output_csv}")


if __name__ == "__main__":
    main()
