import torch
import numpy as np

import argparse
import os

from models import model_zoo
from MiCoEval import MiCoEval
from MiCoUtils import fuse_model, fuse_model_seq
from MiCoMisc import _normalize_attention_quant
from MiCoSmoothQuant import (
    apply_smoothquant,
    collect_smoothquant_act_scales,
    find_smoothquant_mappings,
)


def parse_bits(bits_text: str, n_layers: int, arg_name: str):
    values = [float(v.strip()) for v in bits_text.split(",") if v.strip()]
    if len(values) == 0:
        raise ValueError(f"{arg_name} must contain at least one bitwidth.")
    if len(values) == 1:
        return values * n_layers
    if len(values) != n_layers:
        raise ValueError(
            f"{arg_name} must provide 1 value or exactly {n_layers} values, got {len(values)}."
        )
    return values


def build_attention_qscheme(args, n_layers):
    q_quant = _normalize_attention_quant(args.q_quant)
    k_quant = _normalize_attention_quant(args.k_quant)
    v_quant = _normalize_attention_quant(args.v_quant)
    score_quant = _normalize_attention_quant(args.score_quant)
    none_quant = _normalize_attention_quant("none")

    qscheme = {
        "q": [q_quant] * n_layers,
        "k": [k_quant] * n_layers,
        "v": [v_quant] * n_layers,
        "score": [score_quant] * n_layers,
    }
    for idx in attention_keep_indices(n_layers, args.attn_keep_first, args.attn_keep_last):
        qscheme["q"][idx] = none_quant
        qscheme["k"][idx] = none_quant
        qscheme["v"][idx] = none_quant
        qscheme["score"][idx] = none_quant
    return qscheme


def attention_keep_indices(n_layers, keep_first=False, keep_last=False):
    keep = []
    if keep_first and n_layers > 0:
        keep.append(0)
    if keep_last and n_layers > 0 and (n_layers - 1) not in keep:
        keep.append(n_layers - 1)
    return keep


def apply_attention_qat(model, args):
    if not args.attn_qat:
        return False

    n_attn_layers = getattr(model, "n_attn_layers", 0)
    if n_attn_layers <= 0:
        print("Attention QAT requested, but no quantizable attention layer was found.")
        return False

    quantize_k = (not args.no_quantize_kv) and (not args.no_quantize_k)
    quantize_v = (not args.no_quantize_kv) and (not args.no_quantize_v)
    attn_qscheme = build_attention_qscheme(args, n_attn_layers)
    model.set_attn_qscheme(
        attn_qscheme,
        qat=True,
        bitnet_scale="max",
        bitnet_group_size=None,
        bitnet_group_dim=-1,
        bitnet_clip=None,
        fp8_dtype=args.fp8_dtype,
        int_dim=None,
        int_dim_q=None,
        int_dim_k=-2,
        int_dim_v=-1,
        int_dim_score=None,
        quantize_q=not args.no_quantize_q,
        quantize_k=quantize_k,
        quantize_v=quantize_v,
        quantize_score=args.quantize_score,
        llama_kv_quant_scope=args.llama_kv_quant_scope,
        llama_kv_group_size=args.llama_kv_group_size,
    )
    print("Attention QAT qscheme:", attn_qscheme)
    return True


def apply_smoothquant_if_enabled(model, calib_loader, args, device):
    if not args.smoothquant:
        return "disabled"
    act_scales = collect_smoothquant_act_scales(
        model,
        calib_loader,
        num_batches=args.smoothquant_batches,
        device=device,
    )
    mappings = find_smoothquant_mappings(model)
    applied = apply_smoothquant(
        model,
        act_scales,
        alpha=args.smoothquant_alpha,
        mappings=mappings,
        verbose=args.smoothquant_verbose,
    )
    return f"applied {len(applied)}/{len(mappings)} groups, alpha={args.smoothquant_alpha}"


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    model.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, nargs="?")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--device", type=str, default=None,
                        help="Evaluation device. Defaults to cuda when available, otherwise cpu.")

    parser.add_argument("--ckpt", type=str, default=None)

    parser.add_argument("--weight-q", type=str, default="8")
    parser.add_argument("--act-q", type=str, default="8")
    parser.add_argument("--use-norm", action="store_true")
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--keep-last", action="store_true")
    parser.add_argument("--keep-first", action="store_true")
    parser.add_argument("--smoothquant", action="store_true",
                        help="Apply SmoothQuant before linear/attention quantization.")
    parser.add_argument("--smoothquant-alpha", type=float, default=0.5)
    parser.add_argument("--smoothquant-batches", type=int, default=32)
    parser.add_argument("--smoothquant-verbose", action="store_true")

    parser.add_argument("--fuse", action="store_true")
    parser.add_argument("--fuse-seq", action="store_true")

    parser.add_argument("--mode", type=str, default="ptq",
                        choices=["test", "qat-test", "ptq", "qat", "bops", "size",
                                 "torchao", "info", "latency_host",
                                 "latency_mico", "latency_proxy"])
    parser.add_argument("--qat-forward", action="store_true",
                        help="Use QAT fake-quant forward in --mode test without retraining.")
    parser.add_argument("--qat-epochs", type=int, default=10)
    parser.add_argument("--qat-lr", type=float, default=0.0001)

    parser.add_argument("--mico-target", type=str, default="small")
    parser.add_argument("--kivi-attn", action="store_true",
                        help="Shortcut for --attn-qat with Q=int8, K/V=bitnet, score=none, scope=kivi.")
    parser.add_argument("--attn-qat", action="store_true", default=False,
                        help="Enable fake-quant QAT for attention Q/K/V/Score modules.")
    parser.add_argument("--q-quant", type=str, default="int8",
                        help="Attention Q quantization type: none, int8, int7, fp8, bitnet/int1.58, ...")
    parser.add_argument("--k-quant", type=str, default="bitnet",
                        help="Attention K quantization type.")
    parser.add_argument("--v-quant", type=str, default="bitnet",
                        help="Attention V quantization type.")
    parser.add_argument("--score-quant", type=str, default="none",
                        help="Post-softmax score quantization type.")
    parser.add_argument("--quantize-score", action="store_true", default=False,
                        help="Enable post-softmax attention score quantization.")
    parser.add_argument("--attn-keep-first", action="store_true",
                        help="Keep the first attention layer unquantized.")
    parser.add_argument("--attn-keep-last", action="store_true",
                        help="Keep the last attention layer unquantized.")
    parser.add_argument("--no-quantize-q", action="store_true", default=False)
    parser.add_argument("--no-quantize-k", action="store_true", default=False)
    parser.add_argument("--no-quantize-v", action="store_true", default=False)
    parser.add_argument("--no-quantize-kv", action="store_true", default=False)
    parser.add_argument("--fp8-dtype", type=str, default="e4m3fn", choices=["e4m3", "e4m3fn", "e5m2"],
                        help="FP8 dtype for attention quantization.")
    parser.add_argument("--llama-kv-quant-scope", type=str, default="kivi",
                        choices=["all", "current_group", "history", "past", "cache", "kivi", "residual"],
                        help="For LLaMa attention, quantize all KV or only closed KV groups.")
    parser.add_argument("--llama-kv-group-size", type=int, default=32,
                        help="Current-group size for LLaMa KIVI-style KV quantization.")

    args = parser.parse_args()

    if args.list_models:
        for name in model_zoo.list_zoo_models():
            print(name)
        return

    if args.model_name is None:
        parser.error("model_name is required unless --list-models is used.")

    if args.fuse and args.fuse_seq:
        raise ValueError("Please use only one of --fuse or --fuse-seq.")
    if args.qat_forward and args.mode != "test":
        raise ValueError("--qat-forward is only valid with --mode test. Use --mode qat-test otherwise.")
    if args.smoothquant and args.mode not in ["test", "qat-test"]:
        raise ValueError("--smoothquant is supported in direct --mode test/qat-test paths.")
    if args.kivi_attn:
        args.attn_qat = True
        args.q_quant = "int8"
        args.k_quant = "bitnet"
        args.v_quant = "bitnet"
        args.score_quant = "int8"
        args.quantize_score = False
        args.llama_kv_quant_scope = "kivi"
    run_device = torch.device(
        args.device if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model, train_loader, test_loader = model_zoo.from_zoo(
        args.model_name, shuffle=False, batch_size=args.batch_size
    )
    model = model.to(run_device)

    ckpt_path = args.ckpt or f"output/ckpt/{args.model_name}.pth"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. "
            "Use --ckpt to set a checkpoint path."
        )

    n_layers = model.n_layers
    weight_q = parse_bits(args.weight_q, n_layers, "--weight-q")
    act_q = parse_bits(args.act_q, n_layers, "--act-q")
    if args.keep_last:
        weight_q[-1] = 8
        act_q[-1] = 8
    if args.keep_first:
        weight_q[0] = 8
        act_q[0] = 8

    qscheme = weight_q + act_q

    print(f"\nQScheme w: {weight_q}")
    print(f"QScheme a: {act_q}")
    print(f"Mode: {args.mode}")

    if args.mode in ["test", "qat-test"]:
        load_checkpoint(model, ckpt_path, run_device)
        smoothquant_status = apply_smoothquant_if_enabled(model, train_loader, args, run_device)
        print(f"SmoothQuant: {smoothquant_status}")
        if args.fuse:
            model = fuse_model(model)
        elif args.fuse_seq:
            model = fuse_model_seq(model)
        model = model.to(run_device)
        use_qat_forward = args.mode == "qat-test" or args.qat_forward

        model.set_qscheme(
            [weight_q, act_q],
            qat=use_qat_forward,
            device=run_device,
            group_size=args.group_size,
            use_norm=args.use_norm,
        )
        model = model.to(run_device)
        if args.attn_qat:
            apply_attention_qat(model, args)
            model = model.to(run_device)
        model.eval()
        result = model.test(test_loader)
        print(f"Test Results: {result}")
        return

    evaluator = MiCoEval(model, args.qat_epochs, train_loader, test_loader,
                         ckpt_path, lr=args.qat_lr, model_name=args.model_name, 
                         linear_group_size=args.group_size)

    model_info = evaluator.get_layer_info()
    layer_counts = {}
    print("\nLayer Info:")
    for info in model_info:
        print(info)
        key = str(info["Layer Features"])
        layer_counts[key] = layer_counts.get(key, 0) + 1

    print("\nLayer Counts Summary:")
    for layer_type, count in layer_counts.items():
        print(f"  {layer_type}: {count}")

    if args.mode == "info":
        return

    if args.fuse:
        model = fuse_model(model)
    elif args.fuse_seq:
        model = fuse_model_seq(model)

    model.eval()


    if args.mode == "ptq":
        result = evaluator.eval(qscheme)
        print(f"PTQ Accuracy: {result}")

    elif args.mode == "qat":
        result = evaluator.eval_qat(qscheme, epochs=args.qat_epochs)
        print(f"QAT Accuracy: {result}")

    elif args.mode == "torchao":
        result = evaluator.eval_torchao(qscheme)
        print(f"TorchAO Accuracy: {result}")

    elif args.mode == "bops":
        bops = evaluator.eval_bops(qscheme)
        macs = np.sum(evaluator.layer_macs)
        print(f"BOPs: {bops:,}")
        print(f"MACs: {macs:,}")

    elif args.mode == "size":
        size = evaluator.eval_size(qscheme)
        params = np.sum(evaluator.layer_params)
        print(f"Model Size: {size:,} bits")
        print(f"FP Params: {params:,}")

    elif args.mode.startswith("latency"):
        target = args.mode.replace("latency_", "")
        if target == "mico":
            evaluator.set_mico_target(args.mico_target)
        result = evaluator.eval_latency(qscheme, target=target)
        print(f"Latency ({target}): {result}")


if __name__ == "__main__":
    main()
