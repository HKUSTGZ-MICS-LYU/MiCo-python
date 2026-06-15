import torch
import numpy as np

import sys
import os
import argparse

from models import model_zoo
from MiCoMisc import _normalize_attention_quant


def parse_optional_int(value):
    if value is None:
        return None
    value = str(value).strip().lower()
    if value in ["none", "null", ""]:
        return None
    return int(value)


def build_attention_qscheme(args, n_layers):
    q_quant = _normalize_attention_quant(args.q_quant)
    k_quant = _normalize_attention_quant(args.k_quant)
    v_quant = _normalize_attention_quant(args.v_quant)
    score_quant = _normalize_attention_quant(args.score_quant)

    return {
        "q": [q_quant] * n_layers,
        "k": [k_quant] * n_layers,
        "v": [v_quant] * n_layers,
        "score": [score_quant] * n_layers,
    }


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
        bitnet_scale=args.bitnet_scale,
        bitnet_group_size=parse_optional_int(args.bitnet_group_size),
        bitnet_group_dim=args.bitnet_group_dim,
        bitnet_clip=args.bitnet_clip,
        fp8_dtype=args.fp8_dtype,
        int_dim=parse_optional_int(args.int_dim),
        int_dim_q=parse_optional_int(args.int_dim_q),
        int_dim_k=parse_optional_int(args.int_dim_k),
        int_dim_v=parse_optional_int(args.int_dim_v),
        int_dim_score=parse_optional_int(args.int_dim_score),
        quantize_q=not args.no_quantize_q,
        quantize_k=quantize_k,
        quantize_v=quantize_v,
        quantize_score=args.quantize_score,
        llama_kv_quant_scope=args.llama_kv_quant_scope,
        llama_kv_group_size=args.llama_kv_group_size,
    )
    print("Attention QAT qscheme:", attn_qscheme)
    return True


def checkpoint_state_dict(ckpt):
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    return ckpt


def format_counts(values):
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return ", ".join(f"{value}:{count}" for value, count in sorted(counts.items()))


def llama_mixed_seed(preset):
    if preset == "search":
        return [
            # WQ, WK, WV, WO, FF1, FF2, FF3, Output
            4, 4, 4, 4, 4, 4, 4, 8,
            8, 8, 8, 8, 4, 4, 8, 8,
        ]
    if preset == "w4a8":
        return [
            # WQ, WK, WV, WO, FF1, FF2, FF3, Output
            4, 4, 4, 4, 4, 4, 4, 8,
            8, 8, 8, 8, 8, 8, 8, 8,
        ]
    raise ValueError(f"Unknown LLaMa mixed preset: {preset}")


def print_qscheme_summary(qscheme):
    print(f"Weight qscheme counts: {format_counts(qscheme[0])}")
    print(f"Activation qscheme counts: {format_counts(qscheme[1])}")
    block_width = 7
    if len(qscheme[0]) > block_width:
        first_block = list(zip(qscheme[0][:block_width], qscheme[1][:block_width]))
        print(f"First transformer block W/A pairs: {first_block}; output W/A: {(qscheme[0][-1], qscheme[1][-1])}")


argsparse = argparse.ArgumentParser()
argsparse.add_argument("model_name", type=str)
argsparse.add_argument("epoches", type=int, default=10000)
argsparse.add_argument("--batch-size", type=int, default=32)
argsparse.add_argument("--lr", type=float, default=0.001)
argsparse.add_argument("-q", "--weight_quant", type=float, choices=[1,1.5,2,4,8], default=1)
argsparse.add_argument("-aq", "--act_quant", type=float, choices=[1,1.5,2,4,8], default=8)
argsparse.add_argument("--use-norm", action="store_true", default=False)
argsparse.add_argument("--group-size", type=int, default=0)
argsparse.add_argument("--pretrained", action="store_true", default=False)
argsparse.add_argument("--keep-last", action="store_true", default=False)
argsparse.add_argument("--keep-first", action="store_true", default=False)
argsparse.add_argument(
    "--scheduler",
    type=str,
    default="none",
    help="Non-LLaMa: none/cosine/step/cifar100-step. LLaMa: none/step/cosine/linear/cosine-warmup/linear-warmup.",
)
argsparse.add_argument("--warmup-epochs", type=int, default=3)
argsparse.add_argument("--warmup-lr", type=float, default=1e-6)
argsparse.add_argument("--mixed-llama", action="store_true", default=False)
argsparse.add_argument("--llama-mixed-preset", type=str, default="search", choices=["search", "w4a8"],
                       help="Preset used when --mixed-llama is enabled.")
argsparse.add_argument("--attn-qat", action="store_true", default=False,
                       help="Enable fake-quant QAT for attention Q/K/V/Score modules.")
argsparse.add_argument("--q-quant", type=str, default="int8",
                       help="Attention Q quantization type: none, int8, int7, fp8, bitnet/int1.58, ...")
argsparse.add_argument("--k-quant", type=str, default="bitnet",
                       help="Attention K quantization type.")
argsparse.add_argument("--v-quant", type=str, default="bitnet",
                       help="Attention V quantization type.")
argsparse.add_argument("--score-quant", type=str, default="none",
                       help="Post-softmax score quantization type.")
argsparse.add_argument("--quantize-score", action="store_true", default=False,
                       help="Enable post-softmax attention score quantization.")
argsparse.add_argument("--no-quantize-q", action="store_true", default=False)
argsparse.add_argument("--no-quantize-k", action="store_true", default=False)
argsparse.add_argument("--no-quantize-v", action="store_true", default=False)
argsparse.add_argument("--no-quantize-kv", action="store_true", default=False)
argsparse.add_argument("--bitnet-scale", type=str, default="max", choices=["max", "mean"],
                       help="Scale estimator for BitNet attention quantization.")
argsparse.add_argument("--bitnet-group-size", type=str, default="none",
                       help="Optional group size for BitNet attention quantization; use none to disable.")
argsparse.add_argument("--bitnet-group-dim", type=int, default=-1,
                       help="Dimension used for BitNet group-wise quantization.")
argsparse.add_argument("--bitnet-clip", type=float, default=None,
                       help="Optional abs clipping value before BitNet attention quantization.")
argsparse.add_argument("--fp8-dtype", type=str, default="e4m3fn", choices=["e4m3", "e4m3fn", "e5m2"],
                       help="FP8 dtype for attention quantization.")
argsparse.add_argument("--int-dim", type=str, default="none",
                       help="Shared reduce dim for integer attention quantization scale; none means tensor-wise.")
argsparse.add_argument("--int-dim-q", type=str, default="none",
                       help="Reduce dim for Q integer quantization scale.")
argsparse.add_argument("--int-dim-k", type=str, default="-2",
                       help="Reduce dim for K integer quantization scale. Default -2 fits LLaMa per-channel K.")
argsparse.add_argument("--int-dim-v", type=str, default="-1",
                       help="Reduce dim for V integer quantization scale. Default -1 fits LLaMa per-token V.")
argsparse.add_argument("--int-dim-score", type=str, default="none",
                       help="Reduce dim for post-softmax score integer quantization scale.")
argsparse.add_argument("--llama-kv-quant-scope", type=str, default="all",
                       choices=["all", "current_group", "history", "past", "cache", "kivi", "residual"],
                       help="For LLaMa attention, quantize all KV or only closed KV groups.")
argsparse.add_argument("--llama-kv-group-size", type=int, default=32,
                       help="Current-group size for LLaMa KIVI-style KV quantization.")

args = argsparse.parse_args()
batch_size = args.batch_size
model_name = args.model_name
epoches = args.epoches
lr = args.lr
scheduler = args.scheduler
warmup_epochs = args.warmup_epochs
warmup_lr = args.warmup_lr
weight_quant = args.weight_quant
act_quant = args.act_quant
use_norm = args.use_norm
group_size = args.group_size
pretrained = args.pretrained
keep_last = args.keep_last
keep_first = args.keep_first
use_mixed_llama = args.mixed_llama

if __name__ == "__main__":

    model, train_loader, test_loader = model_zoo.from_zoo(
        model_name, shuffle=True, batch_size=batch_size)

    qscheme = [
        [weight_quant] * model.n_layers, # weight qscheme
        [act_quant] * model.n_layers, # activation qscheme
    ]
    
    if keep_first:
        qscheme[0][0] = 8
        qscheme[1][0] = 8

    if keep_last:
        # Retain Last Layer in W8A8
        qscheme[0][-1] = 8
        qscheme[1][-1] = 8


    if use_mixed_llama and "llama" in model_name:
        from DimTransform import LLaMaInBlockTransformer
        mixed_scheme = llama_mixed_seed(args.llama_mixed_preset)
        out_scheme = LLaMaInBlockTransformer(model.n_layers * 2)(mixed_scheme)
        qscheme = [out_scheme[:model.n_layers], out_scheme[model.n_layers:]]

    print_qscheme_summary(qscheme)
    model.set_qscheme(qscheme, qat=True, use_norm=use_norm, group_size=args.group_size)
    apply_attention_qat(model, args)
    print("Model Param Size:", sum(p.numel() for p in model.parameters()))
    # Detect if there is a full precision checkpoint
    if pretrained:
        if os.path.exists(f"output/ckpt/{model_name}.pth"):
            print("Full Precision Checkpoint Loaded.")
            ckpt = torch.load(f"output/ckpt/{model_name}.pth")
            model.load_state_dict(checkpoint_state_dict(ckpt))

    if "llama" in model_name:
        res = model.train_loop(n_iter=int(epoches),
                train_loader=train_loader, 
                test_loader=test_loader,
                lr = lr, 
                eval_interval = epoches // 5,
                scheduler = scheduler,
                warmup_iters = warmup_epochs,
                warmup_lr = warmup_lr,
                verbose=True)
    else:
        res = model.train_loop(n_epoch=int(epoches),
                        train_loader=train_loader, 
                        test_loader=test_loader,
                        lr = lr, 
                        scheduler = scheduler,
                        warmup_epochs = warmup_epochs,
                        warmup_lr = warmup_lr,
                        early_stopping = False,
                        verbose=True)
        
    torch.save(model.state_dict(), f"output/ckpt/{model_name}_qat_g{group_size}.pth")
    print("Model Train Results: ", res)

    res = model.test(test_loader)

    print("Model Test Results: ", res)
