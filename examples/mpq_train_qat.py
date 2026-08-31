import argparse
import json
import os
import random

import numpy as np
import torch

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
    # Paper TKV uses per-token absmax quantization for fixed-length
    # attention vectors. LLaMa keeps KIVI's per-channel K dimension.
    if args.tkv:
        tkv_q_dim = -1
        tkv_k_dim = -2 if "llama" in args.model_name.lower() else -1
        tkv_v_dim = -1
        tkv_score_dim = -1
    else:
        tkv_q_dim = None
        tkv_k_dim = -2
        tkv_v_dim = -1
        tkv_score_dim = None
    model.set_attn_qscheme(
        attn_qscheme,
        qat=True,
        bitnet_scale=args.bitnet_scale,
        bitnet_group_size=parse_optional_int(args.bitnet_group_size),
        bitnet_group_dim=args.bitnet_group_dim,
        bitnet_clip=args.bitnet_clip,
        fp8_dtype=args.fp8_dtype,
        int_dim=parse_optional_int(args.int_dim),
        int_dim_q=tkv_q_dim if args.tkv else parse_optional_int(args.int_dim_q),
        int_dim_k=tkv_k_dim if args.tkv else parse_optional_int(args.int_dim_k),
        int_dim_v=tkv_v_dim if args.tkv else parse_optional_int(args.int_dim_v),
        int_dim_score=tkv_score_dim if args.tkv else parse_optional_int(args.int_dim_score),
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
argsparse.add_argument("epochs", type=int, nargs="?", default=None)
argsparse.add_argument("--batch-size", type=int, default=None)
argsparse.add_argument("--lr", type=float, default=None)
argsparse.add_argument("--weight-decay", type=float, default=None)
argsparse.add_argument("--optimizer", choices=["adamw", "adam"], default="adamw")
argsparse.add_argument("--grad-clip", type=float, default=None)
argsparse.add_argument("--num-workers", type=int, default=0)
argsparse.add_argument("--seed", type=int, default=42)
argsparse.add_argument("--output", type=str, default=None)
argsparse.add_argument("--label-smoothing", type=float, default=None)
argsparse.add_argument("--mixup-alpha", type=float, default=None)
argsparse.add_argument("--cutmix-alpha", type=float, default=None)
argsparse.add_argument("--ema-decay", type=float, default=None)
argsparse.add_argument("-q", "--weight_quant", type=float, choices=[1,1.5,2,4,8], default=1.5)
argsparse.add_argument("-aq", "--act_quant", type=float, choices=[1,1.5,2,4,8], default=8)
argsparse.add_argument("--use-norm", action="store_true", default=False)
argsparse.add_argument("--group-size", type=int, default=0)
argsparse.add_argument(
    "--pretrained",
    nargs="?",
    const=True,
    default=None,
    metavar="CKPT_PATH",
    help="Load a checkpoint before QAT; without a value uses output/ckpt/{model_name}.pth.",
)
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
argsparse.add_argument("--tkv", "--tkv-attn", dest="tkv", action="store_true",
                       help="Paper TKV: ternary K/V, Int8 Q/score, and model-specific KV granularity.")
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
if args.tkv:
    args.attn_qat = True
    args.q_quant = "int8"
    args.k_quant = "bitnet"
    args.v_quant = "bitnet"
    args.score_quant = "int8"
    args.quantize_score = True
    args.llama_kv_quant_scope = "kivi"

model_name = args.model_name
scheduler = args.scheduler
warmup_epochs = args.warmup_epochs
weight_quant = args.weight_quant
act_quant = args.act_quant
use_norm = args.use_norm
group_size = args.group_size
pretrained = args.pretrained
keep_last = args.keep_last
keep_first = args.keep_first
use_mixed_llama = args.mixed_llama


def case_defaults(name):
    if name == "cct2_cifar10":
        return {"epochs": 100, "batch_size": 128, "lr": 1e-4,
                "weight_decay": 1e-4, "scheduler": "cosine", "warmup": 8,
                "label_smoothing": 0.05, "mixup_alpha": 0.1,
                "cutmix_alpha": 0.5, "ema_decay": 0.9998, "grad_clip": 1.0,
                "train_recipe": "cct_original"}
    if name == "cct7_cifar100":
        return {"epochs": 150, "batch_size": 128, "lr": 1e-4,
                "weight_decay": 1e-4, "scheduler": "cosine", "warmup": 10,
                "label_smoothing": 0.05, "mixup_alpha": 0.1,
                "cutmix_alpha": 0.5, "ema_decay": 0.9998, "grad_clip": 1.0,
                "train_recipe": "cct7_official"}
    if name in ["kwt", "tiny_kwt"]:
        return {"epochs": 100, "batch_size": 128, "lr": 5e-5,
                "weight_decay": 1e-4, "scheduler": "cosine", "warmup": 8,
                "label_smoothing": 0.05, "mixup_alpha": 0.1,
                "cutmix_alpha": 0.0, "ema_decay": 0.9998, "grad_clip": 1.0}
    if "tinyllama" in name or name == "tinyllama":
        return {"epochs": 100000, "batch_size": 32, "lr": 5e-5,
                "weight_decay": 1e-2, "scheduler": "cosine-warmup", "warmup": 2000,
                "label_smoothing": 0.0, "mixup_alpha": 0.0,
                "cutmix_alpha": 0.0, "ema_decay": None, "grad_clip": 1.0}
    return {"epochs": 40, "batch_size": 32, "lr": 1e-4,
            "weight_decay": 1e-4, "scheduler": scheduler, "warmup": warmup_epochs,
            "label_smoothing": 0.0, "mixup_alpha": 0.0,
            "cutmix_alpha": 0.0, "ema_decay": None, "grad_clip": None}


defaults = case_defaults(model_name)
batch_size = defaults["batch_size"] if args.batch_size is None else args.batch_size
epochs = defaults["epochs"] if args.epochs is None else args.epochs
lr = defaults["lr"] if args.lr is None else args.lr
weight_decay = (defaults["weight_decay"] if args.weight_decay is None
                else args.weight_decay)
resolved_scheduler = (defaults["scheduler"] if args.scheduler == "none"
                      and (model_name in ["cct2_cifar10", "cct7_cifar100", "kwt", "tiny_kwt"]
                           or "tinyllama" in model_name)
                      else args.scheduler)
resolved_warmup = (defaults["warmup"] if args.warmup_epochs == 3
                   else args.warmup_epochs)
label_smoothing = (defaults["label_smoothing"] if args.label_smoothing is None
                   else args.label_smoothing)
mixup_alpha = defaults["mixup_alpha"] if args.mixup_alpha is None else args.mixup_alpha
cutmix_alpha = defaults["cutmix_alpha"] if args.cutmix_alpha is None else args.cutmix_alpha
ema_decay = defaults["ema_decay"] if args.ema_decay is None else args.ema_decay
grad_clip = defaults["grad_clip"] if args.grad_clip is None else args.grad_clip
train_recipe = defaults.get("train_recipe")
if ema_decay is not None and ema_decay <= 0:
    ema_decay = None

if __name__ == "__main__":

    model, train_loader, test_loader = model_zoo.from_zoo(
        model_name, shuffle=True, batch_size=batch_size,
        num_workers=args.num_workers,
        train_recipe=train_recipe)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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
    # Load the default FP32 checkpoint or an explicitly supplied checkpoint.
    if pretrained:
        pretrained_path = (
            f"output/ckpt/{model_name}.pth"
            if pretrained is True
            else os.path.expanduser(pretrained)
        )
        if os.path.exists(pretrained_path):
            print(f"Pretrained Checkpoint Loaded: {pretrained_path}")
            ckpt = torch.load(pretrained_path, map_location=model.current_device())
            model.load_state_dict(checkpoint_state_dict(ckpt))
        else:
            raise FileNotFoundError(
                f"Missing pretrained checkpoint {pretrained_path}; "
                "train the FP32 model first or pass a compatible checkpoint."
            )

    os.makedirs("output/ckpt", exist_ok=True)
    default_output_name = f"{model_name}_qat{'_tkv' if args.tkv else ''}_g{group_size}.pth"
    output_path = args.output or f"output/ckpt/{default_output_name}"
    best_path = output_path.replace(".pth", "_best.pth")

    if "llama" in model_name:
        res = model.train_loop(n_iter=int(epochs),
                train_loader=train_loader, 
                test_loader=test_loader,
                lr = lr, 
                eval_interval = max(1, epochs // 10),
                scheduler = resolved_scheduler,
                warmup_iters = resolved_warmup,
                warmup_lr = args.warmup_lr,
                weight_decay=weight_decay,
                grad_clip=args.grad_clip,
                seed=args.seed,
                best_checkpoint_path=best_path,
                verbose=True)
    else:
        res = model.train_loop(n_epoch=int(epochs),
                        train_loader=train_loader, 
                        test_loader=test_loader,
                        lr = lr, 
                        scheduler = resolved_scheduler,
                        warmup_epochs = resolved_warmup,
                        warmup_lr = args.warmup_lr,
                        optimizer_name=args.optimizer,
                        weight_decay=weight_decay,
                        grad_clip=grad_clip,
                        label_smoothing=label_smoothing,
                        mixup_alpha=mixup_alpha,
                        cutmix_alpha=cutmix_alpha,
                        ema_decay=ema_decay,
                        best_checkpoint_path=best_path,
                        seed=args.seed,
                        early_stopping = False,
                        verbose=True)
        
    torch.save(model.state_dict(), output_path)
    with open(output_path.replace(".pth", ".json"), "w") as f:
        json.dump({"model": model_name, "epochs": epochs, "batch_size": batch_size,
                   "weight_quant": weight_quant,
                   "act_quant": act_quant, "lr": lr, "weight_decay": weight_decay,
                   "tkv": args.tkv,
                   "attention_q_quant": args.q_quant,
                   "attention_k_quant": args.k_quant,
                   "attention_v_quant": args.v_quant,
                   "attention_score_quant": args.score_quant,
                   "optimizer": args.optimizer, "scheduler": resolved_scheduler,
                   "warmup": resolved_warmup, "keep_last": keep_last,
                   "label_smoothing": label_smoothing, "mixup_alpha": mixup_alpha,
                   "cutmix_alpha": cutmix_alpha, "ema_decay": ema_decay,
                   "grad_clip": grad_clip, "train_recipe": train_recipe,
                   "seed": args.seed, "result": res}, f, indent=2)
    print("Model Train Results: ", res)

    res = model.test(test_loader)

    print("Model Test Results: ", res)
