import argparse
import copy
import os
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import TinyLLaMa1M, TinyLLaMa3M, TinyLLaMa447K, TinyViT1M, cct_2, cct_7, model_zoo, tiny_waveformer
from models.LLaMa import TinyLLaMa2c110M, TinyLLaMa11M, TinyLLaMa28M
from models.utils import AttentionQuantMixin, set_attention_quantization
from MiCoSmoothQuant import apply_smoothquant, collect_smoothquant_act_scales, find_smoothquant_mappings

QUANT_CHOICES = ["none", "int8", "fp8", "bitnet"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one model with multiple attention quantization modes."
    )
    parser.add_argument("model_name", type=str)
    parser.add_argument("--weight-q", type=int, default=8)
    parser.add_argument("--act-q", type=int, default=8)
    parser.add_argument(
        "--quant",
        nargs="+",
        default=["none", "int8", "fp8", "bitnet"],
        choices=QUANT_CHOICES,
        help="Default attention quantization modes to test.",
    )
    parser.add_argument(
        "--q-quant",
        nargs="+",
        default=None,
        choices=QUANT_CHOICES,
        help="Q quantization modes. Defaults to --quant.",
    )
    parser.add_argument(
        "--kv-quant",
        nargs="+",
        default=None,
        choices=QUANT_CHOICES,
        help="K/V quantization modes. Defaults to --quant.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batches", type=int, default=None, help="Limit test batches.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override model_zoo NUM_WORKERS.")
    parser.add_argument("--checkpoint-dir", type=str, default="output/ckpt")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--strict-load", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--fp8-dtype", type=str, default="e4m3fn", choices=["e4m3fn", "e5m2"])
    parser.add_argument(
        "--int-dim",
        type=str,
        default="none",
        help='Default INT reduction dim. Use "none" for per-tensor, "-1" for per-last-dim, or comma list.',
    )
    parser.add_argument(
        "--int-dim-q",
        type=str,
        default=None,
        help='INT reduction dim for Q (overrides --int-dim).',
    )
    parser.add_argument(
        "--int-dim-k",
        type=str,
        default=None,
        help='INT reduction dim for K (overrides --int-dim).',
    )
    parser.add_argument(
        "--int-dim-v",
        type=str,
        default=None,
        help='INT reduction dim for V (overrides --int-dim).',
    )
    parser.add_argument("--quantize-score", action="store_true")
    parser.add_argument("--no-quantize-q", action="store_true")
    parser.add_argument("--no-quantize-kv", action="store_true")
    parser.add_argument("--no-quantize-output", action="store_true")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use random inputs/labels instead of model_zoo datasets.",
    )
    parser.add_argument("--input-len", type=int, default=512, help="Synthetic WaveFormer length or LLaMa token length.")
    parser.add_argument("--smoothquant", action="store_true", help="Apply SmoothQuant before attention quantization.")
    parser.add_argument("--smoothquant-alpha", type=float, default=0.5)
    parser.add_argument("--smoothquant-batches", type=int, default=32)
    parser.add_argument("--compare-smoothquant", action="store_true", help="Evaluate without and with SmoothQuant.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def parse_int_dim(value):
    value = str(value).strip().lower()
    if value in ["none", "null", ""]:
        return None
    dims = tuple(int(item) for item in value.split(","))
    return dims[0] if len(dims) == 1 else dims


def build_synthetic_model_and_loader(model_name, batch_size, input_len, device):
    name = model_name.lower()
    if name in ["cct2", "cct2_cifar10"]:
        model = cct_2(10).to(device)
        x = torch.randn(batch_size, 3, 32, 32)
        y = torch.randint(0, 10, (batch_size,))
    elif name in ["cct7", "cct7_cifar10"]:
        model = cct_7(10).to(device)
        x = torch.randn(batch_size, 3, 32, 32)
        y = torch.randint(0, 10, (batch_size,))
    elif name == "cct7_cifar100":
        model = cct_7(100).to(device)
        x = torch.randn(batch_size, 3, 32, 32)
        y = torch.randint(0, 100, (batch_size,))
    elif name in ["vit1m", "vit1m_cifar10"]:
        model = TinyViT1M(10).to(device)
        x = torch.randn(batch_size, 3, 32, 32)
        y = torch.randint(0, 10, (batch_size,))
    elif name == "waveformer":
        model = tiny_waveformer(n_classes=35).to(device)
        x = torch.randn(batch_size, 1, input_len)
        y = torch.randint(0, 35, (batch_size,))
    elif name in [
        "tinyllama",
        "llama_tiny",
        "tinyllama_447k",
        "tinyllama_1m",
        "tinyllama_3m",
        "tinyllama_11m",
        "tinyllama_28m",
        "tinyllama_110m",
    ]:
        model_builders = {
            "tinyllama": TinyLLaMa1M,
            "llama_tiny": TinyLLaMa1M,
            "tinyllama_447k": TinyLLaMa447K,
            "tinyllama_1m": TinyLLaMa1M,
            "tinyllama_3m": TinyLLaMa3M,
            "tinyllama_11m": TinyLLaMa11M,
            "tinyllama_28m": TinyLLaMa28M,
            "tinyllama_110m": TinyLLaMa2c110M,
        }
        model = model_builders[name]().to(device)
        seq_len = max(1, min(input_len, model.params.max_seq_len))
        x = torch.randint(0, model.params.vocab_size, (batch_size, seq_len), dtype=torch.long)
        y = torch.randint(0, model.params.vocab_size, (batch_size, seq_len), dtype=torch.long)
    else:
        raise ValueError(
            f"Synthetic mode does not know input shape for {model_name}. "
            "Use a model_zoo name without --synthetic, or add it here."
        )

    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)
    return model, loader, loader


def load_model_and_loader(args, device):
    if args.synthetic:
        return build_synthetic_model_and_loader(args.model_name, args.batch_size, args.input_len, device)

    if args.num_workers is not None:
        model_zoo.NUM_WORKERS = args.num_workers
    model, _train_loader, test_loader = model_zoo.from_zoo(
        args.model_name, shuffle=False, batch_size=args.batch_size
    )
    return model.to(device), _train_loader, test_loader


def maybe_limit_loader(loader, batches):
    if batches is None and not hasattr(loader, "__len__"):
        batches = 100
    if batches is None:
        return loader

    xs, ys = [], []
    for i, (x, y) in enumerate(loader):
        xs.append(x)
        ys.append(y)
        if i + 1 >= batches:
            break
    batch_size = getattr(loader, "batch_size", xs[0].size(0))
    return DataLoader(TensorDataset(torch.cat(xs, dim=0), torch.cat(ys, dim=0)), batch_size=batch_size)


def checkpoint_path(args):
    if args.checkpoint is not None:
        return args.checkpoint
    return os.path.join(args.checkpoint_dir, f"{args.model_name}.pth")


def load_checkpoint_if_available(model, args, device):
    if args.no_checkpoint:
        return "checkpoint loading disabled"

    path = checkpoint_path(args)
    if not os.path.exists(path):
        return f"checkpoint not found: {path}"

    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]

    if args.strict_load:
        model.load_state_dict(state, strict=True)
        return f"loaded {path}"

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        return f"loaded {path} with missing={len(missing)}, unexpected={len(unexpected)}"
    return f"loaded {path}"


def count_attention_modules(model):
    return sum(1 for module in model.modules() if isinstance(module, AttentionQuantMixin))


def quant_configs(args):
    if args.q_quant is None and args.kv_quant is None:
        return [(quant, quant, quant) for quant in args.quant]

    q_quants = args.q_quant if args.q_quant is not None else args.quant
    kv_quants = args.kv_quant if args.kv_quant is not None else args.quant
    configs = []
    for q_quant in q_quants:
        for kv_quant in kv_quants:
            configs.append((f"q={q_quant},kv={kv_quant}", q_quant, kv_quant))
    return configs


def apply_smoothquant_if_enabled(model, calib_loader, args, device):
    if not args.smoothquant and not args.compare_smoothquant:
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
        verbose=False,
    )
    return f"applied {len(applied)}/{len(mappings)} groups, alpha={args.smoothquant_alpha}"


def test_model(model, loader, device):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_count = 0
    total_correct = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            if x.dtype == torch.long and y.dtype == torch.long and x.dim() == 2 and y.dim() == 2:
                logits = model(x, y)
                loss = model.last_loss
                mask = y.view(-1) != -1
                predictions = logits.view(-1, logits.size(-1)).argmax(dim=-1)
                targets = y.view(-1)
                count = mask.sum().item()
                total_loss += loss.item() * count
                total_count += count
                total_correct += (predictions[mask] == targets[mask]).sum().item()
            else:
                logits = model(x)
                loss = criterion(logits, y)
                total_loss += loss.item() * y.size(0)
                total_count += y.size(0)
                total_correct += (logits.argmax(dim=1) == y).sum().item()

    return {
        "TestLoss": total_loss / total_count,
        "TestAcc": total_correct / total_count,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    base_model, calib_loader, test_loader = load_model_and_loader(args, device)
    test_loader = maybe_limit_loader(test_loader, args.batches)
    checkpoint_status = load_checkpoint_if_available(base_model, args, device)
    attention_modules = count_attention_modules(base_model)

    print(f"Model: {args.model_name}")
    print(f"Device: {device}")
    print(f"Attention modules: {attention_modules}")
    print(f"Checkpoint: {checkpoint_status}")
    print(f"SmoothQuant: {'compare' if args.compare_smoothquant else ('enabled' if args.smoothquant else 'disabled')}")
    print()

    base_variants = [("base", base_model)]
    if args.smoothquant or args.compare_smoothquant:
        smooth_model = copy.deepcopy(base_model).to(device)
        smooth_status = apply_smoothquant_if_enabled(smooth_model, calib_loader, args, device)
        print(f"SmoothQuant status: {smooth_status}")
        if args.compare_smoothquant:
            base_variants.append(("smooth", smooth_model))
        else:
            base_variants = [("smooth", smooth_model)]

    qscheme = [
        [args.weight_q] * base_model.n_layers, # weight qscheme
        [args.act_q] * base_model.n_layers, # activation qscheme
    ]
    # Keep Last Layer in 8-bit
    qscheme[0][-1] = 8
    qscheme[1][-1] = 8

    results = []
    int_dim = parse_int_dim(args.int_dim)
    int_dim_q = parse_int_dim(args.int_dim_q) if args.int_dim_q is not None else None
    int_dim_k = parse_int_dim(args.int_dim_k) if args.int_dim_k is not None else None
    int_dim_v = parse_int_dim(args.int_dim_v) if args.int_dim_v is not None else None
    for variant_name, variant_model in base_variants:
        for label, q_quant, kv_quant in quant_configs(args):
            model = copy.deepcopy(variant_model).to(device)
            model.set_qscheme(qscheme)
            set_attention_quantization(
                model,
                quant=kv_quant,
                q_quant=q_quant,
                kv_quant=kv_quant,
                fp8_dtype=args.fp8_dtype,
                int_dim=int_dim,
                int_dim_q=int_dim_q,
                int_dim_k=int_dim_k,
                int_dim_v=int_dim_v,
                quantize_q=not args.no_quantize_q,
                quantize_kv=not args.no_quantize_kv,
                quantize_score=args.quantize_score,
                quantize_output=not args.no_quantize_output,
            )
            result = test_model(model, test_loader, device)
            result["Variant"] = variant_name
            result["Quant"] = label
            result["QQuant"] = q_quant
            result["KVQuant"] = kv_quant
            results.append(result)
            print(f"{variant_name}/{label}: {result}")

    print("\nSummary")
    print(f"{'Variant':<8} {'QQuant':<8} {'KVQuant':<8} {'TestLoss':>12} {'TestAcc':>12}")
    for result in results:
        print(f"{result['Variant']:<8} {result['QQuant']:<8} {result['KVQuant']:<8} {result['TestLoss']:>12.6f} {result['TestAcc']:>12.6f}")


if __name__ == "__main__":
    main()
