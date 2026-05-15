import torch
import numpy as np

import argparse
import os

from models import model_zoo
from MiCoEval import MiCoEval
from MiCoUtils import fuse_model, fuse_model_seq


def parse_bits(bits_text: str, n_layers: int, arg_name: str):
    values = [int(v.strip()) for v in bits_text.split(",") if v.strip()]
    if len(values) == 0:
        raise ValueError(f"{arg_name} must contain at least one integer.")
    if len(values) == 1:
        return values * n_layers
    if len(values) != n_layers:
        raise ValueError(
            f"{arg_name} must provide 1 value or exactly {n_layers} values, got {len(values)}."
        )
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, nargs="?")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--list-models", action="store_true")

    parser.add_argument("--ckpt", type=str, default=None)

    parser.add_argument("--weight-q", type=str, default="8")
    parser.add_argument("--act-q", type=str, default="8")
    parser.add_argument("--use-norm", action="store_true")
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--keep-last", action="store_true")
    parser.add_argument("--keep-first", action="store_true")

    parser.add_argument("--fuse", action="store_true")
    parser.add_argument("--fuse-seq", action="store_true")

    parser.add_argument("--mode", type=str, default="ptq",
                        choices=["test", "ptq", "qat", "bops", "size",
                                 "torchao", "info", "latency_host",
                                 "latency_mico", "latency_proxy"])
    parser.add_argument("--qat-epochs", type=int, default=10)
    parser.add_argument("--qat-lr", type=float, default=0.0001)

    parser.add_argument("--mico-target", type=str, default="small")

    args = parser.parse_args()

    if args.list_models:
        for name in model_zoo.list_zoo_models():
            print(name)
        return

    if args.model_name is None:
        parser.error("model_name is required unless --list-models is used.")

    if args.fuse and args.fuse_seq:
        raise ValueError("Please use only one of --fuse or --fuse-seq.")

    model, train_loader, test_loader = model_zoo.from_zoo(
        args.model_name, shuffle=False, batch_size=args.batch_size
    )
    model = model.to("cpu")

    ckpt_path = args.ckpt or f"output/ckpt/{args.model_name}.pth"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. "
            "Use --ckpt to set a checkpoint path."
        )

    evaluator = MiCoEval(model, args.qat_epochs, train_loader, test_loader,
                         ckpt_path, lr=args.qat_lr, model_name=args.model_name)

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

    n_layers = evaluator.n_layers

    weight_q = parse_bits(args.weight_q, n_layers, "--weight-q")
    act_q = parse_bits(args.act_q, n_layers, "--act-q")
    if args.keep_last:
        weight_q[-1] = 8
        act_q[-1] = 8
    if args.keep_first:
        weight_q[0] = 8
        act_q[0] = 8

    qscheme = weight_q + act_q

    if args.fuse:
        model = fuse_model(model)
    elif args.fuse_seq:
        model = fuse_model_seq(model)

    model.eval()

    print(f"\nQScheme w: {weight_q}")
    print(f"QScheme a: {act_q}")
    print(f"Mode: {args.mode}")

    if args.mode == "test":
        model.set_qscheme([weight_q, act_q], group_size=args.group_size, use_norm=args.use_norm)
        result = model.test(test_loader)
        print(f"Test Results: {result}")

    elif args.mode == "ptq":
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
