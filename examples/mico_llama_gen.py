import argparse
import os

import torch

from MiCoLLaMaGen import mico_llama_export
from MiCoUtils import fuse_model, fuse_model_seq
from models import model_zoo


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


def load_model_ckpt(model, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(ckpt)}")

    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        ckpt = ckpt["state_dict"]
    elif "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
        ckpt = ckpt["model_state_dict"]
    elif "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]

    if all(key.startswith("module.") for key in ckpt.keys()):
        ckpt = {key[7:]: value for key, value in ckpt.items()}

    model.load_state_dict(ckpt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, nargs="?")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--list-models", action="store_true")

    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--skip-ckpt", action="store_true")

    parser.add_argument("--weight-q", type=str, default="8")
    parser.add_argument("--act-q", type=str, default="8")
    parser.add_argument("--use-norm", action="store_true")
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--skip-qscheme", action="store_true")
    parser.add_argument("--keep-last", action="store_true")
    parser.add_argument("--keep-first", action="store_true")

    parser.add_argument("--fuse", action="store_true")
    parser.add_argument("--fuse-seq", action="store_true")

    parser.add_argument("--one-layer", action="store_true")
    parser.add_argument("--output-path", type=str, default="project/model.bin")
    parser.add_argument("--quantize-classifier", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()

    if args.list_models:
        print("tinyllama_1m  tinyllama_3m  tinyllama_11m  tinyllama_28m  tinyllama_110m")
        return

    if args.model_name is None:
        parser.error("model_name is required unless --list-models is used.")

    if "tinyllama" not in args.model_name:
        parser.error("Only tinyllama models are supported by this script.")

    if args.fuse and args.fuse_seq:
        raise ValueError("Please use only one of --fuse or --fuse-seq.")

    model, _, _ = model_zoo.from_zoo(
        args.model_name, shuffle=False, batch_size=args.batch_size
    )
    model = model.to("cpu")

    if not args.skip_ckpt:
        ckpt_path = args.ckpt or f"output/ckpt/{args.model_name}.pth"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}. "
                "Use --ckpt to set a checkpoint path or --skip-ckpt to skip loading."
            )
        load_model_ckpt(model, ckpt_path)

    if not args.skip_qscheme:
        n_layers = model.n_layers
        weight_q = parse_bits(args.weight_q, n_layers, "--weight-q")
        act_q_text = args.act_q if args.act_q is not None else args.weight_q
        act_q = parse_bits(act_q_text, n_layers, "--act-q")
        if args.keep_last:
            weight_q[-1] = 8
            act_q[-1] = 8
        if args.keep_first:
            weight_q[0] = 8
            act_q[0] = 8
        model.set_qscheme([weight_q, act_q], group_size=args.group_size, use_norm=args.use_norm)

    if args.fuse:
        model = fuse_model(model)
    elif args.fuse_seq:
        model = fuse_model_seq(model)

    if args.one_layer:
        model.layers = torch.nn.ModuleList([model.layers[0]])
        model.params.n_layers = 1

    model.eval()

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    mico_llama_export(
        model,
        args.output_path,
        quantize_final_classifier=args.quantize_classifier,
    )


if __name__ == "__main__":
    main()
