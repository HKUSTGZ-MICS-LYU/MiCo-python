import argparse
import os

import torch

from MiCoCodeGen import MiCoCodeGen
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


def parse_shape(shape_text: str):
    shape = tuple(int(v.strip()) for v in shape_text.split(",") if v.strip())
    if len(shape) == 0:
        raise ValueError("--example-shape must provide at least one dimension.")
    if any(d <= 0 for d in shape):
        raise ValueError("--example-shape dimensions must be positive.")
    return shape


def get_batch_input(batch):
    if torch.is_tensor(batch):
        return batch

    if isinstance(batch, (list, tuple)):
        if len(batch) == 0:
            return None
        if torch.is_tensor(batch[0]):
            return batch[0]
        return get_batch_input(batch[0])

    if isinstance(batch, dict):
        preferred_keys = ["input_ids", "inputs", "x", "image", "images", "data"]
        for key in preferred_keys:
            value = batch.get(key)
            if torch.is_tensor(value):
                return value
        for value in batch.values():
            if torch.is_tensor(value):
                return value
        return None

    return None


def get_example_input(test_loader):
    batch = next(iter(test_loader))
    x = get_batch_input(batch)
    if x is None:
        raise TypeError(
            f"Unsupported batch type for codegen input extraction: {type(batch)}"
        )
    if x.dim() > 0:
        x = x[:1]
    return x.to("cpu")


def load_model_ckpt(model, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(ckpt)}")

    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        ckpt = ckpt["state_dict"]
    elif "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
        ckpt = ckpt["model_state_dict"]

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
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--skip-qscheme", action="store_true")
    parser.add_argument("--keep-last", action="store_true")
    parser.add_argument("--keep-first", action="store_true")
    parser.add_argument("--test", action="store_true")

    parser.add_argument("--fuse", action="store_true")
    parser.add_argument("--fuse-seq", action="store_true")
    parser.add_argument("--align-to", type=int, default=32)
    parser.add_argument("--gemmini-mode", action="store_true")

    parser.add_argument("--output-dir", type=str, default="project")
    parser.add_argument("--output-name", type=str, default="model")
    parser.add_argument(
        "--mem-pool",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--example-shape", type=str, default=None)
    parser.add_argument(
        "--example-dtype",
        type=str,
        default="float32",
        choices=["float32", "int64"],
    )
    parser.add_argument("--print-graph", action="store_true")
    parser.add_argument("--dag-file", type=str, default=None)
    parser.add_argument("--dag-simplified", action="store_true")

    args = parser.parse_args()

    if args.list_models:
        for name in model_zoo.list_zoo_models():
            print(name)
        return

    if args.model_name is None:
        parser.error("model_name is required unless --list-models is used.")

    if args.fuse and args.fuse_seq:
        raise ValueError("Please use only one of --fuse or --fuse-seq.")

    model, _, test_loader = model_zoo.from_zoo(
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
        model.set_qscheme([weight_q, act_q], group_size=args.group_size)

    if args.fuse:
        model = fuse_model(model)
    elif args.fuse_seq:
        model = fuse_model_seq(model)

    model.eval()

    if args.example_shape is not None:
        input_shape = parse_shape(args.example_shape)
        if args.example_dtype == "int64":
            example_input = torch.zeros(input_shape, dtype=torch.int64)
        else:
            example_input = torch.randn(input_shape, dtype=torch.float32)
    else:
        if test_loader is None:
            raise ValueError(
                "No test loader available from model_zoo. "
                "Please provide --example-shape and --example-dtype."
            )
        example_input = get_example_input(test_loader)

    codegen = MiCoCodeGen(
        model,
        align_to=args.align_to,
        gemmini_mode=args.gemmini_mode,
    )
    if args.print_graph:
        codegen.print_graph()
    codegen.forward(example_input)

    if args.dag_file:
        codegen.visualize_dag(args.dag_file, simplified=args.dag_simplified)

    codegen.convert(
        output_directory=args.output_dir,
        model_name=args.output_name,
        verbose=args.verbose,
        mem_pool=args.mem_pool,
    )


if __name__ == "__main__":
    main()
