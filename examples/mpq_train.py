import argparse
import json
import os
import random

import numpy as np
import torch

from models import model_zoo

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
argsparse.add_argument("--train-recipe", type=str, default=None,
                       help="Dataset recipe, e.g. cct_original")
argsparse.add_argument("--warmup-mode", choices=["linear", "shilabs"], default=None)
argsparse.add_argument("--weight-decay-mode", choices=["filtered", "all"], default=None)
argsparse.add_argument("--label-smoothing-impl", choices=["torch", "shilabs"], default=None)
argsparse.add_argument(
    "--scheduler",
    type=str,
    default="none",
    help="Non-LLaMa: none/cosine/step/cifar100-step. LLaMa: none/step/cosine/linear/cosine-warmup/linear-warmup.",
)
argsparse.add_argument("--warmup-epochs", type=int, default=3)
argsparse.add_argument("--warmup-lr", type=float, default=1e-6)
argsparse.add_argument("--min-lr", type=float, default=None)

args = argsparse.parse_args()
model_name = args.model_name
scheduler = args.scheduler
warmup_epochs = args.warmup_epochs
warmup_lr = args.warmup_lr


def case_defaults(name):
    if name == "cct2_cifar10":
        return {"epochs": 200, "batch_size": 128, "lr": 5e-4,
                "weight_decay": 3e-2, "scheduler": "cosine", "warmup": 5,
                "min_lr": 0.0,
                "label_smoothing": 0.1, "mixup_alpha": 0.0,
                "cutmix_alpha": 0.0, "ema_decay": None, "grad_clip": None,
                "train_recipe": "cct_original", "warmup_mode": "shilabs",
                "weight_decay_mode": "all", "label_smoothing_impl": "shilabs"}
    if name == "cct7_cifar100":
        return {"epochs": 300, "batch_size": 128, "lr": 6e-4,
                "weight_decay": 6e-2, "scheduler": "cosine", "warmup": 10,
                "min_lr": 1e-5,
                "label_smoothing": 0.1, "mixup_alpha": 0.8,
                "cutmix_alpha": 1.0, "ema_decay": None, "grad_clip": None,
                "train_recipe": "cct7_official", "warmup_mode": "linear",
                "weight_decay_mode": "filtered", "label_smoothing_impl": "torch"}
    if name in ["kwt", "tiny_kwt"]:
        return {"epochs": 150, "batch_size": 128, "lr": 3e-4,
                "weight_decay": 1e-4, "scheduler": "cosine", "warmup": 8,
                "min_lr": 0.0,
                "label_smoothing": 0.05, "mixup_alpha": 0.2,
                "cutmix_alpha": 0.0, "ema_decay": 0.9998, "grad_clip": 1.0,
                "train_recipe": None, "warmup_mode": "linear",
                "weight_decay_mode": "filtered", "label_smoothing_impl": "torch"}
    if "tinyllama" in name or name == "tinyllama":
        return {"epochs": 100000, "batch_size": 32, "lr": 3e-4,
                "weight_decay": 1e-2, "scheduler": "cosine-warmup", "warmup": 2000,
                "min_lr": 0.0,
                "label_smoothing": 0.0, "mixup_alpha": 0.0,
                "cutmix_alpha": 0.0, "ema_decay": None, "grad_clip": 1.0,
                "train_recipe": None, "warmup_mode": "linear",
                "weight_decay_mode": "filtered", "label_smoothing_impl": "torch"}
    return {"epochs": 40, "batch_size": 32, "lr": 1e-3,
            "weight_decay": 1e-4, "scheduler": scheduler, "warmup": warmup_epochs,
            "min_lr": 0.0,
            "label_smoothing": 0.0, "mixup_alpha": 0.0,
            "cutmix_alpha": 0.0, "ema_decay": None, "grad_clip": None,
            "train_recipe": None, "warmup_mode": "linear",
            "weight_decay_mode": "filtered", "label_smoothing_impl": "torch"}


defaults = case_defaults(model_name)
batch_size = defaults["batch_size"] if args.batch_size is None else args.batch_size
epochs = defaults["epochs"] if args.epochs is None else args.epochs
lr = defaults["lr"] if args.lr is None else args.lr
weight_decay = (defaults["weight_decay"] if args.weight_decay is None
                else args.weight_decay)
min_lr = defaults["min_lr"] if args.min_lr is None else args.min_lr
resolved_scheduler = (defaults["scheduler"] if args.scheduler == "none"
                      and model_name in ["cct2_cifar10", "cct7_cifar100", "kwt", "tiny_kwt", "tinyllama", "tinyllama_3m"]
                      else args.scheduler)
resolved_warmup = (defaults["warmup"] if args.warmup_epochs == 3
                   else args.warmup_epochs)
label_smoothing = (defaults["label_smoothing"] if args.label_smoothing is None
                   else args.label_smoothing)
mixup_alpha = defaults["mixup_alpha"] if args.mixup_alpha is None else args.mixup_alpha
cutmix_alpha = defaults["cutmix_alpha"] if args.cutmix_alpha is None else args.cutmix_alpha
ema_decay = defaults["ema_decay"] if args.ema_decay is None else args.ema_decay
grad_clip = defaults["grad_clip"] if args.grad_clip is None else args.grad_clip
train_recipe = (defaults["train_recipe"] if args.train_recipe is None
                else args.train_recipe)
warmup_mode = (defaults["warmup_mode"] if args.warmup_mode is None
               else args.warmup_mode)
weight_decay_mode = (defaults["weight_decay_mode"]
                     if args.weight_decay_mode is None else args.weight_decay_mode)
label_smoothing_impl = (defaults["label_smoothing_impl"]
                        if args.label_smoothing_impl is None
                        else args.label_smoothing_impl)
if ema_decay is not None and ema_decay <= 0:
    ema_decay = None

if __name__ == "__main__":

    model, train_loader, test_loader = model_zoo.from_zoo(
        model_name, shuffle=True, batch_size=batch_size,
        num_workers=args.num_workers, train_recipe=train_recipe)
    os.makedirs("output/ckpt", exist_ok=True)
    output_path = args.output or f"output/ckpt/{model_name}.pth"
    best_path = output_path.replace(".pth", "_best.pth")
    if "llama" in model_name:
        res = model.train_loop(n_iter=int(epochs),
                train_loader=train_loader, 
                test_loader=test_loader,
                lr = lr, 
                eval_interval = max(1, epochs // 10),
                scheduler = resolved_scheduler,
                warmup_iters = resolved_warmup,
                warmup_lr = warmup_lr,
                weight_decay=weight_decay,
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
                        warmup_lr = warmup_lr,
                        min_lr = min_lr,
                        optimizer_name=args.optimizer,
                        weight_decay=weight_decay,
                        grad_clip=grad_clip,
                        label_smoothing=label_smoothing,
                        mixup_alpha=mixup_alpha,
                        cutmix_alpha=cutmix_alpha,
                        ema_decay=ema_decay,
                        warmup_mode=warmup_mode,
                        weight_decay_mode=weight_decay_mode,
                        label_smoothing_impl=label_smoothing_impl,
                        best_checkpoint_path=best_path,
                        seed=args.seed,
                        early_stopping = False,
                        verbose=True)

    torch.save(model.state_dict(), output_path)
    with open(output_path.replace(".pth", ".json"), "w") as f:
        json.dump({"model": model_name, "epochs": epochs, "batch_size": batch_size, "lr": lr,
                   "weight_decay": weight_decay, "optimizer": args.optimizer,
                   "scheduler": resolved_scheduler, "warmup": resolved_warmup,
                   "min_lr": min_lr,
                   "label_smoothing": label_smoothing, "mixup_alpha": mixup_alpha,
                   "cutmix_alpha": cutmix_alpha, "ema_decay": ema_decay,
                   "grad_clip": grad_clip, "train_recipe": train_recipe,
                   "warmup_mode": warmup_mode, "weight_decay_mode": weight_decay_mode,
                   "label_smoothing_impl": label_smoothing_impl,
                   "seed": args.seed, "result": res}, f, indent=2)

    print("Model Train Results: ", res)
