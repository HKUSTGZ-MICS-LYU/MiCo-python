import argparse
import os

import torch

from MiCoDatasets import local_mlm_text
from models.BERT import MicroBERTLocal, TinyBERTLocal


def main():
    parser = argparse.ArgumentParser(description="Train native MiCo BERT on local byte-level MLM data.")
    parser.add_argument("--model", choices=["micro", "tiny"], default="micro")
    parser.add_argument("--text", default="data/local_mlm.txt", help="Local UTF-8 text file. Uses fallback text if absent.")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--out", default=None, help="Checkpoint output path.")
    args = parser.parse_args()

    model = MicroBERTLocal() if args.model == "micro" else TinyBERTLocal()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader, test_loader = local_mlm_text(
        batch_size=args.batch_size,
        max_seq_len=model.params.max_seq_len,
        root=args.text,
        num_works=0,
        shuffle=True,
    )

    results = model.train_loop(
        n_iter=args.steps,
        train_loader=train_loader,
        test_loader=test_loader,
        lr=args.lr,
        eval_interval=args.eval_interval,
        verbose=True,
    )
    print("Final:", results)

    out = args.out or f"output/ckpt/bert_{args.model}_local.pth"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(model.state_dict(), out)
    print(f"Saved checkpoint to {out}")


if __name__ == "__main__":
    main()
