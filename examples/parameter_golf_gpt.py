"""
Train and evaluate a ParameterGolfGPT model on the FineWeb dataset.

This example follows the OpenAI Parameter Golf challenge setup:
    https://github.com/openai/parameter-golf

The baseline configuration uses:
  - 9 transformer layers, 512 hidden dim
  - 8 query heads, 4 KV heads (Grouped Query Attention)
  - relu² MLP with 2x expansion
  - Vocabulary size 1024, sequence length 1024
  - Tied input/output embeddings

Usage::

    python examples/parameter_golf_gpt.py

Set environment variables to override defaults:
    MODEL_SIZE=small   – use the smaller 6-layer variant
    VOCAB_SIZE=50257   – switch to GPT-2 vocabulary (with GPT-2 tokenizer)
    NUM_ITER=5000      – number of gradient steps
    BATCH_SIZE=8       – batch size
"""

import os
import time
import json

import torch

from models.ParameterGolfGPT import (
    ParameterGolfGPT,
    ParameterGolfConfig,
    ParameterGolfBaseline,
    ParameterGolfSmall,
)
from MiCoDatasets import fineweb
from MiCoUtils import list_quantize_layers

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

model_size = os.environ.get("MODEL_SIZE", "baseline")   # "baseline" | "small"
num_iter   = int(os.environ.get("NUM_ITER",  "2000"))
batch_size = int(os.environ.get("BATCH_SIZE", "8"))
lr         = float(os.environ.get("LR",      "3e-4"))
vocab_size = int(os.environ.get("VOCAB_SIZE", "1024"))  # 1024 for parameter golf tokenizer
max_seq_len = int(os.environ.get("MAX_SEQ_LEN", "1024"))

model_name = f"parameter_golf_gpt_{model_size}"

torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    if model_size == "small":
        config = ParameterGolfConfig(
            vocab_size=vocab_size,
            num_layers=6,
            model_dim=256,
            num_heads=4,
            num_kv_heads=2,
            mlp_mult=2,
            max_seq_len=max_seq_len,
            tie_embeddings=True,
        )
        model = ParameterGolfGPT(config).to(device)
    else:
        config = ParameterGolfConfig(
            vocab_size=vocab_size,
            num_layers=9,
            model_dim=512,
            num_heads=8,
            num_kv_heads=4,
            mlp_mult=2,
            max_seq_len=max_seq_len,
            tie_embeddings=True,
        )
        model = ParameterGolfGPT(config).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_qlayers = len(list_quantize_layers(model))
    print(f"Model parameters : {n_params:,}")
    print(f"Quantizable layers: {n_qlayers}")

    # ------------------------------------------------------------------
    # Dataset  (FineWeb, tokenised with GPT-2 tokenizer by default)
    # ------------------------------------------------------------------
    print("Loading FineWeb dataset …")
    train_loader, test_loader = fineweb(
        batch_size=batch_size,
        max_seq_len=model.config.max_seq_len,
        shuffle=True,
    )
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    print(f"Training for {num_iter} iterations (lr={lr}) …")
    t0 = time.time()
    results = model.train_loop(
        n_iter=num_iter,
        train_loader=train_loader,
        test_loader=test_loader,
        verbose=True,
        lr=lr,
        eval_interval=max(num_iter // 10, 50),
    )
    elapsed = time.time() - t0
    print(f"Training done in {elapsed:.1f}s")
    print(f"Final val_loss: {results['TestLoss']:.4f}, "
          f"perplexity: {results['Perplexity']:.2f}")

    # ------------------------------------------------------------------
    # Save checkpoint
    # ------------------------------------------------------------------
    os.makedirs("output/ckpt", exist_ok=True)
    ckpt_path = f"output/ckpt/{model_name}.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")

    # ------------------------------------------------------------------
    # Evaluation (reload and verify)
    # ------------------------------------------------------------------
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt)
    eval_results = model.test(test_loader)
    print("Evaluation results:", eval_results)

    # ------------------------------------------------------------------
    # MPQ search example (INT8 baseline, optional)
    # ------------------------------------------------------------------
    try:
        from MiCoEval import MiCoEval
        from searchers import MiCoSearcher

        evaluator = MiCoEval(
            model,
            epochs=1,
            train_loader=train_loader,
            test_loader=test_loader,
            pretrained_model=ckpt_path,
            objective="ptq_acc",   # uses test-loss as proxy
        )
        int8_bops = evaluator.eval_bops([8] * model.n_layers * 2)
        print(f"INT8 BOPs: {int8_bops:.3e}")

        searcher = MiCoSearcher(evaluator, n_inits=5, qtypes=[4, 6, 8])
        best_config, best_val = searcher.search(
            n_iterations=20,
            objective="ptq_acc",
            constraint="bops",
            constraint_value=int8_bops * 0.5,
        )
        print(f"Best MPQ config: {best_config}")
        print(f"Best val metric: {best_val:.4f}")

        os.makedirs("output/json", exist_ok=True)
        with open(f"output/json/{model_name}_search.json", "w") as f:
            json.dump({"best_config": best_config, "best_val": best_val}, f)
        print(f"Search results saved to output/json/{model_name}_search.json")

    except Exception as exc:
        print(f"MPQ search skipped: {exc}")
