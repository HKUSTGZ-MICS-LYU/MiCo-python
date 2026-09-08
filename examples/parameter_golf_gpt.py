"""
Train and evaluate a ParameterGolfGPT model, aligned with the OpenAI
Parameter Golf challenge baseline (https://github.com/openai/parameter-golf).

The model exposes MiCoModel-compatible train_loop / test entrypoints: test()
returns {TestLoss, TestAcc, Perplexity} (so MiCoEval / searchers work) and
train_loop() trains with Muon on matrix params + Adam on scalars/embeddings,
bf16 autocast and gradient accumulation to TRAIN_BATCH_TOKENS - matching the
official parameter-golf train_gpt.py baseline.

Usage::

    python examples/parameter_golf_gpt.py

Environment overrides:
    DATA_PATH        dir with fineweb_train_*.bin / fineweb_val_*.bin
    TOKENIZER_PATH   SentencePiece .model (default ./data/tokenizers/fineweb_1024_bpe.model)
    MODEL_SIZE       baseline | small
    VOCAB_SIZE       must match the tokenizer (default 1024)
    NUM_ITER         gradient steps                    (default 200)
    BATCH_SIZE       sequences per micro-batch         (default 8)
    MAX_SEQ_LEN      sequence length                   (default 1024)
    TRAIN_BATCH_TOKENS  tokens per optimizer step      (default = BATCH_SIZE*MAX_SEQ_LEN)
    VAL_LOSS_EVERY   validation cadence                (default = NUM_ITER/10)
    N_EVAL_BATCHES   validation batch cap              (default 100)
    LR               scalar learning rate              (default 3e-4)

Download the official export with:
    python data/cached_challenge_fineweb.py --variant sp1024 --train-shards N
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
from MiCoDatasets import (
    fineweb_shards,
    load_fineweb_tokenizer,
    fineweb_bpb_luts,
)
from MiCoUtils import list_quantize_layers

model_size   = os.environ.get("MODEL_SIZE", "baseline")
vocab_size   = int(os.environ.get("VOCAB_SIZE", "1024"))
num_iter     = int(os.environ.get("NUM_ITER", "200"))
batch_size   = int(os.environ.get("BATCH_SIZE", "8"))
max_seq_len  = int(os.environ.get("MAX_SEQ_LEN", "1024"))
lr           = float(os.environ.get("LR", "3e-4"))
train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", str(batch_size * max_seq_len)))
eval_interval = int(os.environ.get("VAL_LOSS_EVERY", str(max(num_iter // 10, 1))))
n_eval_batches = int(os.environ.get("N_EVAL_BATCHES", "100"))
data_path    = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")

model_name = f"parameter_golf_gpt_{model_size}"
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print(f"Using device: {device}")

    if model_size == "small":
        config = ParameterGolfConfig(
            vocab_size=vocab_size, num_layers=6, model_dim=256,
            num_heads=4, num_kv_heads=2, mlp_mult=2,
            max_seq_len=max_seq_len, tie_embeddings=True,
        )
    else:
        config = ParameterGolfConfig(
            vocab_size=vocab_size, num_layers=9, model_dim=512,
            num_heads=8, num_kv_heads=4, mlp_mult=2,
            max_seq_len=max_seq_len, tie_embeddings=True,
        )
    model = ParameterGolfGPT(config).to(device)
    print(f"Model parameters : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Quantizable layers: {len(list_quantize_layers(model))}")

    print(f"Loading FineWeb shards from {data_path} ...")
    train_loader, test_loader = fineweb_shards(
        data_path=data_path, tokenizer_path=tokenizer_path,
        batch_size=batch_size, max_seq_len=config.max_seq_len, device=device,
    )
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    sp = load_fineweb_tokenizer(tokenizer_path, vocab_size=config.vocab_size)
    bpb_luts = fineweb_bpb_luts(sp, config.vocab_size, device)

    t0 = time.time()
    results = model.train_loop(
        n_iter=num_iter,
        train_loader=train_loader,
        test_loader=test_loader,
        verbose=True,
        lr=lr,
        eval_interval=eval_interval,
        train_batch_tokens=train_batch_tokens,
        bpb_luts=bpb_luts,
        n_eval_batches=n_eval_batches,
    )
    print(f"Training done in {time.time() - t0:.1f}s")
    print(f"Final results: {results}")

    os.makedirs("output/ckpt", exist_ok=True)
    ckpt_path = f"output/ckpt/{model_name}.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")
    return
    try:
        from MiCoEval import MiCoEval
        from searchers import MiCoSearcher
        evaluator = MiCoEval(
            model, epochs=1,
            train_loader=train_loader, test_loader=test_loader,
            pretrained_model=ckpt_path, objective="ptq_acc",
        )
        int8_bops = evaluator.eval_bops([8] * model.n_layers * 2)
        print(f"INT8 BOPs: {int8_bops:.3e}")
        searcher = MiCoSearcher(evaluator, n_inits=5, qtypes=[4, 6, 8])
        best_config, best_val = searcher.search(20, "ptq_acc", "bops", int8_bops * 0.5)
        print(f"Best MPQ config: {best_config}, best val {best_val:.4f}")
        os.makedirs("output/json", exist_ok=True)
        with open(f"output/json/{model_name}_search.json", "w") as f:
            json.dump({"best_config": best_config, "best_val": best_val}, f)
    except Exception as exc:
        print(f"MPQ search skipped: {exc}")


if __name__ == "__main__":
    main()
