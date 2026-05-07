import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from MiCoModel import MiCoModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class BertArgs:
    vocab_size: int = 261
    max_seq_len: int = 128
    dim: int = 128
    n_layers: int = 2
    n_heads: int = 2
    hidden_dim: int = 512
    type_vocab_size: int = 2
    dropout: float = 0.1
    norm_eps: float = 1e-5
    pad_token_id: int = 0


class BertSelfAttention(nn.Module):
    def __init__(self, args: BertArgs):
        super().__init__()
        if args.dim % args.n_heads != 0:
            raise ValueError("BERT dim must be divisible by n_heads")
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.scale = math.sqrt(self.head_dim)

        self.q = nn.Linear(args.dim, args.dim)
        self.k = nn.Linear(args.dim, args.dim)
        self.v = nn.Linear(args.dim, args.dim)
        self.o = nn.Linear(args.dim, args.dim)
        self.dropout = nn.Dropout(args.dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        return x.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, s, _ = x.shape
        q = self._split_heads(self.q(x))
        k = self._split_heads(self.k(x))
        v = self._split_heads(self.v(x))

        score = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if attention_mask is not None:
            mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
            score = score.masked_fill(~mask, torch.finfo(score.dtype).min)

        attn = F.softmax(score, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, s, self.n_heads * self.head_dim)
        return self.o(out)


class BertEncoderLayer(nn.Module):
    def __init__(self, args: BertArgs):
        super().__init__()
        self.attn = BertSelfAttention(args)
        self.attn_norm = nn.LayerNorm(args.dim, eps=args.norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(args.dim, args.hidden_dim),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_dim, args.dim),
        )
        self.ffn_norm = nn.LayerNorm(args.dim, eps=args.norm_eps)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.attn_norm(x + self.dropout(self.attn(x, attention_mask)))
        x = self.ffn_norm(x + self.dropout(self.ffn(x)))
        return x


class BertMLMHead(nn.Module):
    def __init__(self, args: BertArgs, embedding_weight: nn.Parameter):
        super().__init__()
        self.dense = nn.Linear(args.dim, args.dim)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(args.dim, eps=args.norm_eps)
        self.decoder = nn.Linear(args.dim, args.vocab_size)
        self.decoder.weight = embedding_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense(x)
        x = self.act(x)
        x = self.norm(x)
        return self.decoder(x)


class MiCoBERT(MiCoModel):
    last_loss: Optional[torch.Tensor]

    def __init__(self, args: BertArgs = BertArgs()):
        super().__init__()
        self.params = args
        self.default_dataset = "local_mlm"
        self.word_embeddings = nn.Embedding(args.vocab_size, args.dim, padding_idx=args.pad_token_id)
        self.position_embeddings = nn.Embedding(args.max_seq_len, args.dim)
        self.token_type_embeddings = nn.Embedding(args.type_vocab_size, args.dim)
        self.emb_norm = nn.LayerNorm(args.dim, eps=args.norm_eps)
        self.dropout = nn.Dropout(args.dropout)
        self.layers = nn.ModuleList([BertEncoderLayer(args) for _ in range(args.n_layers)])
        self.mlm_head = BertMLMHead(args, self.word_embeddings.weight)
        self.last_loss = None

        self.apply(self._init_weights)
        with torch.no_grad():
            self.word_embeddings.weight[args.pad_token_id].zero_()
        self.n_layers = len(self.get_qlayers())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, s = input_ids.shape
        if s > self.params.max_seq_len:
            raise ValueError(f"Sequence length {s} exceeds max_seq_len={self.params.max_seq_len}")

        if attention_mask is None:
            attention_mask = input_ids.ne(self.params.pad_token_id)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        pos = torch.arange(s, device=input_ids.device).unsqueeze(0).expand(b, s)
        x = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(pos)
            + self.token_type_embeddings(token_type_ids)
        )
        x = self.dropout(self.emb_norm(x))

        for layer in self.layers:
            x = layer(x, attention_mask)

        logits = self.mlm_head(x)
        if labels is not None:
            self.last_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        else:
            self.last_loss = None
        return logits

    @torch.no_grad()
    def test(self, test_loader, n_eval_batches: int = 100):
        self.eval()
        losses = []
        correct = 0
        total = 0
        for i, (input_ids, labels) in enumerate(test_loader):
            if i >= n_eval_batches:
                break
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            logits = self(input_ids, labels)
            losses.append(self.last_loss.item())
            masked = labels.ne(-100)
            if masked.any():
                pred = logits.argmax(dim=-1)
                correct += pred[masked].eq(labels[masked]).sum().item()
                total += masked.sum().item()

        return {
            "TestLoss": float(np.mean(losses)) if losses else float("nan"),
            "TestAcc": correct / total if total > 0 else 0.0,
        }

    def train_loop(
        self,
        n_epoch: int = None,
        train_loader=None,
        test_loader=None,
        verbose: bool = False,
        lr: float = 3e-4,
        scheduler: str = "cosine",
        early_stopping: bool = False,
        n_iter: int = None,
        eval_interval: Optional[int] = None,
    ):
        if train_loader is None or test_loader is None:
            raise ValueError("train_loader and test_loader are required")
        if n_iter is None:
            if n_epoch is None:
                raise ValueError("Either n_epoch or n_iter must be provided")
            n_iter = max(1, n_epoch * len(train_loader))
        if eval_interval is None:
            eval_interval = max(n_iter // 10, 50)

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        lr_scheduler = None
        if scheduler == "cosine":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_iter)
        elif scheduler == "step":
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, n_iter // 4), gamma=0.5)
        elif scheduler != "none":
            raise ValueError(f"Scheduler {scheduler} not recognized.")

        train_iter = iter(train_loader)

        def get_batch():
            nonlocal train_iter
            try:
                return next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                return next(train_iter)

        self.train()
        loop = tqdm(range(n_iter), disable=not verbose)
        loss = torch.tensor(float("inf"))
        results = {"TestLoss": float("nan"), "TestAcc": 0.0}
        for step in loop:
            loop.set_description(f"Step {step} Loss: {loss.item():.4f}")
            input_ids, labels = get_batch()
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            self(input_ids, labels)
            loss = self.last_loss
            loss.backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            if step % eval_interval == 0:
                results = self.test(test_loader, n_eval_batches=20)
                if verbose:
                    print(
                        f"step {step}: val_loss {results['TestLoss']:.4f}, "
                        f"masked_acc {results['TestAcc'] * 100:.2f}%"
                    )
                self.train()

        results = self.test(test_loader, n_eval_batches=100)
        return results


def TinyBERTLocal() -> MiCoBERT:
    return MiCoBERT(BertArgs(dim=128, n_layers=2, n_heads=2, hidden_dim=512, max_seq_len=128))


def MicroBERTLocal() -> MiCoBERT:
    return MiCoBERT(BertArgs(dim=64, n_layers=2, n_heads=2, hidden_dim=256, max_seq_len=64))


def BertNAS(args: dict) -> MiCoBERT:
    return MiCoBERT(BertArgs(**args))
