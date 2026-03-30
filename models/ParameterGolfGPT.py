"""
GPT language model from the OpenAI Parameter Golf challenge.
Reference: https://github.com/openai/parameter-golf

Baseline configuration (``ParameterGolfBaseline``):
- 9 transformer layers, 512 hidden dim
- 8 query heads, 4 KV heads (Grouped Query Attention)
- 2x MLP expansion with relu² activation
- Vocabulary size 1024, sequence length 1024
- Tied input/output embeddings
- U-Net style skip connections between encoder and decoder halves
"""

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from MiCoModel import MiCoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ParameterGolfConfig:
    vocab_size: int = 1024
    num_layers: int = 9
    model_dim: int = 512
    num_heads: int = 8
    num_kv_heads: int = 4
    mlp_mult: int = 2
    max_seq_len: int = 1024
    tie_embeddings: bool = True
    tied_embed_init_std: float = 0.005
    logit_softcap: float = 30.0
    rope_base: float = 10000.0
    qk_gain_init: float = 1.5


# ---------------------------------------------------------------------------
# Version-safe helpers
# ---------------------------------------------------------------------------

def _rms_norm(x: torch.Tensor, eps: Optional[float] = None) -> torch.Tensor:
    """RMS normalisation – uses F.rms_norm when available (PyTorch >= 2.4)."""
    if hasattr(F, "rms_norm"):
        return F.rms_norm(x, (x.size(-1),), eps=eps)
    variance = x.float().pow(2).mean(-1, keepdim=True)
    _eps = eps if eps is not None else 1e-6
    return (x * torch.rsqrt(variance + _eps)).to(x.dtype)


def _sdpa_gqa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
) -> torch.Tensor:
    """
    Scaled dot-product attention with Grouped Query Attention support.

    Uses the native ``enable_gqa`` flag when running on PyTorch >= 2.5;
    otherwise falls back to manually repeating K/V to match the query heads.
    """
    if num_kv_heads == num_heads:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
    try:
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=True, enable_gqa=True
        )
    except TypeError:
        pass
    # Manual fallback for PyTorch < 2.5
    n_rep = num_heads // num_kv_heads
    bsz, _, seqlen, head_dim = k.shape
    k = k[:, :, None, :, :].expand(bsz, num_kv_heads, n_rep, seqlen, head_dim)
    k = k.reshape(bsz, num_heads, seqlen, head_dim)
    v = v[:, :, None, :, :].expand(bsz, num_kv_heads, n_rep, seqlen, head_dim)
    v = v.reshape(bsz, num_heads, seqlen, head_dim)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)


class RMSNorm(nn.Module):
    """RMSNorm without learnable scale parameters."""

    def __init__(self, eps: Optional[float] = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _rms_norm(x, self.eps)


class Rotary(nn.Module):
    """Rotary positional embeddings (RoPE) with cached cos/sin tables."""

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Optional[torch.Tensor] = None
        self._sin_cached: Optional[torch.Tensor] = None

    def forward(
        self, seq_len: int, dev: torch.device, dtype: torch.dtype
    ):
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != dev
        ):
            t = torch.arange(seq_len, device=dev, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(dev))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    """Causal self-attention with Grouped Query Attention (GQA) and RoPE."""

    def __init__(self, config: ParameterGolfConfig):
        super().__init__()
        dim = config.model_dim
        if dim % config.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if config.num_heads % config.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = dim // config.num_heads
        kv_dim = config.num_kv_heads * self.head_dim

        self.c_q = nn.Linear(dim, dim, bias=False)
        self.c_k = nn.Linear(dim, kv_dim, bias=False)
        self.c_v = nn.Linear(dim, kv_dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.q_gain = nn.Parameter(
            torch.full((config.num_heads,), config.qk_gain_init, dtype=torch.float32)
        )
        self.rotary = Rotary(self.head_dim, base=config.rope_base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = _rms_norm(q)
        k = _rms_norm(k)

        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = _apply_rotary_emb(q, cos, sin)
        k = _apply_rotary_emb(k, cos, sin)

        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]

        y = _sdpa_gqa(q, k, v, self.num_heads, self.num_kv_heads)
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class GolfMLP(nn.Module):
    """Two-layer MLP with relu² activation."""

    def __init__(self, config: ParameterGolfConfig):
        super().__init__()
        hidden = config.mlp_mult * config.model_dim
        self.fc = nn.Linear(config.model_dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, config.model_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    """Transformer block with residual mixing and learnable per-channel scales."""

    def __init__(self, config: ParameterGolfConfig):
        super().__init__()
        dim = config.model_dim
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(config)
        self.mlp = GolfMLP(config)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        # resid_mix[0] weights the running residual; resid_mix[1] weights x0 (initial embedding)
        self.resid_mix = nn.Parameter(
            torch.stack((torch.ones(dim), torch.zeros(dim))).float()
        )

    def forward(self, x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self.attn(self.attn_norm(x))
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class ParameterGolfGPT(MiCoModel):
    """
    GPT language model from the OpenAI Parameter Golf challenge.

    Architecture highlights:
    - Grouped Query Attention (GQA) with learnable per-head Q gain and RoPE
    - relu² MLP expansion
    - U-Net style skip connections: the first ``num_encoder_layers`` blocks store
      skip tensors that are re-injected in reverse order by the decoder half
    - Learnable per-channel residual mix that blends the running residual with the
      initial post-embedding representation ``x0``
    - Optional tied input/output embeddings with logit softcap

    Reference: https://github.com/openai/parameter-golf
    """

    def __init__(self, config: Optional[ParameterGolfConfig] = None):
        super().__init__()
        if config is None:
            config = ParameterGolfConfig()
        self.config = config

        if config.logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {config.logit_softcap}")

        self.tok_emb = nn.Embedding(config.vocab_size, config.model_dim)
        self.num_encoder_layers = config.num_layers // 2
        self.num_decoder_layers = config.num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(
            torch.ones(self.num_skip_weights, config.model_dim, dtype=torch.float32)
        )
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.final_norm = RMSNorm()
        self.lm_head = (
            None
            if config.tie_embeddings
            else nn.Linear(config.model_dim, config.vocab_size, bias=False)
        )

        self._init_weights()
        self.n_layers = len(self.get_qlayers())
        self.default_dataset = "FINEWEB"
        self.last_loss: Optional[torch.Tensor] = None

    def _init_weights(self) -> None:
        if self.config.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.config.tied_embed_init_std)
        else:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
            if self.lm_head is not None:
                nn.init.zeros_(self.lm_head.weight)

    def _compute_logits(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        return self.config.logit_softcap * torch.tanh(logits_proj / self.config.logit_softcap)

    def forward(
        self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            tokens: (batch, seq_len) integer token indices.
            targets: (batch, seq_len) target token indices for next-token prediction.
                     When provided the cross-entropy loss is stored in ``self.last_loss``
                     and all-position logits are returned. When ``None``, only the
                     last-position logits are returned (for inference).

        Returns:
            logits tensor.
        """
        x = self.tok_emb(tokens)
        x = _rms_norm(x)
        x0 = x
        skips: List[torch.Tensor] = []

        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)

        x = self.final_norm(x)

        if targets is not None:
            logits = self._compute_logits(x.reshape(-1, x.size(-1)))
            self.last_loss = F.cross_entropy(
                logits.float(), targets.reshape(-1), reduction="mean"
            )
            return logits.view(tokens.size(0), tokens.size(1), -1)
        else:
            logits = self._compute_logits(x[:, [-1], :])
            self.last_loss = None
            return logits

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate ``max_new_tokens`` continuation tokens given ``idx``."""
        for _ in range(max_new_tokens):
            idx_cond = (
                idx
                if idx.size(1) <= self.config.max_seq_len
                else idx[:, -self.config.max_seq_len :]
            )
            logits = self(idx_cond)[:, -1, :]
            if temperature == 0.0:
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def test(self, test_loader, n_eval_batches: int = 100):
        """
        Evaluate the model on ``test_loader``.

        Returns a dict with ``TestLoss`` (cross-entropy) and ``Perplexity``.
        """
        self.eval()
        losses = []
        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                if i >= n_eval_batches:
                    break
                x, y = x.to(device), y.to(device)
                self(x, y)
                if self.last_loss is not None:
                    losses.append(self.last_loss.item())
        if not losses:
            return {"TestLoss": float("nan"), "Perplexity": float("nan")}
        avg_loss = float(np.mean(losses))
        perplexity = math.exp(avg_loss)
        return {"TestLoss": avg_loss, "Perplexity": perplexity}

    def train_loop(
        self,
        n_iter: int,
        train_loader,
        test_loader,
        verbose: bool = False,
        lr: float = 3e-4,
        eval_interval: Optional[int] = None,
    ):
        """
        Iterate-based training loop for the language model.

        Args:
            n_iter: Total number of gradient steps.
            train_loader: DataLoader yielding (input_ids, labels) batches.
            test_loader: DataLoader used for periodic validation.
            verbose: Whether to show a tqdm progress bar.
            lr: Peak learning rate for AdamW.
            eval_interval: How often (in steps) to run validation.
                           Defaults to every 10 % of ``n_iter``.

        Returns:
            dict with the final ``TestLoss`` and ``Perplexity``.
        """
        if eval_interval is None:
            eval_interval = max(n_iter // 10, 50)

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_iter)

        train_iter = iter(train_loader)

        def get_batch():
            nonlocal train_iter
            try:
                return next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                return next(train_iter)

        self.train()
        iters = tqdm(range(n_iter), disable=not verbose)
        running_loss = float("inf")
        for iter_num in iters:
            iters.set_description(f"Step {iter_num} Loss: {running_loss:.4f}")

            if iter_num % eval_interval == 0:
                results = self.test(test_loader, n_eval_batches=50)
                print(
                    f"step {iter_num}: val_loss {results['TestLoss']:.4f}, "
                    f"perplexity {results['Perplexity']:.2f}"
                )
                self.train()

            x, y = get_batch()
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            self(x, y)
            loss = self.last_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss = loss.item()

        final_results = self.test(test_loader, n_eval_batches=100)
        print(
            f"Final: val_loss {final_results['TestLoss']:.4f}, "
            f"perplexity {final_results['Perplexity']:.2f}"
        )
        return final_results


# ---------------------------------------------------------------------------
# Pre-built model configurations
# ---------------------------------------------------------------------------

def ParameterGolfBaseline() -> ParameterGolfGPT:
    """Baseline from the parameter golf leaderboard (9 layers, 512 dim, vocab 1024)."""
    return ParameterGolfGPT(
        ParameterGolfConfig(
            vocab_size=1024,
            num_layers=9,
            model_dim=512,
            num_heads=8,
            num_kv_heads=4,
            mlp_mult=2,
            max_seq_len=1024,
            tie_embeddings=True,
        )
    )


def ParameterGolfSmall() -> ParameterGolfGPT:
    """Smaller configuration suitable for quick local experiments."""
    return ParameterGolfGPT(
        ParameterGolfConfig(
            vocab_size=1024,
            num_layers=6,
            model_dim=256,
            num_heads=4,
            num_kv_heads=2,
            mlp_mult=2,
            max_seq_len=512,
            tie_embeddings=True,
        )
    )
