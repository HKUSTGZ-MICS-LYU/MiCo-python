# Collection of some shared kernels/modules

import torch
import torch.nn as nn
import torch.nn.functional as F

from MiCoModel import MiCoModel, MiCoFunc

ATTENTION_QUANT_NONE = "none"
ATTENTION_QUANT_INT8 = "int8"
ATTENTION_QUANT_BITNET = "int1.58"
ATTENTION_QUANT_FP8 = "fp8"


def _normalize_attention_quant(quant):
    if quant is None or quant is False:
        return ATTENTION_QUANT_NONE
    if quant is True:
        return ATTENTION_QUANT_INT8
    quant = str(quant).lower()
    aliases = {
        "none": ATTENTION_QUANT_NONE,
        "fp32": ATTENTION_QUANT_NONE,
        "float32": ATTENTION_QUANT_NONE,
        "bitnet": ATTENTION_QUANT_BITNET,
        "int8": ATTENTION_QUANT_INT8,
        "i8": ATTENTION_QUANT_INT8,
        "fp8": ATTENTION_QUANT_FP8,
        "float8": ATTENTION_QUANT_FP8,
        "e4m3": ATTENTION_QUANT_FP8,
        "e4m3fn": ATTENTION_QUANT_FP8,
        "e5m2": ATTENTION_QUANT_FP8,
    }
    if quant not in aliases:
        raise ValueError(f"Unsupported attention quantization mode: {quant}")
    return aliases[quant]


def _resolve_fp8_dtype(fp8_dtype):
    if isinstance(fp8_dtype, torch.dtype):
        return fp8_dtype
    fp8_dtype = str(fp8_dtype).lower()
    if fp8_dtype in ["e4m3", "e4m3fn", "float8_e4m3fn"]:
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("torch.float8_e4m3fn is not available in this PyTorch build")
        return torch.float8_e4m3fn
    if fp8_dtype in ["e5m2", "float8_e5m2"]:
        if not hasattr(torch, "float8_e5m2"):
            raise RuntimeError("torch.float8_e5m2 is not available in this PyTorch build")
        return torch.float8_e5m2
    raise ValueError(f"Unsupported FP8 dtype: {fp8_dtype}")


def fake_quant_int8(x, dim=None, eps=1e-8):
    """
    Symmetric fake INT8 quantization. Returns dequantized floating-point values
    so the existing PyTorch attention kernels can still run.
    """
    reduce_dims = dim
    if dim is None:
        max_abs = x.detach().abs().amax()
    else:
        if isinstance(dim, int):
            reduce_dims = (dim,)
        max_abs = x.detach().abs().amax(dim=reduce_dims, keepdim=True)

    scale = max_abs.clamp(min=eps) / 127.0
    q = torch.round(x / scale).clamp(-128, 127)
    return q * scale

def fake_quant_bitnet(x, dim=None, eps=1e-8):
    reduce_dims = dim
    if dim is None:
        max_abs = x.detach().abs().amax()
    else:
        if isinstance(dim, int):
            reduce_dims = (dim,)
        max_abs = x.detach().abs().amax(dim=reduce_dims, keepdim=True)

    scale = max_abs.clamp(min=eps) / 1.0
    q = torch.round(x / scale).clamp(-1, 1)
    return q * scale


def fake_quant_fp8(x, fp8_dtype="e4m3fn"):
    """
    FP8 fake quantization using PyTorch float8 storage formats. The tensor is
    cast to FP8 and immediately dequantized back to the original dtype.
    """
    dtype = _resolve_fp8_dtype(fp8_dtype)
    original_dtype = x.dtype
    return x.to(dtype).to(original_dtype)


class AttentionQuantMixin:
    def _init_attention_quant(
        self,
        quant=ATTENTION_QUANT_NONE,
        fp8_dtype="e4m3fn",
        int_dim=None,
        int_dim_q=None,
        int_dim_k=None,
        int_dim_v=None,
        quantize_q=True,
        quantize_kv=True,
        quantize_score=False,
        quantize_output=True,
    ):
        self.attention_quant = _normalize_attention_quant(quant)
        self.fp8_dtype = fp8_dtype
        self.int_dim = int_dim
        self.int_dim_q = int_dim_q if int_dim_q is not None else int_dim
        self.int_dim_k = int_dim_k if int_dim_k is not None else int_dim
        self.int_dim_v = int_dim_v if int_dim_v is not None else int_dim
        self.quantize_q = quantize_q
        self.quantize_kv = quantize_kv
        self.quantize_score = quantize_score
        self.quantize_output = quantize_output

    def set_quantization(
        self,
        quant=ATTENTION_QUANT_NONE,
        fp8_dtype=None,
        int_dim=None,
        int_dim_q=None,
        int_dim_k=None,
        int_dim_v=None,
        quantize_q=None,
        quantize_kv=None,
        quantize_score=None,
        quantize_output=None,
    ):
        self.attention_quant = _normalize_attention_quant(quant)
        if fp8_dtype is not None:
            self.fp8_dtype = fp8_dtype
        if int_dim is not None:
            self.int_dim = int_dim
            self.int_dim_q = int_dim
            self.int_dim_k = int_dim
            self.int_dim_v = int_dim
        if int_dim_q is not None:
            self.int_dim_q = int_dim_q
        if int_dim_k is not None:
            self.int_dim_k = int_dim_k
        if int_dim_v is not None:
            self.int_dim_v = int_dim_v
        if quantize_q is not None:
            self.quantize_q = quantize_q
        if quantize_kv is not None:
            self.quantize_kv = quantize_kv
        if quantize_score is not None:
            self.quantize_score = quantize_score
        if quantize_output is not None:
            self.quantize_output = quantize_output

    def _quantize_attention_tensor(self, x, int_dim=None):
        dim = int_dim
        if self.attention_quant == ATTENTION_QUANT_INT8:
            return fake_quant_int8(x, dim=dim)
        if self.attention_quant == ATTENTION_QUANT_BITNET:
            return fake_quant_bitnet(x, dim=dim)
        if self.attention_quant == ATTENTION_QUANT_FP8:
            return fake_quant_fp8(x, self.fp8_dtype)
        return x

    def _quantize_q(self, q):
        if not self.quantize_q or self.attention_quant == ATTENTION_QUANT_NONE:
            return q
        return self._quantize_attention_tensor(q, int_dim=self.int_dim_q)

    def _quantize_kv(self, k, v):
        if not self.quantize_kv or self.attention_quant == ATTENTION_QUANT_NONE:
            return k, v
        return (
            self._quantize_attention_tensor(k, int_dim=self.int_dim_k),
            self._quantize_attention_tensor(v, int_dim=self.int_dim_v),
        )


class AttentionScore(nn.Module, AttentionQuantMixin):
    def __init__(self, scale: float, quant=ATTENTION_QUANT_NONE, fp8_dtype="e4m3fn",
                 int_dim=None, int_dim_q=None, int_dim_k=None, int_dim_v=None,
                 quantize_q=True, quantize_kv=True,
                 quantize_score=False, quantize_output=True):
        super().__init__()
        self.scale = float(scale)
        self.MiCo_func = MiCoFunc(
            "MiCo_ViT_attention_{dtype}",
            params=[self.scale]
        )
        self._init_attention_quant(
            quant=quant,
            fp8_dtype=fp8_dtype,
            int_dim=int_dim,
            int_dim_q=int_dim_q,
            int_dim_k=int_dim_k,
            int_dim_v=int_dim_v,
            quantize_q=quantize_q,
            quantize_kv=quantize_kv,
            quantize_score=quantize_score,
            quantize_output=quantize_output,
        )

    def forward(self, q, k, v):
        q = self._quantize_q(q)
        k, v = self._quantize_kv(k, v)
        score = torch.einsum("bhif, bhjf->bhij", q, k) / self.scale
        if self.quantize_score and self.attention_quant != ATTENTION_QUANT_NONE:
            score = self._quantize_attention_tensor(score)
        score = F.softmax(score, dim=-1)
        out = torch.einsum("bhij, bhjf->bihf", score, v)
        if self.quantize_output and self.attention_quant != ATTENTION_QUANT_NONE:
            out = self._quantize_attention_tensor(out)
        return out
    

class LinearAttentionScore(nn.Module, AttentionQuantMixin):
    def __init__(self, eps=1e-6, quant=ATTENTION_QUANT_NONE, fp8_dtype="e4m3fn",
                 int_dim=None, int_dim_q=None, int_dim_k=None, int_dim_v=None,
                 quantize_q=True, quantize_kv=True,
                 quantize_score=False, quantize_output=True):
        super().__init__()
        self.eps = eps
        self.MiCo_func = MiCoFunc(
            "MiCo_linear_attention_{dtype}",
            params=[self.eps]
        )
        self._init_attention_quant(
            quant=quant,
            fp8_dtype=fp8_dtype,
            int_dim=int_dim,
            int_dim_q=int_dim_q,
            int_dim_k=int_dim_k,
            int_dim_v=int_dim_v,
            quantize_q=quantize_q,
            quantize_kv=quantize_kv,
            quantize_score=quantize_score,
            quantize_output=quantize_output,
        )

    def forward(self, q, k, v):
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        q = self._quantize_q(q)
        k, v = self._quantize_kv(k, v)

        context = torch.einsum('bhnd,bhnm->bhdm', k, v)
        if self.quantize_score and self.attention_quant != ATTENTION_QUANT_NONE:
            context = self._quantize_attention_tensor(context)
        k_sum = k.sum(dim=2)

        num = torch.einsum('bhnd,bhdm->bnhm', q, context)
        den = torch.einsum('bhnd,bhd->bnh', q, k_sum).unsqueeze(-1)

        out = num / (den + self.eps)
        if self.quantize_output and self.attention_quant != ATTENTION_QUANT_NONE:
            out = self._quantize_attention_tensor(out)
        return out


class LinearAttention(nn.Module):
    """
    Drop-in linear-attention replacement for CCT.Attention.

    Input/output contract:
      x: [batch, tokens, dim] -> [batch, tokens, dim]
    """

    def __init__(self, dim, num_heads=8, attention_dropout=0.1,
                 projection_dropout=0.1, eps=1e-6,
                 quant=ATTENTION_QUANT_NONE, fp8_dtype="e4m3fn",
                 int_dim=None, int_dim_q=None, int_dim_k=None, int_dim_v=None,
                 quantize_q=True, quantize_kv=True,
                 quantize_score=False, quantize_output=True):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads

        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.attn_score = LinearAttentionScore(
            eps=eps,
            quant=quant,
            fp8_dtype=fp8_dtype,
            int_dim=int_dim,
            int_dim_q=int_dim_q,
            int_dim_k=int_dim_k,
            int_dim_v=int_dim_v,
            quantize_q=quantize_q,
            quantize_kv=quantize_kv,
            quantize_score=quantize_score,
            quantize_output=quantize_output,
        )
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        x = self.attn_score(q, k, v).flatten(2)
        x = self.attn_drop(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def set_attention_quantization(model, quant=ATTENTION_QUANT_INT8, fp8_dtype="e4m3fn",
                               int_dim=None, int_dim_q=None, int_dim_k=None,
                               int_dim_v=None, quantize_q=True, quantize_kv=True,
                               quantize_score=False, quantize_output=True):
    """
    Enable fake quantization for all MiCo attention score modules in a model.

    Args:
        model: PyTorch module containing AttentionScore or LinearAttentionScore.
        quant: "none", "int8", or "fp8".
        fp8_dtype: "e4m3fn" or "e5m2" for FP8 fake quantization.
        int_dim: default reduction dim(s) for INT8/BitNet scales. None means per-tensor.
        int_dim_q: override reduction dim for Q.
        int_dim_k: override reduction dim for K.
        int_dim_v: override reduction dim for V.
        quantize_q: quantize Q before attention math.
        quantize_kv: quantize K/V before attention math.
        quantize_score: quantize attention logits for softmax attention or context for linear attention.
        quantize_output: quantize attention output before returning.
    """
    for module in model.modules():
        if isinstance(module, AttentionQuantMixin):
            module.set_quantization(
                quant=quant,
                fp8_dtype=fp8_dtype,
                int_dim=int_dim,
                int_dim_q=int_dim_q,
                int_dim_k=int_dim_k,
                int_dim_v=int_dim_v,
                quantize_q=quantize_q,
                quantize_kv=quantize_kv,
                quantize_score=quantize_score,
                quantize_output=quantize_output,
            )
    return model
