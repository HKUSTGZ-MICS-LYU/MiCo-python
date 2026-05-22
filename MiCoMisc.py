# Collection of shared modules and lightweight tensor quantization helpers.

import torch
import torch.nn as nn
import torch.nn.functional as F


class MiCoFunc:
    def __init__(self, name, input_names=None, params=None):
        self.name = name
        self.input_names = [] if input_names is None else input_names
        self.params = [] if params is None else params


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
        "32": ATTENTION_QUANT_NONE,
        "32.0": ATTENTION_QUANT_NONE,
        "bitnet": ATTENTION_QUANT_BITNET,
        "int1.58": ATTENTION_QUANT_BITNET,
        "1.58": ATTENTION_QUANT_BITNET,
        "1.5": ATTENTION_QUANT_BITNET,
        "1.58bit": ATTENTION_QUANT_BITNET,
        "ternary": ATTENTION_QUANT_BITNET,
        "int8": ATTENTION_QUANT_INT8,
        "i8": ATTENTION_QUANT_INT8,
        "8": ATTENTION_QUANT_INT8,
        "8.0": ATTENTION_QUANT_INT8,
        "fp8": ATTENTION_QUANT_FP8,
        "float8": ATTENTION_QUANT_FP8,
        "e4m3": ATTENTION_QUANT_FP8,
        "e4m3fn": ATTENTION_QUANT_FP8,
        "e5m2": ATTENTION_QUANT_FP8,
    }
    if quant not in aliases:
        raise ValueError(f"Unsupported attention quantization mode: {quant}")
    return aliases[quant]


def attention_qtype_to_quant(qtype):
    if isinstance(qtype, str):
        return _normalize_attention_quant(qtype)
    if qtype is None or qtype is False:
        return ATTENTION_QUANT_NONE
    if qtype >= 32:
        return ATTENTION_QUANT_NONE
    if qtype == 8:
        return ATTENTION_QUANT_INT8
    if 0 < qtype < 2:
        return ATTENTION_QUANT_BITNET
    if qtype == 2:
        return ATTENTION_QUANT_BITNET
    raise ValueError(f"Unsupported attention qtype: {qtype}")


def attention_quant_to_bits(quant):
    quant = _normalize_attention_quant(quant)
    if quant == ATTENTION_QUANT_NONE:
        return 32
    if quant == ATTENTION_QUANT_INT8 or quant == ATTENTION_QUANT_FP8:
        return 8
    if quant == ATTENTION_QUANT_BITNET:
        return 1.58
    raise ValueError(f"Unsupported attention quantization mode: {quant}")


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

    scale = max_abs.clamp(min=eps)
    q = torch.round(x / scale).clamp(-1, 1)
    return q * scale


def fake_quant_fp8(x, fp8_dtype="e4m3fn"):
    dtype = _resolve_fp8_dtype(fp8_dtype)
    return x.to(dtype).to(x.dtype)


class AttentionQuantMixin:
    def _init_attention_quant(
        self,
        quant=ATTENTION_QUANT_NONE,
        q_quant=None,
        kv_quant=None,
        k_quant=None,
        v_quant=None,
        score_quant=None,
        fp8_dtype="e4m3fn",
        int_dim=None,
        int_dim_q=None,
        int_dim_k=None,
        int_dim_v=None,
        int_dim_score=None,
        quantize_q=True,
        quantize_kv=True,
        quantize_k=True,
        quantize_v=True,
        quantize_score=False,
        quantize_output=False,
    ):
        self.attention_quant = _normalize_attention_quant(quant)
        self.q_attention_quant = _normalize_attention_quant(q_quant if q_quant is not None else quant)
        kv_quant = kv_quant if kv_quant is not None else quant
        self.k_attention_quant = _normalize_attention_quant(k_quant if k_quant is not None else kv_quant)
        self.v_attention_quant = _normalize_attention_quant(v_quant if v_quant is not None else kv_quant)
        self.kv_attention_quant = self.k_attention_quant
        self.score_attention_quant = _normalize_attention_quant(
            score_quant if score_quant is not None else quant
        )
        self.fp8_dtype = fp8_dtype
        self.int_dim = int_dim
        self.int_dim_q = int_dim_q if int_dim_q is not None else int_dim
        self.int_dim_k = int_dim_k if int_dim_k is not None else int_dim
        self.int_dim_v = int_dim_v if int_dim_v is not None else int_dim
        self.int_dim_score = int_dim_score if int_dim_score is not None else int_dim
        self.quantize_q = quantize_q
        self.quantize_k = quantize_kv and quantize_k
        self.quantize_v = quantize_kv and quantize_v
        self.quantize_kv = quantize_kv
        self.quantize_score = quantize_score
        self.quantize_output = quantize_output

    def set_quantization(
        self,
        quant=ATTENTION_QUANT_NONE,
        q_quant=None,
        kv_quant=None,
        k_quant=None,
        v_quant=None,
        score_quant=None,
        fp8_dtype=None,
        int_dim=None,
        int_dim_q=None,
        int_dim_k=None,
        int_dim_v=None,
        int_dim_score=None,
        quantize_q=None,
        quantize_kv=None,
        quantize_k=None,
        quantize_v=None,
        quantize_score=None,
        quantize_output=None,
    ):
        self.attention_quant = _normalize_attention_quant(quant)
        self.q_attention_quant = _normalize_attention_quant(q_quant if q_quant is not None else quant)
        kv_quant = kv_quant if kv_quant is not None else quant
        self.k_attention_quant = _normalize_attention_quant(k_quant if k_quant is not None else kv_quant)
        self.v_attention_quant = _normalize_attention_quant(v_quant if v_quant is not None else kv_quant)
        self.kv_attention_quant = self.k_attention_quant
        self.score_attention_quant = _normalize_attention_quant(
            score_quant if score_quant is not None else quant
        )
        if fp8_dtype is not None:
            self.fp8_dtype = fp8_dtype
        if int_dim is not None:
            self.int_dim = int_dim
            self.int_dim_q = int_dim
            self.int_dim_k = int_dim
            self.int_dim_v = int_dim
            self.int_dim_score = int_dim
        if int_dim_q is not None:
            self.int_dim_q = int_dim_q
        if int_dim_k is not None:
            self.int_dim_k = int_dim_k
        if int_dim_v is not None:
            self.int_dim_v = int_dim_v
        if int_dim_score is not None:
            self.int_dim_score = int_dim_score
        if quantize_q is not None:
            self.quantize_q = quantize_q
        if quantize_kv is not None:
            self.quantize_kv = quantize_kv
            self.quantize_k = quantize_kv
            self.quantize_v = quantize_kv
        if quantize_k is not None:
            self.quantize_k = quantize_k
        if quantize_v is not None:
            self.quantize_v = quantize_v
        if quantize_score is not None:
            self.quantize_score = quantize_score
        if quantize_output is not None:
            self.quantize_output = quantize_output

    def _quantize_attention_tensor(self, x, int_dim=None, quant=None):
        quant = self.attention_quant if quant is None else _normalize_attention_quant(quant)
        if quant == ATTENTION_QUANT_INT8:
            return fake_quant_int8(x, dim=int_dim)
        if quant == ATTENTION_QUANT_BITNET:
            return fake_quant_bitnet(x, dim=int_dim)
        if quant == ATTENTION_QUANT_FP8:
            return fake_quant_fp8(x, self.fp8_dtype)
        return x

    def _quantize_q(self, q):
        if not self.quantize_q or self.q_attention_quant == ATTENTION_QUANT_NONE:
            return q
        return self._quantize_attention_tensor(q, int_dim=self.int_dim_q, quant=self.q_attention_quant)

    def _quantize_k(self, k):
        if not self.quantize_k or self.k_attention_quant == ATTENTION_QUANT_NONE:
            return k
        return self._quantize_attention_tensor(k, int_dim=self.int_dim_k, quant=self.k_attention_quant)

    def _quantize_v(self, v):
        if not self.quantize_v or self.v_attention_quant == ATTENTION_QUANT_NONE:
            return v
        return self._quantize_attention_tensor(v, int_dim=self.int_dim_v, quant=self.v_attention_quant)

    def _quantize_kv(self, k, v):
        return self._quantize_k(k), self._quantize_v(v)

    def _quantize_score(self, score):
        if not self.quantize_score or self.score_attention_quant == ATTENTION_QUANT_NONE:
            return score
        return self._quantize_attention_tensor(
            score, int_dim=self.int_dim_score, quant=self.score_attention_quant
        )

    def _quantize_output(self, output):
        if not self.quantize_output or self.attention_quant == ATTENTION_QUANT_NONE:
            return output
        return self._quantize_attention_tensor(output)


class AttentionScore(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = float(scale)
        self.MiCo_func = MiCoFunc("MiCo_ViT_attention_{dtype}", params=[self.scale])

    def forward(self, q, k, v):
        score = torch.einsum("bhif, bhjf->bhij", q, k) / self.scale
        score = F.softmax(score, dim=-1)
        return torch.einsum("bhij, bhjf->bihf", score, v)


class LinearAttentionScore(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.MiCo_func = MiCoFunc("MiCo_linear_attention_{dtype}", params=[self.eps])

    def forward(self, q, k, v):
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        context = torch.einsum("bhnd,bhnm->bhdm", k, v)
        k_sum = k.sum(dim=2)
        num = torch.einsum("bhnd,bhdm->bnhm", q, context)
        den = torch.einsum("bhnd,bhd->bnh", q, k_sum).unsqueeze(-1)
        return num / (den + self.eps)


class LinearAttention(nn.Module):
    def __init__(self, dim, num_heads=8, attention_dropout=0.1,
                 projection_dropout=0.1, eps=1e-6, **kwargs):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads

        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.attn_score = LinearAttentionScore(eps=eps)
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


def set_attention_quantization(model, quant=ATTENTION_QUANT_INT8,
                               q_quant=None, kv_quant=None, k_quant=None, v_quant=None,
                               score_quant=None, fp8_dtype="e4m3fn",
                               int_dim=None, int_dim_q=None, int_dim_k=None,
                               int_dim_v=None, int_dim_score=None,
                               quantize_q=True, quantize_kv=True,
                               quantize_k=True, quantize_v=True,
                               quantize_score=False, quantize_output=False):
    for module in model.modules():
        if isinstance(module, AttentionQuantMixin):
            module.set_quantization(
                quant=quant,
                q_quant=q_quant,
                kv_quant=kv_quant,
                k_quant=k_quant,
                v_quant=v_quant,
                score_quant=score_quant,
                fp8_dtype=fp8_dtype,
                int_dim=int_dim,
                int_dim_q=int_dim_q,
                int_dim_k=int_dim_k,
                int_dim_v=int_dim_v,
                int_dim_score=int_dim_score,
                quantize_q=quantize_q,
                quantize_kv=quantize_kv,
                quantize_k=quantize_k,
                quantize_v=quantize_v,
                quantize_score=quantize_score,
                quantize_output=quantize_output,
            )
    return model
