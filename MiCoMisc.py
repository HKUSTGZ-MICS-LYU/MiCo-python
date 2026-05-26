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


def _parse_int_quant(quant):
    text = str(quant).lower()
    if text.startswith("int") and text[3:].isdigit():
        bits = int(text[3:])
    elif text.startswith("i") and text[1:].isdigit():
        bits = int(text[1:])
    elif text.isdigit():
        bits = int(text)
    else:
        return None
    if 1 <= bits <= 31:
        return bits
    return None


def _normalize_attention_quant(quant):
    if quant is None or quant is False:
        return ATTENTION_QUANT_NONE
    if quant is True:
        return ATTENTION_QUANT_INT8
    quant = str(quant).lower()
    int_bits = _parse_int_quant(quant)
    if int_bits is not None:
        if int_bits >= 32:
            return ATTENTION_QUANT_NONE
        return f"int{int_bits}"
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
    if 0 < qtype < 2:
        return ATTENTION_QUANT_BITNET
    if float(qtype).is_integer():
        return f"int{int(qtype)}"
    raise ValueError(f"Unsupported attention qtype: {qtype}")


def attention_quant_to_bits(quant):
    quant = _normalize_attention_quant(quant)
    if quant == ATTENTION_QUANT_NONE:
        return 32
    if quant == ATTENTION_QUANT_FP8:
        return 8
    if quant == ATTENTION_QUANT_BITNET:
        return 1.58
    int_bits = _parse_int_quant(quant)
    if int_bits is not None:
        return int_bits
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
    return fake_quant_int(x, qbit=8, dim=dim, eps=eps)


def _normalize_reduce_dims(dim):
    if dim is None:
        return None
    if isinstance(dim, int):
        return (dim,)
    return tuple(dim)


def _apply_group_quant(x, quant_fn, dim=-1, group_size=None):
    if group_size is None or group_size <= 0:
        return quant_fn(x)

    dim = dim % x.dim()
    size = x.size(dim)
    if size % group_size != 0:
        raise ValueError(f"group_size {group_size} must divide tensor dim {dim} size {size}")

    num_groups = size // group_size
    shape = list(x.shape)
    shape[dim] = num_groups
    shape.insert(dim + 1, group_size)
    grouped = x.reshape(shape)
    quantized = quant_fn(grouped, reduce_dim=dim + 1)
    return quantized.reshape_as(x)


def _clip_by_mean_abs(x, alpha=None, dim=None, eps=1e-8):
    if alpha is None or alpha <= 0:
        return x
    reduce_dims = _normalize_reduce_dims(dim)
    if reduce_dims is None:
        limit = alpha * x.detach().abs().mean().clamp(min=eps)
    else:
        limit = alpha * x.detach().abs().mean(dim=reduce_dims, keepdim=True).clamp(min=eps)
    return x.clamp(-limit, limit)


def fake_quant_int(x, qbit=8, dim=None, eps=1e-8):
    reduce_dims = _normalize_reduce_dims(dim)
    if dim is None:
        max_abs = x.detach().abs().amax()
    else:
        max_abs = x.detach().abs().amax(dim=reduce_dims, keepdim=True)

    qbit = int(qbit)
    if qbit <= 0:
        raise ValueError(f"qbit must be positive, got {qbit}")
    if qbit == 1:
        if dim is None:
            scale = x.detach().abs().mean().clamp(min=eps)
        else:
            scale = x.detach().abs().mean(dim=reduce_dims, keepdim=True).clamp(min=eps)
        q = torch.sign(x)
        q = torch.where(q == 0.0, torch.ones_like(q), q)
        return q * scale

    qmax = 2 ** (qbit - 1) - 1
    qmin = -(2 ** (qbit - 1))
    scale = max_abs.clamp(min=eps) / float(qmax)
    q = torch.round(x / scale).clamp(qmin, qmax)
    return q * scale


def _normalize_bitnet_scale(bitnet_scale):
    bitnet_scale = str(bitnet_scale).lower()
    if bitnet_scale not in ["max", "mean"]:
        raise ValueError(f"Unsupported bitnet scale mode: {bitnet_scale}")
    return bitnet_scale


def fake_quant_bitnet(x, dim=None, eps=1e-8, mode="max",
                      group_size=None, group_dim=-1, clip_alpha=None):
    x = _clip_by_mean_abs(x, alpha=clip_alpha, dim=dim, eps=eps)

    def quant_fn(tensor, reduce_dim=None):
        return _fake_quant_bitnet_impl(
            tensor,
            dim=reduce_dim if reduce_dim is not None else dim,
            eps=eps,
            mode=mode,
        )

    if group_size is not None and group_size > 0:
        return _apply_group_quant(x, quant_fn, dim=group_dim, group_size=group_size)
    return quant_fn(x)


def _fake_quant_bitnet_impl(x, dim=None, eps=1e-8, mode="max"):
    reduce_dims = _normalize_reduce_dims(dim)
    mode = _normalize_bitnet_scale(mode)

    if mode == "max":
        if dim is None:
            denom = x.detach().abs().amax()
        else:
            denom = x.detach().abs().amax(dim=reduce_dims, keepdim=True)
    else:
        if dim is None:
            denom = x.detach().abs().mean()
        else:
            denom = x.detach().abs().mean(dim=reduce_dims, keepdim=True)

    scale = denom.clamp(min=eps)
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
        bitnet_scale="max",
        bitnet_group_size=None,
        bitnet_group_dim=-1,
        bitnet_clip=None,
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
        llama_kv_quant_scope="all",
        llama_kv_group_size=32,
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
        self.bitnet_scale = _normalize_bitnet_scale(bitnet_scale)
        self.bitnet_group_size = bitnet_group_size
        self.bitnet_group_dim = bitnet_group_dim
        self.bitnet_clip = bitnet_clip
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
        self.llama_kv_quant_scope = self._normalize_llama_kv_quant_scope(llama_kv_quant_scope)
        self.llama_kv_group_size = int(llama_kv_group_size)

    def set_quantization(
        self,
        quant=ATTENTION_QUANT_NONE,
        q_quant=None,
        kv_quant=None,
        k_quant=None,
        v_quant=None,
        score_quant=None,
        bitnet_scale=None,
        bitnet_group_size=None,
        bitnet_group_dim=None,
        bitnet_clip=None,
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
        llama_kv_quant_scope=None,
        llama_kv_group_size=None,
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
        if bitnet_scale is not None:
            self.bitnet_scale = _normalize_bitnet_scale(bitnet_scale)
        if bitnet_group_size is not None:
            self.bitnet_group_size = bitnet_group_size
        if bitnet_group_dim is not None:
            self.bitnet_group_dim = bitnet_group_dim
        if bitnet_clip is not None:
            self.bitnet_clip = bitnet_clip
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
        if llama_kv_quant_scope is not None:
            self.llama_kv_quant_scope = self._normalize_llama_kv_quant_scope(llama_kv_quant_scope)
        if llama_kv_group_size is not None:
            self.llama_kv_group_size = int(llama_kv_group_size)

    @staticmethod
    def _normalize_llama_kv_quant_scope(scope):
        scope = str(scope).lower()
        aliases = {
            "all": "all",
            "full": "all",
            "history": "current_group",
            "past": "current_group",
            "cache": "current_group",
            "cached": "current_group",
            "current_group": "current_group",
            "group": "current_group",
            "kivi": "current_group",
            "residual": "current_group",
        }
        if scope not in aliases:
            raise ValueError(f"Unsupported LLaMa KV quantization scope: {scope}")
        return aliases[scope]

    def _quantize_attention_tensor(self, x, int_dim=None, quant=None):
        quant = self.attention_quant if quant is None else _normalize_attention_quant(quant)
        int_bits = _parse_int_quant(quant)
        if int_bits is not None:
            return fake_quant_int(x, qbit=int_bits, dim=int_dim)
        if quant == ATTENTION_QUANT_BITNET:
            return fake_quant_bitnet(
                x,
                dim=int_dim,
                mode=self.bitnet_scale,
                group_size=self.bitnet_group_size,
                group_dim=self.bitnet_group_dim,
                clip_alpha=self.bitnet_clip,
            )
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


class LLaMaAttention(nn.Module):
    def __init__(self, head_dim: int, dropout: float = 0.0,
                 max_seq_len: int = 256, use_flash: bool = True):
        super().__init__()
        self.head_dim = head_dim
        self.dropout = dropout
        self.flash = use_flash and hasattr(torch.nn.functional, "scaled_dot_product_attention")
        mask = torch.full((1, 1, max_seq_len, max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, q, k, v):
        if self.flash:
            return torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )

        seqlen = q.shape[2]
        scores = torch.matmul(q, k.transpose(2, 3)) / (self.head_dim ** 0.5)
        mask = self.mask[:, :, :seqlen, :seqlen].to(device=scores.device)
        scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(q)
        scores = F.dropout(scores, p=self.dropout, training=self.training)
        return torch.matmul(scores, v)


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
                               score_quant=None, bitnet_scale="max",
                               bitnet_group_size=None, bitnet_group_dim=-1,
                               bitnet_clip=None, fp8_dtype="e4m3fn",
                               int_dim=None, int_dim_q=None, int_dim_k=None,
                               int_dim_v=None, int_dim_score=None,
                               quantize_q=True, quantize_kv=True,
                               quantize_k=True, quantize_v=True,
                               quantize_score=False, quantize_output=False,
                               llama_kv_quant_scope=None,
                               llama_kv_group_size=None):
    for module in model.modules():
        if isinstance(module, AttentionQuantMixin):
            module.set_quantization(
                quant=quant,
                q_quant=q_quant,
                kv_quant=kv_quant,
                k_quant=k_quant,
                v_quant=v_quant,
                score_quant=score_quant,
                bitnet_scale=bitnet_scale,
                bitnet_group_size=bitnet_group_size,
                bitnet_group_dim=bitnet_group_dim,
                bitnet_clip=bitnet_clip,
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
                llama_kv_quant_scope=llama_kv_quant_scope,
                llama_kv_group_size=llama_kv_group_size,
            )
    return model
