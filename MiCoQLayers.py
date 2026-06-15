import torch
import torch.nn as nn
import torch.nn.utils.fusion as fusion
import torch.nn.functional as F
from MiCoMisc import (
    AttentionQuantMixin,
    AttentionScore,
    LLaMaAttention,
    LinearAttentionScore,
    ATTENTION_QUANT_NONE,
    attention_qtype_to_quant,
    attention_quant_to_bits,
)

DEFAULT_W_Q =8
DEFAULT_ACT_Q = 8

def activation_nquant(x: torch.Tensor, qbit = 8):
    if qbit == 1:
        # TODO: This is the most straightforward binarization
        x_absmean = x.abs().mean(dim=-1, keepdim=True)
        y = x.sign()
        y = y * x_absmean
        # Replace 0.0 with -x_absmean
        y = torch.where(y == 0.0, -x_absmean, y)
    elif qbit < 2:
        x_absmean = x.abs().mean(dim=-1, keepdim=True)
        scale = 1.0 / x_absmean.clamp_(min=1e-5)
        y = (x * scale).round().clamp_(-1, 1) / scale
    elif qbit == 2:
        x_absmean = x.abs().mean(dim=-1, keepdim=True)
        scale = 1.0 / x_absmean.clamp_(min=1e-5)
        y = (x * scale).round().clamp_(-2, 1) / scale
    else:
        x_absmax = x.abs().max(dim=-1, keepdim=True).values
        scale = (2**(qbit-1) - 1) / x_absmax.clamp_(min=1e-5)
        y = (x * scale).round().clamp_(-(2**(qbit-1)), 2**(qbit-1) - 1) / scale
    return y

def activation_nquant_2d(x: torch.Tensor, qbit = 8):
    if qbit == 1:
        x_absmean = torch.mean(x.abs(), dim=(-2,-1), keepdim=True)
        y = x.sign() * x_absmean
        y = torch.where(y == 0.0, -x_absmean, y)
    elif qbit < 2: # Ternary quantization
        x_absmean = torch.mean(x.abs(), dim=(-2,-1), keepdim=True)
        scale = 1.0 / x_absmean.clamp_(min=1e-5)
        y = (x * scale).round().clamp_(-1, 1) / scale
    elif qbit == 2:
        x_absmean = torch.mean(x.abs(), dim=(-2,-1), keepdim=True)
        scale = 1.0 / x_absmean.clamp_(min=1e-5)
        y = (x * scale).round().clamp_(-2, 1) / scale
    else:
        x_absmax = torch.amax(x.abs(), dim=(-2,-1), keepdim=True)
        scale = (2**(qbit-1) - 1) / x_absmax.clamp_(min=1e-5)
        y = (x * scale).round().clamp_(-(2**(qbit-1)), 2**(qbit-1) - 1) / scale
    return y

def activation_pact_quant(x: torch.Tensor, qbit = 8):
    # Parameterized Clipping Activation for Quantized Neural Networks
    # https://arxiv.org/pdf/1805.06085
    # TODO: implement PACT
    pass


def activation_nquant_group(x: torch.Tensor, qbit: int = 8, mode: str = "max",
                            dim: int = -1, group_size: int = 32) -> torch.Tensor:
    """
    Group-wise symmetric activation quantization.
    - dim: dimension to group over
    - group_size: number of contiguous elements per group along 'dim'
    - mode: "max" (per-group max) for qbit > 2, otherwise "mean"
    Returns:
      y: dequantized activation with same shape as x
    """
    assert qbit > 0, "qbit should be positive"
    assert isinstance(group_size, int) and group_size > 0, "group_size must be a positive int"

    ndim = x.dim()
    dim = dim % ndim
    n = x.size(dim)
    assert n % group_size == 0, f"group_size {group_size} must divide size along dim {dim} ({n})"

    # Reshape to expose [num_groups, group_size] along 'dim'
    num_groups = n // group_size
    new_shape = list(x.shape)
    new_shape[dim] = num_groups
    new_shape.insert(dim + 1, group_size)  # [.., num_groups, group_size, ..]
    g = x.reshape(new_shape)
    reduce_dim = dim + 1  # the 'group_size' axis

    if qbit == 1:
        m = g.abs().mean(dim=reduce_dim, keepdim=True).clamp(min=1e-5)
        yg = g.sign() * m
        yg = torch.where(yg == 0.0, -m, yg)
    elif qbit < 2:
        m = g.abs().mean(dim=reduce_dim, keepdim=True).clamp(min=1e-5)
        scale = 1.0 / m
        yg = (g * scale).round().clamp_(-1, 1) / scale
    elif qbit == 2:
        m = g.abs().mean(dim=reduce_dim, keepdim=True).clamp(min=1e-5)
        scale = 1.0 / m
        yg = (g * scale).round().clamp_(-2, 1) / scale
    else:
        if (mode == "max"):
            denom = g.abs().amax(dim=reduce_dim, keepdim=True)
        elif mode == "mean":
            denom = g.abs().mean(dim=reduce_dim, keepdim=True)
        else:
            raise ValueError("Invalid mode")
        scale = (2**(qbit - 1) - 1) / denom.clamp(min=1e-5)
        yg = (g * scale).round().clamp_(-(2**(qbit - 1)), 2**(qbit - 1) - 1) / scale

    y = yg.reshape_as(x)
    return y

def weight_quant1b(w: torch.Tensor):
    # 1-bit quantization
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = w.sign()
    return u, 1/scale

def weight_quant158b(w: torch.Tensor):
    # 1.5-bit quantization
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1)
    return u, 1/scale


def weight_quantnb(w: torch.Tensor, qbit = 8, mode = "max"):
    assert qbit > 1, "qbit should be larger than 1"
    if (mode == "max") and (qbit > 2):
        scale = (2**(qbit-1) - 1) / w.abs().max().clamp_(min=1e-5)
    elif mode == "mean" or (qbit <= 2):
        scale = (2**(qbit-1) - 1) / w.abs().mean().clamp_(min=1e-5)
    else:
        raise ValueError("Invalid mode")
    u = (w * scale).round().clamp_(-(2**(qbit-1)), 2**(qbit-1) - 1)
    return u, 1/scale

def weight_quantnb_group(w: torch.Tensor, qbit: int = 8, mode: str = "max",
                         dim: int = -1, group_size: int = 32, return_expanded: bool = True):
    """
    Group-wise weight quantization for qbit >= 1.
    - dim: dimension to group over
    - group_size: number of contiguous elements per group along 'dim'
    - mode: "max" (per-group max) for qbit > 2, otherwise "mean"
    - return_expanded: if True, inv_scale is expanded to w.shape; else it has one fewer elements along 'dim'
    Returns:
      u: integer-quantized weights (same shape as w)
      inv_scale: inverse scale tensor (broadcastable to w if return_expanded=True)
    """
    assert qbit >= 1, "qbit should be larger than or equal to 1"
    assert isinstance(group_size, int) and group_size > 0, "group_size must be a positive int"

    # Normalize dim to positive index
    ndim = w.dim()
    dim = dim % ndim
    n = w.size(dim)
    assert n % group_size == 0, f"group_size {group_size} must divide size along dim {dim} ({n})"

    # Reshape to expose [num_groups, group_size] along 'dim'
    num_groups = n // group_size
    new_shape = list(w.shape)
    new_shape[dim] = num_groups
    new_shape.insert(dim + 1, group_size)  # [.., num_groups, group_size, ..]
    x = w.reshape(new_shape)

    reduce_dim = dim + 1  # the 'group_size' axis
    if qbit == 1:
        denom = x.abs().mean(dim=reduce_dim, keepdim=True)
        scale = 1.0 / denom.clamp(min=1e-5)
        u_group = x.sign()
    elif 1 < qbit < 2:
        denom = x.abs().mean(dim=reduce_dim, keepdim=True)
        scale = 1.0 / denom.clamp(min=1e-5)
        u_group = (x * scale).round().clamp_(-1, 1)
    elif (mode == "max") and (qbit > 2):
        denom = x.abs().amax(dim=reduce_dim, keepdim=True)
        scale = (2**(qbit - 1) - 1) / denom.clamp(min=1e-5)
        u_group = (x * scale).round().clamp_(-(2**(qbit - 1)), 2**(qbit - 1) - 1)
    elif (mode == "mean") or (qbit <= 2):
        denom = x.abs().mean(dim=reduce_dim, keepdim=True)
        scale = (2**(qbit - 1) - 1) / denom.clamp(min=1e-5)
        u_group = (x * scale).round().clamp_(-(2**(qbit - 1)), 2**(qbit - 1) - 1)
    else:
        raise ValueError("Invalid mode")

    u = u_group.reshape_as(w)

    inv_scale_group = 1.0 / scale
    if return_expanded:
        inv_scale = inv_scale_group.expand_as(x).reshape_as(w)
    else:
        # Keep one scale per group: shape same as w but with 'group_size' axis collapsed
        # i.e., shape replaces size 'n' at 'dim' with 'num_groups'
        squeeze_shape = list(new_shape)
        # collapse group_size axis
        squeeze_shape.pop(reduce_dim)
        inv_scale = inv_scale_group.reshape(squeeze_shape)

    return u, inv_scale

def weight_quant(w: torch.Tensor, qtype, mode = "max"):
    if qtype == 1:
        u,s = weight_quant1b(w)
    elif (qtype >= 1.5) and (qtype < 2):
        u,s = weight_quant158b(w)
    elif qtype >= 2:
        u,s = weight_quantnb(w, int(qtype), mode=mode)
    else:
        raise ValueError("Invalid quantization type")
    return u,s

class SimpleRMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**-0.5

    def forward(self, x):
        """Forward method of SimpleRMSNorm"""
        return F.normalize(x, dim=-1) * self.scale

class HadamardTransform(nn.Module):
    # https://github.com/Dao-AILab/fast-hadamard-transform
    def __init__(self, *args, **kwargs):
        self.func = None
        try:
            from fast_hadamard_transform import hadamard_transform
            self.func = hadamard_transform
        except:
            assert "Please install fast hardamard transform from https://github.com/Dao-AILab/fast-hadamard-transform"
        super().__init__(*args, **kwargs)
    def forward(self, x, scale=1.0):
        return self.func(x, scale)

class BitQLayer:
    def __init__(self, qtype=DEFAULT_W_Q, act_q=DEFAULT_ACT_Q, qat=False, 
                 use_norm=False, hadamard=False, group_size=1):
        super().__init__()
        self.qtype = qtype
        self.act_q = act_q
        self.qat = qat
        self.use_norm = use_norm
        self.haramard = hadamard
        self.group_size = group_size
        
        self.qforward = False
        self.qw = None
        self.qw_scale = None
        self.macs = None
        self.act_l2 = None

    def get_bops(self):
        return self.get_mac() * self.qtype * self.act_q
    
    def get_mac(self):
        if self.macs is None:
            return 0
        return self.macs
    
    def get_params(self):
        return self.weight.numel()
    
    def _weight_quant_impl(self, w):
        if self.group_size > 1:
            return weight_quantnb_group(w, self.qtype, dim=1, group_size=self.group_size)
        else:
            return weight_quant(w, self.qtype)

    def weight_quant(self, w: torch.Tensor):
        u, s = self._weight_quant_impl(w)
        return u * s
    
    def save_qweight(self):
        self.qw, self.qw_scale = self._weight_quant_impl(self.weight.data)

    def qweight(self):
        if self.qw is None or self.qw_scale is None:
            self.save_qweight()
        if self.qw.device != self.weight.device:
            self.qw = self.qw.to(self.weight.device)
        if self.qw_scale.device != self.weight.device:
            self.qw_scale = self.qw_scale.to(self.weight.device)
        return self.qw * self.qw_scale

    def ste_weight_quant(self):
        w = self.weight
        return w + (self.weight_quant(w) - w).detach()
    
    def export_qweight(self):
        return {
            "QType": self.qtype, 
            "ActQType": self.act_q,
            "Scale": self.qw_scale.item(), 
            "Weight": self.qw.cpu().tolist()
        }

class BitLinear(nn.Linear, BitQLayer):

    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 device=None, dtype=None,
                 qat = False,
                 use_norm = False,
                 hadamard = False,
                 group_size = 1,
                 qtype = DEFAULT_W_Q,
                 act_q = DEFAULT_ACT_Q) -> None:
        
        nn.Linear.__init__(self, in_features, out_features, bias, device, dtype)
        BitQLayer.__init__(self, qtype, act_q, qat, use_norm, hadamard, group_size)

        self.in_w = in_features
        self.in_h = 1

        self.layer_features = [in_features, out_features] # M, K
        self.layer_type = 'Linear'

    def get_mac(self):
        return self.in_features * self.out_features
    
    def act_quant(self, x: torch.Tensor):
        if self.group_size > 1:
            return activation_nquant_group(x, self.act_q, dim=-1, group_size=self.group_size)
        else:
            return activation_nquant(x, self.act_q)

    def export_qweight(self):
        res = super().export_qweight()
        res["LayerType"] = "Linear"
        return res
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        if self.qat:
            # Forward with Quantization Aware Training (QAT)
            # Using Straight-Through-Estimator (STE)
            x_norm = SimpleRMSNorm(self.in_features)(x) if self.use_norm else x
            x_norm = HadamardTransform()(x) if self.haramard else x_norm
            x_quant = x_norm + (self.act_quant(x_norm) - x_norm).detach()
            w_quant = self.ste_weight_quant()
            y = F.linear(x_quant, w_quant, bias=self.bias)
            return y
        elif self.qforward is True:
            # Forward with Post Training Quantization (PTQ)
            # Only for inference
            x_norm = SimpleRMSNorm(self.in_features)(x) if self.use_norm else x
            x_norm = HadamardTransform()(x) if self.haramard else x_norm
            qx = self.act_quant(x_norm)
            y = F.linear(qx, self.qweight(), bias=self.bias)
            return y
        else:
            return F.linear(x, w, bias=self.bias)


class BitConv2d(nn.Conv2d, BitQLayer):

    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size, 
                 stride = 1, 
                 padding = 0, 
                 dilation = 1, 
                 groups= 1, bias = True, 
                 padding_mode: str = 'zeros', 
                 device=None, dtype=None,
                 qat = False,
                 use_norm = False,
                 qtype = DEFAULT_W_Q,
                 act_q = DEFAULT_ACT_Q) -> None:
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, 
                         padding, dilation, groups, bias, padding_mode, 
                         device, dtype)
        BitQLayer.__init__(self, qtype, act_q, qat, use_norm)
        self.stride = stride
        self.padding = padding

        self.layer_type = 'Conv2D'
        self.in_w = None
        self.in_h = None
    
    def export_qweight(self):
        res = super().export_qweight()
        res.update({
            "LayerType": "Conv2d",
            "Stride": self.stride,
            "Padding": self.padding
        })
        return res

    def rmsnorm(self, x: torch.Tensor, eps = 1e-5):
        y = torch.sqrt(torch.mean(x**2, dim=(-2,-1), keepdim=True))            
        z = x / (y+eps)
        return z

    def forward(self, x: torch.Tensor):
        w = self.weight
        # Get MAC counts
        if self.macs is None:
            inc, inw, inh = self.in_channels, x.shape[2], x.shape[3]
            self.in_w = inw
            self.in_h = inh
            outc = self.out_channels
            outw = (inw - self.kernel_size[0] + 2*self.padding[0]) / self.stride[0] + 1
            outh = (inh - self.kernel_size[1] + 2*self.padding[1]) / self.stride[1] + 1
            self.macs = (self.kernel_size[0]*self.kernel_size[1]) * inc / self.groups * outh * outw * outc
            self.layer_features = [inh, inw, inc / self.groups, outc, self.kernel_size[0], self.stride[0]]
            # self.layer_features = (
            #     inc, inw, outc, outw, self.kernel_size[0])
            # self.layer_features = (self.kernel_size[0]*self.kernel_size[1]*inc, outh * outw, outc)
        if self.qat:
            # Forward with Quantization Aware Training (QAT)
            # Using Straight-Through-Estimator (STE) 
            x_norm = self.rmsnorm(x) if self.use_norm else x
            x_quant = x_norm + (activation_nquant_2d(x_norm, self.act_q) - x_norm).detach()
            w_quant = self.ste_weight_quant()
            y = F.conv2d(x_quant, w_quant, self.bias, self.stride, self.padding, 
                        self.dilation, self.groups)
            return y
        elif self.qforward:
            # Forward with Post Training Quantization (PTQ)
            # Only for inference
            x_norm = self.rmsnorm(x) if self.use_norm else x
            qx = activation_nquant_2d(x_norm, self.act_q)
            y = F.conv2d(qx, self.qweight(), self.bias, self.stride, self.padding, 
                        self.dilation, self.groups)
            return y
        else:
            return F.conv2d(x, w, self.bias, self.stride, self.padding, 
                            self.dilation, self.groups)


class BitConv1d(nn.Conv1d, BitQLayer):

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, padding_mode: str = 'zeros',
                 device=None, dtype=None,
                 qat=False, use_norm=False,
                 qtype=DEFAULT_W_Q, act_q=DEFAULT_ACT_Q) -> None:
        nn.Conv1d.__init__(self, in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode,
                         device, dtype)
        BitQLayer.__init__(self, qtype, act_q, qat, use_norm)
        self.stride = stride if isinstance(stride, tuple) else (stride,)
        self.padding = padding if isinstance(padding, tuple) else (padding,)

        self.layer_type = 'Conv1D'
        self.in_l = None

    def export_qweight(self):
        res = super().export_qweight()
        res.update({
            "LayerType": "Conv1d",
            "Stride": self.stride,
            "Padding": self.padding
        })
        return res

    def rmsnorm(self, x: torch.Tensor, eps=1e-5):
        y = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True))
        return x / (y + eps)

    def forward(self, x: torch.Tensor):
        w = self.weight
        if self.macs is None:
            inl = x.shape[2]
            self.in_l = inl
            outc = self.out_channels
            k = self.kernel_size[0]
            stride = self.stride[0]
            pad = self.padding[0]
            outl = (inl - k + 2 * pad) / stride + 1
            self.macs = k * self.in_channels * outl * outc
            self.layer_features = [inl, self.in_channels, outc, k]
        if self.qat:
            x_norm = self.rmsnorm(x) if self.use_norm else x
            x_quant = x_norm + (activation_nquant(x_norm, self.act_q) - x_norm).detach()
            w_quant = self.ste_weight_quant()
            return F.conv1d(x_quant, w_quant, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        elif self.qforward is True:
            x_norm = self.rmsnorm(x) if self.use_norm else x
            qx = activation_nquant(x_norm, self.act_q)
            return F.conv1d(qx, self.qweight(), self.bias,
                            self.stride, self.padding, self.dilation, self.groups)
        else:
            return F.conv1d(x, w, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)


class _BitAttentionBase(BitQLayer, AttentionQuantMixin):
    layer_type = "AttentionScore"

    def _init_bit_attention(
        self,
        q_qtype=DEFAULT_ACT_Q,
        k_qtype=DEFAULT_W_Q,
        v_qtype=DEFAULT_W_Q,
        score_qtype=DEFAULT_ACT_Q,
        qat=False,
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
        quantize_k=True,
        quantize_v=True,
        quantize_score=True,
        llama_kv_quant_scope="all",
        llama_kv_group_size=32,
    ):
        # Keep BitQLayer's legacy qtype/act_q fields meaningful enough for
        # generic reporting: qtype tracks K/V-side precision, act_q tracks Q.
        BitQLayer.__init__(self, qtype=k_qtype, act_q=q_qtype, qat=qat)
        self.q_qtype = q_qtype
        self.k_qtype = k_qtype
        self.v_qtype = v_qtype
        self.score_qtype = score_qtype
        self.score_macs = 0
        self.context_macs = 0
        self._init_attention_quant(
            quant=ATTENTION_QUANT_NONE,
            q_quant=attention_qtype_to_quant(q_qtype),
            k_quant=attention_qtype_to_quant(k_qtype),
            v_quant=attention_qtype_to_quant(v_qtype),
            score_quant=attention_qtype_to_quant(score_qtype),
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
            quantize_kv=True,
            quantize_k=quantize_k,
            quantize_v=quantize_v,
            quantize_score=quantize_score,
            quantize_output=False,
            llama_kv_quant_scope=llama_kv_quant_scope,
            llama_kv_group_size=llama_kv_group_size,
        )

    def set_attn_qscheme(self, q_qtype=None, k_qtype=None, v_qtype=None, score_qtype=None):
        if q_qtype is not None:
            self.q_qtype = q_qtype
            self.act_q = q_qtype
            self.q_attention_quant = attention_qtype_to_quant(q_qtype)
        if k_qtype is not None:
            self.k_qtype = k_qtype
            self.qtype = k_qtype
            self.k_attention_quant = attention_qtype_to_quant(k_qtype)
            self.kv_attention_quant = self.k_attention_quant
        if v_qtype is not None:
            self.v_qtype = v_qtype
            self.v_attention_quant = attention_qtype_to_quant(v_qtype)
        if score_qtype is not None:
            self.score_qtype = score_qtype
            self.score_attention_quant = attention_qtype_to_quant(score_qtype)

    def get_params(self):
        return 0

    def save_qweight(self):
        self.qw = None
        self.qw_scale = None

    def export_qweight(self):
        return {
            "LayerType": self.layer_type,
            "QType": self.q_qtype,
            "KType": self.k_qtype,
            "VType": self.v_qtype,
            "ScoreType": self.score_qtype,
        }

    def get_mac(self):
        return self.score_macs + self.context_macs

    def get_bops(self):
        k_bits = attention_quant_to_bits(self.k_attention_quant)
        v_bits = attention_quant_to_bits(self.v_attention_quant)
        if self.layer_type == "LLaMaAttention" and self.llama_kv_quant_scope == "current_group":
            group_size = max(int(self.llama_kv_group_size), 1)
            seq_len = self.layer_features[2] if len(self.layer_features) > 2 else 0
            if seq_len > 0:
                quant_pairs = sum((idx // group_size) * group_size + 1 for idx in range(seq_len))
                total_pairs = seq_len * (seq_len + 1) / 2
                quant_ratio = min(max(quant_pairs / max(total_pairs, 1), 0.0), 1.0)
                k_bits = quant_ratio * k_bits + (1.0 - quant_ratio) * 32
                v_bits = quant_ratio * v_bits + (1.0 - quant_ratio) * 32
        return (
            self.score_macs * attention_quant_to_bits(self.q_attention_quant) *
            k_bits
            + self.context_macs * attention_quant_to_bits(self.score_attention_quant) *
            v_bits
        )


class BitAttentionScore(AttentionScore, _BitAttentionBase):
    def __init__(self, scale: float,
                 q_qtype=DEFAULT_ACT_Q,
                 k_qtype=DEFAULT_W_Q,
                 v_qtype=DEFAULT_W_Q,
                 score_qtype=DEFAULT_ACT_Q,
                 qat=False,
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
                 quantize_k=True,
                 quantize_v=True,
                 quantize_score=True,
                 llama_kv_quant_scope="all",
                 llama_kv_group_size=32):
        AttentionScore.__init__(self, scale)
        self.layer_type = "AttentionScore"
        self._init_bit_attention(
            q_qtype=q_qtype,
            k_qtype=k_qtype,
            v_qtype=v_qtype,
            score_qtype=score_qtype,
            qat=qat,
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
            quantize_k=quantize_k,
            quantize_v=quantize_v,
            quantize_score=quantize_score,
            llama_kv_quant_scope=llama_kv_quant_scope,
            llama_kv_group_size=llama_kv_group_size,
        )

    def forward(self, q, k, v):
        B, H, I, Fdim = q.shape
        J = k.shape[2]
        self.score_macs = B * H * I * J * Fdim
        self.context_macs = B * H * I * J * Fdim
        self.macs = self.score_macs + self.context_macs
        self.layer_features = [B, H, I, J, Fdim]

        q = self._quantize_q(q)
        k = self._quantize_k(k)
        v = self._quantize_v(v)
        score = torch.einsum("bhif, bhjf->bhij", q, k) / self.scale
        score = F.softmax(score, dim=-1)
        score = self._quantize_score(score)
        return torch.einsum("bhij, bhjf->bihf", score, v)


class BitLinearAttentionScore(LinearAttentionScore, _BitAttentionBase):
    def __init__(self, eps=1e-6,
                 q_qtype=DEFAULT_ACT_Q,
                 k_qtype=DEFAULT_W_Q,
                 v_qtype=DEFAULT_W_Q,
                 score_qtype=DEFAULT_ACT_Q,
                 qat=False,
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
                 quantize_k=True,
                 quantize_v=True,
                 quantize_score=True,
                 llama_kv_quant_scope="all",
                 llama_kv_group_size=32):
        LinearAttentionScore.__init__(self, eps)
        self.layer_type = "LinearAttentionScore"
        self._init_bit_attention(
            q_qtype=q_qtype,
            k_qtype=k_qtype,
            v_qtype=v_qtype,
            score_qtype=score_qtype,
            qat=qat,
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
            quantize_k=quantize_k,
            quantize_v=quantize_v,
            quantize_score=quantize_score,
            llama_kv_quant_scope=llama_kv_quant_scope,
            llama_kv_group_size=llama_kv_group_size,
        )

    def forward(self, q, k, v):
        B, H, N, D = q.shape
        M = v.shape[-1]
        self.score_macs = B * H * N * D * M
        self.context_macs = B * H * N * D * M
        self.macs = self.score_macs + self.context_macs
        self.layer_features = [B, H, N, D, M]

        q = F.elu(q) + 1
        k = F.elu(k) + 1
        q = self._quantize_q(q)
        k = self._quantize_k(k)
        v = self._quantize_v(v)

        context = torch.einsum("bhnd,bhnm->bhdm", k, v)
        context = self._quantize_score(context)
        k_sum = k.sum(dim=2)
        num = torch.einsum("bhnd,bhdm->bnhm", q, context)
        den = torch.einsum("bhnd,bhd->bnh", q, k_sum).unsqueeze(-1)
        return num / (den + self.eps)


class BitLLaMaAttention(LLaMaAttention, _BitAttentionBase):
    def __init__(self, head_dim: int, dropout: float = 0.0,
                 max_seq_len: int = 256,
                 q_qtype=DEFAULT_ACT_Q,
                 k_qtype=DEFAULT_W_Q,
                 v_qtype=DEFAULT_W_Q,
                 score_qtype=DEFAULT_ACT_Q,
                 qat=False,
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
                 quantize_k=True,
                 quantize_v=True,
                 quantize_score=True,
                 llama_kv_quant_scope="all",
                 llama_kv_group_size=32):
        LLaMaAttention.__init__(
            self,
            head_dim=head_dim,
            dropout=dropout,
            max_seq_len=max_seq_len,
            use_flash=False,
        )
        self.layer_type = "LLaMaAttention"
        self._init_bit_attention(
            q_qtype=q_qtype,
            k_qtype=k_qtype,
            v_qtype=v_qtype,
            score_qtype=score_qtype,
            qat=qat,
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
            quantize_k=quantize_k,
            quantize_v=quantize_v,
            quantize_score=quantize_score,
            llama_kv_quant_scope=llama_kv_quant_scope,
            llama_kv_group_size=llama_kv_group_size,
        )

    def forward(self, q, k, v):
        B, H, I, Fdim = q.shape
        J = k.shape[2]
        self.score_macs = B * H * I * J * Fdim
        self.context_macs = B * H * I * J * Fdim
        self.macs = self.score_macs + self.context_macs
        self.layer_features = [B, H, I, J, Fdim]

        q = self._quantize_q(q)
        if self.llama_kv_quant_scope == "current_group":
            scores, v_for_context = self._forward_current_group_kv(q, k, v, I, J)
        else:
            k = self._quantize_k(k)
            v_for_context = self._quantize_v(v)
            scores = torch.matmul(q, k.transpose(2, 3))
        scores = scores / (self.head_dim ** 0.5)
        mask = self.mask[:, :, :I, :J].to(device=scores.device)
        scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(q)
        scores = self._quantize_score(scores)
        scores = F.dropout(scores, p=self.dropout, training=self.training)
        if isinstance(v_for_context, torch.Tensor):
            return torch.matmul(scores, v_for_context)
        return self._matmul_current_group_v(scores, v, v_for_context, I, J)

    def _query_positions(self, I, J, device):
        start = max(J - I, 0)
        return torch.arange(start, start + I, device=device)

    def _group_prefix_lengths(self, I, J, device):
        group_size = max(int(self.llama_kv_group_size), 1)
        q_pos = self._query_positions(I, J, device)
        return (q_pos // group_size) * group_size

    def _forward_current_group_kv(self, q, k, v, I, J):
        prefix_lengths = self._group_prefix_lengths(I, J, q.device).clamp(max=J)
        max_prefix = int(prefix_lengths.max().item()) if prefix_lengths.numel() > 0 else 0

        if max_prefix <= 0:
            scores = torch.matmul(q, k.transpose(2, 3))
            return scores, None

        k_mixed = k
        if self.quantize_k and self.k_attention_quant != ATTENTION_QUANT_NONE:
            k_quant = self._quantize_k(k[:, :, :max_prefix, :])
            k_mixed = torch.cat([k_quant, k[:, :, max_prefix:, :]], dim=2)
        scores = torch.matmul(q, k_mixed.transpose(2, 3))

        if self.quantize_k and self.k_attention_quant != ATTENTION_QUANT_NONE:
            key_pos = torch.arange(J, device=q.device).view(1, 1, 1, J)
            row_prefix = prefix_lengths.view(1, 1, I, 1)
            fp_mask = (key_pos >= row_prefix) & (key_pos < max_prefix)
            if fp_mask.any():
                fp_scores = torch.matmul(q, k[:, :, :max_prefix, :].transpose(2, 3))
                scores[:, :, :, :max_prefix] = torch.where(
                    fp_mask[:, :, :, :max_prefix],
                    fp_scores,
                    scores[:, :, :, :max_prefix],
                )

        if not self.quantize_v or self.v_attention_quant == ATTENTION_QUANT_NONE:
            return scores, v
        return scores, (self._quantize_v(v[:, :, :max_prefix, :]), prefix_lengths, max_prefix)

    def _matmul_current_group_v(self, scores, v, v_context, I, J):
        v_quant_prefix, prefix_lengths, max_prefix = v_context
        output = torch.matmul(scores, v)
        if max_prefix <= 0:
            return output

        quant_output = torch.matmul(scores[:, :, :, :max_prefix], v_quant_prefix)
        fp_prefix_output = torch.matmul(scores[:, :, :, :max_prefix], v[:, :, :max_prefix, :])
        output = output + quant_output - fp_prefix_output

        key_pos = torch.arange(max_prefix, device=scores.device).view(1, 1, 1, max_prefix)
        row_prefix = prefix_lengths.view(1, 1, I, 1)
        fp_mask = key_pos >= row_prefix
        if fp_mask.any():
            fp_score = scores[:, :, :, :max_prefix] * fp_mask.to(dtype=scores.dtype)
            quant_fp_group_output = torch.matmul(fp_score, v_quant_prefix)
            true_fp_group_output = torch.matmul(fp_score, v[:, :, :max_prefix, :])
            output = output + true_fp_group_output - quant_fp_group_output
        return output
