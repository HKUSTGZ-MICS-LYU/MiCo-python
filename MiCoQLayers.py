import torch
import torch.nn as nn
import torch.nn.utils.fusion as fusion
import torch.nn.functional as F
import copy

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
    elif qbit < 2:
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

class BitLinear(nn.Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 device=None, dtype=None,
                 qat = False,
                 use_norm = False,
                 qtype = DEFAULT_W_Q,
                 act_q = DEFAULT_ACT_Q) -> None:
        
        self.qtype = qtype
        self.act_q = act_q
        self.qat = qat
        self.use_norm = use_norm
        self.act_l2 = None
        self.in_w = in_features
        self.in_h = 1

        self.layer_features = (1, in_features, out_features) # N, M, K
        self.layer_type = 'Linear'

        self.qforward = False
        super().__init__(in_features, out_features, bias, device, dtype)

    def get_bops(self):
        return self.get_mac() * self.qtype * self.act_q
    
    def get_mac(self):
        return self.in_features * self.out_features
    
    def get_params(self):
        return self.weight.numel()
    
    def weight_quant(self, w: torch.Tensor):
        u, s = weight_quant(w, self.qtype)
        return u*s
    
    def save_qweight(self):
        self.qw, self.qw_scale = weight_quant(self.weight.data, self.qtype)
        return
    
    def export_qweight(self):
        # Export quantized weight in binary format
        return {"LayerType": "Linear",
                "QType": self.qtype, 
                "ActQType": self.act_q,
                "Scale": self.qw_scale.item(), 
                "Weight": self.qw.cpu().tolist()}
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        if self.qat:
            # Forward with Quantization Aware Training (QAT)
            # Using Straight-Through-Estimator (STE) 
            x_norm = SimpleRMSNorm(self.in_features)(x) if self.use_norm else x
            x_quant = x_norm + (activation_nquant(x_norm, self.act_q) - x_norm).detach()
            w_quant = w + (self.weight_quant(w) - w).detach()
            y = F.linear(x_quant, w_quant, bias=self.bias)
            return y
        elif self.qforward is True:
            # Forward with Post Training Quantization (PTQ)
            # Only for inference
            qx = activation_nquant(x, self.act_q)
            y = F.linear(qx, self.qw, bias=self.bias) * self.qw_scale
            return y
        else:
            return F.linear(x, w, bias=self.bias)


class BitConv2d(nn.Conv2d):

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
        super().__init__(in_channels, out_channels, kernel_size, stride, 
                         padding, dilation, groups, bias, padding_mode, 
                         device, dtype)
        self.stride = stride
        self.padding = padding
        self.qat = qat
        self.qtype = qtype
        self.act_q = act_q
        self.use_norm = use_norm

        self.qforward = False
        self.qw = None
        self.qw_scale = None
        self.macs = None
        self.act_l2 = None
        self.layer_type = 'Conv2D'

        self.in_w = None
        self.in_h = None

    def get_bops(self):
        return self.macs * self.qtype * self.act_q

    def get_mac(self):
        return self.macs

    def get_params(self):
        return self.weight.numel()

    def weight_quant(self, w: torch.Tensor):
        u, s = weight_quant(w, self.qtype)
        return u*s
    
    def save_qweight(self):
        self.qw, self.qw_scale = weight_quant(self.weight.data, self.qtype)
        return
    
    def export_qweight(self):
        # Export quantized weight in binary format
        return {"LayerType": "Conv2d",
                "QType": self.qtype, 
                "ActQType": self.act_q,
                "Scale": self.qw_scale.item(),
                "Stride": self.stride,
                "Padding": self.padding,
                "Weight": self.qw.cpu().tolist()}

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
            self.macs = (self.kernel_size[0]*self.kernel_size[1]) * inc * outh * outw * outc
            self.layer_features = (
                inc, inw, outc, outw, self.kernel_size[0])
            # self.layer_features = (self.kernel_size[0]*self.kernel_size[1]*inc, outh * outw, outc)
        if self.qat:
            # Forward with Quantization Aware Training (QAT)
            # Using Straight-Through-Estimator (STE) 
            x_norm = self.rmsnorm(x) if self.use_norm else x
            x_quant = x_norm + (activation_nquant_2d(x_norm, self.act_q) - x_norm).detach()
            w_quant = w + (self.weight_quant(w) - w).detach()
            y = F.conv2d(x_quant, w_quant, self.bias, self.stride, self.padding, 
                        self.dilation, self.groups)
            return y
        elif self.qforward:
            # Forward with Post Training Quantization (PTQ)
            # Only for inference
            qx = activation_nquant_2d(x, self.act_q)
            y = F.conv2d(qx, self.qw * self.qw_scale, self.bias, self.stride, self.padding, 
                        self.dilation, self.groups)
            return y
        else:
            return F.conv2d(x, w, self.bias, self.stride, self.padding, 
                    self.dilation, self.groups)
