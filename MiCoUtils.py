from MiCoQLayers import (
    BitLinear,
    BitConv2d,
    BitConv1d,
    BitQLayer,
    BitAttentionScore,
    BitLinearAttentionScore,
    BitLLaMaAttention,
)
from MiCoMisc import AttentionScore, LLaMaAttention, LinearAttentionScore

import copy
import struct
import torch
import torch.nn as nn
import torch.nn.utils.fusion as fusion

from torchao.quantization.quant_api import(
    quantize_,
    Int8DynamicActivationIntxWeightConfig,
    Int8DynamicActivationInt8WeightConfig,
    Int4WeightOnlyConfig,
    Int8WeightOnlyConfig,
)

def list_quantize_layers(model: nn.Module):
    layers = []
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) \
            or isinstance(module, nn.Conv2d) \
            or isinstance(module, nn.Conv1d):
            layers.append(module)
        else:
            layers += list_quantize_layers(module)
    return layers

def list_quantize_attn_layers(model: nn.Module):
    layers = []
    for name, module in model.named_children():
        if isinstance(module, (BitAttentionScore, BitLinearAttentionScore, BitLLaMaAttention)):
            layers.append(module)
        elif isinstance(module, (AttentionScore, LinearAttentionScore, LLaMaAttention)):
            layers.append(module)
        else:
            layers += list_quantize_attn_layers(module)
    return layers

def list_qlayers(model: nn.Module):
    layers = []
    for name, module in model.named_children():
        if isinstance(module, BitQLayer):
            layers.append(module)
        else:
            layers += list_qlayers(module)
    return layers

def list_qlayers_names(model: nn.Module):
    layers = []
    for name, module in model.named_children():
        if isinstance(module, BitQLayer):
            layers.append(name)
        else:
            layers += list_quantize_layers(module)
    return layers

def torchao_quantize(qw, qa):
    if (qw == 8) and (qa == 8):
        return Int8DynamicActivationInt8WeightConfig()
    elif (qw == 4) and (qa == 8):
        return Int8DynamicActivationIntxWeightConfig(weight_dtype=torch.int4)
    elif (qw == 4) and (qa == 4):
        raise NotImplementedError("INT4 dynamic activations are not supported in torchao 0.11.0+")
    elif (qw == 8) and (qa == 4):
        raise NotImplementedError
    elif (qw == 8) and (qa == 32):
        return Int8WeightOnlyConfig()
    elif (qw == 4) and (qa == 32):
        return Int4WeightOnlyConfig()
    else:
        raise NotImplementedError

# Note: Torch AO only quantizes linear layers
def replace_quantize_layers_torchao(model: nn.Module,
                            weight_types_list: list, 
                            act_types_list: list,
                            device=None):
    
    wlist = copy.deepcopy(weight_types_list)
    alist = copy.deepcopy(act_types_list)

    __replace_layer_torchao(model,
                            wlist,
                            alist,
                            device)


def __replace_layer_torchao(model: nn.Module,
                            weight_types_list: list, 
                            act_types_list: list,
                            device):

    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            weight_type = weight_types_list.pop(0)
            act_type = act_types_list.pop(0)
            quantize_(module, torchao_quantize(weight_type, act_type), 
                      device=device)
        else:
            __replace_layer_torchao(
                module, weight_types_list, act_types_list, device)


def replace_quantize_layers(model: nn.Module,
                            weight_types_list: list, 
                            act_types_list: list,
                            quant_aware = False,
                            group_size = 1,
                            use_norm = False,
                            use_bias = False,
                            device=None):
    
    wlist = copy.deepcopy(weight_types_list)
    alist = copy.deepcopy(act_types_list)

    __replace_layer(model,
                    wlist, 
                    alist,
                    quant_aware,
                    group_size,
                    use_norm,
                    use_bias,
                    device)
    
    return

def _expand_attn_qscheme(attn_qscheme, n_layers):
    if attn_qscheme is None:
        raise ValueError("attn_qscheme must be provided")
    if not isinstance(attn_qscheme, dict):
        raise TypeError("attn_qscheme must be a dict with q/k/v/score keys")

    def expand(name, default=None):
        value = attn_qscheme.get(name, default)
        if value is None:
            raise ValueError(f"Missing attention qscheme key: {name}")
        if isinstance(value, (list, tuple)):
            if len(value) != n_layers:
                raise ValueError(
                    f"attention qscheme '{name}' length {len(value)} != {n_layers}"
                )
            return list(value)
        return [value] * n_layers

    return {
        "q": expand("q"),
        "k": expand("k", attn_qscheme.get("kv")),
        "v": expand("v", attn_qscheme.get("kv")),
        "score": expand("score"),
    }


def replace_quantize_attn_layers(model: nn.Module,
                                 attn_qscheme: dict,
                                 quant_aware=False,
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
    n_layers = len(list_quantize_attn_layers(model))
    qscheme = _expand_attn_qscheme(attn_qscheme, n_layers)
    __replace_attn_layer(
        model,
        qscheme["q"],
        qscheme["k"],
        qscheme["v"],
        qscheme["score"],
        quant_aware=quant_aware,
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
    return


def set_attn_qscheme(model: nn.Module, attn_qscheme: dict, **kwargs):
    replace_quantize_attn_layers(model, attn_qscheme, **kwargs)
    return


def __replace_attn_layer(model: nn.Module,
                         q_list: list,
                         k_list: list,
                         v_list: list,
                         score_list: list,
                         quant_aware=False,
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
    def module_device(module):
        for tensor in list(module.parameters(recurse=False)) + list(module.buffers(recurse=False)):
            return tensor.device
        return None

    def maybe_to_source_device(new_module, old_module):
        device = module_device(old_module)
        if device is not None:
            new_module = new_module.to(device)
        return new_module

    for name, module in model.named_children():
        if isinstance(module, BitAttentionScore):
            module.set_attn_qscheme(q_list.pop(0), k_list.pop(0), v_list.pop(0), score_list.pop(0))
            module.set_quantization(
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
        elif isinstance(module, BitLinearAttentionScore):
            module.set_attn_qscheme(q_list.pop(0), k_list.pop(0), v_list.pop(0), score_list.pop(0))
            module.set_quantization(
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
        elif isinstance(module, BitLLaMaAttention):
            module.set_attn_qscheme(q_list.pop(0), k_list.pop(0), v_list.pop(0), score_list.pop(0))
            module.set_quantization(
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
        elif isinstance(module, AttentionScore):
            new_module = BitAttentionScore(
                    module.scale,
                    q_qtype=q_list.pop(0),
                    k_qtype=k_list.pop(0),
                    v_qtype=v_list.pop(0),
                    score_qtype=score_list.pop(0),
                    qat=quant_aware,
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
            setattr(model, name, maybe_to_source_device(new_module, module))
        elif isinstance(module, LinearAttentionScore):
            new_module = BitLinearAttentionScore(
                    module.eps,
                    q_qtype=q_list.pop(0),
                    k_qtype=k_list.pop(0),
                    v_qtype=v_list.pop(0),
                    score_qtype=score_list.pop(0),
                    qat=quant_aware,
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
            setattr(model, name, maybe_to_source_device(new_module, module))
        elif isinstance(module, LLaMaAttention):
            new_module = BitLLaMaAttention(
                    head_dim=module.head_dim,
                    dropout=module.dropout,
                    max_seq_len=module.mask.shape[-1],
                    q_qtype=q_list.pop(0),
                    k_qtype=k_list.pop(0),
                    v_qtype=v_list.pop(0),
                    score_qtype=score_list.pop(0),
                    qat=quant_aware,
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
            setattr(model, name, maybe_to_source_device(new_module, module))
        else:
            __replace_attn_layer(
                module,
                q_list,
                k_list,
                v_list,
                score_list,
                quant_aware=quant_aware,
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
    return

def __replace_layer(model: nn.Module,
            weight_types_list: list, 
            act_types_list: list,
            quant_aware = False,
            group_size = 1,
            use_norm = False,
            use_bias = False,
            device=None):
    
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            weight_type = weight_types_list.pop(0)
            act_type = act_types_list.pop(0)
            weights = module.weight
            bias = module.bias
            has_bias = True if (use_bias) and (module.bias is not None) else False
            setattr(
                model, name,
                BitLinear(module.in_features, module.out_features, bias=has_bias,
                          qtype = weight_type, act_q = act_type, group_size=group_size,
                          qat = quant_aware, use_norm=use_norm,
                          device=device)
            )
            getattr(model, name).weight = weights
            getattr(model, name).bias = bias

        elif isinstance(module, nn.Conv2d):
            weight_type = weight_types_list.pop(0)
            act_type = act_types_list.pop(0)
            weights = module.weight
            bias = module.bias
            has_bias = True if (use_bias) and (module.bias is not None) else False
            setattr(
                model, name,
                BitConv2d(module.in_channels, module.out_channels, module.kernel_size, 
                          module.stride, module.padding, module.dilation, module.groups, 
                          bias = has_bias, qtype = weight_type, act_q = act_type,
                          qat = quant_aware, use_norm=use_norm,
                          device=device)
            )
            getattr(model, name).weight = weights
            getattr(model, name).bias = bias

        elif isinstance(module, nn.Conv1d):
            weight_type = weight_types_list.pop(0)
            act_type = act_types_list.pop(0)
            weights = module.weight
            bias = module.bias
            has_bias = True if (use_bias) and (module.bias is not None) else False
            setattr(
                model, name,
                BitConv1d(module.in_channels, module.out_channels, module.kernel_size, 
                          module.stride, module.padding, module.dilation, module.groups, 
                          bias = has_bias, qtype = weight_type, act_q = act_type,
                          qat = quant_aware, use_norm=use_norm,
                          device=device)
            )
            getattr(model, name).weight = weights
            getattr(model, name).bias = bias
        
        else:
            __replace_layer(module, weight_types_list, act_types_list, 
                                    quant_aware=quant_aware, device=device, group_size=group_size,
                                    use_norm=use_norm, use_bias=use_bias)
    return

def set_to_qforward(model: nn.Module):
    for name, module in model.named_children():
        if isinstance(module, BitQLayer):
            module.qforward = True
            module.save_qweight()
        else:
            set_to_qforward(module)
    return

def unset_qforward(model: nn.Module):
    for name, module in model.named_children():
        if isinstance(module, BitQLayer):
            module.qforward = False
            module.qw = None
            module.qw_scale = None
        else:
            unset_qforward(module)
    return

# WEIGHT_SIZE = {
#     "1b": 1,
#     "1.5b": 1.6, # Consider 5 weights compressed into 8 bits
#     "2b": 2,
#     "3b": 3,
#     "4b": 4,
#     "5b": 5,
#     "6b": 6,
#     "7b": 7,
#     "8b": 8
# }

def get_model_params(model: nn.Module):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    params = sum(p.numel() for p in param_dict.values())
    return params

def get_model_size(model: nn.Module):
    size = 0
    for name, module in model.named_children():
        if isinstance(module, BitQLayer):
            params = module.get_params()
            if params:
                size += params * module.qtype
        else:
            size += get_model_size(module)
    return size

def get_model_macs(model: nn.Module):
    macs = 0
    for name, module in model.named_children():
        if isinstance(module, BitQLayer):
            macs += module.get_mac()
        else:
            macs += get_model_macs(module)
    return macs

def export_layer_weights(model: nn.Module):
    data = {}
    for name, module in model.named_children():
        if isinstance(module, BitQLayer):
            data[name] = module.export_qweight()
        else:
            sub_data = export_layer_weights(module)
            if sub_data != {}:
                data[name] = sub_data
    return data

def weight_export(weight: torch.Tensor, qtype: int, align_to=32):
    """
    Export weights to binary format with K-dimension aligned to multiple of `align_to`
    
    Args:
        weight: Weight array to export
        qtype: Quantization type (1, 2, 4, or 8 bits)
        align_to: Alignment boundary (default 32)
        
    Returns:
        Binary packed data
    """
    # For 2D weights, ensure K dimension is aligned
    if len(weight.shape) > 1:
        print(f"Exporting Weight shape: {weight.shape}")
        weight = weight.flatten(start_dim=1)
        M, K = weight.shape
        if K % align_to != 0:
            # Calculate padding needed
            pad_size = align_to - (K % align_to)
            # Create padded weight array
            padded_weight = torch.zeros((M, K + pad_size), dtype=weight.dtype)
            padded_weight[:, :K] = weight
            print(f"Padding weight from {weight.shape} to {padded_weight.shape}")
            # Flatten for processing
            weight = padded_weight.flatten()
        else:
            # Flatten the already aligned weight
            weight = weight.flatten()
    weight = weight.cpu().to(int).tolist()
    # Process based on quantization type
    if qtype == 8:
        return struct.pack(f'{len(weight)}b', *weight)
    elif qtype == 4:
        data = []
        num_loop = len(weight) // 2
        for i in range(num_loop):
            w = ((weight[i*2+1] & 0xF) << 4) | (weight[i*2] & 0xF)
            data.append(w)
        if len(weight) % 2 == 1:
            w = weight[-1] & 0xF
            data.append(w)
        return struct.pack(f'{len(data)}B', *data)
    elif (qtype == 2) or (qtype > 1 and qtype < 2): # Ternary (1.58b) treated as 2b
        data = []
        num_loop = len(weight) // 4
        for i in range(num_loop):
            w = (weight[i*4] & 0x3) | \
                ((weight[i*4+1] & 0x3) << 2) | \
                ((weight[i*4+2] & 0x3) << 4) | \
                ((weight[i*4+3] & 0x3) << 6)
            data.append(w)
        rem = len(weight) % 4
        if rem != 0:
            w = 0
            for i in range(rem):
                w |= (weight[-rem+i] & 0x3) << (i*2)
            data.append(w)
        return struct.pack(f'{len(data)}B', *data)
    elif qtype == 1:
        data = []
        num_loop = len(weight) // 8
        for i in range(num_loop):
            binary = ""
            for b in range(8):
                binary += "1" if weight[i*8+b] < 0 else "0"
            data.append(int(binary[::-1], 2))
        rem = len(weight) % 8
        if rem != 0:
            binary = ""
            for b in range(rem):
                binary += "1" if weight[-rem+b] < 0 else "0"
            for b in range(8-rem):
                binary += "0"
            data.append(int(binary[::-1], 2))
        return struct.pack(f'{len(data)}B', *data)
    else:
        raise NotImplementedError(f"Quantization type {qtype} not supported")

# Fuse Conv2D+BatchNorm2D layers
# Note: DeepSeek generated, not sure if it fully works.
def fuse_model(model: nn.Module):

    model.eval()
    for name, module in model.named_children():
        fuse_model(module)
        
        children = list(module.named_children())
        new_children = []
        i = 0
        while i < len(children):
            child_name, child_module = children[i]
            if isinstance(child_module, nn.Conv2d):
                if i + 1 < len(children):
                    next_name, next_module = children[i + 1]
                    if isinstance(next_module, nn.BatchNorm2d):
                        fused_conv = fusion.fuse_conv_bn_eval(child_module, next_module)
                        new_children.append((child_name, fused_conv))
                        i += 2
                        continue
            new_children.append((child_name, child_module))
            i += 1
        
        if len(new_children) != len(children):
            for child_name, _ in children:
                delattr(module, child_name)
            for new_name, new_module in new_children:
                module.add_module(new_name, new_module)
    
    return model

# Old version of fuse_model
# Note: This function should be called before replacing layers into Q-layers
# Also you need to make sure Conv2D+BatchNorm2D are in Sequential containers
def fuse_model_seq(model: nn.Module):
    model.eval()
    # Recursively fuse Conv2D + BatchNorm2D layers
    for name, module in model.named_children():
        if isinstance(module, nn.Sequential):
            # Traverse the sequential container
            new_module = nn.Sequential()
            skip_next = False
            for i, sub_module in enumerate(module):
                if skip_next:
                    skip_next = False
                    continue
                if isinstance(sub_module, nn.Conv2d):
                    # Check if the next module is a BatchNorm2d
                    if i + 1 < len(module) and isinstance(module[i + 1], nn.BatchNorm2d):
                        # Fuse the Conv2d and BatchNorm2d layers
                        fused_layer = fusion.fuse_conv_bn_eval(sub_module, 
                                                               module[i + 1])
                        new_module.append(fused_layer)
                        skip_next = True  # Skip the next BatchNorm2d layer
                    else:
                        new_module.append(sub_module)
                else:
                    new_module.append(sub_module)
            setattr(model, name, new_module)
        else:
            # If the module is not a Sequential, recursively fuse its children
            fuse_model_seq(module)

    return model
