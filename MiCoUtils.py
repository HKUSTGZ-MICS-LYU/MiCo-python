from MiCoQLayers import BitLinear, BitConv2d

import copy
import struct
import torch
import torch.nn as nn
import torch.nn.utils.fusion as fusion

from torchao.quantization.quant_api import(
    quantize_,
    Int4DynamicActivationInt4WeightConfig,
    Int8DynamicActivationInt4WeightConfig,
    Int8DynamicActivationInt8WeightConfig,
    Int4WeightOnlyConfig,
    Int8WeightOnlyConfig,
)

def list_quantize_layers(model: nn.Module):
    layers = []
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            layers.append(module)
        else:
            layers += list_quantize_layers(module)
    return layers

def list_qlayers(model: nn.Module):
    layers = []
    for name, module in model.named_children():
        if isinstance(module, BitLinear) or isinstance(module, BitConv2d):
            layers.append(module)
        else:
            layers += list_qlayers(module)
    return layers

def list_qlayers_names(model: nn.Module):
    layers = []
    for name, module in model.named_children():
        if isinstance(module, BitLinear) or isinstance(module, BitConv2d):
            layers.append(name)
        else:
            layers += list_quantize_layers(module)
    return layers

def torchao_quantize(qw, qa):
    if (qw == 8) and (qa == 8):
        return Int8DynamicActivationInt8WeightConfig()
    elif (qw == 4) and (qa == 8):
        return Int8DynamicActivationInt4WeightConfig()
    elif (qw == 4) and (qa == 4):
        return Int4DynamicActivationInt4WeightConfig()
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
        else:
            __replace_layer(module, weight_types_list, act_types_list, 
                                    quant_aware=quant_aware, device=device, group_size=group_size,
                                    use_norm=use_norm, use_bias=use_bias)
    return

def set_to_qforward(model: nn.Module):
    for name, module in model.named_children():
        if isinstance(module, BitLinear) or isinstance(module, BitConv2d):
            module.qforward = True
            module.save_qweight()
        else:
            set_to_qforward(module)
    return

def unset_qforward(model: nn.Module):
    for name, module in model.named_children():
        if isinstance(module, BitLinear) or isinstance(module, BitConv2d):
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
        if isinstance(module, BitLinear) or isinstance(module, BitConv2d):
            size += module.weight.numel() * module.qtype
        else:
            size += get_model_size(module)
    return size

def get_model_macs(model: nn.Module):
    macs = 0
    for name, module in model.named_children():
        if isinstance(module, BitLinear):
            macs += module.get_mac()
        elif isinstance(module, BitConv2d):
            macs += module.get_mac()
        else:
            macs += get_model_macs(module)
    return macs

def export_layer_weights(model: nn.Module):
    data = {}
    for name, module in model.named_children():
        if isinstance(module, BitLinear) or isinstance(module, BitConv2d):
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
    import numpy as np
    
    # For 2D weights, ensure K dimension is aligned
    if len(weight.shape) > 1:
        print(f"Exporting Weight shape: {weight.shape}")
        weight = weight.flatten(start_dim=1)
        M, K = weight.shape
        if K % align_to != 0:
            # Calculate padding needed
            pad_size = align_to - (K % align_to)
            # Use torch.nn.functional.pad for efficient padding
            weight = torch.nn.functional.pad(weight, (0, pad_size), mode='constant', value=0)
            print(f"Padded weight to shape {weight.shape}")
        # Flatten the weight
        weight = weight.flatten()
    
    # Convert to NumPy array for efficient processing
    weight_np = weight.cpu().to(torch.int32).numpy()
    
    # Process based on quantization type
    if qtype == 8:
        return weight_np.astype(np.int8).tobytes()
    elif qtype == 4:
        # Pad to even length if necessary
        if len(weight_np) % 2 == 1:
            weight_np = np.append(weight_np, 0)
        # Reshape to pairs and pack using vectorized operations
        weight_pairs = weight_np.reshape(-1, 2)
        packed = ((weight_pairs[:, 1] & 0xF) << 4) | (weight_pairs[:, 0] & 0xF)
        return packed.astype(np.uint8).tobytes()
    elif (qtype == 2) or (qtype > 1 and qtype < 2):  # Ternary (1.58b) treated as 2b
        # Pad to multiple of 4 if necessary
        rem = len(weight_np) % 4
        if rem != 0:
            weight_np = np.append(weight_np, np.zeros(4 - rem, dtype=weight_np.dtype))
        # Reshape to groups of 4 and pack using vectorized operations
        weight_groups = weight_np.reshape(-1, 4)
        packed = ((weight_groups[:, 0] & 0x3) |
                  ((weight_groups[:, 1] & 0x3) << 2) |
                  ((weight_groups[:, 2] & 0x3) << 4) |
                  ((weight_groups[:, 3] & 0x3) << 6))
        return packed.astype(np.uint8).tobytes()
    elif qtype == 1:
        # Pad to multiple of 8 if necessary
        rem = len(weight_np) % 8
        if rem != 0:
            weight_np = np.append(weight_np, np.zeros(8 - rem, dtype=weight_np.dtype))
        # Reshape to groups of 8 and pack using vectorized operations
        weight_groups = weight_np.reshape(-1, 8)
        # Create sign bits: 1 if negative, 0 otherwise
        sign_bits = (weight_groups < 0).astype(np.uint8)
        # Pack 8 bits into one byte using fully vectorized operations with dot product
        # Bit positions: [1, 2, 4, 8, 16, 32, 64, 128] = [2^0, 2^1, ..., 2^7]
        bit_positions = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
        packed = np.dot(sign_bits, bit_positions).astype(np.uint8)
        return packed.tobytes()
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