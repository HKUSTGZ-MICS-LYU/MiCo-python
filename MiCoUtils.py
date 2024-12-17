from MiCoQLayers import BitLinear, BitConv2d

import copy
import struct
import torch
import torch.nn as nn
import torch.nn.utils.fusion as fusion

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

# Note: this function will clear the weight_types_list and act_types_list
def replace_quantize_layers(model: nn.Module,
                            weight_types_list: list, 
                            act_types_list: list,
                            quant_aware = False,
                            use_norm = False,
                            use_bias = False,
                            device=None):
    
    wlist = copy.deepcopy(weight_types_list)
    alist = copy.deepcopy(act_types_list)

    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            weight_type = wlist.pop(0)
            act_type = alist.pop(0)
            weights = module.weight
            bias = module.bias
            has_bias = True if (use_bias) and (module.bias is not None) else False
            setattr(
                model, name,
                BitLinear(module.in_features, module.out_features, bias=has_bias,
                          qtype = weight_type, act_q = act_type, 
                          qat = quant_aware, use_norm=use_norm,
                          device=device)
            )
            getattr(model, name).weight = weights
            getattr(model, name).bias = bias

        elif isinstance(module, nn.Conv2d):
            weight_type = wlist.pop(0)
            act_type = alist.pop(0)
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
            replace_quantize_layers(module, weight_types_list, act_types_list, 
                                    quant_aware=quant_aware, device=device, 
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

WEIGHT_SIZE = {
    "1b": 1,
    "1.5b": 1.6, # Consider 5 weights compressed into 8 bits
    "2b": 2,
    "3b": 3,
    "4b": 4,
    "5b": 5,
    "6b": 6,
    "7b": 7,
    "8b": 8
}

def get_model_size(model: nn.Module):
    size = 0
    for name, module in model.named_children():
        if isinstance(module, BitLinear) or isinstance(module, BitConv2d):
            size += module.weight.numel() * WEIGHT_SIZE[module.qtype]
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

def weight_export(weight, qtype):
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
    elif (qtype <=2) and (qtype > 1):
        data = []
        num_loop = len(weight) // 4
        for i in range(num_loop):
            w = (weight[i*4] & 0x3) | \
                ((weight[i*4+1] & 0x3) << 2) | \
                ((weight[i*4+2] & 0x3) << 4) | \
                ((weight[i*4+3] & 0x3) << 6)
            data.append(w)
        rem = len(weight) % 4
        w = 0
        if rem != 0:
            for i in range(rem):
                w |= (weight[-rem+i] & 0x3) << (i*2)
            data.append(w)
    # elif qtype == "1.5b":
        # TODO: 1.58-bit compression, 5 weights in 8-bit
        # pass
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
    else:
        raise NotImplementedError
    return struct.pack(f'{len(data)}B', *data)

# Fuse Conv2D+BatchNorm2D layers
# Note: This function should be called before replacing layers into Q-layers
# Also you need to make sure Conv2D+BatchNorm2D are in Sequential containers
def fuse_model(model: nn.Module):
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
            fuse_model(module)

    return model