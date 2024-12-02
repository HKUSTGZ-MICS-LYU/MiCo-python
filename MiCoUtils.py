from MiCoQLayers import BitLinear, BitConv2d

import torch
import torch.nn as nn

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
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            weight_type = weight_types_list.pop(0)
            act_type = act_types_list.pop(0)
            weights = module.weight
            bias = module.bias if use_bias else None
            setattr(
                model, name,
                BitLinear(module.in_features, module.out_features, bias=use_bias,
                          qtype = weight_type, act_q = act_type, 
                          qat = quant_aware, use_norm=use_norm,
                          device=device)
            )
            getattr(model, name).weight = weights
            getattr(model, name).bias = bias
        elif isinstance(module, nn.Conv2d):
            weight_type = weight_types_list.pop(0)
            act_type = act_types_list.pop(0)
            weights = module.weight
            bias = module.bias if use_bias else None
            setattr(
                model, name,
                BitConv2d(module.in_channels, module.out_channels, module.kernel_size, 
                          module.stride, module.padding, module.dilation, module.groups, 
                          bias = use_bias, qtype = weight_type, act_q = act_type,
                          qat = quant_aware, use_norm=use_norm,
                          device=device)
            )
            getattr(model, name).weight = weights
            getattr(model, name).bias = bias
        else:
            replace_quantize_layers(module, weight_types_list, act_types_list, 
                                    quant_aware=quant_aware, device=device)
    return

# Note: this function will clear the weight_types_list and act_types_list
def renew_quantize_layers(model: nn.Module,
                            weight_types_list: list, 
                            act_types_list: list,
                            quant_aware = False,
                            device=None):
    for name, module in model.named_children():
        if isinstance(module, BitLinear) or isinstance(module, BitConv2d):
            weight_type = weight_types_list.pop(0)
            act_type = act_types_list.pop(0)
            module.qat = quant_aware
            module.qtype = weight_type
            module.act_q = act_type
            module.save_qweight()
        else:
            renew_quantize_layers(module, weight_types_list, act_types_list, 
                                    quant_aware=quant_aware, device=device)     
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