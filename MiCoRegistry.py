"""
MiCo Operation Registry

This module implements a registry pattern for PyTorch operations,
allowing extensible code generation without modifying the core MiCoCodeGen class.
"""

import operator
from typing import Any, Dict, List, Tuple, Callable
import torch
import torch.nn
import torch.nn.functional as F

from MiCoQLayers import BitConv2d, BitLinear


class MiCoOpRegistry:
    """
    Registry for PyTorch operation handlers.
    
    This class maintains static dictionaries of function and module handlers,
    allowing users to register custom operation handlers without modifying core code.
    """
    
    # Static dictionaries to store handlers
    _function_handlers: Dict[Callable, Callable] = {}
    _module_handlers: Dict[type, Callable] = {}
    
    @classmethod
    def register_function(cls, torch_fn: Callable) -> Callable:
        """
        Decorator to register a function handler.
        
        Args:
            torch_fn: The PyTorch function to register a handler for
                     (e.g., torch.add, torch.nn.functional.relu)
        
        Returns:
            Decorator function
            
        Example:
            @MiCoOpRegistry.register_function(torch.add)
            def handle_add(codegen, n, out, input_names, input_args):
                codegen.add_uninitialized_tensor(n.name, out)
                codegen.add_forward_call("MiCo_add{dim}d_{dtype}", out, n.name, input_names)
        """
        def decorator(handler_func: Callable) -> Callable:
            cls._function_handlers[torch_fn] = handler_func
            return handler_func
        return decorator
    
    @classmethod
    def register_module(cls, module_type: type) -> Callable:
        """
        Decorator to register a module handler.
        
        Args:
            module_type: The PyTorch module type to register a handler for
                        (e.g., torch.nn.Conv2d, torch.nn.ReLU)
        
        Returns:
            Decorator function
            
        Example:
            @MiCoOpRegistry.register_module(torch.nn.Conv2d)
            def handle_conv2d_module(codegen, n, out, module, input_names):
                # handler implementation
                pass
        """
        def decorator(handler_func: Callable) -> Callable:
            cls._module_handlers[module_type] = handler_func
            return handler_func
        return decorator
    
    @classmethod
    def get_function_handler(cls, torch_fn: Callable) -> Callable:
        """
        Get the registered handler for a PyTorch function.
        
        Args:
            torch_fn: The PyTorch function to get a handler for
        
        Returns:
            The registered handler function, or None if not found
        """
        return cls._function_handlers.get(torch_fn)
    
    @classmethod
    def get_module_handler(cls, module: torch.nn.Module) -> Callable:
        """
        Get the registered handler for a PyTorch module.
        
        Args:
            module: The PyTorch module instance to get a handler for
        
        Returns:
            The registered handler function, or None if not found
        """
        return cls._module_handlers.get(type(module))


# ============================================================================
# Function Handlers
# ============================================================================

@MiCoOpRegistry.register_function(operator.__add__)
@MiCoOpRegistry.register_function(torch.add)
def handle_add(codegen, n, out, input_names, input_args):
    """Handler for addition operations."""
    codegen.add_uninitialized_tensor(n.name, out)
    codegen.add_forward_call("MiCo_add{dim}d_{dtype}", out, n.name, input_names)


@MiCoOpRegistry.register_function(operator.__mul__)
def handle_mul(codegen, n, out, input_names, input_args):
    """Handler for multiplication operations."""
    codegen.add_uninitialized_tensor(n.name, out)
    codegen.add_forward_call("MiCo_mul{dim}d_{dtype}", out, n.name, input_names)


@MiCoOpRegistry.register_function(torch.nn.functional.relu)
def handle_relu(codegen, n, out, input_names, input_args):
    """Handler for ReLU activation function."""
    codegen.add_uninitialized_tensor(n.name, out)
    codegen.add_forward_call("MiCo_relu{dim}d_{dtype}", out, n.name, input_names)


@MiCoOpRegistry.register_function(torch.nn.functional.relu6)
def handle_relu6(codegen, n, out, input_names, input_args):
    """Handler for ReLU6 activation function."""
    codegen.add_uninitialized_tensor(n.name, out)
    codegen.add_forward_call("MiCo_relu6{dim}d_{dtype}", out, n.name, input_names)


@MiCoOpRegistry.register_function(torch.nn.functional.tanh)
def handle_tanh(codegen, n, out, input_names, input_args):
    """Handler for Tanh activation function."""
    codegen.add_uninitialized_tensor(n.name, out)
    codegen.add_forward_call("MiCo_tanh{dim}d_{dtype}", out, n.name, input_names)


@MiCoOpRegistry.register_function(torch.nn.functional.linear)
def handle_linear(codegen, n, out, input_names, input_args):
    """Handler for linear (fully connected) layer function."""
    weight = codegen.model.state_dict()[input_args[1].target]
    bias = codegen.model.state_dict()[input_args[2].target]
    codegen.add_uninitialized_tensor(n.name, out)
    codegen.add_initialized_tensor(f"{input_names[1]}", weight)
    codegen.add_initialized_tensor(f"{input_names[2]}", bias)
    codegen.add_forward_call("MiCo_addmm_{dtype}", out, n.name, input_names)


@MiCoOpRegistry.register_function(torch.nn.functional.avg_pool2d)
def handle_avg_pool2d(codegen, n, out, input_names, input_args):
    """Handler for 2D average pooling function."""
    codegen.add_uninitialized_tensor(n.name, out)
    if isinstance(input_args[1], Tuple):
        kernel_size = input_args[1][0]
    elif isinstance(input_args[1], int):
        kernel_size = input_args[1]
    if len(input_args) > 2:
        stride = input_args[2]
    else:
        stride = 1
    codegen.add_forward_call("MiCo_avgpool{dim}d_{dtype}", out, n.name, input_names, 
                             [kernel_size, stride])


@MiCoOpRegistry.register_function(torch.nn.functional.max_pool2d)
def handle_max_pool2d(codegen, n, out, input_names, input_args):
    """Handler for 2D max pooling function."""
    codegen.add_uninitialized_tensor(n.name, out)
    if isinstance(input_args[1], Tuple):
        kernel_size = input_args[1][0]
    elif isinstance(input_args[1], int):
        kernel_size = input_args[1]
    if len(input_args) > 2:
        stride = input_args[2]
    else:
        stride = 1
    codegen.add_forward_call("MiCo_maxpool{dim}d_{dtype}", out, n.name, input_names,
                             [kernel_size, stride])


@MiCoOpRegistry.register_function(torch.nn.functional.adaptive_avg_pool2d)
def handle_adaptive_avg_pool2d(codegen, n, out, input_names, input_args):
    """Handler for 2D adaptive average pooling function."""
    codegen.add_uninitialized_tensor(n.name, out)
    if isinstance(input_args[1], Tuple):
        output_size = input_args[1][0]
    elif isinstance(input_args[1], int):
        output_size = input_args[1]
    codegen.add_forward_call("MiCo_adaptive_avgpool{dim}d_{dtype}", out, n.name, input_names, [output_size])


@MiCoOpRegistry.register_function(torch.flatten)
def handle_flatten(codegen, n, out, input_names, input_args):
    """Handler for flatten operation."""
    codegen.add_connect_tensor(n.name, out)
    codegen.add_forward_call("MiCo_CONNECT", out, n.name, input_names)


@MiCoOpRegistry.register_function(torch.cat)
def handle_cat(codegen, n, out, input_names, input_args):
    """Handler for concatenation operation."""
    codegen.add_uninitialized_tensor(n.name, out)
    codegen.add_forward_call("MiCo_concat{dim}d_{dtype}", out, n.name, input_names)


# ============================================================================
# Module Handlers
# ============================================================================

@MiCoOpRegistry.register_module(BitConv2d)
def handle_bitconv2d_module(codegen, n, out, module, input_names):
    """Handler for BitConv2d quantized convolution module."""
    layer_name = n.name
    weight = module.weight
    bias = module.bias
    input_names.append(f"{layer_name}_weight")
    input_names.append(f"{layer_name}_bias")

    codegen.add_uninitialized_tensor(layer_name, out)
    codegen.add_initialized_tensor(f"{layer_name}_weight", weight, 
                                    quant=module.qtype, scale=module.qw_scale)
    codegen.add_initialized_tensor(f"{layer_name}_bias", bias)

    codegen.add_forward_call("MiCo_bitconv2d_{dtype}", out, layer_name, input_names, [
        round(module.qtype),
        round(module.act_q),
        module.stride[0],   # assume same stride for both dimensions
        module.padding[0],  # assume same padding for both dimensions
        module.dilation[0], # assume same dilation for both dimensions
        module.groups,
        codegen.align_to
    ])


@MiCoOpRegistry.register_module(BitLinear)
def handle_bitlinear_module(codegen, n, out, module, input_names):
    """Handler for BitLinear quantized linear module."""
    layer_name = n.name
    weight = module.weight
    bias = module.bias
    input_names.append(f"{layer_name}_weight")
    input_names.append(f"{layer_name}_bias")

    codegen.add_uninitialized_tensor(layer_name, out)
    codegen.add_initialized_tensor(f"{layer_name}_weight", weight, 
                                    quant=module.qtype, scale=module.qw_scale)
    codegen.add_initialized_tensor(f"{layer_name}_bias", bias)

    codegen.add_forward_call("MiCo_bitlinear_{dtype}", out, layer_name, input_names, [
        round(module.qtype),
        round(module.act_q),
        codegen.align_to])


@MiCoOpRegistry.register_module(torch.nn.Conv2d)
def handle_conv2d_module(codegen, n, out, module, input_names):
    """Handler for Conv2d module."""
    layer_name = n.name
    weight = module.weight
    bias = module.bias
    input_names.append(f"{layer_name}_weight")
    input_names.append(f"{layer_name}_bias")

    codegen.add_uninitialized_tensor(layer_name, out)
    codegen.add_initialized_tensor(f"{layer_name}_weight", module.weight)
    codegen.add_initialized_tensor(f"{layer_name}_bias", module.bias)
    
    codegen.add_forward_call("MiCo_conv2d_{dtype}", out, layer_name, input_names, [
        module.stride[0],   # assume same stride for both dimensions
        module.padding[0],  # assume same padding for both dimensions
        module.dilation[0], # assume same dilation for both dimensions
        module.groups
    ])


@MiCoOpRegistry.register_module(torch.nn.BatchNorm2d)
def handle_batchnorm2d_module(codegen, n, out, module, input_names):
    """Handler for BatchNorm2d module."""
    layer_name = n.name
    input_names.append(f"{layer_name}_weight")
    input_names.append(f"{layer_name}_bias")
    input_names.append(f"{layer_name}_running_mean")
    input_names.append(f"{layer_name}_running_var")
    codegen.add_uninitialized_tensor(layer_name, out)
    codegen.add_initialized_tensor(f"{layer_name}_weight", module.weight)
    codegen.add_initialized_tensor(f"{layer_name}_bias", module.bias)
    codegen.add_initialized_tensor(f"{layer_name}_running_mean", module.running_mean)
    codegen.add_initialized_tensor(f"{layer_name}_running_var", module.running_var)
    codegen.add_forward_call("MiCo_batchnorm2d_{dtype}", out, layer_name, input_names, [module.eps])


@MiCoOpRegistry.register_module(torch.nn.ELU)
def handle_elu_module(codegen, n, out, module, input_names):
    """Handler for ELU activation module."""
    layer_name = n.name
    codegen.add_uninitialized_tensor(layer_name, out)
    codegen.add_forward_call("MiCo_elu{dim}d_{dtype}", out, layer_name, input_names, [module.alpha])


@MiCoOpRegistry.register_module(torch.nn.ReLU)
def handle_relu_module(codegen, n, out, module, input_names):
    """Handler for ReLU activation module."""
    layer_name = n.name
    codegen.add_uninitialized_tensor(layer_name, out)
    codegen.add_forward_call("MiCo_relu{dim}d_{dtype}", out, layer_name, input_names)


@MiCoOpRegistry.register_module(torch.nn.ReLU6)
def handle_relu6_module(codegen, n, out, module, input_names):
    """Handler for ReLU6 activation module."""
    layer_name = n.name
    codegen.add_uninitialized_tensor(layer_name, out)
    codegen.add_forward_call("MiCo_relu6{dim}d_{dtype}", out, layer_name, input_names)


@MiCoOpRegistry.register_module(torch.nn.Tanh)
def handle_tanh_module(codegen, n, out, module, input_names):
    """Handler for Tanh activation module."""
    layer_name = n.name
    codegen.add_uninitialized_tensor(layer_name, out)
    codegen.add_forward_call("MiCo_tanh{dim}d_{dtype}", out, layer_name, input_names)


@MiCoOpRegistry.register_module(torch.nn.AvgPool2d)
def handle_avgpool2d_module(codegen, n, out, module, input_names):
    """Handler for AvgPool2d module."""
    layer_name = n.name
    codegen.add_uninitialized_tensor(layer_name, out)
    if isinstance(module.kernel_size, Tuple):
        kernel_size = module.kernel_size[0]
    elif isinstance(module.kernel_size, int):
        kernel_size = module.kernel_size
    codegen.add_forward_call("MiCo_avgpool{dim}d_{dtype}", out, layer_name, input_names, 
                             [kernel_size, module.stride, module.padding])


@MiCoOpRegistry.register_module(torch.nn.MaxPool2d)
def handle_maxpool2d_module(codegen, n, out, module, input_names):
    """Handler for MaxPool2d module."""
    layer_name = n.name
    if isinstance(module.kernel_size, Tuple):
        kernel_size = module.kernel_size[0]
    elif isinstance(module.kernel_size, int):
        kernel_size = module.kernel_size
    codegen.add_uninitialized_tensor(layer_name, out)
    codegen.add_forward_call("MiCo_maxpool{dim}d_{dtype}", out, layer_name, input_names, 
                             [kernel_size, module.stride, module.padding])


@MiCoOpRegistry.register_module(torch.nn.AdaptiveAvgPool2d)
def handle_adaptive_avgpool2d_module(codegen, n, out, module, input_names):
    """Handler for AdaptiveAvgPool2d module."""
    layer_name = n.name
    if isinstance(module.output_size, Tuple):
        output_size = module.output_size[0]
    elif isinstance(module.output_size, int):
        output_size = module.output_size
    codegen.add_uninitialized_tensor(layer_name, out)
    codegen.add_forward_call("MiCo_adaptive_avgpool{dim}d_{dtype}", out, layer_name, input_names, [output_size])


@MiCoOpRegistry.register_module(torch.nn.Flatten)
def handle_flatten_module(codegen, n, out, module, input_names):
    """Handler for Flatten module."""
    layer_name = n.name
    codegen.add_connect_tensor(layer_name, out)
    codegen.add_forward_call("MiCo_CONNECT", out, layer_name, input_names)


@MiCoOpRegistry.register_module(torch.nn.Linear)
def handle_linear_module(codegen, n, out, module, input_names):
    """Handler for Linear module."""
    layer_name = n.name
    weight = module.weight
    bias = module.bias
    input_names.append(f"{layer_name}_weight")
    input_names.append(f"{layer_name}_bias")

    codegen.add_uninitialized_tensor(layer_name, out)
    codegen.add_initialized_tensor(f"{layer_name}_weight", weight)
    codegen.add_initialized_tensor(f"{layer_name}_bias", bias)
    codegen.add_forward_call("MiCo_linear_{dtype}", out, layer_name, input_names)


@MiCoOpRegistry.register_module(torch.nn.Identity)
@MiCoOpRegistry.register_module(torch.nn.Dropout)
def handle_identity_module(codegen, n, out, module, input_names):
    """Handler for Identity and Dropout modules (pass-through operations)."""
    layer_name = n.name
    codegen.add_connect_tensor(layer_name, out)
    codegen.add_forward_call("MiCo_CONNECT", out, layer_name, input_names)
