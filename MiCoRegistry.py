"""
MiCo Operation Registry

This module implements a registry pattern for PyTorch operations,
allowing extensible code generation without modifying the core MiCoCodeGen class.

Example Usage:
    
    # Register a custom function handler
    from MiCoRegistry import MiCoOpRegistry
    import torch
    
    @MiCoOpRegistry.register_function(torch.sigmoid)
    def handle_sigmoid(codegen, n, out, input_names, input_args):
        '''Handler for sigmoid activation function.'''
        codegen.add_uninitialized_tensor(n.name, out)
        codegen.add_forward_call("MiCo_sigmoid{dim}d_{dtype}", out, n.name, input_names)
    
    # Register a custom module handler
    @MiCoOpRegistry.register_module(MyCustomLayer)
    def handle_my_custom_layer(codegen, n, out, module, input_names):
        '''Handler for MyCustomLayer module.'''
        layer_name = n.name
        codegen.add_uninitialized_tensor(layer_name, out)
        codegen.add_forward_call("MiCo_custom_{dtype}", out, layer_name, input_names,
                                 [module.some_param])
"""

import operator
from typing import Any, Dict, List, Tuple, Callable
import torch
import torch.nn
import torch.nn.functional as F

from MiCoQLayers import BitConv1d, BitConv2d, BitLinear


try:
    from models.KWT import KWTPatchEmbedding
except ImportError:
    KWTPatchEmbedding = None


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
    if not isinstance(out, torch.Tensor):
        codegen.scalar_values[n.name] = out
        return
    codegen.add_uninitialized_tensor(n.name, out)
    codegen.add_forward_call("MiCo_add{dim}d_{dtype}", out, n.name, input_names)


@MiCoOpRegistry.register_function(operator.__mul__)
def handle_mul(codegen, n, out, input_names, input_args):
    """Handler for multiplication operations."""
    if not isinstance(out, torch.Tensor):
        codegen.scalar_values[n.name] = out
        return
    codegen.add_uninitialized_tensor(n.name, out)
    codegen.add_forward_call("MiCo_mul{dim}d_{dtype}", out, n.name, input_names)


@MiCoOpRegistry.register_function(getattr)
def handle_getattr(codegen, n, out, input_names, input_args):
    """Handler for tensor metadata getattr calls such as x.shape."""
    codegen.scalar_values[n.name] = out


@MiCoOpRegistry.register_function(operator.floordiv)
@MiCoOpRegistry.register_function(operator.__floordiv__)
def handle_floordiv(codegen, n, out, input_names, input_args):
    """Handler for scalar floor division used by traced shape expressions."""
    codegen.scalar_values[n.name] = out

@MiCoOpRegistry.register_function(operator.getitem)
def handle_getitem(codegen, n, out, input_names, input_args):
    """Handler for getitem operations (tensor slicing and shape metadata indexing)."""
    if not isinstance(out, torch.Tensor):
        codegen.scalar_values[n.name] = out
        return

    index = input_args[1]
    if out.dim() == 3 and isinstance(index, tuple) and len(index) == 2:
        first, second = index
        is_full_first = isinstance(first, slice) and first.start is None and first.stop is None and first.step is None
        is_prefix_second = (
            isinstance(second, slice)
            and second.start is None
            and second.step is None
        )
        if is_full_first and is_prefix_second:
            src_name = input_names[0]
            src_tensor = codegen.tensors[src_name]["tensor"]
            codegen.add_connect_tensor(n.name, out)
            if tuple(src_tensor.shape) == tuple(out.shape):
                codegen.add_forward_call("MiCo_CONNECT", out, n.name, [src_name])
            else:
                codegen.add_forward_call("MiCo_getitem3d_prefix_{dtype}", out, n.name, [src_name])
            return

    if out.dim() == 2 and isinstance(index, tuple) and len(index) == 2 and isinstance(index[1], int):
        # ViT class-token extraction: x[:, idx, :]
        codegen.add_uninitialized_tensor(n.name, out)
        codegen.add_forward_call("MiCo_getitem3d_to2d_{dtype}", out, n.name, input_names, [index[1]])
        return

    raise NotImplementedError(f"Unsupported tensor getitem pattern: index={index}, out_dim={out.dim()}")


@MiCoOpRegistry.register_function(operator.__truediv__)
@MiCoOpRegistry.register_function(operator.truediv)
def handle_truediv(codegen, n, out, input_names, input_args):
    """Handler for tensor/scalar true division."""
    if len(input_args) != 2:
        raise ValueError(f"Unsupported truediv arguments: {input_args}")
    scalar = input_args[1]
    if isinstance(scalar, torch.fx.node.Node):
        scalar = codegen.scalar_values.get(scalar.name, None)
    if not isinstance(scalar, (int, float)):
        raise NotImplementedError(f"Only tensor/scalar truediv is supported, got scalar={scalar}")
    codegen.add_uninitialized_tensor(n.name, out)
    codegen.add_forward_call("MiCo_div{dim}d_scalar_{dtype}", out, n.name, input_names, [float(scalar)])


@MiCoOpRegistry.register_function(torch.nn.functional.softmax)
@MiCoOpRegistry.register_function(torch.softmax)
def handle_softmax(codegen, n, out, input_names, input_args):
    """Handler for softmax."""
    dim = n.kwargs.get("dim", -1)
    codegen.add_uninitialized_tensor(n.name, out)
    codegen.add_forward_call("MiCo_softmax{dim}d_{dtype}", out, n.name, input_names, [dim])


@MiCoOpRegistry.register_function(torch.einsum)
@MiCoOpRegistry.register_function(torch.functional.einsum)
def handle_einsum(codegen, n, out, input_names, input_args):
    """Handler for ViT attention einsum patterns."""
    equation = input_args[0].replace(" ", "")
    codegen.add_uninitialized_tensor(n.name, out)
    if equation == "bhif,bhjf->bhij":
        codegen.add_forward_call("MiCo_einsum_bhif_bhjf_bhij_{dtype}", out, n.name, input_names)
    elif equation == "bhij,bhjf->bihf":
        codegen.add_forward_call("MiCo_einsum_bhij_bhjf_bihf_{dtype}", out, n.name, input_names)
    elif equation == "bkn,bnd->bd":
        codegen.add_forward_call("MiCo_einsum_bkn_bnd_bd_{dtype}", out, n.name, input_names)
    else:
        raise NotImplementedError(f"Unsupported einsum equation: {equation}")


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


def _extract_scalar_param(param, param_name, default=None):
    """Extract a scalar C API parameter from PyTorch int/tuple pooling args."""
    if param is None:
        if default is None:
            raise ValueError(f"{param_name} cannot be None")
        param = default

    if isinstance(param, torch.fx.node.Node):
        raise ValueError(f"Unresolved FX node for {param_name}: {param}")

    if isinstance(param, torch.Size):
        param = tuple(param)

    if isinstance(param, (tuple, list)):
        if len(param) == 0:
            raise ValueError(f"{param_name} cannot be empty")
        first = param[0]
        if any(value != first for value in param):
            raise NotImplementedError(
                f"MiCo C pooling kernels only support scalar/symmetric {param_name}, got {param}"
            )
        param = first

    if isinstance(param, bool) or not isinstance(param, int):
        raise ValueError(f"Unexpected {param_name} type: {type(param)}")
    return param


def _extract_kernel_size(param):
    """Helper to extract scalar kernel size for the C pooling API."""
    return _extract_scalar_param(param, "kernel_size")


def _extract_output_size(param):
    """Helper to extract scalar output size for the C adaptive pooling API."""
    return _extract_scalar_param(param, "output_size")


def _pool_arg(n, input_args, index, name, default=None):
    """Read pooling arg from positional or keyword FX args and normalize it."""
    if len(input_args) > index:
        value = input_args[index]
    else:
        value = n.kwargs.get(name, default)
    return _extract_scalar_param(value, name, default)


@MiCoOpRegistry.register_function(torch.nn.functional.linear)
def handle_linear(codegen, n, out, input_names, input_args):
    """Handler for linear (fully connected) layer function."""
    if len(input_args) < 3 or not hasattr(input_args[1], 'target') or not hasattr(input_args[2], 'target'):
        raise ValueError(f"Invalid arguments for linear function: {input_args}")
    weight = codegen.model.state_dict()[input_args[1].target]
    bias = codegen.model.state_dict()[input_args[2].target]
    
    # Gemmini mode: transpose weight from [M, K] to [K, M]
    if codegen.gemmini_mode:
        weight = weight.t().contiguous()
    
    codegen.add_uninitialized_tensor(n.name, out)
    codegen.add_initialized_tensor(f"{input_names[1]}", weight)
    codegen.add_initialized_tensor(f"{input_names[2]}", bias)
    codegen.add_forward_call("MiCo_addmm_{dtype}", out, n.name, input_names)


@MiCoOpRegistry.register_function(torch.nn.functional.avg_pool2d)
def handle_avg_pool2d(codegen, n, out, input_names, input_args):
    """Handler for 2D average pooling function."""
    codegen.add_uninitialized_tensor(n.name, out)
    kernel_size = _pool_arg(n, input_args, 1, "kernel_size")
    stride = _pool_arg(n, input_args, 2, "stride", kernel_size)
    padding = _pool_arg(n, input_args, 3, "padding", 0)
    codegen.add_forward_call("MiCo_avgpool{dim}d_{dtype}", out, n.name, input_names, 
                             [kernel_size, stride, padding])


@MiCoOpRegistry.register_function(torch.nn.functional.max_pool2d)
def handle_max_pool2d(codegen, n, out, input_names, input_args):
    """Handler for 2D max pooling function."""
    codegen.add_uninitialized_tensor(n.name, out)
    kernel_size = _pool_arg(n, input_args, 1, "kernel_size")
    stride = _pool_arg(n, input_args, 2, "stride", kernel_size)
    padding = _pool_arg(n, input_args, 3, "padding", 0)
    codegen.add_forward_call("MiCo_maxpool{dim}d_{dtype}", out, n.name, input_names,
                             [kernel_size, stride, padding])


@MiCoOpRegistry.register_function(torch.nn.functional.adaptive_avg_pool2d)
def handle_adaptive_avg_pool2d(codegen, n, out, input_names, input_args):
    """Handler for 2D adaptive average pooling function."""
    codegen.add_uninitialized_tensor(n.name, out)
    output_size = _extract_output_size(input_args[1])
    codegen.add_forward_call("MiCo_adaptive_avgpool{dim}d_{dtype}", out, n.name, input_names, [output_size])


@MiCoOpRegistry.register_function(torch.nn.functional.avg_pool1d)
def handle_avg_pool1d(codegen, n, out, input_names, input_args):
    """Handler for 1D average pooling function."""
    codegen.add_uninitialized_tensor(n.name, out)
    kernel_size = _pool_arg(n, input_args, 1, "kernel_size")
    stride = _pool_arg(n, input_args, 2, "stride", kernel_size)
    padding = _pool_arg(n, input_args, 3, "padding", 0)
    codegen.add_forward_call("MiCo_avgpool{dim}d_{dtype}", out, n.name, input_names, 
                             [kernel_size, stride, padding])


@MiCoOpRegistry.register_function(torch.nn.functional.max_pool1d)
def handle_max_pool1d(codegen, n, out, input_names, input_args):
    """Handler for 1D max pooling function."""
    codegen.add_uninitialized_tensor(n.name, out)
    kernel_size = _pool_arg(n, input_args, 1, "kernel_size")
    stride = _pool_arg(n, input_args, 2, "stride", kernel_size)
    padding = _pool_arg(n, input_args, 3, "padding", 0)
    codegen.add_forward_call("MiCo_maxpool{dim}d_{dtype}", out, n.name, input_names,
                             [kernel_size, stride, padding])


@MiCoOpRegistry.register_function(torch.nn.functional.adaptive_avg_pool1d)
def handle_adaptive_avg_pool1d(codegen, n, out, input_names, input_args):
    """Handler for 1D adaptive average pooling function."""
    codegen.add_uninitialized_tensor(n.name, out)
    output_size = _extract_output_size(input_args[1])
    codegen.add_forward_call("MiCo_adaptive_avgpool{dim}d_{dtype}", out, n.name, input_names, [output_size])


@MiCoOpRegistry.register_function(torch.flatten)
def handle_flatten(codegen, n, out, input_names, input_args):
    """Handler for flatten operation."""
    codegen.add_connect_tensor(n.name, out)
    if codegen.gemmini_mode:
        codegen.add_forward_call("MiCo_NHWC2NCHW_flatten_{dtype}", out, n.name, input_names)
    else:
        codegen.add_forward_call("MiCo_CONNECT", out, n.name, input_names)


@MiCoOpRegistry.register_function(torch.cat)
def handle_cat(codegen, n, out, input_names, input_args):
    """Handler for concatenation operation."""
    codegen.add_uninitialized_tensor(n.name, out)
    codegen.add_forward_call("MiCo_concat{dim}d_{dtype}", out, n.name, input_names)


# ============================================================================
# Module Handlers
# ============================================================================

if KWTPatchEmbedding is not None:
    @MiCoOpRegistry.register_module(KWTPatchEmbedding)
    def handle_kwt_patch_embedding_module(codegen, n, out, module, input_names):
        """Handler for KWT patch extraction plus projection."""
        layer_name = n.name
        input_name = input_names[0]
        weight_name = f"{layer_name}_proj_weight"
        bias_name = f"{layer_name}_proj_bias"

        if isinstance(module.proj, BitLinear):
            patch_dim = module.channels * module.patch_res[0] * module.patch_res[1]
            patches_name = f"{layer_name}_patches"
            patches = torch.empty(
                (out.shape[0], module.num_patches, patch_dim),
                dtype=out.dtype,
                device=out.device,
            )
            codegen.add_uninitialized_tensor(patches_name, patches)
            codegen.add_forward_call(
                "MiCo_kwt_patch_extract_{dtype}",
                patches,
                patches_name,
                [input_name],
                [module.patch_res[0], module.patch_res[1]],
            )

            codegen.add_uninitialized_tensor(layer_name, out)
            codegen.add_initialized_tensor(weight_name, module.proj.weight, quant=module.proj.qtype, scale=module.proj.qw_scale)
            codegen.add_initialized_tensor(bias_name, module.proj.bias)
            codegen.add_forward_call(
                "MiCo_bitlinear3d_{dtype}",
                out,
                layer_name,
                [patches_name, weight_name, bias_name],
                [round(module.proj.qtype), round(module.proj.act_q), codegen.align_to],
            )
            return

        input_names.append(weight_name)
        input_names.append(bias_name)

        codegen.add_uninitialized_tensor(layer_name, out)
        codegen.add_initialized_tensor(weight_name, module.proj.weight)
        codegen.add_initialized_tensor(bias_name, module.proj.bias)
        codegen.add_forward_call(
            "MiCo_kwt_patch_embedding_{dtype}",
            out,
            layer_name,
            input_names,
            [module.patch_res[0], module.patch_res[1]],
        )


def _maybe_insert_simple_rmsnorm(codegen, layer_name, module, input_names):
    """Insert simple RMSNorm before quantized layer input when requested."""
    if not getattr(module, "use_norm", False):
        return input_names
    if len(input_names) == 0:
        raise ValueError(f"BitQLayer '{layer_name}' has no input tensor for RMSNorm")

    norm_name = f"{layer_name}_rmsnorm"
    input_name = input_names[0]
    input_tensor = codegen.tensors[input_name]["tensor"]

    codegen.add_uninitialized_tensor(norm_name, input_tensor)
    codegen.add_forward_call("MiCo_simple_rmsnorm{dim}d_{dtype}", input_tensor, norm_name, [input_name])

    return [norm_name] + input_names[1:]


@MiCoOpRegistry.register_module(BitConv1d)
def handle_bitconv1d_module(codegen, n, out, module, input_names):
    """Handler for BitConv1d quantized convolution module."""
    layer_name = n.name
    weight = module.weight
    bias = module.bias
    input_names = _maybe_insert_simple_rmsnorm(codegen, layer_name, module, input_names)
    input_names.append(f"{layer_name}_weight")
    input_names.append(f"{layer_name}_bias")

    # Gemmini mode: weight format is OIK (OutChannels, InChannels, KernelSize)
    # No transformation needed as this is already the PyTorch format

    codegen.add_uninitialized_tensor(layer_name, out)
    codegen.add_initialized_tensor(f"{layer_name}_weight", weight, 
                                    quant=module.qtype, scale=module.qw_scale)
    codegen.add_initialized_tensor(f"{layer_name}_bias", bias)

    codegen.add_forward_call("MiCo_bitconv1d_{dtype}", out, layer_name, input_names, [
        round(module.qtype),
        round(module.act_q),
        module.stride[0],   # assume same stride for both dimensions
        module.padding[0],  # assume same padding for both dimensions
        module.dilation[0], # assume same dilation for both dimensions
        module.groups,
        codegen.align_to
    ])

@MiCoOpRegistry.register_module(BitConv2d)
def handle_bitconv2d_module(codegen, n, out, module, input_names):
    """Handler for BitConv2d quantized convolution module."""
    layer_name = n.name
    weight = module.weight
    bias = module.bias
    input_names = _maybe_insert_simple_rmsnorm(codegen, layer_name, module, input_names)
    input_names.append(f"{layer_name}_weight")
    input_names.append(f"{layer_name}_bias")

    # Gemmini mode: permute weight from OIHW [O, I, Kh, Kw] to KhKwIO [Kh, Kw, I, O]
    if codegen.gemmini_mode:
        weight = weight.permute(2, 3, 1, 0).contiguous()

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
    input_names = _maybe_insert_simple_rmsnorm(codegen, layer_name, module, input_names)
    input_names.append(f"{layer_name}_weight")
    input_names.append(f"{layer_name}_bias")

    # Gemmini mode: transpose weight from [M, K] to [K, M]
    if codegen.gemmini_mode:
        weight = weight.t().contiguous()

    codegen.add_uninitialized_tensor(layer_name, out)
    codegen.add_initialized_tensor(f"{layer_name}_weight", weight, 
                                    quant=module.qtype, scale=module.qw_scale)
    codegen.add_initialized_tensor(f"{layer_name}_bias", bias)

    bitlinear_name = "MiCo_bitlinear_{dtype}" if out.dim() == 2 else "MiCo_bitlinear3d_{dtype}"
    codegen.add_forward_call(bitlinear_name, out, layer_name, input_names, [
        round(module.qtype),
        round(module.act_q),
        codegen.align_to])

@MiCoOpRegistry.register_module(torch.nn.Conv1d)
def handle_conv1d_module(codegen, n, out, module, input_names):
    """Handler for Conv1d module."""
    layer_name = n.name
    weight = module.weight
    bias = module.bias
    input_names.append(f"{layer_name}_weight")
    input_names.append(f"{layer_name}_bias")

    # Gemmini mode: weight format is OIK (OutChannels, InChannels, KernelSize)
    # No transformation needed as this is already the PyTorch format

    codegen.add_uninitialized_tensor(layer_name, out)
    codegen.add_initialized_tensor(f"{layer_name}_weight", module.weight)
    codegen.add_initialized_tensor(f"{layer_name}_bias", module.bias)
    
    codegen.add_forward_call("MiCo_conv1d_{dtype}", out, layer_name, input_names, [
        module.stride[0],   # assume same stride for both dimensions
        module.padding[0],  # assume same padding for both dimensions
        module.dilation[0], # assume same dilation for both dimensions
        module.groups
    ])

@MiCoOpRegistry.register_module(torch.nn.Conv2d)
def handle_conv2d_module(codegen, n, out, module, input_names):
    """Handler for Conv2d module."""
    layer_name = n.name
    weight = module.weight
    bias = module.bias
    input_names.append(f"{layer_name}_weight")
    input_names.append(f"{layer_name}_bias")

    # Gemmini mode: permute weight from OIHW [O, I, Kh, Kw] to KhKwIO [Kh, Kw, I, O]
    if codegen.gemmini_mode:
        weight = weight.permute(2, 3, 1, 0).contiguous()

    codegen.add_uninitialized_tensor(layer_name, out)
    codegen.add_initialized_tensor(f"{layer_name}_weight", weight)
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
    kernel_size = _extract_kernel_size(module.kernel_size)
    stride = _extract_scalar_param(module.stride, "stride", kernel_size)
    padding = _extract_scalar_param(module.padding, "padding", 0)
    codegen.add_forward_call("MiCo_avgpool{dim}d_{dtype}", out, layer_name, input_names, 
                             [kernel_size, stride, padding])


@MiCoOpRegistry.register_module(torch.nn.MaxPool2d)
def handle_maxpool2d_module(codegen, n, out, module, input_names):
    """Handler for MaxPool2d module."""
    layer_name = n.name
    codegen.add_uninitialized_tensor(layer_name, out)
    kernel_size = _extract_kernel_size(module.kernel_size)
    stride = _extract_scalar_param(module.stride, "stride", kernel_size)
    padding = _extract_scalar_param(module.padding, "padding", 0)
    codegen.add_forward_call("MiCo_maxpool{dim}d_{dtype}", out, layer_name, input_names, 
                             [kernel_size, stride, padding])


@MiCoOpRegistry.register_module(torch.nn.AdaptiveAvgPool2d)
def handle_adaptive_avgpool2d_module(codegen, n, out, module, input_names):
    """Handler for AdaptiveAvgPool2d module."""
    layer_name = n.name
    codegen.add_uninitialized_tensor(layer_name, out)
    output_size = _extract_output_size(module.output_size)
    codegen.add_forward_call("MiCo_adaptive_avgpool{dim}d_{dtype}", out, layer_name, input_names, [output_size])


@MiCoOpRegistry.register_module(torch.nn.AvgPool1d)
def handle_avgpool1d_module(codegen, n, out, module, input_names):
    """Handler for AvgPool1d module."""
    layer_name = n.name
    codegen.add_uninitialized_tensor(layer_name, out)
    kernel_size = _extract_kernel_size(module.kernel_size)
    stride = _extract_scalar_param(module.stride, "stride", kernel_size)
    padding = _extract_scalar_param(module.padding, "padding", 0)
    codegen.add_forward_call("MiCo_avgpool{dim}d_{dtype}", out, layer_name, input_names, 
                             [kernel_size, stride, padding])


@MiCoOpRegistry.register_module(torch.nn.MaxPool1d)
def handle_maxpool1d_module(codegen, n, out, module, input_names):
    """Handler for MaxPool1d module."""
    layer_name = n.name
    codegen.add_uninitialized_tensor(layer_name, out)
    kernel_size = _extract_kernel_size(module.kernel_size)
    stride = _extract_scalar_param(module.stride, "stride", kernel_size)
    padding = _extract_scalar_param(module.padding, "padding", 0)
    codegen.add_forward_call("MiCo_maxpool{dim}d_{dtype}", out, layer_name, input_names, 
                             [kernel_size, stride, padding])


@MiCoOpRegistry.register_module(torch.nn.AdaptiveAvgPool1d)
def handle_adaptive_avgpool1d_module(codegen, n, out, module, input_names):
    """Handler for AdaptiveAvgPool1d module."""
    layer_name = n.name
    codegen.add_uninitialized_tensor(layer_name, out)
    output_size = _extract_output_size(module.output_size)
    codegen.add_forward_call("MiCo_adaptive_avgpool{dim}d_{dtype}", out, layer_name, input_names, [output_size])


@MiCoOpRegistry.register_module(torch.nn.Flatten)
def handle_flatten_module(codegen, n, out, module, input_names):
    """Handler for Flatten module."""
    layer_name = n.name
    codegen.add_connect_tensor(layer_name, out)
    if codegen.gemmini_mode:
        codegen.add_forward_call("MiCo_NHWC2NCHW_flatten_{dtype}", out, layer_name, input_names)
    else:
        codegen.add_forward_call("MiCo_CONNECT", out, layer_name, input_names)


@MiCoOpRegistry.register_module(torch.nn.Linear)
def handle_linear_module(codegen, n, out, module, input_names):
    """Handler for Linear module."""
    layer_name = n.name
    weight = module.weight
    bias = module.bias
    input_names.append(f"{layer_name}_weight")
    input_names.append(f"{layer_name}_bias")

    # Gemmini mode: transpose weight from [M, K] to [K, M]
    if codegen.gemmini_mode:
        weight = weight.t().contiguous()

    codegen.add_uninitialized_tensor(layer_name, out)
    codegen.add_initialized_tensor(f"{layer_name}_weight", weight)
    codegen.add_initialized_tensor(f"{layer_name}_bias", bias)
    linear_name = "MiCo_linear_{dtype}" if out.dim() == 2 else "MiCo_linear3d_{dtype}"
    codegen.add_forward_call(linear_name, out, layer_name, input_names)


@MiCoOpRegistry.register_module(torch.nn.LayerNorm)
def handle_layernorm_module(codegen, n, out, module, input_names):
    """Handler for LayerNorm module."""
    layer_name = n.name
    input_names.append(f"{layer_name}_weight")
    input_names.append(f"{layer_name}_bias")

    codegen.add_uninitialized_tensor(layer_name, out)
    codegen.add_initialized_tensor(f"{layer_name}_weight", module.weight)
    codegen.add_initialized_tensor(f"{layer_name}_bias", module.bias)

    if out.dim() == 2:
        codegen.add_forward_call("MiCo_layernorm2d_{dtype}", out, layer_name, input_names, [
            module.normalized_shape[0], module.eps
        ])
    elif out.dim() == 3:
        codegen.add_forward_call("MiCo_layernorm3d_{dtype}", out, layer_name, input_names, [
            module.normalized_shape[0], module.eps
        ])
    else:
        raise NotImplementedError(f"LayerNorm only supports 2D/3D outputs, got {out.dim()}D")


@MiCoOpRegistry.register_module(torch.nn.GELU)
def handle_gelu_module(codegen, n, out, module, input_names):
    """Handler for GELU activation module."""
    layer_name = n.name
    codegen.add_uninitialized_tensor(layer_name, out)
    if out.dim() == 2:
        codegen.add_forward_call("MiCo_gelu2d_{dtype}", out, layer_name, input_names)
    elif out.dim() == 3:
        codegen.add_forward_call("MiCo_gelu3d_{dtype}", out, layer_name, input_names)
    else:
        raise NotImplementedError(f"GELU only supports 2D/3D outputs, got {out.dim()}D")


@MiCoOpRegistry.register_module(torch.nn.Identity)
@MiCoOpRegistry.register_module(torch.nn.Dropout)
def handle_identity_module(codegen, n, out, module, input_names):
    """Handler for Identity and Dropout modules (pass-through operations)."""
    layer_name = n.name
    codegen.add_connect_tensor(layer_name, out)
    codegen.add_forward_call("MiCo_CONNECT", out, layer_name, input_names)
