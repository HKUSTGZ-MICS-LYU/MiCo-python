# MiCo Registry Pattern Usage Guide

## Overview

The MiCo code generator now uses a registry pattern to manage PyTorch operation handlers. This makes the codebase more extensible, maintainable, and allows users to add custom operations without modifying the core code.

## Benefits

1. **Extensibility**: Register custom operations without modifying `MiCoCodeGen.py`
2. **Maintainability**: Each operation handler is isolated and self-contained
3. **Testability**: Individual handlers can be unit tested
4. **Clear Documentation**: Decorators serve as documentation of supported operations

## Architecture

### Core Components

- **`MiCoOpRegistry`**: Central registry class that maintains dictionaries of handlers
- **`@register_function`**: Decorator for registering PyTorch function handlers (e.g., `torch.nn.functional.relu`)
- **`@register_module`**: Decorator for registering PyTorch module handlers (e.g., `torch.nn.Conv2d`)

### Pre-registered Operations

The following operations are pre-registered in `MiCoRegistry.py`:

**Function Handlers** (for `torch.nn.functional.*` and operators):
- `operator.__add__`, `torch.add`
- `operator.__mul__`
- `torch.nn.functional.relu`
- `torch.nn.functional.relu6`
- `torch.nn.functional.tanh`
- `torch.nn.functional.linear`
- `torch.nn.functional.avg_pool2d`
- `torch.nn.functional.max_pool2d`
- `torch.nn.functional.adaptive_avg_pool2d`
- `torch.flatten`
- `torch.cat`

**Module Handlers** (for `torch.nn.*` modules):
- `BitConv2d` (custom quantized conv)
- `BitLinear` (custom quantized linear)
- `torch.nn.Conv2d`
- `torch.nn.BatchNorm2d`
- `torch.nn.ELU`
- `torch.nn.ReLU`
- `torch.nn.ReLU6`
- `torch.nn.Tanh`
- `torch.nn.AvgPool2d`
- `torch.nn.MaxPool2d`
- `torch.nn.AdaptiveAvgPool2d`
- `torch.nn.Flatten`
- `torch.nn.Linear`
- `torch.nn.Identity`
- `torch.nn.Dropout`

## How to Register Custom Operations

### Registering a Custom Function

```python
from MiCoRegistry import MiCoOpRegistry
import torch

@MiCoOpRegistry.register_function(torch.sigmoid)
def handle_sigmoid(codegen, n, out, input_names, input_args):
    """
    Handler for sigmoid activation function.
    
    Args:
        codegen: The MiCoCodeGen instance
        n: The FX node being processed
        out: The output tensor
        input_names: List of input tensor names
        input_args: Original arguments from the node
    """
    codegen.add_uninitialized_tensor(n.name, out)
    codegen.add_forward_call("MiCo_sigmoid{dim}d_{dtype}", out, n.name, input_names)
```

### Registering a Custom Module

```python
from MiCoRegistry import MiCoOpRegistry
import torch.nn as nn

class MyCustomLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x):
        return x * self.alpha

@MiCoOpRegistry.register_module(MyCustomLayer)
def handle_my_custom_layer(codegen, n, out, module, input_names):
    """
    Handler for MyCustomLayer module.
    
    Args:
        codegen: The MiCoCodeGen instance
        n: The FX node being processed
        out: The output tensor
        module: The module instance
        input_names: List of input tensor names
    """
    layer_name = n.name
    codegen.add_uninitialized_tensor(layer_name, out)
    codegen.add_forward_call("MiCo_custom_activation{dim}d_{dtype}", 
                             out, layer_name, input_names, [module.alpha])
```

### Using Custom Operations

Once registered, your custom operations can be used in any PyTorch model:

```python
from MiCoCodeGen import MiCoCodeGen

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, 3)
        self.custom = MyCustomLayer(alpha=2.0)
        self.n_layers = 2
    
    def forward(self, x):
        x = self.conv(x)
        x = torch.sigmoid(x)  # Uses registered handler
        x = self.custom(x)    # Uses registered handler
        return x

model = MyModel()
codegen = MiCoCodeGen(model, align_to=32)
codegen.forward(torch.randn(1, 1, 28, 28))
codegen.convert("output", "model")
```

## Handler Function Signatures

### Function Handler Signature

```python
def handler_function(codegen, n, out, input_names, input_args):
    """
    Args:
        codegen (MiCoCodeGen): The code generator instance
        n (torch.fx.node.Node): The FX graph node
        out (torch.Tensor): The output tensor
        input_names (List[str]): Names of input tensors
        input_args (Tuple): Original arguments from the function call
    """
    pass
```

### Module Handler Signature

```python
def handler_module(codegen, n, out, module, input_names):
    """
    Args:
        codegen (MiCoCodeGen): The code generator instance
        n (torch.fx.node.Node): The FX graph node
        out (torch.Tensor): The output tensor
        module (torch.nn.Module): The module instance
        input_names (List[str]): Names of input tensors
    """
    pass
```

## Error Handling

If you try to use an unregistered operation, you'll get a clear error message:

```python
# Using unregistered torch.sqrt
RuntimeError: Function 'torch.sqrt' is not registered. 
Use @MiCoOpRegistry.register_function() decorator to add support.
```

## Helper Methods

When writing handlers, you can use these `MiCoCodeGen` methods:

- `add_uninitialized_tensor(name, tensor)`: Register an uninitialized tensor
- `add_initialized_tensor(name, tensor, quant=0, scale=0.0)`: Register an initialized tensor
- `add_connect_tensor(name, tensor)`: Register a pass-through tensor
- `add_forward_call(function_name, out, layer_name, input_names, parameters=None)`: Generate a forward call

## Best Practices

1. **Keep handlers simple**: Each handler should do one thing well
2. **Use helper functions**: Extract common logic (see `_extract_kernel_size` in `MiCoRegistry.py`)
3. **Add validation**: Check arguments before using them
4. **Document parameters**: Explain what each parameter means
5. **Follow naming conventions**: Use `handle_<operation_name>` for handler function names
6. **Test your handlers**: Write unit tests for custom operations

## Migration from Old Code

If you have old code that directly modified `MiCoCodeGen.py`, you can migrate it by:

1. Creating a handler function
2. Registering it with the appropriate decorator
3. Removing the if-else branch from the core code

Example:

**Old code** (modifying `MiCoCodeGen.py`):
```python
elif function == torch.my_custom_op:
    self.add_uninitialized_tensor(layer_name, out)
    self.add_forward_call("MiCo_custom{dim}d_{dtype}", out, layer_name, input_names)
```

**New code** (in separate file or at module level):
```python
from MiCoRegistry import MiCoOpRegistry
import torch

@MiCoOpRegistry.register_function(torch.my_custom_op)
def handle_my_custom_op(codegen, n, out, input_names, input_args):
    codegen.add_uninitialized_tensor(n.name, out)
    codegen.add_forward_call("MiCo_custom{dim}d_{dtype}", out, n.name, input_names)
```

## Troubleshooting

### Handler not being called

- Make sure the decorator is executed before using the operation
- Check that you're registering the correct function/module type
- Verify the import order (registry must be imported before use)

### AttributeError in handler

- Check that input_args has the expected structure
- Add validation at the start of your handler
- Use `hasattr()` to check for attributes before accessing them

### Generated code incorrect

- Verify the function name template (e.g., `"MiCo_op{dim}d_{dtype}"`)
- Check parameter order in `add_forward_call()`
- Ensure tensors are initialized correctly

## Additional Resources

- See `MiCoRegistry.py` for examples of all pre-registered handlers
- Check test files in `/tmp/test_*.py` for usage examples
- Refer to `MiCoCodeGen.py` for available helper methods
