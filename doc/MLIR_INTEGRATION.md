# MLIR Integration Proposal for MiCo Framework

## Overview

This document proposes the integration of MLIR (Multi-Level Intermediate Representation) as a mid-end representation for the MiCo framework. MLIR is widely used in modern AI compiler systems and provides a flexible, extensible infrastructure for building domain-specific compilers.

## Goals

1. **Unified IR**: Use MLIR as a common intermediate representation between PyTorch models and various backend targets
2. **Sub-byte Data Types**: Enable mixed-precision quantization with sub-byte (1-8 bit) weight and activation representations
3. **Hardware Portability**: Leverage MLIR's infrastructure for targeting multiple hardware backends
4. **Optimization Pipeline**: Enable MLIR-based optimization passes for quantized neural networks

## Architecture

MiCo provides two MLIR generation paths:

### Path 1: Standalone MiCoMLIRGen

```
┌─────────────────────┐
│   PyTorch Model     │
│  (MiCo Framework)   │
└─────────┬───────────┘
          │ MiCoMLIRGen
          ▼
┌─────────────────────┐
│    MiCo Dialect     │
│  (Sub-byte types,   │
│   Quant operations) │
└─────────┬───────────┘
          │ Lowering
          ▼
┌─────────────────────┐
│   Arith + Tensor    │
│    + MemRef         │
└─────────┬───────────┘
          │ Lowering
          ▼
┌─────────────────────┐
│      LLVM IR        │
│   or Hardware IR    │
└─────────┴───────────┘
```

### Path 2: Torch-MLIR Integration (MiCoTorchMLIRGen)

```
┌─────────────────────┐
│   PyTorch Model     │
│  (MiCo Framework)   │
└─────────┬───────────┘
          │ torch-mlir (first pass)
          ▼
┌─────────────────────┐
│   Torch Dialect     │
│  (Standard PyTorch  │
│   operations)       │
└─────────┬───────────┘
          │ MiCo quantization overlay
          ▼
┌─────────────────────┐
│    MiCo Dialect     │
│  (Sub-byte types,   │
│   Quant metadata)   │
└─────────┬───────────┘
          │ Lowering
          ▼
┌─────────────────────┐
│  Linalg / StableHLO │
│    + MiCo-Lib       │
└─────────┴───────────┘
```

## Torch-MLIR Integration

### Installation

```bash
pip install torch-mlir -f https://github.com/llvm/torch-mlir-release/releases
```

### Usage with Torch-MLIR Backend

```python
from models import LeNet
from MiCoTorchMLIRGen import MiCoTorchMLIRGen
from MiCoUtils import fuse_model
import torch

model = LeNet(1)
model.set_qscheme([[8, 6, 6, 4, 4], [8, 8, 8, 8, 8]])
model = fuse_model(model)
model.eval()

# Use torch-mlir backend with Torch dialect output
mlir_gen = MiCoTorchMLIRGen(model, output_type="torch")
mlir_gen.forward(torch.randn(1, 1, 28, 28))
mlir_path = mlir_gen.convert("output", "lenet_mnist")

# Check if torch-mlir was used
print(f"Backend: {'torch-mlir' if mlir_gen.use_torch_mlir else 'standalone'}")
```

### Supported Output Types

| Output Type | Description |
|-------------|-------------|
| `torch` | Torch dialect (default) - closest to PyTorch semantics |
| `linalg` | Linalg on Tensors - good for custom lowering |
| `stablehlo` | StableHLO - compatible with XLA ecosystem |

### Benefits of Torch-MLIR Integration

1. **Production-ready lowering**: Leverages torch-mlir's mature PyTorch → MLIR conversion
2. **Ecosystem compatibility**: Integrates with IREE, Blade, and other MLIR-based compilers
3. **Reduced maintenance**: Core operation lowering handled by torch-mlir
4. **MiCo-specific extensions**: Sub-byte quantization metadata preserved via MiCo dialect overlay

## MiCo MLIR Dialect

### Custom Types

The MiCo dialect introduces sub-byte integer types for mixed-precision quantization:

```mlir
// Sub-byte integer types
!mico.int<1>   // 1-bit (binary)
!mico.int<2>   // 2-bit (ternary)
!mico.int<4>   // 4-bit
!mico.int<8>   // 8-bit (standard)

// Quantized tensor type with scale factor
!mico.qtensor<shape x element_type, scale: f32>
```

### Operations

#### Quantized Linear Operations

```mlir
// Quantized matrix multiplication
// Inputs: activation tensor, weight tensor, bias tensor
// Attributes: weight_bits, activation_bits
%result = mico.bitlinear(%activation, %weight, %bias) {
    weight_bits = 4 : i32,
    act_bits = 8 : i32
} : (tensor<1x256xf32>, tensor<64x256x!mico.int<4>>, tensor<64xf32>) -> tensor<1x64xf32>

// Quantized 2D convolution
%result = mico.bitconv2d(%input, %weight, %bias) {
    weight_bits = 4 : i32,
    act_bits = 8 : i32,
    stride = [1, 1],
    padding = [1, 1],
    dilation = [1, 1],
    groups = 1
} : (tensor<1x32x28x28xf32>, tensor<64x32x3x3x!mico.int<4>>, tensor<64xf32>) -> tensor<1x64x28x28xf32>

// Quantized 1D convolution  
%result = mico.bitconv1d(%input, %weight, %bias) {
    weight_bits = 8 : i32,
    act_bits = 8 : i32,
    stride = [1],
    padding = [0],
    dilation = [1],
    groups = 1
} : (tensor<1x32x128xf32>, tensor<64x32x3x!mico.int<8>>, tensor<64xf32>) -> tensor<1x64x126xf32>
```

#### Quantization Operations

```mlir
// Weight quantization (per-tensor symmetric)
%qweight, %scale = mico.weight_quant(%weight) {
    bits = 4 : i32,
    mode = "max"  // or "mean"
} : (tensor<64x256xf32>) -> (tensor<64x256x!mico.int<4>>, f32)

// Activation quantization (per-tensor symmetric)
%qact = mico.activation_quant(%activation) {
    bits = 8 : i32
} : (tensor<1x256xf32>) -> tensor<1x256xf32>
```

#### Standard Operations

```mlir
// ReLU activation
%result = mico.relu(%input) : (tensor<1x64x28x28xf32>) -> tensor<1x64x28x28xf32>

// ReLU6 activation
%result = mico.relu6(%input) : (tensor<1x64x28x28xf32>) -> tensor<1x64x28x28xf32>

// Max pooling
%result = mico.maxpool2d(%input) {
    kernel_size = [2, 2],
    stride = [2, 2]
} : (tensor<1x64x28x28xf32>) -> tensor<1x64x14x14xf32>

// Average pooling
%result = mico.avgpool2d(%input) {
    kernel_size = [2, 2],
    stride = [2, 2]
} : (tensor<1x64x28x28xf32>) -> tensor<1x64x14x14xf32>

// Flatten
%result = mico.flatten(%input) {
    start_dim = 1 : i32
} : (tensor<1x64x7x7xf32>) -> tensor<1x3136xf32>

// Element-wise add
%result = mico.add(%lhs, %rhs) : (tensor<1x64x28x28xf32>, tensor<1x64x28x28xf32>) -> tensor<1x64x28x28xf32>

// Concatenation
%result = mico.concat(%t1, %t2) {
    dim = 1 : i32
} : (tensor<1x32x28x28xf32>, tensor<1x32x28x28xf32>) -> tensor<1x64x28x28xf32>
```

## Implementation

### MiCoMLIRGen Class

The `MiCoMLIRGen` class extends the existing code generation infrastructure:

```python
from MiCoCodeGen import MiCoCodeGen

class MiCoMLIRGen(MiCoCodeGen):
    """
    MLIR code generator for MiCo models.
    Generates MLIR code using the MiCo dialect for mixed-precision quantized models.
    """
    
    def __init__(self, model, dialect="mico"):
        super().__init__(model)
        self.dialect = dialect
        self.mlir_operations = []
        self.mlir_types = {}
    
    def convert(self, output_directory, model_name):
        """Generate MLIR code from the traced model."""
        # Generate MLIR module
        mlir_code = self.generate_mlir_module(model_name)
        
        # Write to file
        mlir_path = os.path.join(output_directory, f"{model_name}.mlir")
        with open(mlir_path, "w") as f:
            f.write(mlir_code)
        
        return mlir_path
```

### File Structure

```
MiCo-python/
├── MiCoMLIRGen.py           # Main MLIR code generator
├── mlir/                    # MLIR dialect definitions (optional, for MLIR-based compilation)
│   └── MiCoDialect.td       # TableGen dialect definition
├── doc/
│   └── MLIR_INTEGRATION.md  # This document
├── examples/
│   └── mlir_example.py      # Example usage
└── tests/
    └── test_mlir_codegen.py # Unit tests
```

## Usage Example

```python
import torch
from models import LeNet
from MiCoMLIRGen import MiCoMLIRGen
from MiCoUtils import fuse_model

# Load and prepare model
model = LeNet(1)
model.set_qscheme([[8, 6, 6, 4, 4], [8, 8, 8, 8, 8]])  # Mixed precision config
model.load_state_dict(torch.load('output/ckpt/lenet_mnist.pth'))
model = fuse_model(model)
model.eval()

# Generate MLIR
mlir_gen = MiCoMLIRGen(model)
mlir_gen.forward(torch.randn(1, 1, 28, 28))
mlir_path = mlir_gen.convert("output", "lenet_mnist")

print(f"MLIR code generated at: {mlir_path}")
```

### Generated MLIR Output Example

```mlir
// MiCo MLIR Module for LeNet-5 with Mixed Precision
module @lenet_mnist {
    // Weight constants
    mico.constant @conv1_weight : tensor<6x1x5x5x!mico.int<8>> = dense<...>
    mico.constant @conv1_bias : tensor<6xf32> = dense<...>
    mico.constant @conv2_weight : tensor<16x6x5x5x!mico.int<6>> = dense<...>
    mico.constant @conv2_bias : tensor<16xf32> = dense<...>
    mico.constant @fc1_weight : tensor<120x256x!mico.int<6>> = dense<...>
    mico.constant @fc1_bias : tensor<120xf32> = dense<...>
    mico.constant @fc2_weight : tensor<84x120x!mico.int<4>> = dense<...>
    mico.constant @fc2_bias : tensor<84xf32> = dense<...>
    mico.constant @fc3_weight : tensor<10x84x!mico.int<4>> = dense<...>
    mico.constant @fc3_bias : tensor<10xf32> = dense<...>
    
    func.func @forward(%input: tensor<1x1x28x28xf32>) -> tensor<1x10xf32> {
        // Conv1: W8A8
        %conv1 = mico.bitconv2d(%input, @conv1_weight, @conv1_bias) {
            weight_bits = 8, act_bits = 8, stride = [1, 1], padding = [0, 0]
        } : (tensor<1x1x28x28xf32>) -> tensor<1x6x24x24xf32>
        %relu1 = mico.relu(%conv1) : tensor<1x6x24x24xf32> -> tensor<1x6x24x24xf32>
        %pool1 = mico.maxpool2d(%relu1) {kernel_size = [2, 2], stride = [2, 2]} 
            : tensor<1x6x24x24xf32> -> tensor<1x6x12x12xf32>
        
        // Conv2: W6A8
        %conv2 = mico.bitconv2d(%pool1, @conv2_weight, @conv2_bias) {
            weight_bits = 6, act_bits = 8, stride = [1, 1], padding = [0, 0]
        } : (tensor<1x6x12x12xf32>) -> tensor<1x16x8x8xf32>
        %relu2 = mico.relu(%conv2) : tensor<1x16x8x8xf32> -> tensor<1x16x8x8xf32>
        %pool2 = mico.maxpool2d(%relu2) {kernel_size = [2, 2], stride = [2, 2]}
            : tensor<1x16x8x8xf32> -> tensor<1x16x4x4xf32>
        
        // Flatten
        %flat = mico.flatten(%pool2) {start_dim = 1} 
            : tensor<1x16x4x4xf32> -> tensor<1x256xf32>
        
        // FC1: W6A8
        %fc1 = mico.bitlinear(%flat, @fc1_weight, @fc1_bias) {
            weight_bits = 6, act_bits = 8
        } : (tensor<1x256xf32>) -> tensor<1x120xf32>
        %relu3 = mico.relu(%fc1) : tensor<1x120xf32> -> tensor<1x120xf32>
        
        // FC2: W4A8
        %fc2 = mico.bitlinear(%relu3, @fc2_weight, @fc2_bias) {
            weight_bits = 4, act_bits = 8
        } : (tensor<1x120xf32>) -> tensor<1x84xf32>
        %relu4 = mico.relu(%fc2) : tensor<1x84xf32> -> tensor<1x84xf32>
        
        // FC3: W4A8
        %fc3 = mico.bitlinear(%relu4, @fc3_weight, @fc3_bias) {
            weight_bits = 4, act_bits = 8
        } : (tensor<1x10xf32>) -> tensor<1x10xf32>
        
        return %fc3 : tensor<1x10xf32>
    }
}
```

## Lowering Passes

### Stage 1: MiCo Dialect → Linalg + Arith

Lower MiCo high-level operations to Linalg generic operations with quantization-aware transformations:

```mlir
// Before (MiCo dialect)
%result = mico.bitlinear(%act, %weight, %bias) {weight_bits = 4, act_bits = 8}

// After (Linalg + Arith)
%qact = linalg.generic ... // Quantize activation
%dequant_weight = arith.mulf %weight, %scale : f32
%matmul = linalg.matmul ins(%qact, %dequant_weight) outs(%result)
%biased = linalg.generic ... // Add bias
```

### Stage 2: Linalg → Affine/SCF

Lower Linalg operations to loop-based representations for further optimization:

```mlir
// Affine loops with tiling for cache optimization
affine.for %i = 0 to 64 step 16 {
    affine.for %j = 0 to 256 step 32 {
        // Tiled matrix multiplication
    }
}
```

### Stage 3: Affine → LLVM/Hardware IR

Final lowering to LLVM IR or target-specific IR (e.g., for VexiiRiscv, Gemmini):

```mlir
// LLVM dialect
llvm.func @forward(%arg0: !llvm.ptr) -> !llvm.ptr {
    // Low-level implementation
}
```

## Hardware Backend Targets

### 1. CPU (LLVM)
Standard LLVM-based compilation for x86, ARM, RISC-V CPUs.

### 2. VexiiRiscv (Custom SIMD)
Target MiCo's custom SIMD instructions through LLVM RISC-V backend with custom intrinsics.

### 3. Gemmini Accelerator
Generate code compatible with Gemmini's systolic array architecture.

### 4. BitFusion Accelerator
Lower to BitFusion-compatible operations using existing `MiCoGraphGen` as reference.

## Benefits

1. **Portability**: MLIR provides a standard infrastructure for targeting multiple hardware backends
2. **Optimization**: Leverage MLIR's optimization passes (loop fusion, tiling, vectorization)
3. **Interoperability**: Compatible with other MLIR-based tools (IREE, TVM-MLIR, torch-mlir)
4. **Extensibility**: Easy to add new operations and lowering passes
5. **Debugging**: MLIR's textual format enables easier debugging and analysis

## Future Work

1. **Group Quantization**: Add support for group-wise quantization in the dialect
2. **Transformer Support**: Add operations for attention mechanisms (for LLaMa, ViT)
3. **Auto-tuning**: Integration with auto-tuning frameworks for optimal lowering decisions
4. **JIT Compilation**: Enable just-in-time compilation using MLIR's execution engine
5. **Hardware Synthesis**: Explore MLIR-to-RTL lowering for FPGA/ASIC targets

## Dependencies

- **mlir-python-bindings**: Python bindings for MLIR (optional, for advanced compilation)
- **torch-mlir**: For potential integration with torch-mlir ecosystem (optional)

## References

1. [MLIR: Multi-Level Intermediate Representation](https://mlir.llvm.org/)
2. [MLIR Dialects Documentation](https://mlir.llvm.org/docs/Dialects/)
3. [torch-mlir Project](https://github.com/llvm/torch-mlir)
4. [IREE: Intermediate Representation Execution Environment](https://github.com/iree-org/iree)
5. [MiCo Framework Paper](https://github.com/HKUSTGZ-MICS-LYU/MiCo-python)

---

**Author**: MiCo Development Team  
**Version**: 1.0  
**Date**: January 2026
