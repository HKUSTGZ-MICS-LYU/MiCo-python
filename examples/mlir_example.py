#!/usr/bin/env python3
"""
MLIR Code Generation Example

This script demonstrates how to use MiCoMLIRGen to generate MLIR code
from PyTorch models with mixed-precision quantization.

Example usage:
    python examples/mlir_example.py

The script will:
1. Load a LeNet model with mixed-precision quantization
2. Generate MLIR code with the MiCo dialect
3. Save the output to output/lenet_mnist.mlir
"""

import os
import sys
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import MLP, LeNet, VGG
from MiCoMLIRGen import MiCoMLIRGen
from MiCoUtils import fuse_model


def example_mlp():
    """Generate MLIR for a simple MLP model."""
    print("=" * 60)
    print("Example 1: MLP with Mixed Precision")
    print("=" * 60)
    
    # Create MLP model
    model = MLP(in_features=256, config={"Layers": [128, 64, 10]})
    
    # Set mixed precision: different bit widths per layer
    # Weights: [8-bit, 6-bit, 4-bit]
    # Activations: [8-bit, 8-bit, 8-bit]
    weight_bits = [8, 6, 4]
    activation_bits = [8, 8, 8]
    model.set_qscheme([weight_bits, activation_bits])
    
    # Fuse batch normalization if present
    model = fuse_model(model)
    model.eval()
    
    # Create MLIR generator
    mlir_gen = MiCoMLIRGen(model)
    
    # Trace the model with example input
    example_input = torch.randn(1, 256)
    mlir_gen.forward(example_input)
    
    # Generate MLIR code
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    mlir_path = mlir_gen.convert(output_dir, "mlp_example")
    
    print(f"\nGenerated MLIR file: {mlir_path}")
    
    # Print first few lines of the generated code
    with open(mlir_path, 'r') as f:
        lines = f.readlines()
        print("\nPreview of generated MLIR:")
        print("-" * 40)
        for line in lines[:30]:
            print(line, end='')
        if len(lines) > 30:
            print("... (truncated)")
    
    print()
    return mlir_path


def example_lenet():
    """Generate MLIR for LeNet model."""
    print("=" * 60)
    print("Example 2: LeNet with Mixed Precision")
    print("=" * 60)
    
    # Create LeNet model (1 input channel for MNIST)
    model = LeNet(in_channels=1)
    
    # Set mixed precision configuration
    # LeNet has 5 quantizable layers: conv1, conv2, fc1, fc2, fc3
    # Using progressively lower precision for later layers
    weight_bits = [8, 6, 6, 4, 4]
    activation_bits = [8, 8, 8, 8, 8]
    model.set_qscheme([weight_bits, activation_bits])
    
    # Fuse batch normalization
    model = fuse_model(model)
    model.eval()
    
    # Create MLIR generator
    mlir_gen = MiCoMLIRGen(model)
    
    # Trace with 28x28 input (MNIST dimensions)
    example_input = torch.randn(1, 1, 28, 28)
    mlir_gen.forward(example_input)
    
    # Generate MLIR code
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    mlir_path = mlir_gen.convert(output_dir, "lenet_mnist_mlir")
    
    print(f"\nGenerated MLIR file: {mlir_path}")
    
    # Print the generated code
    with open(mlir_path, 'r') as f:
        content = f.read()
        print("\nGenerated MLIR code:")
        print("-" * 40)
        print(content)
    
    print()
    return mlir_path


def example_vgg_partial():
    """Generate MLIR for VGG (partial - first few layers)."""
    print("=" * 60)
    print("Example 3: VGG with Mixed Precision")
    print("=" * 60)
    
    # Create VGG model for CIFAR-10
    model = VGG(in_channels=3, num_class=10)
    
    # Set uniform 8-bit precision (can be modified for MPQ)
    n_layers = model.n_layers
    weight_bits = [8] * n_layers
    activation_bits = [8] * n_layers
    model.set_qscheme([weight_bits, activation_bits])
    
    # Fuse batch normalization
    model = fuse_model(model)
    model.eval()
    
    # Create MLIR generator
    mlir_gen = MiCoMLIRGen(model)
    
    # Trace with CIFAR-10 dimensions (32x32)
    example_input = torch.randn(1, 3, 32, 32)
    mlir_gen.forward(example_input)
    
    # Generate MLIR code
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    mlir_path = mlir_gen.convert(output_dir, "vgg_cifar10_mlir")
    
    print(f"\nGenerated MLIR file: {mlir_path}")
    print(f"Number of operations: {len(mlir_gen.mlir_ops)}")
    print(f"Number of weights: {len(mlir_gen.mlir_weights)}")
    
    print()
    return mlir_path


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("MiCo MLIR Code Generation Examples")
    print("=" * 60 + "\n")
    
    # Run examples
    mlp_path = example_mlp()
    lenet_path = example_lenet()
    
    # VGG example (optional, can be slow)
    try:
        vgg_path = example_vgg_partial()
    except Exception as e:
        print(f"VGG example skipped due to: {e}")
        vgg_path = None
    
    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print("\nGenerated MLIR files:")
    print(f"  - MLP:   {mlp_path}")
    print(f"  - LeNet: {lenet_path}")
    if vgg_path:
        print(f"  - VGG:   {vgg_path}")
    
    print("\nThe generated MLIR files use the MiCo dialect which supports:")
    print("  - Sub-byte integer types (!mico.int<N>)")
    print("  - Quantized operations (mico.bitlinear, mico.bitconv2d)")
    print("  - Standard neural network operations (mico.relu, mico.maxpool2d)")
    print("\nSee doc/MLIR_INTEGRATION.md for dialect specification and usage.")
    
    print("\n" + "=" * 60)
    print("Examples Complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
