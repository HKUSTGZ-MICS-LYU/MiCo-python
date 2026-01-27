#!/usr/bin/env python3
"""
Torch-MLIR Code Generation Example

This script demonstrates how to use MiCoTorchMLIRGen to generate MLIR code
from PyTorch models using torch-mlir as the first pass.

Example usage:
    python examples/torch_mlir_example.py

The script will:
1. Check if torch-mlir is installed
2. Load a LeNet model with mixed-precision quantization
3. Generate MLIR code using torch-mlir backend (or fallback to standalone)
4. Save the output to output/lenet_torch_mlir.mlir
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models import MLP, LeNet, VGG
from MiCoTorchMLIRGen import MiCoTorchMLIRGen, check_torch_mlir_installation
from MiCoUtils import fuse_model


def check_installation():
    """Check and report torch-mlir installation status."""
    print("=" * 60)
    print("Torch-MLIR Installation Status")
    print("=" * 60)
    
    status = check_torch_mlir_installation()
    
    print(f"Available: {status['available']}")
    if status['available']:
        print(f"Version: {status['version']}")
        print(f"Supported output types: {', '.join(status['output_types'])}")
    else:
        print(f"\nTo install torch-mlir:")
        print(f"  {status['install_command']}")
    
    print()
    return status['available']


def example_mlp():
    """Generate MLIR for a simple MLP model."""
    print("=" * 60)
    print("Example 1: MLP with Torch-MLIR Backend")
    print("=" * 60)
    
    # Create MLP model
    model = MLP(in_features=256, config={"Layers": [128, 64, 10]})
    
    # Set mixed precision
    weight_bits = [8, 6, 4]
    activation_bits = [8, 8, 8]
    model.set_qscheme([weight_bits, activation_bits])
    
    model = fuse_model(model)
    model.eval()
    
    # Create torch-mlir generator
    mlir_gen = MiCoTorchMLIRGen(model, output_type="torch")
    
    # Trace the model
    example_input = torch.randn(1, 256)
    mlir_gen.forward(example_input)
    
    # Generate MLIR code
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    mlir_path = mlir_gen.convert(output_dir, "mlp_torch_mlir")
    
    print(f"\nGenerated MLIR file: {mlir_path}")
    print(f"Backend used: {'torch-mlir' if mlir_gen.use_torch_mlir else 'standalone (fallback)'}")
    
    # Print first few lines
    with open(mlir_path, 'r') as f:
        lines = f.readlines()
        print("\nPreview of generated MLIR:")
        print("-" * 40)
        for line in lines[:25]:
            print(line, end='')
        if len(lines) > 25:
            print("... (truncated)")
    
    print()
    return mlir_path


def example_lenet():
    """Generate MLIR for LeNet model."""
    print("=" * 60)
    print("Example 2: LeNet with Torch-MLIR Backend")
    print("=" * 60)
    
    # Create LeNet model
    model = LeNet(in_channels=1)
    
    # Set mixed precision
    weight_bits = [8, 6, 6, 4, 4]
    activation_bits = [8, 8, 8, 8, 8]
    model.set_qscheme([weight_bits, activation_bits])
    
    model = fuse_model(model)
    model.eval()
    
    # Create torch-mlir generator with linalg output
    mlir_gen = MiCoTorchMLIRGen(model, output_type="torch")
    
    # Trace with MNIST dimensions
    example_input = torch.randn(1, 1, 28, 28)
    mlir_gen.forward(example_input)
    
    # Generate MLIR code
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    mlir_path = mlir_gen.convert(output_dir, "lenet_torch_mlir")
    
    print(f"\nGenerated MLIR file: {mlir_path}")
    print(f"Backend used: {'torch-mlir' if mlir_gen.use_torch_mlir else 'standalone (fallback)'}")
    
    print()
    return mlir_path


def example_compare_backends():
    """Compare torch-mlir and standalone backend outputs."""
    print("=" * 60)
    print("Example 3: Compare Backends")
    print("=" * 60)
    
    # Create model
    model = MLP(in_features=64, config={"Layers": [32, 10]})
    model.set_qscheme([[8, 8], [8, 8]])
    model = fuse_model(model)
    model.eval()
    
    example_input = torch.randn(1, 64)
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate with torch-mlir backend (reusing the imported class from line 26)
    torch_mlir_gen = MiCoTorchMLIRGen(model, output_type="torch")
    torch_mlir_gen.forward(example_input)
    torch_mlir_path = torch_mlir_gen.convert(output_dir, "compare_torch_mlir")
    torch_mlir_used = torch_mlir_gen.use_torch_mlir
    
    # Generate with standalone backend
    from MiCoMLIRGen import MiCoMLIRGen
    standalone_gen = MiCoMLIRGen(model)
    standalone_gen.forward(example_input)
    standalone_path = standalone_gen.convert(output_dir, "compare_standalone")
    
    print(f"\nTorch-MLIR output: {torch_mlir_path}")
    print(f"  Backend actually used: {'torch-mlir' if torch_mlir_used else 'standalone (fallback)'}")
    print(f"\nStandalone output: {standalone_path}")
    
    # Get file sizes
    torch_mlir_size = os.path.getsize(torch_mlir_path)
    standalone_size = os.path.getsize(standalone_path)
    
    print(f"\nFile sizes:")
    print(f"  Torch-MLIR: {torch_mlir_size} bytes")
    print(f"  Standalone: {standalone_size} bytes")
    
    print()
    return torch_mlir_path, standalone_path


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("MiCo Torch-MLIR Code Generation Examples")
    print("=" * 60 + "\n")
    
    # Check installation
    is_available = check_installation()
    
    if not is_available:
        print("Note: torch-mlir not installed. Examples will use standalone fallback.")
        print()
    
    # Run examples
    mlp_path = example_mlp()
    lenet_path = example_lenet()
    torch_mlir_path, standalone_path = example_compare_backends()
    
    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print("\nGenerated MLIR files:")
    print(f"  - MLP:        {mlp_path}")
    print(f"  - LeNet:      {lenet_path}")
    print(f"  - Comparison: {torch_mlir_path}")
    print(f"                {standalone_path}")
    
    print("\nThe torch-mlir backend provides:")
    print("  - Production-ready PyTorch → MLIR lowering")
    print("  - Torch dialect output compatible with IREE, Blade, etc.")
    print("  - MiCo dialect overlay for sub-byte quantization metadata")
    
    if not is_available:
        print("\nTo enable torch-mlir backend:")
        print("  pip install torch-mlir -f https://github.com/llvm/torch-mlir-release/releases")
    
    print("\n" + "=" * 60)
    print("Examples Complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
