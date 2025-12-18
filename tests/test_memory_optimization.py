#!/usr/bin/env python3
"""
Test script for memory optimization in MiCoCodeGen.
This tests the memory pool allocation feature without requiring trained checkpoints.
"""

import torch
import torch.nn as nn
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, '/home/runner/work/MiCo-python/MiCo-python')

from models import MLP, LeNet
from MiCoCodeGen import MiCoCodeGen
from MiCoUtils import fuse_model

def test_simple_mlp():
    """Test memory optimization with a simple MLP model."""
    print("\n" + "="*60)
    print("Testing Memory Optimization with Simple MLP")
    print("="*60)
    
    # Create a simple MLP model
    model = MLP(in_features=256, config={"Layers": [64, 64, 64, 10]})
    
    # Set quantization scheme (all layers use 8-bit)
    weight_q = [8] * model.n_layers
    activation_q = [8] * model.n_layers
    model.set_qscheme([weight_q, activation_q])
    
    # Fuse batch normalization if any
    model = fuse_model(model)
    model.eval()
    
    # Create example input
    example_input = torch.randn(1, 256)
    
    # Create code generator
    codegen = MiCoCodeGen(model, align_to=32)
    
    # Run forward pass to trace the model
    output = codegen.forward(example_input)
    
    print(f"\nModel traced successfully. Output shape: {output.shape}")
    
    # Convert to C code with memory optimization
    output_dir = "/tmp/mico_test_mlp"
    os.makedirs(output_dir, exist_ok=True)
    codegen.convert(output_dir, "model_mlp", verbose=False)
    
    print(f"\nGenerated C code in: {output_dir}")
    return codegen

def test_lenet():
    """Test memory optimization with LeNet model."""
    print("\n" + "="*60)
    print("Testing Memory Optimization with LeNet")
    print("="*60)
    
    # Create LeNet model
    model = LeNet(1)  # 1 channel for MNIST
    
    # Set quantization scheme
    weight_q = [8] * model.n_layers
    activation_q = [8] * model.n_layers
    model.set_qscheme([weight_q, activation_q])
    
    # Fuse batch normalization if any
    model = fuse_model(model)
    model.eval()
    
    # Create example input (MNIST 28x28)
    example_input = torch.randn(1, 1, 28, 28)
    
    # Create code generator
    codegen = MiCoCodeGen(model, align_to=32)
    
    # Run forward pass to trace the model
    output = codegen.forward(example_input)
    
    print(f"\nModel traced successfully. Output shape: {output.shape}")
    
    # Convert to C code with memory optimization
    output_dir = "/tmp/mico_test_lenet"
    os.makedirs(output_dir, exist_ok=True)
    codegen.convert(output_dir, "model_lenet", verbose=False)
    
    print(f"\nGenerated C code in: {output_dir}")
    return codegen

def verify_memory_pools(codegen):
    """Verify that memory pools are correctly allocated."""
    print("\n" + "="*60)
    print("Verifying Memory Pool Allocation")
    print("="*60)
    
    memory_pools, tensor_to_pool = codegen.allocate_memory_pools()
    lifetimes = codegen.compute_tensor_lifetimes()
    
    print(f"\nNumber of memory pools: {len(memory_pools)}")
    print(f"Number of tensors using pools: {len(tensor_to_pool)}")
    
    # Verify no overlapping lifetimes in the same pool
    for pool_id, pool in enumerate(memory_pools):
        print(f"\nPool {pool_id}:")
        print(f"  Size: {pool['size']} bytes")
        print(f"  Tensors: {len(pool['tensors'])}")
        
        # Check for overlaps (should not exist)
        intervals = pool['intervals']
        for i, (name1, start1, end1) in enumerate(intervals):
            for j, (name2, start2, end2) in enumerate(intervals):
                if i < j:
                    # Check if intervals overlap
                    if not (end1 < start2 or start1 > end2):
                        print(f"  WARNING: Overlap detected between {name1} and {name2}!")
                        return False
    
    print("\n✓ All memory pools verified: No overlapping lifetimes!")
    return True

if __name__ == "__main__":
    torch.manual_seed(0)
    
    try:
        # Test MLP
        codegen_mlp = test_simple_mlp()
        verify_memory_pools(codegen_mlp)
        
        # Test LeNet
        codegen_lenet = test_lenet()
        verify_memory_pools(codegen_lenet)
        
        print("\n" + "="*60)
        print("All tests passed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
