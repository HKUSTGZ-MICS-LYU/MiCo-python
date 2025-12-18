#!/usr/bin/env python3
"""
Test memory optimization on non-sequential models like ResNet8.
ResNet has skip connections which create more complex dependency patterns.
"""

import torch
import sys
import os

sys.path.insert(0, '/home/runner/work/MiCo-python/MiCo-python')

from models import resnet_alt_8
from MiCoCodeGen import MiCoCodeGen
from MiCoUtils import fuse_model

def test_resnet8_memory_optimization():
    """Test memory optimization with ResNet8 (non-sequential model with skip connections)."""
    print("\n" + "="*70)
    print("Testing Memory Optimization with ResNet8 (Non-Sequential Model)")
    print("="*70)
    
    # Create ResNet8 model for CIFAR-10
    model = resnet_alt_8(n_class=10)
    model.default_dataset = "CIFAR10"
    
    # Set quantization scheme
    weight_q = [8] * model.n_layers
    activation_q = [8] * model.n_layers
    model.set_qscheme([weight_q, activation_q])
    
    # Fuse batch normalization
    model = fuse_model(model)
    model.eval()
    
    # Create example input (CIFAR-10: 3x32x32)
    example_input = torch.randn(1, 3, 32, 32)
    
    # Create code generator
    codegen = MiCoCodeGen(model, align_to=32)
    
    # Run forward pass to trace the model
    output = codegen.forward(example_input)
    
    print(f"\nModel traced successfully.")
    print(f"Input shape: {example_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Analyze memory statistics
    memory_pools, tensor_to_pool = codegen.allocate_memory_pools()
    lifetimes = codegen.compute_tensor_lifetimes()
    
    # Calculate total memory without pooling
    total_without_pooling = 0
    uninitialized_tensors = []
    for name, tensor_dict in codegen.tensors.items():
        if not tensor_dict.get("initialized", False) and tensor_dict["tensor"] is not None:
            size = tensor_dict["tensor"].nelement() * tensor_dict["tensor"].element_size()
            total_without_pooling += size
            uninitialized_tensors.append((name, size))
    
    # Calculate total memory with pooling
    total_with_pooling = sum(pool['size'] for pool in memory_pools)
    
    # Calculate savings
    saved = total_without_pooling - total_with_pooling
    percent_saved = 100.0 * saved / total_without_pooling if total_without_pooling > 0 else 0
    
    print(f"\nMemory Statistics:")
    print(f"  Model: ResNet8 (non-sequential with skip connections)")
    print(f"  Uninitialized tensors: {len(uninitialized_tensors)}")
    print(f"  Memory without pooling: {total_without_pooling:,} bytes ({total_without_pooling/1024:.2f} KB)")
    print(f"  Memory with pooling: {total_with_pooling:,} bytes ({total_with_pooling/1024:.2f} KB)")
    print(f"  Memory saved: {saved:,} bytes ({saved/1024:.2f} KB)")
    print(f"  Percentage saved: {percent_saved:.1f}%")
    print(f"  Number of memory pools: {len(memory_pools)}")
    
    # Verify no overlapping lifetimes in pools
    print(f"\nVerifying Memory Pool Allocation:")
    all_valid = True
    for pool_id, pool in enumerate(memory_pools):
        print(f"  Pool {pool_id}: {pool['size']:,} bytes ({pool['size']/1024:.2f} KB), {len(pool['tensors'])} tensors")
        
        # Check for overlaps (should not exist)
        intervals = pool['intervals']
        for i, (name1, start1, end1) in enumerate(intervals):
            for j, (name2, start2, end2) in enumerate(intervals):
                if i < j:
                    # Check if intervals overlap
                    if not (end1 < start2 or start1 > end2):
                        print(f"    ✗ WARNING: Overlap detected between {name1} [{start1},{end1}] and {name2} [{start2},{end2}]!")
                        all_valid = False
    
    if all_valid:
        print("\n✓ All memory pools verified: No overlapping lifetimes!")
    else:
        print("\n✗ Memory pool verification FAILED: Overlapping lifetimes detected!")
        return False
    
    # Generate C code
    output_dir = "/tmp/mico_test_resnet8"
    os.makedirs(output_dir, exist_ok=True)
    codegen.convert(output_dir, "model_resnet8", verbose=False)
    
    print(f"\nGenerated C code in: {output_dir}")
    
    # Verify generated code has memory pools
    with open(f"{output_dir}/model_resnet8.h", "r") as f:
        content = f.read()
        pool_count = content.count("memory_pool_")
        print(f"Memory pools in generated C code: {pool_count // 2}")  # Divide by 2 because each pool appears twice (declaration + allocation)
    
    print("\n" + "="*70)
    print("✓ ResNet8 (non-sequential model) test PASSED!")
    print("="*70)
    
    return True

if __name__ == "__main__":
    torch.manual_seed(0)
    
    try:
        success = test_resnet8_memory_optimization()
        if success:
            print("\nTest completed successfully!")
            sys.exit(0)
        else:
            print("\nTest failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
