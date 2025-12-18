#!/usr/bin/env python3
"""
Test script to compare memory usage before and after optimization.
"""

import torch
import sys
sys.path.insert(0, '/home/runner/work/MiCo-python/MiCo-python')

from models import MLP, LeNet, VGG
from MiCoCodeGen import MiCoCodeGen
from MiCoUtils import fuse_model

def test_model_memory(model_class, model_kwargs, input_shape, model_name):
    """Test memory optimization for a given model."""
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print(f"{'='*70}")
    
    # Create model
    model = model_class(**model_kwargs)
    
    # Set quantization scheme
    weight_q = [8] * model.n_layers
    activation_q = [8] * model.n_layers
    model.set_qscheme([weight_q, activation_q])
    model = fuse_model(model)
    model.eval()
    
    # Create example input
    example_input = torch.randn(*input_shape)
    
    # Create code generator
    codegen = MiCoCodeGen(model, align_to=32)
    output = codegen.forward(example_input)
    
    # Calculate memory statistics
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
    
    print(f"\nModel: {model_name}")
    print(f"Input shape: {input_shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nMemory Statistics:")
    print(f"  Uninitialized tensors: {len(uninitialized_tensors)}")
    print(f"  Memory without pooling: {total_without_pooling:,} bytes ({total_without_pooling/1024:.2f} KB)")
    print(f"  Memory with pooling: {total_with_pooling:,} bytes ({total_with_pooling/1024:.2f} KB)")
    print(f"  Memory saved: {saved:,} bytes ({saved/1024:.2f} KB)")
    print(f"  Percentage saved: {percent_saved:.1f}%")
    print(f"  Number of memory pools: {len(memory_pools)}")
    
    # Show pool details
    print(f"\nMemory Pool Details:")
    for pool_id, pool in enumerate(memory_pools):
        print(f"  Pool {pool_id}: {pool['size']:,} bytes ({pool['size']/1024:.2f} KB), {len(pool['tensors'])} tensors")
        # Show which tensors share this pool
        tensor_names = pool['tensors'][:5]  # Show first 5
        if len(pool['tensors']) > 5:
            print(f"    Tensors: {', '.join(tensor_names)}, ... (+{len(pool['tensors'])-5} more)")
        else:
            print(f"    Tensors: {', '.join(tensor_names)}")
    
    return {
        'model_name': model_name,
        'total_without_pooling': total_without_pooling,
        'total_with_pooling': total_with_pooling,
        'saved': saved,
        'percent_saved': percent_saved,
        'num_pools': len(memory_pools),
        'num_tensors': len(uninitialized_tensors)
    }

if __name__ == "__main__":
    torch.manual_seed(0)
    
    results = []
    
    # Test MLP
    results.append(test_model_memory(
        MLP,
        {'in_features': 256, 'config': {"Layers": [64, 64, 64, 10]}},
        (1, 256),
        "MLP (4 layers)"
    ))
    
    # Test LeNet
    results.append(test_model_memory(
        LeNet,
        {'in_channels': 1},
        (1, 1, 28, 28),
        "LeNet (MNIST)"
    ))
    
    # Test VGG (smaller version for CIFAR)
    results.append(test_model_memory(
        VGG,
        {'in_channels': 3, 'num_class': 10},
        (1, 3, 32, 32),
        "VGG (CIFAR-10)"
    ))
    
    # Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'Before (KB)':<15} {'After (KB)':<15} {'Saved (KB)':<15} {'Saved %':<10}")
    print(f"{'-'*70}")
    for r in results:
        print(f"{r['model_name']:<20} "
              f"{r['total_without_pooling']/1024:<15.2f} "
              f"{r['total_with_pooling']/1024:<15.2f} "
              f"{r['saved']/1024:<15.2f} "
              f"{r['percent_saved']:<10.1f}")
    
    print(f"\n{'='*70}")
    print("✓ All models tested successfully!")
    print(f"{'='*70}")
