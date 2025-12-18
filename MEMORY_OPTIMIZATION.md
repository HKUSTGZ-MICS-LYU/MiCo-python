# Memory Optimization in MiCoCodeGen

## Overview

The MiCoCodeGen now includes an ahead-of-time memory allocation optimization that significantly reduces runtime memory usage by reusing buffers for tensors with non-overlapping lifetimes.

## Problem Statement

Previously, MiCoCodeGen would allocate separate memory for every layer's output using individual `malloc` calls:

```c
// Old approach - each tensor gets its own malloc
model->layer1_output.data = (float *)malloc(1024);
model->layer2_output.data = (float *)malloc(1024);
model->layer3_output.data = (float *)malloc(1024);
// ... total: 3072 bytes allocated
```

This approach is wasteful because:
1. Intermediate layer outputs are only needed temporarily
2. Once a layer's output is consumed, its memory buffer is no longer needed
3. Multiple layers could share the same memory buffer if their lifetimes don't overlap

## Solution

The new implementation:
1. **Analyzes tensor lifetimes**: Determines when each tensor is created and last used
2. **Allocates memory pools**: Groups tensors with non-overlapping lifetimes into shared pools
3. **Generates optimized code**: Uses pre-allocated pools instead of individual mallocs

```c
// New approach - tensors with non-overlapping lifetimes share pools
model->memory_pool_0 = (float *)malloc(1024);  // Shared by multiple tensors
model->memory_pool_1 = (float *)malloc(256);   // Shared by multiple tensors
model->layer1_output.data = model->memory_pool_0;
model->layer2_output.data = model->memory_pool_1;
model->layer3_output.data = model->memory_pool_0;  // Reuses pool 0
// ... total: 1280 bytes allocated (58% savings!)
```

## Implementation Details

### 1. Tensor Lifetime Analysis

The `compute_tensor_lifetimes()` method analyzes the computation graph to determine:
- **First use**: When a tensor is created
- **Last use**: When a tensor is no longer needed

Example timeline for a simple model:
```
Time:  0    1    2    3    4    5    6    7    8
       x → layer1 → relu1 → layer2 → relu2 → output
```

Lifetimes:
- `x`: [0, 1] - Created at time 0, last used at time 1
- `layer1`: [1, 2] - Created at time 1, last used at time 2
- `relu1`: [2, 3] - Created at time 2, last used at time 3
- etc.

### 2. Memory Pool Allocation

The `allocate_memory_pools()` method uses a greedy interval-based algorithm:

1. Sort tensors by size (descending) for better packing
2. For each tensor:
   - Try to find an existing pool where the tensor's lifetime doesn't overlap with any tensor already in that pool
   - If found, add tensor to that pool and update pool size if needed
   - If not found, create a new pool

### 3. Code Generation

The `convert()` method generates C code with:
- Memory pool declarations in the model struct
- Pool allocations in `model_init()`
- Tensor data pointers assigned to pools

## Results

Memory savings achieved on various models:

| Model | Before | After | Saved | Savings % |
|-------|--------|-------|-------|-----------|
| MLP (4 layers) | 2.58 KB | 1.25 KB | 1.33 KB | 51.5% |
| LeNet (MNIST) | 59.30 KB | 41.34 KB | 17.95 KB | 30.3% |
| VGG (CIFAR-10) | 1212.08 KB | 576.00 KB | 636.08 KB | 52.5% |

### Example: MLP Memory Pool Usage

For a 4-layer MLP:
- **Pool 0** (1024 bytes): Shared by 5 tensors
  - `x` (input)
  - `layer1_output`
  - `layer3_output`
  - `layer5_output`
  - `layer7_output`
  
- **Pool 1** (256 bytes): Shared by 4 tensors
  - `layer2_output`
  - `layer4_output`
  - `layer6_output`
  - `output`

These tensors can share pools because their lifetimes don't overlap - by the time `layer3_output` is computed, `x` and `layer1_output` are no longer needed.

## Usage

The optimization is automatic. Just use MiCoCodeGen as before:

```python
from MiCoCodeGen import MiCoCodeGen
from models import LeNet
from MiCoUtils import fuse_model

# Create and prepare model
model = LeNet(1)
model.set_qscheme([[8]*model.n_layers, [8]*model.n_layers])
model = fuse_model(model)
model.eval()

# Generate optimized C code
codegen = MiCoCodeGen(model, align_to=32)
codegen.forward(torch.randn(1, 1, 28, 28))
codegen.convert("output_dir", "model_name")
```

The generated code will automatically use memory pools with optimal memory usage.

## Testing

To verify the optimization on your own models:

```bash
python test_memory_optimization.py  # Run basic tests
python test_memory_stats.py         # See detailed statistics
```

## Technical Notes

### Why offset is always 0

In the current implementation, `tensor_to_pool` maps tensors to `(pool_id, offset)` where offset is always 0. This is correct because:

1. Tensors with non-overlapping lifetimes can reuse the **entire** pool buffer
2. There's no need for offsets within the pool because tensors never coexist
3. Each pool's size is set to the maximum size of all tensors using it

Example:
```
Pool 0 (size: 1024 bytes)
  Time 0-1: tensor A uses pool [0:512]    → uses full pool capacity
  Time 2-3: tensor B uses pool [0:1024]   → uses full pool capacity
  Time 4-5: tensor C uses pool [0:256]    → uses full pool capacity
```

### Future Enhancements

Possible future improvements:
1. Support for offsets within pools (for partially overlapping tensors)
2. Better packing algorithms (e.g., optimal interval scheduling)
3. Pool size alignment optimizations
4. Support for different data types in the same pool

## References

- See `MiCoCodeGen.py` for implementation details
- See `test_memory_optimization.py` for usage examples
- See `test_memory_stats.py` for benchmarking code
