"""
Profiler Module

This module provides tools for benchmarking and profiling hardware targets.

Available samplers:
- ProfileSampler: Abstract base class for profile samplers
- MatMulSampler: Sampler for matrix multiplication operations
- Conv2DSampler: Sampler for 2D convolution operations  
- PoolingSampler: Sampler for pooling operations

Sampling strategies:
- 'random': Random sampling with log-scale support
- 'corner': Boundary corner samples
- 'prior': Prior distribution from common DNN configs
- 'lhs': Latin Hypercube Sampling for better coverage
- 'adaptive': Combines corner, prior, and LHS (recommended)

Adaptive Profilers (integrate with MiCoProxy for in-time error feedback):
- AdaptiveProfiler: Abstract base class
- AdaptiveMatMulProfiler: For matmul operations
- AdaptiveConv2DProfiler: For conv2d operations
- AdaptivePoolingProfiler: For pooling operations

Example (Sampler only):
    from profiler import MatMulSampler, Conv2DSampler
    
    sampler = MatMulSampler(
        ranges={'N': [16], 'M': (16, 4096), 'K': (16, 4096)},
        strategy='adaptive'
    )
    samples = sampler.generate(num_samples=100)

Example (Adaptive Profiler with MiCoProxy):
    from profiler import AdaptiveMatMulProfiler
    
    profiler = AdaptiveMatMulProfiler(
        ranges={'N': [16], 'M': (16, 4096), 'K': (16, 4096)},
        benchmark_fn=my_benchmark_function
    )
    dataset = profiler.run(
        init_samples=50,
        iterations=3,
        error_threshold=0.1
    )
    proxy = profiler.get_proxy()  # Get trained LogRandomForest with cbops+
"""

from .sampler import (
    ProfileSampler,
    MatMulSampler,
    Conv2DSampler,
    PoolingSampler,
)

from .adaptive import (
    AdaptiveProfiler,
    AdaptiveMatMulProfiler,
    AdaptiveConv2DProfiler,
    AdaptivePoolingProfiler,
)

__all__ = [
    # Samplers
    'ProfileSampler',
    'MatMulSampler',
    'Conv2DSampler',
    'PoolingSampler',
    # Adaptive Profilers
    'AdaptiveProfiler',
    'AdaptiveMatMulProfiler',
    'AdaptiveConv2DProfiler',
    'AdaptivePoolingProfiler',
]
