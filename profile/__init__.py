"""
Profile Module

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

Example:
    from profile import MatMulSampler, Conv2DSampler
    
    sampler = MatMulSampler(
        ranges={'N': [16], 'M': (16, 4096), 'K': (16, 4096)},
        strategy='adaptive'
    )
    samples = sampler.generate(num_samples=100)
"""

from .sampler import (
    ProfileSampler,
    MatMulSampler,
    Conv2DSampler,
    PoolingSampler,
)

__all__ = [
    'ProfileSampler',
    'MatMulSampler',
    'Conv2DSampler',
    'PoolingSampler',
]
