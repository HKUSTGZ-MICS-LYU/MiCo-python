"""
Profile Sampler Module

This module provides reusable sampling strategies for hardware profiling,
inspired by nn-Meter's adaptive data sampling approach.

Supported sampling strategies:
- Random sampling with uniform or log-scale distribution
- Corner sampling for boundary coverage
- Prior distribution sampling from common DNN configurations
- Fine-grained adaptive sampling for error regions
- Latin Hypercube Sampling (LHS) for better coverage

Usage:
    from profiler.sampler import MatMulSampler, Conv2DSampler

    sampler = MatMulSampler(
        ranges={'N': [16], 'M': (16, 4096), 'K': (16, 4096)},
        strategy='adaptive'
    )
    samples = sampler.generate(num_samples=100)
"""

import random
import math
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Set, Any, Optional
from tqdm import tqdm
from itertools import product

# Type alias for range specification
RangeSpec = Union[int, List[int], Tuple[int, int]]


class ProfileSampler(ABC):
    """
    Abstract base class for profile samplers.
    
    Provides reusable sampling strategies for hardware profiling benchmarks.
    Subclasses implement kernel-specific sample generation logic.
    
    Attributes:
        ranges: Dictionary mapping parameter names to their value ranges.
                Each range can be:
                - int: Fixed value
                - List[int]: Discrete choices
                - Tuple[int, int]: Continuous range (min, max)
        strategy: Sampling strategy to use ('random', 'corner', 'prior', 
                  'adaptive', 'lhs')
        log_scale_params: Set of parameter names to sample in log scale
    """
    
    # Common DNN configuration priors (based on nn-Meter and common architectures)
    PRIOR_HW = [1, 3, 7, 14, 28, 56, 112, 224]
    PRIOR_CHANNELS = [1, 3, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    PRIOR_KERNEL_SIZES = [1, 3, 5, 7]
    PRIOR_STRIDES = [1, 2, 4]
    PRIOR_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
    
    def __init__(
        self,
        ranges: Dict[str, RangeSpec],
        strategy: str = 'adaptive',
        log_scale_params: Optional[Set[str]] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the sampler.
        
        Args:
            ranges: Dictionary of parameter ranges
            strategy: Sampling strategy ('random', 'corner', 'prior', 
                      'adaptive', 'lhs')
            log_scale_params: Set of parameter names to sample in log scale
            seed: Random seed for reproducibility
        """
        self.ranges = ranges
        self.strategy = strategy
        self.log_scale_params = log_scale_params or set()
        
        if seed is not None:
            random.seed(seed)
    
    @staticmethod
    def get_random_val(r: RangeSpec, log_scale: bool = False) -> int:
        """
        Get a random value from a range specification.
        
        Args:
            r: Range specification (int, list, or tuple)
            log_scale: If True, sample in log scale for tuple ranges
            
        Returns:
            Random integer value from the range
        """
        if isinstance(r, list):
            return random.choice(r)
        elif isinstance(r, tuple):
            if log_scale and r[0] > 0:
                val = math.exp(random.uniform(math.log(r[0]), math.log(r[1])))
            else:
                val = random.uniform(r[0], r[1])
            return int(round(val))
        else:
            return r
    
    @staticmethod
    def get_corner_values(r: RangeSpec) -> List[int]:
        """
        Get corner (boundary) values from a range specification.
        
        Args:
            r: Range specification
            
        Returns:
            List of corner values
        """
        if isinstance(r, list):
            return [r[0], r[-1]] if len(r) > 1 else r
        elif isinstance(r, tuple):
            return [r[0], r[1]]
        else:
            return [r]
    
    @staticmethod
    def get_prior_values(r: RangeSpec, priors: List[int]) -> List[int]:
        """
        Get values from prior distribution within the given range.
        
        Args:
            r: Range specification
            priors: List of prior values to filter
            
        Returns:
            List of prior values within range
        """
        if isinstance(r, list):
            return r
        elif isinstance(r, tuple):
            return [p for p in priors if r[0] <= p <= r[1]]
        else:
            return [r]
    
    @abstractmethod
    def _generate_sample(self) -> Tuple:
        """
        Generate a single sample.
        
        Returns:
            Tuple of parameter values
        """
        pass
    
    @abstractmethod
    def _get_corner_samples(self) -> Set[Tuple]:
        """
        Generate corner samples for boundary coverage.
        
        Returns:
            Set of corner sample tuples
        """
        pass
    
    @abstractmethod
    def _get_prior_samples(self) -> Set[Tuple]:
        """
        Generate samples from prior distribution.
        
        Returns:
            Set of prior sample tuples
        """
        pass
    
    def _validate_sample(self, sample: Tuple) -> bool:
        """
        Validate a sample. Override in subclass for kernel-specific constraints.
        
        Args:
            sample: Sample tuple to validate
            
        Returns:
            True if sample is valid
        """
        return True
    
    def _generate_lhs_samples(self, num_samples: int) -> Set[Tuple]:
        """
        Generate samples using Latin Hypercube Sampling.
        
        LHS ensures better coverage of the parameter space by dividing
        each dimension into equal intervals and ensuring one sample per interval.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Set of sample tuples
        """
        samples = set()
        param_names = list(self.ranges.keys())
        n_params = len(param_names)
        
        # Generate LHS indices for each parameter
        lhs_indices = []
        for _ in range(n_params):
            indices = list(range(num_samples))
            random.shuffle(indices)
            lhs_indices.append(indices)
        
        for i in range(num_samples):
            values = []
            for j, param in enumerate(param_names):
                r = self.ranges[param]
                idx = lhs_indices[j][i]
                
                if isinstance(r, list):
                    # Map LHS index to discrete choice
                    choice_idx = idx * len(r) // num_samples
                    values.append(r[min(choice_idx, len(r) - 1)])
                elif isinstance(r, tuple):
                    # Map LHS index to continuous range
                    lo, hi = r[0], r[1]
                    use_log = param in self.log_scale_params and lo > 0
                    
                    # Add small random jitter within the stratum
                    fraction = (idx + random.random()) / num_samples
                    
                    if use_log:
                        val = math.exp(
                            math.log(lo) + fraction * (math.log(hi) - math.log(lo))
                        )
                    else:
                        val = lo + fraction * (hi - lo)
                    values.append(int(round(val)))
                else:
                    values.append(r)
            
            sample = tuple(values)
            if self._validate_sample(sample):
                samples.add(sample)
        
        return samples
    
    def _generate_adaptive_samples(
        self,
        num_samples: int,
        error_samples: Optional[List[Tuple]] = None,
        fine_grained_num: int = 5,
        error_range_factor: float = 0.4
    ) -> Set[Tuple]:
        """
        Generate samples using adaptive sampling strategy.
        
        Combines corner sampling, prior sampling, and fine-grained sampling
        around error regions (if provided).
        
        Args:
            num_samples: Target number of samples
            error_samples: Samples with high prediction error (for refinement)
            fine_grained_num: Number of fine-grained samples per error sample
            error_range_factor: Range factor for fine-grained sampling
            
        Returns:
            Set of sample tuples
        """
        samples = set()
        
        # 1. Add corner samples for boundary coverage
        corner_samples = self._get_corner_samples()
        samples.update(corner_samples)
        
        # 2. Add prior samples from common configurations
        prior_samples = self._get_prior_samples()
        # Limit prior samples to avoid overwhelming the dataset
        prior_list = list(prior_samples)
        if len(prior_list) > num_samples // 3:
            prior_list = random.sample(prior_list, num_samples // 3)
        samples.update(prior_list)
        
        # 3. Add fine-grained samples around error regions
        if error_samples:
            for error_sample in error_samples:
                fine_samples = self._generate_fine_grained_samples(
                    error_sample, fine_grained_num, error_range_factor
                )
                samples.update(fine_samples)
        
        # 4. Fill remaining with LHS samples for coverage
        remaining = num_samples - len(samples)
        if remaining > 0:
            lhs_samples = self._generate_lhs_samples(remaining * 2)
            for sample in lhs_samples:
                if len(samples) >= num_samples:
                    break
                if sample not in samples:
                    samples.add(sample)
        
        return samples
    
    def _generate_fine_grained_samples(
        self,
        center_sample: Tuple,
        num_samples: int,
        range_factor: float = 0.4
    ) -> Set[Tuple]:
        """
        Generate fine-grained samples around a center point.
        
        Used for adaptive sampling to refine predictions in high-error regions.
        
        Args:
            center_sample: Center sample to sample around
            num_samples: Number of samples to generate
            range_factor: Factor to determine sampling range (e.g., 0.4 means
                          sample from [0.6*val, 1.4*val] for each dimension)
            
        Returns:
            Set of fine-grained sample tuples
        """
        samples = set()
        param_names = list(self.ranges.keys())
        
        for _ in range(num_samples * 3):  # Over-generate to handle duplicates
            if len(samples) >= num_samples:
                break
                
            values = []
            for i, (param, center_val) in enumerate(
                zip(param_names, center_sample)
            ):
                r = self.ranges[param]
                
                if isinstance(r, list):
                    # For discrete choices, pick nearby values
                    try:
                        idx = r.index(center_val)
                        delta = random.randint(-1, 1)
                        new_idx = max(0, min(len(r) - 1, idx + delta))
                        values.append(r[new_idx])
                    except ValueError:
                        values.append(random.choice(r))
                elif isinstance(r, tuple):
                    # Sample from [lo, hi] where lo = (1-rf)*val, hi = (1+rf)*val
                    lo = max(r[0], int(center_val * (1 - range_factor)))
                    hi = min(r[1], int(center_val * (1 + range_factor)))
                    
                    # Ensure valid range (lo < hi)
                    if lo >= hi:
                        lo = r[0]
                        hi = r[1]
                    
                    use_log = param in self.log_scale_params and lo > 0 and hi > 0
                    if use_log:
                        val = math.exp(
                            random.uniform(math.log(max(1, lo)), math.log(max(1, hi)))
                        )
                    else:
                        val = random.uniform(lo, hi)
                    values.append(int(round(val)))
                else:
                    values.append(r)
            
            sample = tuple(values)
            if self._validate_sample(sample):
                samples.add(sample)
        
        return samples
    
    def generate(
        self,
        num_samples: int,
        error_samples: Optional[List[Tuple]] = None,
        fine_grained_num: int = 5,
        show_progress: bool = True
    ) -> List[Tuple]:
        """
        Generate samples using the configured strategy.
        
        Args:
            num_samples: Number of samples to generate
            error_samples: Optional list of error samples for adaptive refinement
            fine_grained_num: Number of fine-grained samples per error sample
            show_progress: Whether to show progress bar
            
        Returns:
            List of sample tuples
        """
        samples: Set[Tuple] = set()
        
        if self.strategy == 'corner':
            samples = self._get_corner_samples()
            
        elif self.strategy == 'prior':
            samples = self._get_prior_samples()
            
        elif self.strategy == 'lhs':
            samples = self._generate_lhs_samples(num_samples)
            
        elif self.strategy == 'adaptive':
            samples = self._generate_adaptive_samples(
                num_samples, error_samples, fine_grained_num=fine_grained_num
            )
            
        else:  # 'random' or default
            # Add corners for representativeness
            corner_samples = self._get_corner_samples()
            samples.update(corner_samples)
            
            # Generate random samples
            pbar = tqdm(
                total=num_samples,
                desc=f"Generating {self.__class__.__name__} Samples",
                disable=not show_progress
            )
            pbar.update(len(samples))
            
            attempts = 0
            max_attempts = num_samples * 10
            
            while len(samples) < num_samples and attempts < max_attempts:
                sample = self._generate_sample()
                if sample not in samples and self._validate_sample(sample):
                    samples.add(sample)
                    pbar.update(1)
                attempts += 1
            
            pbar.close()
        
        return list(samples)


class MatMulSampler(ProfileSampler):
    """
    Sampler for matrix multiplication operations.
    
    Parameters:
        N: Batch size / first dimension
        M: Output dimension
        K: Reduction dimension
    """
    
    def __init__(
        self,
        ranges: Optional[Dict[str, RangeSpec]] = None,
        strategy: str = 'adaptive',
        seed: Optional[int] = None
    ):
        # Default ranges for MatMul
        default_ranges = {
            'N': [16],
            'M': (16, 4096),
            'K': (16, 4096)
        }
        ranges = ranges or default_ranges
        
        # M and K typically benefit from log-scale sampling
        log_scale_params = {'M', 'K'}
        
        super().__init__(
            ranges=ranges,
            strategy=strategy,
            log_scale_params=log_scale_params,
            seed=seed
        )
    
    def _generate_sample(self) -> Tuple[int, int, int]:
        """Generate a single MatMul sample (N, M, K)."""
        N = self.get_random_val(self.ranges['N'])
        M = self.get_random_val(self.ranges['M'], log_scale=True)
        K = self.get_random_val(self.ranges['K'], log_scale=True)
        return (N, M, K)
    
    def _get_corner_samples(self) -> Set[Tuple[int, int, int]]:
        """Generate corner samples for MatMul."""
        samples = set()
        
        n_values = self.get_corner_values(self.ranges['N'])
        m_values = self.get_corner_values(self.ranges['M'])
        k_values = self.get_corner_values(self.ranges['K'])
        
        for n, m, k in product(n_values, m_values, k_values):
                    samples.add((n, m, k))
        
        return samples
    
    def _get_prior_samples(self) -> Set[Tuple[int, int, int]]:
        """Generate samples from prior distribution."""
        samples = set()
        
        n_priors = self.get_prior_values(
            self.ranges['N'], self.PRIOR_BATCH_SIZES
        )
        m_priors = self.get_prior_values(
            self.ranges['M'], self.PRIOR_CHANNELS
        )
        k_priors = self.get_prior_values(
            self.ranges['K'], self.PRIOR_CHANNELS
        )
        for n, m, k in product(n_priors, m_priors, k_priors):
            samples.add((n, m, k))
        return samples


class Conv2DSampler(ProfileSampler):
    """
    Sampler for 2D convolution operations.
    
    Parameters:
        HW: Height/Width of input feature map
        C: Input channels
        K: Output channels (filters)
        KS: Kernel size
        S: Stride
    """
    
    def __init__(
        self,
        ranges: Optional[Dict[str, RangeSpec]] = None,
        strategy: str = 'adaptive',
        seed: Optional[int] = None
    ):
        # Default ranges for Conv2D
        default_ranges = {
            'HW': (4, 64),
            'C': (3, 1024),
            'K': (16, 2048),
            'KS': [1, 3, 5, 7],
            'S': [1, 2]
        }
        ranges = ranges or default_ranges
        
        # C and K typically benefit from log-scale sampling
        log_scale_params = {'C', 'K'}
        
        super().__init__(
            ranges=ranges,
            strategy=strategy,
            log_scale_params=log_scale_params,
            seed=seed
        )
    
    def _generate_sample(self) -> Tuple[int, int, int, int]:
        """Generate a single Conv2D sample (HW, C, K, KS, S)."""
        HW = self.get_random_val(self.ranges['HW'], log_scale=False)
        C = self.get_random_val(self.ranges['C'], log_scale=True)
        K = self.get_random_val(self.ranges['K'], log_scale=True)
        KS = self.get_random_val(self.ranges['KS'])
        S = self.get_random_val(self.ranges['S'])
        return (HW, C, K, KS, S)
    
    def _validate_sample(self, sample: Tuple) -> bool:
        """Validate Conv2D sample (kernel size <= feature map size)."""
        HW, C, K, KS, S = sample
        return KS <= HW
    
    def _get_corner_samples(self) -> Set[Tuple[int, int, int, int]]:
        """Generate corner samples for Conv2D."""
        samples = set()
        
        hw_values = self.get_corner_values(self.ranges['HW'])
        c_values = self.get_corner_values(self.ranges['C'])
        k_values = self.get_corner_values(self.ranges['K'])
        ks_values = self.get_corner_values(self.ranges['KS'])
        s_values = self.get_corner_values(self.ranges['S'])
        
        value_space = product(hw_values, c_values, k_values, ks_values, s_values)
        for (hw, c, k, ks, s) in value_space:
            if ks <= hw:
               samples.add((hw, c, k, ks, s))
        
        return samples
    
    def _get_prior_samples(self) -> Set[Tuple[int, int, int, int]]:
        """Generate samples from prior distribution."""
        samples = set()
        
        hw_priors = self.get_prior_values(self.ranges['HW'], self.PRIOR_HW)
        c_priors = self.get_prior_values(self.ranges['C'], self.PRIOR_CHANNELS)
        k_priors = self.get_prior_values(self.ranges['K'], self.PRIOR_CHANNELS)
        ks_priors = self.get_prior_values(
            self.ranges['KS'], self.PRIOR_KERNEL_SIZES
        )
        s_priors = self.get_prior_values(self.ranges['S'], self.PRIOR_STRIDES)
        
        value_space = product(hw_priors, c_priors, k_priors, ks_priors, s_priors)
        for (hw, c, k, ks, s) in value_space:
            if ks <= hw:
               samples.add((hw, c, k, ks, s))
        
        return samples


class PoolingSampler(ProfileSampler):
    """
    Sampler for pooling operations (MaxPool, AvgPool).
    
    Parameters:
        C: Input channels
        HW: Height/Width of input feature map
        K: Kernel size
        S: Stride
    """
    
    def __init__(
        self,
        ranges: Optional[Dict[str, RangeSpec]] = None,
        strategy: str = 'adaptive',
        seed: Optional[int] = None
    ):
        # Default ranges for Pooling
        default_ranges = {
            'C': (1, 256),
            'HW': (8, 112),
            'K': [2, 3, 4],
            'S': [1, 2]
        }
        ranges = ranges or default_ranges
        
        # C benefits from log-scale sampling
        log_scale_params = {'C'}
        
        super().__init__(
            ranges=ranges,
            strategy=strategy,
            log_scale_params=log_scale_params,
            seed=seed
        )
    
    def _generate_sample(self) -> Tuple[int, int, int, int]:
        """Generate a single Pooling sample (C, HW, K, S)."""
        C = self.get_random_val(self.ranges['C'], log_scale=True)
        HW = self.get_random_val(self.ranges['HW'], log_scale=False)
        K = self.get_random_val(self.ranges['K'])
        S = self.get_random_val(self.ranges['S'])
        return (C, HW, K, S)
    
    def _validate_sample(self, sample: Tuple) -> bool:
        """Validate Pooling sample (kernel size <= feature map size)."""
        C, HW, K, S = sample
        return K <= HW
    
    def _get_corner_samples(self) -> Set[Tuple[int, int, int, int]]:
        """Generate corner samples for Pooling."""
        samples = set()
        
        c_values = self.get_corner_values(self.ranges['C'])
        hw_values = self.get_corner_values(self.ranges['HW'])
        k_values = self.get_corner_values(self.ranges['K'])
        s_values = self.get_corner_values(self.ranges['S'])
        
        for c in c_values:
            for hw in hw_values:
                for k in k_values:
                    for s in s_values:
                        if k <= hw:
                            samples.add((c, hw, k, s))
        
        return samples
    
    def _get_prior_samples(self) -> Set[Tuple[int, int, int, int]]:
        """Generate samples from prior distribution."""
        samples = set()
        
        c_priors = self.get_prior_values(self.ranges['C'], self.PRIOR_CHANNELS)
        hw_priors = self.get_prior_values(self.ranges['HW'], self.PRIOR_HW)
        k_priors = self.get_prior_values(self.ranges['K'], [2, 3, 4])
        s_priors = self.get_prior_values(self.ranges['S'], self.PRIOR_STRIDES)
        
        for c in c_priors:
            for hw in hw_priors:
                for k in k_priors:
                    for s in s_priors:
                        if k <= hw:
                            samples.add((c, hw, k, s))
        
        return samples
