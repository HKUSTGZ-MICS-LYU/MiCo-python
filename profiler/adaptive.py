"""
Adaptive Profiler Module

This module provides an adaptive profiling strategy that integrates
MiCoProxy with the sampler for in-time error feedback, inspired by
nn-Meter's iterative refinement approach.

The adaptive profiler:
1. Generates initial samples using the sampler
2. Runs benchmarks to get actual latency measurements  
3. Trains a proxy model on the data
4. Identifies samples with high prediction error
5. Generates fine-grained samples around error regions
6. Repeats until convergence or max iterations

Usage:
    from profiler.adaptive import AdaptiveMatMulProfiler, AdaptiveConv2DProfiler
    
    profiler = AdaptiveMatMulProfiler(
        ranges={'N': [16], 'M': (16, 4096), 'K': (16, 4096)},
        benchmark_fn=my_benchmark_function
    )
    dataset = profiler.run(
        init_samples=50,
        iterations=3,
        error_threshold=0.1
    )
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any
from abc import ABC, abstractmethod

from .sampler import (
    ProfileSampler,
    MatMulSampler,
    Conv2DSampler,
    PoolingSampler,
    RangeSpec
)

# Import MiCoProxy components for error feedback
try:
    from MiCoProxy import MiCoProxy, LogRandomForestRegressor
    HAS_MICO_PROXY = True
except ImportError:
    HAS_MICO_PROXY = False


class AdaptiveProfiler(ABC):
    """
    Abstract base class for adaptive profilers.
    
    Integrates sampling with MiCoProxy for in-time error feedback,
    using LogRandomForest model with cbops+ features as recommended.
    
    Attributes:
        sampler: ProfileSampler instance for sample generation
        benchmark_fn: Function to run benchmarks and return latency
        proxy: MiCoProxy model for error prediction
        error_threshold: Threshold for identifying high-error samples
        preprocess: Feature preprocessing method (default: 'cbops+')
    """
    
    DEFAULT_PREPROCESS = 'cbops+'
    
    def __init__(
        self,
        sampler: ProfileSampler,
        benchmark_fn: Optional[Callable] = None,
        error_threshold: float = 0.1,
        preprocess: str = 'cbops+'
    ):
        """
        Initialize the adaptive profiler.
        
        Args:
            sampler: ProfileSampler instance
            benchmark_fn: Function that takes sample params and returns 
                          list of (params..., QA, QW, latency) tuples
            error_threshold: MAPE threshold for identifying high-error samples
            preprocess: Feature preprocessing ('raw', 'bops+', 'cbops', 'cbops+')
        """
        self.sampler = sampler
        self.benchmark_fn = benchmark_fn
        self.error_threshold = error_threshold
        self.preprocess = preprocess
        self.proxy = None
        self.dataset = []
        
    def _create_proxy(self) -> 'MiCoProxy':
        """Create a MiCoProxy with LogRandomForest and cbops+ features."""
        if not HAS_MICO_PROXY:
            raise ImportError(
                "MiCoProxy not available. Please ensure MiCoProxy.py is accessible."
            )
        return MiCoProxy(
            LogRandomForestRegressor(random_state=42),
            preprocess=self.preprocess
        )
    
    @abstractmethod
    def _prepare_features(self, data: List[Tuple]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and targets from benchmark data.
        
        Args:
            data: List of benchmark result tuples
            
        Returns:
            Tuple of (X features array, y latency array)
        """
        pass
    
    @abstractmethod
    def _extract_sample_params(self, data_row: Tuple) -> Tuple:
        """
        Extract sample parameters from a benchmark data row.
        
        Args:
            data_row: Single benchmark result tuple
            
        Returns:
            Sample parameter tuple (without QA, QW, latency)
        """
        pass
    
    def _identify_error_samples(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        data: List[Tuple]
    ) -> List[Tuple]:
        """
        Identify samples with high prediction error.
        
        Args:
            X: Feature array
            y_true: True latency values
            data: Original benchmark data
            
        Returns:
            List of sample parameter tuples with high error
        """
        if self.proxy is None or len(y_true) == 0:
            return []
        
        y_pred = self.proxy.predict(X)
        
        # Calculate per-sample MAPE
        errors = np.abs(y_pred - y_true) / (y_true + 1e-8)
        
        # Find samples above threshold
        error_indices = np.where(errors > self.error_threshold)[0]
        
        # Extract unique sample parameters
        error_samples = set()
        for idx in error_indices:
            sample_params = self._extract_sample_params(data[idx])
            error_samples.add(sample_params)
        
        return list(error_samples)
    
    def _train_proxy(self, data: List[Tuple]) -> float:
        """
        Train the proxy model on current data.
        
        Args:
            data: List of benchmark result tuples
            
        Returns:
            Cross-validation MAPE score
        """
        if len(data) < 5:
            return float('inf')
        
        X, y = self._prepare_features(data)
        
        self.proxy = self._create_proxy()
        self.proxy.fit(X, y)
        
        # Calculate training MAPE (as estimate)
        y_pred = self.proxy.predict(X)
        mape = np.mean(np.abs(y_pred - y) / (y + 1e-8))
        
        return mape
    
    def run(
        self,
        init_samples: int = 50,
        iterations: int = 3,
        samples_per_iteration: int = 20,
        fine_grained_num: int = 5,
        verbose: bool = True
    ) -> List[Tuple]:
        """
        Run adaptive profiling with iterative refinement.
        
        Args:
            init_samples: Number of initial samples
            iterations: Number of refinement iterations
            samples_per_iteration: New samples per iteration
            fine_grained_num: Fine-grained samples per error sample
            verbose: Whether to print progress
            
        Returns:
            Complete benchmark dataset
        """
        if self.benchmark_fn is None:
            raise ValueError("benchmark_fn must be provided to run profiling")
        
        self.dataset = []
        all_samples = set()
        
        # Phase 1: Initial sampling
        if verbose:
            print(f"Phase 1: Generating {init_samples} initial samples...")
        
        initial_samples = self.sampler.generate(
            num_samples=init_samples,
            show_progress=verbose
        )
        
        # Benchmark initial samples
        if verbose:
            print(f"Running benchmarks on {len(initial_samples)} samples...")
        
        for sample in initial_samples:
            if sample not in all_samples:
                try:
                    results = self.benchmark_fn(*sample)
                    self.dataset.extend(results)
                    all_samples.add(sample)
                except Exception as e:
                    if verbose:
                        print(f"Benchmark failed for {sample}: {e}")
        
        # Train initial proxy
        if HAS_MICO_PROXY and len(self.dataset) >= 5:
            mape = self._train_proxy(self.dataset)
            if verbose:
                print(f"Initial proxy MAPE: {mape*100:.2f}%")
        
        # Phase 2: Iterative refinement
        for i in range(iterations):
            if verbose:
                print(f"\nIteration {i+1}/{iterations}: Adaptive refinement...")
            
            if not HAS_MICO_PROXY:
                if verbose:
                    print("MiCoProxy not available, skipping adaptive refinement")
                break
            
            # Identify error samples
            X, y = self._prepare_features(self.dataset)
            error_samples = self._identify_error_samples(X, y, self.dataset)
            
            if verbose:
                print(f"Found {len(error_samples)} high-error sample regions")
            
            if len(error_samples) == 0:
                if verbose:
                    print("No high-error samples found, convergence achieved!")
                break
            
            # Generate new samples around error regions
            new_samples = self.sampler.generate(
                num_samples=samples_per_iteration,
                error_samples=error_samples,
                show_progress=verbose
            )
            
            # Filter out already-benchmarked samples
            new_samples = [s for s in new_samples if s not in all_samples]
            
            if verbose:
                print(f"Benchmarking {len(new_samples)} new samples...")
            
            for sample in new_samples:
                if sample not in all_samples:
                    try:
                        results = self.benchmark_fn(*sample)
                        self.dataset.extend(results)
                        all_samples.add(sample)
                    except Exception as e:
                        if verbose:
                            print(f"Benchmark failed for {sample}: {e}")
            
            # Retrain proxy
            mape = self._train_proxy(self.dataset)
            if verbose:
                print(f"Updated proxy MAPE: {mape*100:.2f}%")
        
        if verbose:
            print(f"\nAdaptive profiling complete. Total samples: {len(self.dataset)}")
        
        return self.dataset
    
    def get_proxy(self) -> Optional['MiCoProxy']:
        """Get the trained proxy model."""
        return self.proxy
    
    def get_dataset(self) -> List[Tuple]:
        """Get the collected benchmark dataset."""
        return self.dataset


class AdaptiveMatMulProfiler(AdaptiveProfiler):
    """
    Adaptive profiler for matrix multiplication operations.
    
    Benchmark data format: (N, M, K, QA, QW, latency)
    """
    
    def __init__(
        self,
        ranges: Optional[Dict[str, RangeSpec]] = None,
        benchmark_fn: Optional[Callable] = None,
        error_threshold: float = 0.1,
        preprocess: str = 'cbops+',
        seed: Optional[int] = None
    ):
        sampler = MatMulSampler(ranges=ranges, strategy='adaptive', seed=seed)
        super().__init__(
            sampler=sampler,
            benchmark_fn=benchmark_fn,
            error_threshold=error_threshold,
            preprocess=preprocess
        )
    
    def _prepare_features(self, data: List[Tuple]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for MatMul: (MACS, M, K, QA, QW)
        """
        data_array = np.array(data)
        N = data_array[:, 0]
        M = data_array[:, 1]
        K = data_array[:, 2]
        QA = data_array[:, 3]
        QW = data_array[:, 4]
        latency = data_array[:, 5]
        
        # MACS = N * M * K (but N is typically batch/sequence length)
        MACS = N * M * K
        
        X = np.column_stack((MACS, M, K, QA, QW))
        y = latency
        
        return X, y
    
    def _extract_sample_params(self, data_row: Tuple) -> Tuple:
        """Extract (N, M, K) from data row."""
        return (data_row[0], data_row[1], data_row[2])


class AdaptiveConv2DProfiler(AdaptiveProfiler):
    """
    Adaptive profiler for 2D convolution operations.
    
    Benchmark data format: (H, W, C, K, KS, QA, QW, latency)
    """
    
    def __init__(
        self,
        ranges: Optional[Dict[str, RangeSpec]] = None,
        benchmark_fn: Optional[Callable] = None,
        error_threshold: float = 0.1,
        preprocess: str = 'cbops+',
        seed: Optional[int] = None
    ):
        sampler = Conv2DSampler(ranges=ranges, strategy='adaptive', seed=seed)
        super().__init__(
            sampler=sampler,
            benchmark_fn=benchmark_fn,
            error_threshold=error_threshold,
            preprocess=preprocess
        )
    
    def _prepare_features(self, data: List[Tuple]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for Conv2D: (MACS, H, W, C, K, KS, QA, QW)
        """
        data_array = np.array(data)
        H = data_array[:, 0]
        W = data_array[:, 1]
        C = data_array[:, 2]
        K = data_array[:, 3]
        KS = data_array[:, 4]
        QA = data_array[:, 5]
        QW = data_array[:, 6]
        latency = data_array[:, 7]
        
        # MACS for conv2d
        H_out = (H - KS) + 1
        W_out = (W - KS) + 1
        MACS = H_out * W_out * C * K * KS * KS
        
        X = np.column_stack((MACS, H, W, C, K, KS, QA, QW))
        y = latency
        
        return X, y
    
    def _extract_sample_params(self, data_row: Tuple) -> Tuple:
        """Extract (H, C, K, KS) from data row - H is used as HW since H=W."""
        return (data_row[0], data_row[2], data_row[3], data_row[4])


class AdaptivePoolingProfiler(AdaptiveProfiler):
    """
    Adaptive profiler for pooling operations.
    
    Benchmark data format: (C, H, W, K, S, latency)
    """
    
    def __init__(
        self,
        ranges: Optional[Dict[str, RangeSpec]] = None,
        benchmark_fn: Optional[Callable] = None,
        error_threshold: float = 0.1,
        preprocess: str = 'cbops+',
        seed: Optional[int] = None
    ):
        sampler = PoolingSampler(ranges=ranges, strategy='adaptive', seed=seed)
        super().__init__(
            sampler=sampler,
            benchmark_fn=benchmark_fn,
            error_threshold=error_threshold,
            preprocess=preprocess
        )
    
    def _prepare_features(self, data: List[Tuple]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for Pooling.
        Note: Pooling doesn't have QA/QW in same sense, adapt as needed.
        """
        data_array = np.array(data)
        C = data_array[:, 0]
        H = data_array[:, 1]
        W = data_array[:, 2]
        K = data_array[:, 3]
        S = data_array[:, 4]
        latency = data_array[:, 5]
        
        # Simple feature representation for pooling
        X = np.column_stack((C, H, W, K, S))
        y = latency
        
        return X, y
    
    def _extract_sample_params(self, data_row: Tuple) -> Tuple:
        """Extract (C, H, K, S) from data row - H is used as HW since H=W."""
        return (data_row[0], data_row[1], data_row[3], data_row[4])
