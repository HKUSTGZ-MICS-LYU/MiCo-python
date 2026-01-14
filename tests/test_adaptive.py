#!/usr/bin/env python3
"""
Test suite for the Adaptive Profiler module.

Tests the adaptive profiling functionality including:
- Basic initialization
- Feature preparation
- Error sample identification
- Integration with MiCoProxy (when available)
"""

import random
import sys
import os
import unittest
import numpy as np

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profiler.adaptive import (
    AdaptiveProfiler,
    AdaptiveMatMulProfiler,
    AdaptiveConv2DProfiler,
    AdaptivePoolingProfiler,
    HAS_MICO_PROXY
)


class TestAdaptiveMatMulProfiler(unittest.TestCase):
    """Test the AdaptiveMatMulProfiler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        random.seed(42)
        np.random.seed(42)
        self.default_ranges = {
            'N': [16],
            'M': (16, 4096),
            'K': (16, 4096)
        }
    
    def test_initialization(self):
        """Test initialization without benchmark function."""
        profiler = AdaptiveMatMulProfiler(
            ranges=self.default_ranges,
            error_threshold=0.1
        )
        self.assertIsNotNone(profiler)
        self.assertEqual(profiler.error_threshold, 0.1)
        self.assertEqual(profiler.preprocess, 'cbops+')
    
    def test_initialization_custom_preprocess(self):
        """Test initialization with custom preprocessing."""
        profiler = AdaptiveMatMulProfiler(
            ranges=self.default_ranges,
            preprocess='raw'
        )
        self.assertEqual(profiler.preprocess, 'raw')
    
    def test_prepare_features(self):
        """Test feature preparation for matmul."""
        profiler = AdaptiveMatMulProfiler(ranges=self.default_ranges)
        
        # Mock data: (N, M, K, QA, QW, latency)
        data = [
            (16, 64, 128, 8, 8, 1000),
            (16, 128, 256, 4, 4, 2000),
            (16, 256, 512, 2, 2, 4000),
        ]
        
        X, y = profiler._prepare_features(data)
        
        self.assertEqual(X.shape[0], 3)
        self.assertEqual(X.shape[1], 5)  # MACS, M, K, QA, QW
        self.assertEqual(len(y), 3)
        np.testing.assert_array_equal(y, [1000, 2000, 4000])
    
    def test_extract_sample_params(self):
        """Test extracting sample parameters from data row."""
        profiler = AdaptiveMatMulProfiler(ranges=self.default_ranges)
        
        data_row = (16, 128, 256, 8, 8, 1500)
        params = profiler._extract_sample_params(data_row)
        
        self.assertEqual(params, (16, 128, 256))
    
    def test_run_without_benchmark_fn_raises(self):
        """Test that run() raises error without benchmark function."""
        profiler = AdaptiveMatMulProfiler(ranges=self.default_ranges)
        
        with self.assertRaises(ValueError):
            profiler.run(init_samples=10)


class TestAdaptiveConv2DProfiler(unittest.TestCase):
    """Test the AdaptiveConv2DProfiler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        random.seed(42)
        np.random.seed(42)
        self.default_ranges = {
            'HW': (4, 64),
            'C': (3, 256),
            'K': (16, 512),
            'KS': [1, 3, 5],
            'S': [1, 2]
        }
    
    def test_initialization(self):
        """Test initialization."""
        profiler = AdaptiveConv2DProfiler(ranges=self.default_ranges)
        self.assertIsNotNone(profiler)
    
    def test_prepare_features(self):
        """Test feature preparation for conv2d."""
        profiler = AdaptiveConv2DProfiler(ranges=self.default_ranges)
        
        # Mock data: (H, W, C, K, KS, S, QA, QW, latency)
        data = [
            (32, 32, 64, 128, 3, 1, 8, 8, 5000),
            (16, 16, 128, 256, 3, 2, 4, 4, 3000),
        ]
        
        X, y = profiler._prepare_features(data)
        
        self.assertEqual(X.shape[0], 2)
        self.assertEqual(X.shape[1], 9)  # MACS, H, W, C, K, KS, S, QA, QW
        np.testing.assert_array_equal(y, [5000, 3000])
    
    def test_extract_sample_params(self):
        """Test extracting sample parameters from data row."""
        profiler = AdaptiveConv2DProfiler(ranges=self.default_ranges)
        
        # (H, W, C, K, KS, S, QA, QW, latency)
        data_row = (32, 32, 64, 128, 3, 1, 8, 8, 5000)
        params = profiler._extract_sample_params(data_row)
        
        # Should extract (HW, C, K, KS, S)
        self.assertEqual(params, (32, 64, 128, 3, 1))


class TestAdaptivePoolingProfiler(unittest.TestCase):
    """Test the AdaptivePoolingProfiler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        random.seed(42)
        np.random.seed(42)
        self.default_ranges = {
            'C': (1, 256),
            'HW': (8, 112),
            'K': [2, 3, 4],
            'S': [1, 2]
        }
    
    def test_initialization(self):
        """Test initialization."""
        profiler = AdaptivePoolingProfiler(ranges=self.default_ranges)
        self.assertIsNotNone(profiler)
    
    def test_prepare_features(self):
        """Test feature preparation for pooling."""
        profiler = AdaptivePoolingProfiler(ranges=self.default_ranges)
        
        # Mock data: (C, H, W, K, S, latency)
        data = [
            (64, 28, 28, 2, 2, 100),
            (128, 14, 14, 3, 2, 150),
        ]
        
        X, y = profiler._prepare_features(data)
        
        self.assertEqual(X.shape[0], 2)
        self.assertEqual(X.shape[1], 5)  # C, H, W, K, S
        np.testing.assert_array_equal(y, [100, 150])


class TestAdaptiveProfilerWithMockBenchmark(unittest.TestCase):
    """Test adaptive profiler with mock benchmark function."""
    
    def setUp(self):
        """Set up test fixtures."""
        random.seed(42)
        np.random.seed(42)
    
    def mock_matmul_benchmark(self, N, M, K):
        """Mock benchmark that returns synthetic latency data."""
        results = []
        for qa in [2, 4, 8]:
            for qw in [2, 4, 8]:
                # Synthetic latency based on MACS and quantization
                latency = N * M * K * qa * qw // 1000 + random.randint(1, 100)
                results.append((N, M, K, qa, qw, latency))
        return results
    
    def test_run_with_mock_benchmark(self):
        """Test running adaptive profiler with mock benchmark."""
        profiler = AdaptiveMatMulProfiler(
            ranges={'N': [16], 'M': (32, 256), 'K': (32, 256)},
            benchmark_fn=self.mock_matmul_benchmark,
            error_threshold=0.1,
            seed=42
        )
        
        dataset = profiler.run(
            init_samples=5,
            iterations=1,
            samples_per_iteration=3,
            verbose=False
        )
        
        # Should have collected some data
        self.assertGreater(len(dataset), 0)
        
        # Each sample should have 9 results (3 qa x 3 qw)
        self.assertEqual(len(dataset) % 9, 0)
    
    def test_get_dataset(self):
        """Test getting the collected dataset."""
        profiler = AdaptiveMatMulProfiler(
            ranges={'N': [16], 'M': (32, 128), 'K': (32, 128)},
            benchmark_fn=self.mock_matmul_benchmark,
            seed=42
        )
        
        profiler.run(init_samples=3, iterations=0, verbose=False)
        
        dataset = profiler.get_dataset()
        self.assertIsInstance(dataset, list)
        self.assertGreater(len(dataset), 0)


@unittest.skipUnless(HAS_MICO_PROXY, "MiCoProxy not available")
class TestAdaptiveProfilerWithProxy(unittest.TestCase):
    """Test adaptive profiler with actual MiCoProxy integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        random.seed(42)
        np.random.seed(42)
    
    def mock_matmul_benchmark(self, N, M, K):
        """Mock benchmark with deterministic latency."""
        results = []
        for qa in [2, 4, 8]:
            for qw in [2, 4, 8]:
                latency = N * M * K * qa * qw // 1000 + 50
                results.append((N, M, K, qa, qw, latency))
        return results
    
    def test_proxy_training(self):
        """Test that proxy is trained during adaptive profiling."""
        profiler = AdaptiveMatMulProfiler(
            ranges={'N': [16], 'M': (32, 256), 'K': (32, 256)},
            benchmark_fn=self.mock_matmul_benchmark,
            error_threshold=0.1,
            seed=42
        )
        
        profiler.run(
            init_samples=10,
            iterations=1,
            verbose=False
        )
        
        proxy = profiler.get_proxy()
        self.assertIsNotNone(proxy)
    
    def test_error_sample_identification(self):
        """Test identification of high-error samples."""
        profiler = AdaptiveMatMulProfiler(
            ranges={'N': [16], 'M': (32, 256), 'K': (32, 256)},
            benchmark_fn=self.mock_matmul_benchmark,
            error_threshold=0.05,  # Low threshold to find some errors
            seed=42
        )
        
        # Generate some data
        profiler.run(init_samples=5, iterations=0, verbose=False)
        
        # Train proxy manually
        profiler._train_proxy(profiler.dataset)
        
        # Check error identification works
        X, y = profiler._prepare_features(profiler.dataset)
        error_samples = profiler._identify_error_samples(X, y, profiler.dataset)
        
        # Should be a list (may be empty if predictions are good)
        self.assertIsInstance(error_samples, list)


if __name__ == "__main__":
    unittest.main()
