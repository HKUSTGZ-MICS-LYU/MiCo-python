#!/usr/bin/env python3
"""
Test suite for the Profile Sampler module.

Tests the core sampling functionality including:
- Basic sample generation for all samplers
- Different sampling strategies
- Validation constraints
- Corner and prior sampling
- Adaptive sampling with error refinement
"""

import random
import sys
import os
import unittest

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profiler.sampler import (
    ProfileSampler,
    MatMulSampler,
    Conv2DSampler,
    PoolingSampler,
)


class TestMatMulSampler(unittest.TestCase):
    """Test the MatMulSampler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        random.seed(42)
        self.default_ranges = {
            'N': [16],
            'M': (16, 4096),
            'K': (16, 4096)
        }
    
    def test_initialization_default(self):
        """Test default initialization."""
        sampler = MatMulSampler()
        self.assertIsNotNone(sampler)
        self.assertEqual(sampler.strategy, 'adaptive')
        self.assertIn('M', sampler.log_scale_params)
        self.assertIn('K', sampler.log_scale_params)
    
    def test_initialization_custom_ranges(self):
        """Test initialization with custom ranges."""
        custom_ranges = {'N': [8, 16, 32], 'M': (32, 512), 'K': (64, 256)}
        sampler = MatMulSampler(ranges=custom_ranges, strategy='random')
        self.assertEqual(sampler.ranges, custom_ranges)
        self.assertEqual(sampler.strategy, 'random')
    
    def test_generate_random_strategy(self):
        """Test sample generation with random strategy."""
        sampler = MatMulSampler(ranges=self.default_ranges, strategy='random', seed=42)
        samples = sampler.generate(num_samples=10, show_progress=False)
        
        self.assertEqual(len(samples), 10)
        for sample in samples:
            self.assertEqual(len(sample), 3)
            N, M, K = sample
            self.assertIn(N, self.default_ranges['N'])
            self.assertGreaterEqual(M, self.default_ranges['M'][0])
            self.assertLessEqual(M, self.default_ranges['M'][1])
            self.assertGreaterEqual(K, self.default_ranges['K'][0])
            self.assertLessEqual(K, self.default_ranges['K'][1])
    
    def test_generate_corner_strategy(self):
        """Test corner sample generation."""
        sampler = MatMulSampler(ranges=self.default_ranges, strategy='corner', seed=42)
        samples = sampler.generate(num_samples=10, show_progress=False)
        
        # Corner samples should include boundary combinations
        self.assertGreater(len(samples), 0)
        corners = [(16, 16, 16), (16, 4096, 4096), (16, 16, 4096), (16, 4096, 16)]
        for corner in corners:
            self.assertIn(corner, samples)
    
    def test_generate_prior_strategy(self):
        """Test prior distribution sampling."""
        sampler = MatMulSampler(ranges=self.default_ranges, strategy='prior', seed=42)
        samples = sampler.generate(num_samples=100, show_progress=False)
        
        # Prior samples should come from common DNN configurations
        self.assertGreater(len(samples), 0)
        # Check that samples use prior values
        valid_channels = ProfileSampler.PRIOR_CHANNELS
        for sample in samples:
            N, M, K = sample
            self.assertIn(M, valid_channels)
            self.assertIn(K, valid_channels)
    
    def test_generate_lhs_strategy(self):
        """Test Latin Hypercube Sampling."""
        sampler = MatMulSampler(ranges=self.default_ranges, strategy='lhs', seed=42)
        samples = sampler.generate(num_samples=20, show_progress=False)
        
        self.assertEqual(len(samples), 20)
        # Check sample diversity (no duplicates due to LHS nature)
        self.assertEqual(len(set(samples)), len(samples))
    
    def test_generate_adaptive_strategy(self):
        """Test adaptive sampling strategy."""
        sampler = MatMulSampler(ranges=self.default_ranges, strategy='adaptive', seed=42)
        samples = sampler.generate(num_samples=30, show_progress=False)
        
        self.assertEqual(len(samples), 30)
        # Adaptive should include corners
        self.assertIn((16, 16, 16), samples)
        self.assertIn((16, 4096, 4096), samples)
    
    def test_deterministic_with_seed(self):
        """Test that same seed produces same samples."""
        sampler1 = MatMulSampler(ranges=self.default_ranges, strategy='random', seed=123)
        samples1 = sampler1.generate(num_samples=10, show_progress=False)
        
        sampler2 = MatMulSampler(ranges=self.default_ranges, strategy='random', seed=123)
        samples2 = sampler2.generate(num_samples=10, show_progress=False)
        
        self.assertEqual(samples1, samples2)


class TestConv2DSampler(unittest.TestCase):
    """Test the Conv2DSampler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        random.seed(42)
        self.default_ranges = {
            'HW': (4, 64),
            'C': (3, 1024),
            'K': (16, 2048),
            'KS': [1, 3, 5, 7],
            'S': [1, 2]
        }
    
    def test_initialization(self):
        """Test initialization."""
        sampler = Conv2DSampler()
        self.assertIsNotNone(sampler)
        self.assertIn('C', sampler.log_scale_params)
        self.assertIn('K', sampler.log_scale_params)
    
    def test_validation_kernel_size(self):
        """Test that kernel size validation works."""
        sampler = Conv2DSampler(ranges=self.default_ranges, strategy='random', seed=42)
        samples = sampler.generate(num_samples=50, show_progress=False)
        
        for sample in samples:
            HW, C, K, KS, S = sample
            # Kernel size must not exceed feature map size
            self.assertLessEqual(KS, HW)
    
    def test_generate_samples(self):
        """Test basic sample generation."""
        sampler = Conv2DSampler(ranges=self.default_ranges, strategy='adaptive', seed=42)
        samples = sampler.generate(num_samples=20, show_progress=False)
        
        # Adaptive strategy may generate more samples due to corners/priors
        self.assertGreaterEqual(len(samples), 20)
        for sample in samples:
            self.assertEqual(len(sample), 5)
            HW, C, K, KS, S = sample
            self.assertGreaterEqual(HW, self.default_ranges['HW'][0])
            self.assertLessEqual(HW, self.default_ranges['HW'][1])
            self.assertIn(KS, self.default_ranges['KS'])
            self.assertIn(S, self.default_ranges['S'])
    
    def test_corner_samples_valid(self):
        """Test that corner samples respect validation."""
        sampler = Conv2DSampler(ranges=self.default_ranges, strategy='corner', seed=42)
        samples = sampler.generate(num_samples=20, show_progress=False)
        
        for sample in samples:
            HW, C, K, KS, S = sample
            self.assertLessEqual(KS, HW)


class TestPoolingSampler(unittest.TestCase):
    """Test the PoolingSampler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        random.seed(42)
        self.default_ranges = {
            'C': (1, 256),
            'HW': (8, 112),
            'K': [2, 3, 4],
            'S': [1, 2]
        }
    
    def test_initialization(self):
        """Test initialization."""
        sampler = PoolingSampler()
        self.assertIsNotNone(sampler)
        self.assertIn('C', sampler.log_scale_params)
    
    def test_generate_samples(self):
        """Test basic sample generation."""
        sampler = PoolingSampler(ranges=self.default_ranges, strategy='adaptive', seed=42)
        samples = sampler.generate(num_samples=15, show_progress=False)
        
        self.assertGreater(len(samples), 0)
        for sample in samples:
            self.assertEqual(len(sample), 4)
            C, HW, K, S = sample
            # Kernel size must not exceed feature map size
            self.assertLessEqual(K, HW)


class TestProfileSamplerHelpers(unittest.TestCase):
    """Test helper methods in ProfileSampler."""
    
    def test_get_random_val_fixed(self):
        """Test get_random_val with fixed value."""
        val = ProfileSampler.get_random_val(42)
        self.assertEqual(val, 42)
    
    def test_get_random_val_list(self):
        """Test get_random_val with list."""
        choices = [1, 2, 3, 4, 5]
        for _ in range(20):
            val = ProfileSampler.get_random_val(choices)
            self.assertIn(val, choices)
    
    def test_get_random_val_tuple(self):
        """Test get_random_val with tuple range."""
        for _ in range(20):
            val = ProfileSampler.get_random_val((10, 100), log_scale=False)
            self.assertGreaterEqual(val, 10)
            self.assertLessEqual(val, 100)
    
    def test_get_random_val_log_scale(self):
        """Test get_random_val with log scale."""
        for _ in range(20):
            val = ProfileSampler.get_random_val((1, 1000), log_scale=True)
            self.assertGreaterEqual(val, 1)
            self.assertLessEqual(val, 1000)
    
    def test_get_corner_values_fixed(self):
        """Test get_corner_values with fixed value."""
        corners = ProfileSampler.get_corner_values(42)
        self.assertEqual(corners, [42])
    
    def test_get_corner_values_list(self):
        """Test get_corner_values with list."""
        corners = ProfileSampler.get_corner_values([1, 3, 5, 7])
        self.assertEqual(corners, [1, 7])
    
    def test_get_corner_values_tuple(self):
        """Test get_corner_values with tuple."""
        corners = ProfileSampler.get_corner_values((10, 100))
        self.assertEqual(corners, [10, 100])
    
    def test_get_prior_values_tuple(self):
        """Test get_prior_values filtering."""
        priors = [1, 3, 7, 14, 28, 56, 112, 224]
        filtered = ProfileSampler.get_prior_values((10, 60), priors)
        self.assertEqual(filtered, [14, 28, 56])


class TestAdaptiveSampling(unittest.TestCase):
    """Test adaptive sampling with error refinement."""
    
    def test_fine_grained_sampling(self):
        """Test fine-grained sampling around error regions."""
        sampler = MatMulSampler(
            ranges={'N': [16], 'M': (16, 4096), 'K': (16, 4096)},
            strategy='adaptive',
            seed=42
        )
        
        # Simulate error samples that need refinement
        error_samples = [(16, 1000, 2000), (16, 500, 500)]
        
        samples = sampler.generate(
            num_samples=30,
            error_samples=error_samples,
            show_progress=False
        )
        
        self.assertEqual(len(samples), 30)
        
        # Check that some samples are near the error samples
        found_near_error = False
        for sample in samples:
            N, M, K = sample
            for err_n, err_m, err_k in error_samples:
                if (abs(M - err_m) < err_m * 0.5 and 
                    abs(K - err_k) < err_k * 0.5 and
                    N == err_n):
                    found_near_error = True
                    break
        
        self.assertTrue(found_near_error)
    
    def test_fine_grained_boundary_edge_case(self):
        """Test fine-grained sampling near boundaries."""
        sampler = MatMulSampler(
            ranges={'N': [16], 'M': (16, 64), 'K': (16, 64)},
            strategy='adaptive',
            seed=42
        )
        
        # Error samples at boundaries - this should not crash
        error_samples = [(16, 16, 16), (16, 64, 64)]
        
        samples = sampler.generate(
            num_samples=20,
            error_samples=error_samples,
            show_progress=False
        )
        
        self.assertEqual(len(samples), 20)
        # All samples should be within valid ranges
        for sample in samples:
            N, M, K = sample
            self.assertEqual(N, 16)
            self.assertGreaterEqual(M, 16)
            self.assertLessEqual(M, 64)
            self.assertGreaterEqual(K, 16)
            self.assertLessEqual(K, 64)


if __name__ == "__main__":
    unittest.main()
