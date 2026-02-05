#!/usr/bin/env python3
"""
Test suite for MiCoProxy Transfer Learning functionality.

Tests the transfer learning capabilities including:
- Basic transfer learning workflow (pretrain, finetune)
- Different fine-tuning strategies
- Bidirectional transfer (small <-> high)
- Cross-target transfer (bitfusion -> mico)
- Data ratio effects
"""

import sys
import os
import unittest
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MiCoProxy import (
    MiCoProxy,
    LogRandomForestRegressor,
    load_proxy_data,
    get_transfer_proxy,
    evaluate_transfer_learning,
    compare_transfer_directions,
    explore_cross_target_transfer
)
from sklearn.metrics import mean_absolute_percentage_error


class TestMiCoProxyTransferLearning(unittest.TestCase):
    """Test the MiCoProxy transfer learning methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.seed = 42
        np.random.seed(self.seed)
        
        # Load sample data for testing
        self.X_small, self.y_small = load_proxy_data(
            'benchmark_results/mico_small_matmul_zoo.csv', 'matmul'
        )
        self.X_high, self.y_high = load_proxy_data(
            'benchmark_results/mico_high_matmul_zoo.csv', 'matmul'
        )
    
    def test_pretrain_basic(self):
        """Test basic pretraining functionality."""
        model = LogRandomForestRegressor(random_state=self.seed)
        proxy = MiCoProxy(model, preprocess='cbops+', seed=self.seed)
        
        # Should not be pretrained initially
        self.assertFalse(proxy.is_pretrained())
        
        # Pretrain on small data
        proxy.pretrain(self.X_small, self.y_small)
        
        # Should be pretrained now
        self.assertTrue(proxy.is_pretrained())
        
        # Should be able to make predictions
        predictions = proxy.predict(self.X_high)
        self.assertEqual(len(predictions), len(self.y_high))
    
    def test_finetune_combined_strategy(self):
        """Test fine-tuning with combined strategy."""
        model = LogRandomForestRegressor(random_state=self.seed)
        proxy = MiCoProxy(model, preprocess='cbops+', seed=self.seed)
        
        # Pretrain on source
        proxy.pretrain(self.X_small, self.y_small)
        
        # Finetune on target
        result = proxy.finetune(
            self.X_high, self.y_high, 
            finetune_ratio=0.1, 
            strategy='combined'
        )
        
        self.assertEqual(result['strategy'], 'combined')
        self.assertEqual(result['finetune_ratio'], 0.1)
        self.assertGreater(result['target_samples_used'], 0)
        
        # Should be able to predict
        predictions = proxy.predict(self.X_high)
        self.assertEqual(len(predictions), len(self.y_high))
    
    def test_finetune_target_only_strategy(self):
        """Test fine-tuning with target_only strategy."""
        model = LogRandomForestRegressor(random_state=self.seed)
        proxy = MiCoProxy(model, preprocess='cbops+', seed=self.seed)
        
        # target_only doesn't require pretraining
        result = proxy.finetune(
            self.X_high, self.y_high,
            finetune_ratio=0.5,
            strategy='target_only'
        )
        
        self.assertEqual(result['strategy'], 'target_only')
        predictions = proxy.predict(self.X_high)
        self.assertEqual(len(predictions), len(self.y_high))
    
    def test_finetune_weighted_strategy(self):
        """Test fine-tuning with weighted strategy."""
        model = LogRandomForestRegressor(random_state=self.seed)
        proxy = MiCoProxy(model, preprocess='cbops+', seed=self.seed)
        
        # Pretrain first
        proxy.pretrain(self.X_small, self.y_small)
        
        # Finetune with weighted strategy
        result = proxy.finetune(
            self.X_high, self.y_high,
            finetune_ratio=0.2,
            strategy='weighted'
        )
        
        self.assertEqual(result['strategy'], 'weighted')
        predictions = proxy.predict(self.X_high)
        self.assertEqual(len(predictions), len(self.y_high))
    
    def test_finetune_requires_pretrain(self):
        """Test that non-target_only strategies require pretraining."""
        model = LogRandomForestRegressor(random_state=self.seed)
        proxy = MiCoProxy(model, preprocess='cbops+', seed=self.seed)
        
        # Should raise error when trying to finetune with combined without pretraining
        with self.assertRaises(ValueError):
            proxy.finetune(
                self.X_high, self.y_high,
                finetune_ratio=0.1,
                strategy='combined'
            )
    
    def test_finetune_invalid_strategy(self):
        """Test that invalid strategy raises error."""
        model = LogRandomForestRegressor(random_state=self.seed)
        proxy = MiCoProxy(model, preprocess='cbops+', seed=self.seed)
        proxy.pretrain(self.X_small, self.y_small)
        
        with self.assertRaises(ValueError):
            proxy.finetune(
                self.X_high, self.y_high,
                finetune_ratio=0.1,
                strategy='invalid_strategy'
            )
    
    def test_finetune_ratio_affects_samples(self):
        """Test that finetune_ratio correctly controls sample count."""
        model = LogRandomForestRegressor(random_state=self.seed)
        proxy = MiCoProxy(model, preprocess='cbops+', seed=self.seed)
        proxy.pretrain(self.X_small, self.y_small)
        
        total_samples = len(self.X_high)
        
        # Test with 10% ratio
        result_10 = proxy.finetune(
            self.X_high, self.y_high,
            finetune_ratio=0.1,
            strategy='combined'
        )
        expected_10 = int(total_samples * 0.1)
        self.assertEqual(result_10['target_samples_used'], expected_10)
        
        # Recreate and test with 50% ratio
        model2 = LogRandomForestRegressor(random_state=self.seed)
        proxy2 = MiCoProxy(model2, preprocess='cbops+', seed=self.seed)
        proxy2.pretrain(self.X_small, self.y_small)
        
        result_50 = proxy2.finetune(
            self.X_high, self.y_high,
            finetune_ratio=0.5,
            strategy='combined'
        )
        expected_50 = int(total_samples * 0.5)
        self.assertEqual(result_50['target_samples_used'], expected_50)


class TestLoadProxyData(unittest.TestCase):
    """Test the load_proxy_data function."""
    
    def test_load_matmul_data(self):
        """Test loading matmul proxy data."""
        X, y = load_proxy_data(
            'benchmark_results/mico_small_matmul_zoo.csv',
            'matmul'
        )
        
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(len(X), len(y))
        self.assertGreater(len(X), 0)
        # matmul features: MACS, M, K, QA, QW
        self.assertEqual(X.shape[1], 5)
    
    def test_load_conv2d_data(self):
        """Test loading conv2d proxy data."""
        X, y = load_proxy_data(
            'benchmark_results/mico_small_conv2d_zoo.csv',
            'conv2d'
        )
        
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(len(X), len(y))
        self.assertGreater(len(X), 0)
        # conv2d features: MACS, H, W, C, K, Ks, S, QA, QW
        self.assertEqual(X.shape[1], 9)
    
    def test_load_invalid_kernel_type(self):
        """Test that invalid kernel type raises error."""
        with self.assertRaises(ValueError):
            load_proxy_data(
                'benchmark_results/mico_small_matmul_zoo.csv',
                'invalid_kernel'
            )


class TestGetTransferProxy(unittest.TestCase):
    """Test the get_transfer_proxy function."""
    
    def test_basic_transfer(self):
        """Test basic transfer learning functionality."""
        proxy, results = get_transfer_proxy(
            source_type='mico_small',
            target_type='mico_high',
            kernel_type='matmul',
            finetune_ratio=0.1,
            verbose=False
        )
        
        self.assertIsNotNone(proxy)
        self.assertIn('source_type', results)
        self.assertIn('target_type', results)
        self.assertIn('mape_before', results)
        self.assertIn('mape_after', results)
        self.assertIn('mape_improvement', results)
    
    def test_transfer_improves_with_data(self):
        """Test that more target data generally improves performance."""
        _, results_10 = get_transfer_proxy(
            source_type='mico_small',
            target_type='mico_high',
            kernel_type='matmul',
            finetune_ratio=0.1,
            seed=42,
            verbose=False
        )
        
        _, results_50 = get_transfer_proxy(
            source_type='mico_small',
            target_type='mico_high',
            kernel_type='matmul',
            finetune_ratio=0.5,
            seed=42,
            verbose=False
        )
        
        # More data should generally lead to better or equal performance
        # (allowing for some variance)
        self.assertLessEqual(
            results_50['mape_after'], 
            results_10['mape_after'] * 1.5  # Allow 50% tolerance
        )
    
    def test_different_strategies(self):
        """Test that different strategies produce results."""
        strategies = ['combined', 'target_only', 'weighted']
        
        for strategy in strategies:
            proxy, results = get_transfer_proxy(
                source_type='mico_small',
                target_type='mico_high',
                kernel_type='matmul',
                finetune_ratio=0.1,
                strategy=strategy,
                verbose=False
            )
            
            self.assertIsNotNone(proxy)
            self.assertEqual(results['strategy'], strategy)
            self.assertGreater(results['mape_after'], 0)
            self.assertLess(results['mape_after'], 10)  # Sanity check


class TestCompareTransferDirections(unittest.TestCase):
    """Test the compare_transfer_directions function."""
    
    def test_both_directions(self):
        """Test transfer in both directions."""
        results = compare_transfer_directions(
            kernel_type='matmul',
            finetune_ratio=0.1,
            verbose=False
        )
        
        self.assertIn('directions', results)
        self.assertIn('small_to_high', results['directions'])
        self.assertIn('high_to_small', results['directions'])
        
        # Both directions should have valid MAPE
        self.assertGreater(results['directions']['small_to_high']['mape_after'], 0)
        self.assertGreater(results['directions']['high_to_small']['mape_after'], 0)


class TestEvaluateTransferLearning(unittest.TestCase):
    """Test the evaluate_transfer_learning function."""
    
    def test_evaluation_with_multiple_ratios(self):
        """Test evaluation across multiple ratios."""
        results = evaluate_transfer_learning(
            source_type='mico_small',
            target_type='mico_high',
            kernel_type='matmul',
            ratios=[0.1, 0.5],
            strategies=['combined'],
            n_trials=2,
            verbose=False
        )
        
        self.assertEqual(results['source_type'], 'mico_small')
        self.assertEqual(results['target_type'], 'mico_high')
        self.assertIn('experiments', results)
        self.assertEqual(len(results['experiments']), 2)  # 2 ratios * 1 strategy


class TestExploreCrossTargetTransfer(unittest.TestCase):
    """Test the explore_cross_target_transfer function."""
    
    def test_bitfusion_to_mico_transfer(self):
        """Test transfer from BitFusion to MiCo targets."""
        results = explore_cross_target_transfer(
            source_type='bitfusion',
            target_types=['mico_small'],
            kernel_type='matmul',
            finetune_ratios=[0.1],
            verbose=False
        )
        
        self.assertEqual(results['source_type'], 'bitfusion')
        self.assertIn('mico_small', results['targets'])


if __name__ == "__main__":
    unittest.main()
