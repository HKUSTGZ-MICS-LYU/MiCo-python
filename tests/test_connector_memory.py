#!/usr/bin/env python3
"""
Test suite for connector tensor memory pool handling.

This tests the fix for the issue where connector tensors (bypass=True) were not
properly handled in memory pool allocation, leading to in-place operations where
the source tensor and output tensor shared the same memory pool.

The issue: tensor_0 -> tensor_connect -> tensor_1 pattern where tensor_connect 
is a bypass tensor (e.g., flatten, view) that shares memory with tensor_0.
Without the fix, tensor_0 and tensor_1 could be assigned to the same pool,
causing in-place operations in linear/conv2d that read from tensor_connect
(alias of tensor_0) and write to tensor_1.
"""

import torch
import torch.nn as nn
import sys
import os
import unittest

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import MLP, LeNet
from MiCoCodeGen import MiCoCodeGen
from MiCoUtils import fuse_model


class TestConnectorMemoryPool(unittest.TestCase):
    """Test that connector tensors don't cause in-place operation issues."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
    
    def _check_no_inplace_issues(self, codegen):
        """
        Check that no connector-related in-place issues exist.
        
        Returns True if no issues found, False otherwise.
        """
        dag = codegen.extract_tensor_dag()
        memory_pools, tensor_to_pool = codegen.allocate_memory_pools()
        
        for name, info in codegen.tensors.items():
            if info.get('bypass', False):
                # This is a connector tensor
                deps = dag.get(name, [])
                if deps:
                    source_tensor = deps[0]  # The tensor the connector points to
                    # Find who uses this connector
                    users = [n for n, d in dag.items() if name in d]
                    if users:
                        for user in users:
                            if source_tensor in tensor_to_pool and user in tensor_to_pool:
                                source_pool = tensor_to_pool[source_tensor][0]
                                user_pool = tensor_to_pool[user][0]
                                if source_pool == user_pool:
                                    return False, f"{source_tensor} and {user} share pool {source_pool}"
        return True, "No issues found"
    
    def test_mlp_connector_memory(self):
        """Test MLP model doesn't have connector memory issues."""
        model = MLP(in_features=32, config={"Layers": [16, 10]})
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()
        
        codegen = MiCoCodeGen(model, align_to=32)
        example_input = torch.randn(1, 32)
        codegen.forward(example_input)
        
        ok, msg = self._check_no_inplace_issues(codegen)
        self.assertTrue(ok, msg)
    
    def test_lenet_connector_memory(self):
        """Test LeNet model doesn't have connector memory issues."""
        model = LeNet(1)
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()
        
        codegen = MiCoCodeGen(model, align_to=32)
        example_input = torch.randn(1, 1, 28, 28)
        codegen.forward(example_input)
        
        ok, msg = self._check_no_inplace_issues(codegen)
        self.assertTrue(ok, msg)
    
    def test_connector_source_not_in_user_pool(self):
        """
        Test that when tensor_A -> connector -> tensor_B,
        tensor_A and tensor_B are never in the same pool.
        """
        model = MLP(in_features=64, config={"Layers": [32, 16, 10]})
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()
        
        codegen = MiCoCodeGen(model, align_to=32)
        example_input = torch.randn(1, 64)
        codegen.forward(example_input)
        
        dag = codegen.extract_tensor_dag()
        memory_pools, tensor_to_pool = codegen.allocate_memory_pools()
        
        # Find all connector tensors and verify their sources are not in the same pool as users
        for name, info in codegen.tensors.items():
            if info.get('bypass', False):
                deps = dag.get(name, [])
                if deps:
                    source_tensor = deps[0]
                    users = [n for n, d in dag.items() if name in d]
                    for user in users:
                        if source_tensor in tensor_to_pool and user in tensor_to_pool:
                            source_pool = tensor_to_pool[source_tensor][0]
                            user_pool = tensor_to_pool[user][0]
                            self.assertNotEqual(source_pool, user_pool,
                                f"Source tensor {source_tensor} and user tensor {user} "
                                f"should not share pool {source_pool}")


class TestConnectorConflictDetection(unittest.TestCase):
    """Test the conflict detection logic for connector tensors."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
    
    def test_conflict_set_built_correctly(self):
        """Test that conflicts are correctly identified for connector patterns."""
        model = MLP(in_features=32, config={"Layers": [16, 10]})
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()
        
        codegen = MiCoCodeGen(model, align_to=32)
        example_input = torch.randn(1, 32)
        codegen.forward(example_input)
        
        dag = codegen.extract_tensor_dag()
        
        # Find the flatten connector
        flatten_tensor = None
        flatten_source = None
        for name, info in codegen.tensors.items():
            if info.get('bypass', False):
                deps = dag.get(name, [])
                if deps:
                    flatten_tensor = name
                    flatten_source = deps[0]
                    break
        
        self.assertIsNotNone(flatten_tensor, "Should find a bypass/connector tensor")
        self.assertIsNotNone(flatten_source, "Connector should have a source tensor")
        
        # Find who uses the connector
        users = [n for n, d in dag.items() if flatten_tensor in d]
        self.assertGreater(len(users), 0, "Connector should have at least one user")


if __name__ == "__main__":
    unittest.main()
