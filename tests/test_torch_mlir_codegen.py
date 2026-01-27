#!/usr/bin/env python3
"""
Test suite for MiCoTorchMLIRGen class.

Tests the torch-mlir based MLIR code generation functionality including:
- Fallback behavior when torch-mlir is not available
- Basic initialization and configuration
- Output file generation
"""

import os
import sys
import tempfile
import shutil
import unittest

import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import MLP, LeNet
from MiCoTorchMLIRGen import MiCoTorchMLIRGen, check_torch_mlir_installation, TORCH_MLIR_AVAILABLE
from MiCoUtils import fuse_model


class TestMiCoTorchMLIRGenBasics(unittest.TestCase):
    """Test basic MiCoTorchMLIRGen functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test MiCoTorchMLIRGen initialization."""
        model = MLP(in_features=32, config={"Layers": [16, 10]})
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()
        
        mlir_gen = MiCoTorchMLIRGen(model)
        
        self.assertIsNotNone(mlir_gen)
        self.assertEqual(mlir_gen.dialect, "mico")
        self.assertEqual(mlir_gen.output_type, "torch")
    
    def test_initialization_with_output_type(self):
        """Test initialization with different output types."""
        model = MLP(in_features=32, config={"Layers": [16, 10]})
        model.set_qscheme([[8, 8], [8, 8]])
        model = fuse_model(model)
        model.eval()
        
        for output_type in ["torch", "linalg", "stablehlo"]:
            mlir_gen = MiCoTorchMLIRGen(model, output_type=output_type)
            self.assertEqual(mlir_gen.output_type, output_type)
    
    def test_check_torch_mlir_installation(self):
        """Test the installation check function."""
        status = check_torch_mlir_installation()
        
        self.assertIn("available", status)
        self.assertIn("version", status)
        self.assertIn("output_types", status)
        self.assertIn("install_command", status)
        
        self.assertIsInstance(status["available"], bool)
        self.assertIsInstance(status["output_types"], list)
    
    def test_is_torch_mlir_available(self):
        """Test the availability check method."""
        model = MLP(in_features=32, config={"Layers": [16, 10]})
        model.set_qscheme([[8, 8], [8, 8]])
        model = fuse_model(model)
        model.eval()
        
        mlir_gen = MiCoTorchMLIRGen(model)
        
        # Should match the global availability
        self.assertEqual(mlir_gen.is_torch_mlir_available(), TORCH_MLIR_AVAILABLE)


class TestMiCoTorchMLIRGenConvert(unittest.TestCase):
    """Test MLIR code conversion functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_convert_generates_file(self):
        """Test that convert generates an MLIR file."""
        model = MLP(in_features=32, config={"Layers": [16, 10]})
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()
        
        mlir_gen = MiCoTorchMLIRGen(model)
        mlir_gen.forward(torch.randn(1, 32))
        
        # Convert
        mlir_path = mlir_gen.convert(self.temp_dir, "test_model")
        
        # Check file exists (either torch_mlir or standalone)
        self.assertTrue(os.path.exists(mlir_path))
        self.assertTrue(mlir_path.endswith(".mlir"))
    
    def test_convert_without_forward_raises_error(self):
        """Test that convert raises error if forward not called."""
        model = MLP(in_features=32, config={"Layers": [16, 10]})
        model.set_qscheme([[8, 8], [8, 8]])
        model = fuse_model(model)
        model.eval()
        
        mlir_gen = MiCoTorchMLIRGen(model)
        
        with self.assertRaises(ValueError):
            mlir_gen.convert(self.temp_dir, "test_model")
    
    def test_convert_fallback_mode(self):
        """Test that fallback to standalone mode works."""
        model = MLP(in_features=32, config={"Layers": [16, 10]})
        model.set_qscheme([[8, 8], [8, 8]])
        model = fuse_model(model)
        model.eval()
        
        mlir_gen = MiCoTorchMLIRGen(model)
        mlir_gen.forward(torch.randn(1, 32))
        
        # Force fallback mode
        original_use_torch_mlir = mlir_gen.use_torch_mlir
        mlir_gen.use_torch_mlir = False
        
        mlir_path = mlir_gen.convert(self.temp_dir, "fallback_test")
        
        # Restore
        mlir_gen.use_torch_mlir = original_use_torch_mlir
        
        self.assertTrue(os.path.exists(mlir_path))
        
        with open(mlir_path, 'r') as f:
            content = f.read()
        
        # Should have MiCo dialect content
        self.assertIn("module", content)


class TestMiCoTorchMLIRGenLeNet(unittest.TestCase):
    """Test torch-mlir generation for LeNet model."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_lenet_conversion(self):
        """Test LeNet model conversion."""
        model = LeNet(1)
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()
        
        mlir_gen = MiCoTorchMLIRGen(model)
        mlir_gen.forward(torch.randn(1, 1, 28, 28))
        mlir_path = mlir_gen.convert(self.temp_dir, "lenet_test")
        
        self.assertTrue(os.path.exists(mlir_path))
    
    def test_lenet_mixed_precision(self):
        """Test LeNet with mixed precision conversion."""
        model = LeNet(1)
        weight_q = [8, 6, 6, 4, 4]  # Mixed precision
        activation_q = [8, 8, 8, 8, 8]
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()
        
        mlir_gen = MiCoTorchMLIRGen(model)
        mlir_gen.forward(torch.randn(1, 1, 28, 28))
        mlir_path = mlir_gen.convert(self.temp_dir, "lenet_mixed")
        
        self.assertTrue(os.path.exists(mlir_path))


class TestMiCoTorchMLIRGenReset(unittest.TestCase):
    """Test reset functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
    
    def test_reset_clears_state(self):
        """Test that reset clears the generator state."""
        model = MLP(in_features=32, config={"Layers": [16, 10]})
        model.set_qscheme([[8, 8], [8, 8]])
        model = fuse_model(model)
        model.eval()
        
        mlir_gen = MiCoTorchMLIRGen(model)
        mlir_gen.forward(torch.randn(1, 32))
        
        # Check state is populated
        self.assertIsNotNone(mlir_gen.example_inputs)
        
        # Reset
        mlir_gen.reset()
        
        # Check state is cleared
        self.assertIsNone(mlir_gen.example_inputs)
        self.assertIsNone(mlir_gen.torch_mlir_ir)
        self.assertEqual(len(mlir_gen.mico_quantization_ops), 0)


if __name__ == "__main__":
    unittest.main()
