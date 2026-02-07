#!/usr/bin/env python3
"""
Test suite for MiCoMLIRGen class.

Tests the MLIR code generation functionality including:
- Basic initialization and tracing
- Module handler tests (BitLinear, BitConv2d, etc.)
- Function handler tests (relu, add, etc.)
- Full model conversion tests
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
from MiCoMLIRGen import MiCoMLIRGen
from MiCoUtils import fuse_model


class TestMiCoMLIRGenBasics(unittest.TestCase):
    """Test basic MiCoMLIRGen functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test MiCoMLIRGen initialization."""
        model = MLP(in_features=32, config={"Layers": [16, 10]})
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()
        
        mlir_gen = MiCoMLIRGen(model)
        
        self.assertIsNotNone(mlir_gen)
        self.assertEqual(mlir_gen.dialect, "mico")
        self.assertEqual(len(mlir_gen.mlir_ops), 0)
        self.assertEqual(len(mlir_gen.mlir_weights), 0)
    
    def test_reset(self):
        """Test the reset method."""
        model = MLP(in_features=32, config={"Layers": [16, 10]})
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()
        
        mlir_gen = MiCoMLIRGen(model)
        
        # Add some data
        mlir_gen.mlir_ops.append("test_op")
        mlir_gen.mlir_weights["test"] = {}
        
        # Reset
        mlir_gen.reset()
        
        # Verify cleared
        self.assertEqual(len(mlir_gen.mlir_ops), 0)
        self.assertEqual(len(mlir_gen.mlir_weights), 0)
        self.assertEqual(mlir_gen.ssa_counter, 0)
    
    def test_format_tensor_type_float32(self):
        """Test tensor type formatting for float32."""
        model = MLP(in_features=16, config={"Layers": [8]})
        model.set_qscheme([[8], [8]])
        model = fuse_model(model)
        model.eval()
        
        mlir_gen = MiCoMLIRGen(model)
        
        tensor = torch.randn(1, 64, 28, 28)
        type_str = mlir_gen._format_tensor_type(tensor)
        
        self.assertEqual(type_str, "tensor<1x64x28x28xf32>")
    
    def test_format_tensor_type_quantized(self):
        """Test tensor type formatting for quantized types."""
        model = MLP(in_features=16, config={"Layers": [8]})
        model.set_qscheme([[8], [8]])
        model = fuse_model(model)
        model.eval()
        
        mlir_gen = MiCoMLIRGen(model)
        
        tensor = torch.randn(64, 32)
        type_str = mlir_gen._format_tensor_type(tensor, qbit=4)
        
        self.assertEqual(type_str, "tensor<64x32x!mico.int<4>>")
    
    def test_ssa_naming(self):
        """Test SSA value naming."""
        model = MLP(in_features=16, config={"Layers": [8]})
        model.set_qscheme([[8], [8]])
        model = fuse_model(model)
        model.eval()
        
        mlir_gen = MiCoMLIRGen(model)
        
        name1 = mlir_gen._get_ssa_name()
        name2 = mlir_gen._get_ssa_name()
        name3 = mlir_gen._get_ssa_name("conv")
        
        self.assertEqual(name1, "%v0")
        self.assertEqual(name2, "%v1")
        self.assertEqual(name3, "%conv2")


class TestMiCoMLIRGenForward(unittest.TestCase):
    """Test forward pass functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_forward_mlp(self):
        """Test forward pass with MLP model."""
        model = MLP(in_features=32, config={"Layers": [16, 10]})
        weight_q = [8, 6]  # Mixed precision
        activation_q = [8, 8]
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()
        
        mlir_gen = MiCoMLIRGen(model)
        example_input = torch.randn(1, 32)
        
        # Forward pass
        output = mlir_gen.forward(example_input)
        
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (1, 10))
        
        # Check that operations were generated
        self.assertGreater(len(mlir_gen.mlir_ops), 0)
    
    def test_forward_lenet(self):
        """Test forward pass with LeNet model."""
        model = LeNet(1)
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()
        
        mlir_gen = MiCoMLIRGen(model)
        example_input = torch.randn(1, 1, 28, 28)
        
        # Forward pass
        output = mlir_gen.forward(example_input)
        
        self.assertIsNotNone(output)
        self.assertEqual(output.shape[0], 1)
        
        # Check operations generated
        self.assertGreater(len(mlir_gen.mlir_ops), 0)
        
        # Check weight constants generated
        self.assertGreater(len(mlir_gen.mlir_weights), 0)


class TestMiCoMLIRGenConvert(unittest.TestCase):
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
        """Test that convert generates MLIR file."""
        model = MLP(in_features=32, config={"Layers": [16, 10]})
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()
        
        mlir_gen = MiCoMLIRGen(model)
        mlir_gen.forward(torch.randn(1, 32))
        
        # Convert
        mlir_path = mlir_gen.convert(self.temp_dir, "test_model")
        
        # Check file exists
        self.assertTrue(os.path.exists(mlir_path))
        self.assertTrue(mlir_path.endswith(".mlir"))
    
    def test_convert_without_forward_raises_error(self):
        """Test that convert raises error if forward not called."""
        model = MLP(in_features=32, config={"Layers": [16, 10]})
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()
        
        mlir_gen = MiCoMLIRGen(model)
        
        with self.assertRaises(ValueError):
            mlir_gen.convert(self.temp_dir, "test_model")
    
    def test_convert_content_has_module(self):
        """Test that generated MLIR has module structure."""
        model = MLP(in_features=32, config={"Layers": [16, 10]})
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()
        
        mlir_gen = MiCoMLIRGen(model)
        mlir_gen.forward(torch.randn(1, 32))
        mlir_path = mlir_gen.convert(self.temp_dir, "test_model")
        
        with open(mlir_path, 'r') as f:
            content = f.read()
        
        # Check for essential MLIR structures
        self.assertIn("module @test_model", content)
        self.assertIn("func.func @forward", content)
        self.assertIn("return", content)
    
    def test_convert_content_has_mico_ops(self):
        """Test that generated MLIR contains mico dialect operations."""
        model = MLP(in_features=32, config={"Layers": [16, 10]})
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()
        
        mlir_gen = MiCoMLIRGen(model)
        mlir_gen.forward(torch.randn(1, 32))
        mlir_path = mlir_gen.convert(self.temp_dir, "test_model")
        
        with open(mlir_path, 'r') as f:
            content = f.read()
        
        # Check for MiCo dialect operations
        self.assertIn("mico.bitlinear", content)
        self.assertIn("mico.relu", content)
    
    def test_convert_mixed_precision(self):
        """Test MLIR generation with mixed precision."""
        model = MLP(in_features=64, config={"Layers": [32, 16, 10]})
        weight_q = [8, 6, 4]  # Different precisions
        activation_q = [8, 8, 8]
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()
        
        mlir_gen = MiCoMLIRGen(model)
        mlir_gen.forward(torch.randn(1, 64))
        mlir_path = mlir_gen.convert(self.temp_dir, "mixed_precision_model")
        
        with open(mlir_path, 'r') as f:
            content = f.read()
        
        # Check that different bit widths are present
        self.assertIn("weight_bits = 8", content)
        self.assertIn("weight_bits = 6", content)
        self.assertIn("weight_bits = 4", content)


class TestMiCoMLIRGenLeNet(unittest.TestCase):
    """Test MLIR generation for LeNet model."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_lenet_conv_ops(self):
        """Test that LeNet generates conv2d operations."""
        model = LeNet(1)
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()
        
        mlir_gen = MiCoMLIRGen(model)
        mlir_gen.forward(torch.randn(1, 1, 28, 28))
        mlir_path = mlir_gen.convert(self.temp_dir, "lenet_test")
        
        with open(mlir_path, 'r') as f:
            content = f.read()
        
        # Check for conv2d operations
        self.assertIn("mico.bitconv2d", content)
    
    def test_lenet_pooling_ops(self):
        """Test that LeNet generates pooling operations."""
        model = LeNet(1)
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()
        
        mlir_gen = MiCoMLIRGen(model)
        mlir_gen.forward(torch.randn(1, 1, 28, 28))
        mlir_path = mlir_gen.convert(self.temp_dir, "lenet_test")
        
        with open(mlir_path, 'r') as f:
            content = f.read()
        
        # Check for pooling operations (LeNet uses avgpool)
        self.assertIn("mico.avgpool2d", content)
    
    def test_lenet_weight_constants(self):
        """Test that LeNet generates weight constants."""
        model = LeNet(1)
        weight_q = [8, 6, 6, 4, 4]  # Mixed precision
        activation_q = [8, 8, 8, 8, 8]
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()
        
        mlir_gen = MiCoMLIRGen(model)
        mlir_gen.forward(torch.randn(1, 1, 28, 28))
        mlir_path = mlir_gen.convert(self.temp_dir, "lenet_test")
        
        with open(mlir_path, 'r') as f:
            content = f.read()
        
        # Check for weight constants
        self.assertIn("mico.constant", content)
        self.assertIn("_weight", content)
        self.assertIn("_bias", content)


class TestMiCoMLIRGenGetMLIRCode(unittest.TestCase):
    """Test get_mlir_code method."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
    
    def test_get_mlir_code_before_forward_raises(self):
        """Test that get_mlir_code raises error before forward."""
        model = MLP(in_features=32, config={"Layers": [16, 10]})
        model.set_qscheme([[8, 8], [8, 8]])
        model = fuse_model(model)
        model.eval()
        
        mlir_gen = MiCoMLIRGen(model)
        
        with self.assertRaises(ValueError):
            mlir_gen.get_mlir_code()
    
    def test_get_mlir_code_returns_string(self):
        """Test that get_mlir_code returns a string."""
        model = MLP(in_features=32, config={"Layers": [16, 10]})
        model.set_qscheme([[8, 8], [8, 8]])
        model = fuse_model(model)
        model.eval()
        
        mlir_gen = MiCoMLIRGen(model)
        mlir_gen.forward(torch.randn(1, 32))
        
        code = mlir_gen.get_mlir_code()
        
        self.assertIsInstance(code, str)
        self.assertGreater(len(code), 0)


class TestMiCoMLIRGenCustomOperations(unittest.TestCase):
    """Test handling of various PyTorch operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_relu6_operation(self):
        """Test ReLU6 activation handling."""
        class ModelWithReLU6(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(32, 16)
                self.relu6 = nn.ReLU6()
                self.n_layers = 1
            
            def forward(self, x):
                x = self.fc(x)
                x = self.relu6(x)
                return x
        
        from MiCoUtils import replace_quantize_layers
        
        model = ModelWithReLU6()
        replace_quantize_layers(model, [8], [8])  # Modifies model in-place
        model.eval()
        
        mlir_gen = MiCoMLIRGen(model)
        mlir_gen.forward(torch.randn(1, 32))
        mlir_path = mlir_gen.convert(self.temp_dir, "relu6_test")
        
        with open(mlir_path, 'r') as f:
            content = f.read()
        
        self.assertIn("mico.relu6", content)
    
    def test_flatten_operation(self):
        """Test Flatten operation handling."""
        class ModelWithFlatten(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 8, 3)
                self.flatten = nn.Flatten()
                self.fc = nn.Linear(8 * 26 * 26, 10)
                self.n_layers = 2
            
            def forward(self, x):
                x = self.conv(x)
                x = self.flatten(x)
                x = self.fc(x)
                return x
        
        from MiCoUtils import replace_quantize_layers
        
        model = ModelWithFlatten()
        replace_quantize_layers(model, [8, 8], [8, 8])  # Modifies model in-place
        model.eval()
        
        mlir_gen = MiCoMLIRGen(model)
        mlir_gen.forward(torch.randn(1, 1, 28, 28))
        mlir_path = mlir_gen.convert(self.temp_dir, "flatten_test")
        
        with open(mlir_path, 'r') as f:
            content = f.read()
        
        self.assertIn("mico.flatten", content)


if __name__ == "__main__":
    unittest.main()
