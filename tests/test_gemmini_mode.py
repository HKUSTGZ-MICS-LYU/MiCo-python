#!/usr/bin/env python3
"""
Test suite for Gemmini mode in MiCoCodeGen.
Tests weight transposition and layout transformations for Gemmini accelerator.
"""

import torch
import torch.nn as nn
import sys
import os
import tempfile
import shutil
import unittest

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import MLP, LeNet
from MiCoCodeGen import MiCoCodeGen
from MiCoUtils import fuse_model


class TestGemminiModeInitialization(unittest.TestCase):
    """Test Gemmini mode initialization."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
    
    def test_gemmini_mode_default_false(self):
        """Test that gemmini_mode defaults to False."""
        model = MLP(in_features=32, config={"Layers": [16, 10]})
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()
        
        codegen = MiCoCodeGen(model)
        
        self.assertFalse(codegen.gemmini_mode)
    
    def test_gemmini_mode_enabled(self):
        """Test that gemmini_mode can be enabled."""
        model = MLP(in_features=32, config={"Layers": [16, 10]})
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()
        
        codegen = MiCoCodeGen(model, gemmini_mode=True)
        
        self.assertTrue(codegen.gemmini_mode)


class TestGemminiModeLinearWeight(unittest.TestCase):
    """Test weight transposition for Linear layers in Gemmini mode."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_linear_weight_shape_normal_mode(self):
        """Test that Linear weight shape is [M, K] in normal mode."""
        # M = out_features, K = in_features
        model = MLP(in_features=32, config={"Layers": [16, 10]})
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()
        
        codegen = MiCoCodeGen(model, gemmini_mode=False)
        example_input = torch.randn(1, 32)
        codegen.forward(example_input)
        
        # Check that weight tensors have shape [M, K]
        # First layer: in_features=32, out_features=16
        weight_tensor = codegen.tensors['layers_0_weight']['tensor']
        self.assertEqual(weight_tensor.shape, (16, 32))  # [M, K]
    
    def test_linear_weight_shape_gemmini_mode(self):
        """Test that Linear weight shape is [K, M] in Gemmini mode."""
        # M = out_features, K = in_features
        model = MLP(in_features=32, config={"Layers": [16, 10]})
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()
        
        codegen = MiCoCodeGen(model, gemmini_mode=True)
        example_input = torch.randn(1, 32)
        codegen.forward(example_input)
        
        # Check that weight tensors have shape [K, M] (transposed)
        # First layer: in_features=32, out_features=16
        weight_tensor = codegen.tensors['layers_0_weight']['tensor']
        self.assertEqual(weight_tensor.shape, (32, 16))  # [K, M] - transposed
    
    def test_linear_weight_values_transposed(self):
        """Test that Linear weight values are correctly transposed in Gemmini mode."""
        model = MLP(in_features=32, config={"Layers": [16, 10]})
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()
        
        # Get original weight before codegen
        original_weight = model.layers[0].weight.detach().clone()
        
        # Normal mode
        codegen_normal = MiCoCodeGen(model, gemmini_mode=False)
        example_input = torch.randn(1, 32)
        codegen_normal.forward(example_input)
        normal_weight = codegen_normal.tensors['layers_0_weight']['tensor']
        
        # Gemmini mode
        codegen_gemmini = MiCoCodeGen(model, gemmini_mode=True)
        codegen_gemmini.forward(example_input)
        gemmini_weight = codegen_gemmini.tensors['layers_0_weight']['tensor']
        
        # Verify transposition: gemmini_weight should equal normal_weight.T
        self.assertTrue(torch.allclose(gemmini_weight, normal_weight.t()))


class TestGemminiModeConv2dWeight(unittest.TestCase):
    """Test weight layout for Conv2d layers in Gemmini mode."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_conv2d_weight_shape_gemmini_mode(self):
        """Test that Conv2d weight shape is KhKwIO in Gemmini mode."""
        model = LeNet(1)
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()
        
        codegen = MiCoCodeGen(model, gemmini_mode=True)
        example_input = torch.randn(1, 1, 28, 28)
        codegen.forward(example_input)
        
        # Check that weight tensor shape is [Kh, Kw, I, O] (KhKwIO format)
        # First conv layer has out_channels=6, in_channels=1, kernel_size=5
        # Tensor name is layers_0_weight
        weight_tensor = codegen.tensors['layers_0_weight']['tensor']
        self.assertEqual(len(weight_tensor.shape), 4)
        # Shape should be [5, 5, 1, 6] (KhKwIO format)
        self.assertEqual(weight_tensor.shape[0], 5)   # kernel_h
        self.assertEqual(weight_tensor.shape[1], 5)   # kernel_w
        self.assertEqual(weight_tensor.shape[2], 1)   # in_channels
        self.assertEqual(weight_tensor.shape[3], 6)   # out_channels
    
    def test_conv2d_weight_values_permuted(self):
        """Test that Conv2d weight values are correctly permuted in Gemmini mode."""
        model = LeNet(1)
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()
        
        # Get original weight before codegen
        original_weight = model.layers[0].weight.detach().clone()
        
        # Normal mode
        codegen_normal = MiCoCodeGen(model, gemmini_mode=False)
        example_input = torch.randn(1, 1, 28, 28)
        codegen_normal.forward(example_input)
        normal_weight = codegen_normal.tensors['layers_0_weight']['tensor']
        
        # Gemmini mode
        codegen_gemmini = MiCoCodeGen(model, gemmini_mode=True)
        codegen_gemmini.forward(example_input)
        gemmini_weight = codegen_gemmini.tensors['layers_0_weight']['tensor']
        
        # Verify permutation: gemmini_weight should equal normal_weight.permute(2, 3, 1, 0)
        expected_weight = normal_weight.permute(2, 3, 1, 0)
        self.assertTrue(torch.allclose(gemmini_weight, expected_weight))


class TestGemminiModeConvert(unittest.TestCase):
    """Test code generation with Gemmini mode."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_convert_mlp_gemmini_mode(self):
        """Test that convert generates files with Gemmini mode enabled."""
        model = MLP(in_features=32, config={"Layers": [16, 10]})
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()
        
        codegen = MiCoCodeGen(model, gemmini_mode=True)
        example_input = torch.randn(1, 32)
        codegen.forward(example_input)
        
        # Convert to C code
        output_dir = os.path.join(self.temp_dir, "mlp_gemmini")
        os.makedirs(output_dir, exist_ok=True)
        codegen.convert(output_dir, "test_mlp_gemmini", verbose=False, mem_pool=False)
        
        # Check that files were generated
        model_h_path = os.path.join(output_dir, "test_mlp_gemmini.h")
        model_bin_path = os.path.join(output_dir, "test_mlp_gemmini.bin")
        self.assertTrue(os.path.exists(model_h_path))
        self.assertTrue(os.path.exists(model_bin_path))
    
    def test_convert_lenet_gemmini_mode(self):
        """Test that convert generates files for LeNet with Gemmini mode enabled."""
        model = LeNet(1)
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()
        
        codegen = MiCoCodeGen(model, gemmini_mode=True)
        example_input = torch.randn(1, 1, 28, 28)
        codegen.forward(example_input)
        
        # Convert to C code
        output_dir = os.path.join(self.temp_dir, "lenet_gemmini")
        os.makedirs(output_dir, exist_ok=True)
        codegen.convert(output_dir, "test_lenet_gemmini", verbose=False, mem_pool=False)
        
        # Check that files were generated
        model_h_path = os.path.join(output_dir, "test_lenet_gemmini.h")
        model_bin_path = os.path.join(output_dir, "test_lenet_gemmini.bin")
        self.assertTrue(os.path.exists(model_h_path))
        self.assertTrue(os.path.exists(model_bin_path))


class TestGemminiModeWeightBinarySize(unittest.TestCase):
    """Test that weight binaries are properly generated in both modes."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_weight_content_generated_both_modes(self):
        """Test that weight content is generated in both modes."""
        model = MLP(in_features=32, config={"Layers": [16, 10]})
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()
        
        example_input = torch.randn(1, 32)
        
        # Normal mode
        codegen_normal = MiCoCodeGen(model, gemmini_mode=False)
        codegen_normal.forward(example_input)
        output_dir_normal = os.path.join(self.temp_dir, "mlp_normal")
        os.makedirs(output_dir_normal, exist_ok=True)
        codegen_normal.convert(output_dir_normal, "model", verbose=False, mem_pool=False)
        normal_size = len(codegen_normal.weight_content)
        
        # Gemmini mode
        codegen_gemmini = MiCoCodeGen(model, gemmini_mode=True)
        codegen_gemmini.forward(example_input)
        output_dir_gemmini = os.path.join(self.temp_dir, "mlp_gemmini")
        os.makedirs(output_dir_gemmini, exist_ok=True)
        codegen_gemmini.convert(output_dir_gemmini, "model", verbose=False, mem_pool=False)
        gemmini_size = len(codegen_gemmini.weight_content)
        
        # Both modes should generate non-empty weight content
        self.assertGreater(normal_size, 0)
        self.assertGreater(gemmini_size, 0)
        
        # Both binary files should exist and be non-empty
        normal_bin = os.path.join(output_dir_normal, "model.bin")
        gemmini_bin = os.path.join(output_dir_gemmini, "model.bin")
        self.assertTrue(os.path.exists(normal_bin))
        self.assertTrue(os.path.exists(gemmini_bin))
        self.assertGreater(os.path.getsize(normal_bin), 0)
        self.assertGreater(os.path.getsize(gemmini_bin), 0)


if __name__ == "__main__":
    unittest.main()
