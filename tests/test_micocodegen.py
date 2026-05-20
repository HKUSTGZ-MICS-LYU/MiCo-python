#!/usr/bin/env python3
"""
Test suite for MiCoCodeGen class.
Tests core functionality of the code generator including initialization,
graph extraction, tracing, and code generation.
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
from MiCoCodeGen import MiCoCodeGen, MiCoTrace
from MiCoUtils import fuse_model


class Pooling1dModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg1d = nn.AvgPool1d(kernel_size=5, stride=(4,), padding=(2,))

    def forward(self, x):
        return self.avg1d(x)


class Pooling2dModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.max2d = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

    def forward(self, x):
        y2 = torch.nn.functional.avg_pool2d(
            x,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
        )
        y3 = self.max2d(x)
        return y2 + y3


class TestMiCoTrace(unittest.TestCase):
    """Test the MiCoTrace tracer class."""
    
    def test_tracer_initialization(self):
        """Test that MiCoTrace can be instantiated."""
        tracer = MiCoTrace()
        self.assertIsNotNone(tracer)
    
    def test_tracer_simple_model(self):
        """Test tracing a simple model."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        tracer = MiCoTrace()
        graph = tracer.trace(model)
        self.assertIsNotNone(graph)
        # Check that we have the expected number of nodes
        # (placeholder + 3 modules + output)
        node_count = len(list(graph.nodes))
        self.assertGreater(node_count, 0)


class TestMiCoCodeGenBasics(unittest.TestCase):
    """Test basic MiCoCodeGen functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization_mlp(self):
        """Test MiCoCodeGen initialization with MLP model."""
        model = MLP(in_features=64, config={"Layers": [32, 16, 10]})
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()
        
        codegen = MiCoCodeGen(model, align_to=32)
        
        self.assertIsNotNone(codegen)
        self.assertIsNotNone(codegen.model)
        self.assertIsNotNone(codegen.graph)
        self.assertIsNotNone(codegen.gm)
        self.assertEqual(codegen.align_to, 32)
    
    def test_initialization_lenet(self):
        """Test MiCoCodeGen initialization with LeNet model."""
        model = LeNet(1)
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()
        
        codegen = MiCoCodeGen(model, align_to=16)
        
        self.assertIsNotNone(codegen)
        self.assertEqual(codegen.align_to, 16)
    
    def test_reset(self):
        """Test the reset method."""
        model = MLP(in_features=32, config={"Layers": [16, 10]})
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()
        
        codegen = MiCoCodeGen(model)
        
        # Add some data
        codegen.model_struct.append("test")
        codegen.model_init.append("test")
        codegen.model_forward.append("test")
        
        # Reset
        codegen.reset()
        
        # Verify everything is cleared
        self.assertEqual(len(codegen.model_struct), 0)
        self.assertEqual(len(codegen.model_init), 0)
        self.assertEqual(len(codegen.model_forward), 0)
        self.assertEqual(codegen.weight_content, b"")


class TestMiCoCodeGenDtypeConversion(unittest.TestCase):
    """Test dtype conversion utility methods."""
    
    def test_get_dtype_str_float32(self):
        """Test dtype string conversion for float32."""
        dtype_str = MiCoCodeGen.get_dtype_str(torch.float32)
        self.assertEqual(dtype_str, "F32")
    
    def test_get_dtype_str_float16(self):
        """Test dtype string conversion for float16."""
        dtype_str = MiCoCodeGen.get_dtype_str(torch.float16)
        self.assertEqual(dtype_str, "F16")
    
    def test_get_dtype_str_unsupported(self):
        """Test dtype string conversion for unsupported dtype."""
        with self.assertRaises(ValueError):
            MiCoCodeGen.get_dtype_str(torch.int32)
    
    def test_get_ctype_str_float32(self):
        """Test C type string conversion for float32."""
        ctype_str = MiCoCodeGen.get_ctype_str(torch.float32)
        self.assertEqual(ctype_str, "float32")
    
    def test_get_ctype_str_float16(self):
        """Test C type string conversion for float16."""
        ctype_str = MiCoCodeGen.get_ctype_str(torch.float16)
        self.assertEqual(ctype_str, "float16_t")
    
    def test_get_ctype_str_unsupported(self):
        """Test C type string conversion for unsupported dtype."""
        with self.assertRaises(ValueError):
            MiCoCodeGen.get_ctype_str(torch.int64)


class TestMiCoCodeGenForward(unittest.TestCase):
    """Test forward pass functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
    
    def test_forward_mlp(self):
        """Test forward pass with MLP model."""
        model = MLP(in_features=32, config={"Layers": [16, 10]})
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()
        
        codegen = MiCoCodeGen(model)
        example_input = torch.randn(1, 32)
        
        # Forward pass
        output = codegen.forward(example_input)
        
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (1, 10))

    def test_forward_mlp_with_qnorm_emits_rmsnorm(self):
        """Test QLayer use_norm emits RMSNorm before the quantized layer."""
        model = MLP(in_features=32, config={"Layers": [16]})
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q], qat=True, use_norm=True)
        model = fuse_model(model)
        model.eval()

        codegen = MiCoCodeGen(model)
        codegen.forward(torch.randn(1, 32))

        forward_code = "\n".join(codegen.model_forward)
        self.assertIn("MiCo_simple_rmsnorm2d_f32", forward_code)
        self.assertIn("layers_0_rmsnorm", codegen.tensors)
        self.assertIn("&model->layers_0_rmsnorm, &model->layers_0_weight", forward_code)
    
    def test_forward_lenet(self):
        """Test forward pass with LeNet model."""
        model = LeNet(1)
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()
        
        codegen = MiCoCodeGen(model)
        example_input = torch.randn(1, 1, 28, 28)
        
        # Forward pass
        output = codegen.forward(example_input)
        
        self.assertIsNotNone(output)
        self.assertEqual(output.shape[0], 1)

    def test_pooling_parameters_are_emitted_as_c_scalars(self):
        """Tuple pooling parameters should not be stringified into C calls."""
        codegen = MiCoCodeGen(Pooling1dModel().eval())
        codegen.forward(torch.randn(1, 2, 16))
        forward_code = "\n".join(codegen.model_forward)
        self.assertIn("MiCo_avgpool3d_f32", forward_code)
        self.assertNotIn("(4,)", forward_code)
        self.assertNotIn("(2,)", forward_code)

        codegen = MiCoCodeGen(Pooling2dModel().eval())
        codegen.forward(torch.randn(1, 2, 16, 16))
        forward_code = "\n".join(codegen.model_forward)
        self.assertIn("MiCo_avgpool4d_f32", forward_code)
        self.assertIn("MiCo_maxpool4d_f32", forward_code)
        self.assertNotIn("(2, 2)", forward_code)
        self.assertNotIn("(1, 1)", forward_code)


class TestMiCoCodeGenTensorManagement(unittest.TestCase):
    """Test tensor management methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        model = MLP(in_features=16, config={"Layers": [8, 4]})
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()
        self.codegen = MiCoCodeGen(model)
    
    def test_add_uninitialized_tensor(self):
        """Test adding an uninitialized tensor."""
        tensor = torch.randn(4, 4)
        self.codegen.add_uninitialized_tensor("test_tensor", tensor, quant=0)
        
        self.assertIn("test_tensor", self.codegen.tensors)
        self.assertFalse(self.codegen.tensors["test_tensor"]["initialized"])
        self.assertEqual(self.codegen.tensors["test_tensor"]["quantized"], 0)
    
    def test_add_initialized_tensor(self):
        """Test adding an initialized tensor."""
        tensor = torch.randn(4, 4)
        self.codegen.add_initialized_tensor("test_tensor_init", tensor, quant=8, scale=0.1)
        
        self.assertIn("test_tensor_init", self.codegen.tensors)
        self.assertTrue(self.codegen.tensors["test_tensor_init"]["initialized"])
        self.assertEqual(self.codegen.tensors["test_tensor_init"]["quantized"], 8)
        self.assertEqual(self.codegen.tensors["test_tensor_init"]["scale"], 0.1)
        self.assertFalse(self.codegen.tensors["test_tensor_init"]["bypass"])
    
    def test_add_connect_tensor(self):
        """Test adding a connect tensor."""
        tensor = torch.randn(4, 4)
        self.codegen.add_connect_tensor("test_tensor_connect", tensor, quant=8)
        
        self.assertIn("test_tensor_connect", self.codegen.tensors)
        self.assertFalse(self.codegen.tensors["test_tensor_connect"]["initialized"])
        self.assertEqual(self.codegen.tensors["test_tensor_connect"]["quantized"], 8)
        self.assertTrue(self.codegen.tensors["test_tensor_connect"]["bypass"])


class TestMiCoCodeGenConversion(unittest.TestCase):
    """Test code generation and conversion functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_convert_without_forward_raises_error(self):
        """Test that convert raises error if forward hasn't been called."""
        model = MLP(in_features=16, config={"Layers": [8, 4]})
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()
        
        codegen = MiCoCodeGen(model)
        
        # Try to convert without calling forward first
        with self.assertRaises((ValueError, AttributeError)):
            codegen.convert(self.temp_dir, "test_model")
    
    def test_convert_mlp_generates_files(self):
        """Test that convert generates expected C files for MLP."""
        model = MLP(in_features=32, config={"Layers": [16, 10]})
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()
        
        codegen = MiCoCodeGen(model)
        example_input = torch.randn(1, 32)
        codegen.forward(example_input)
        
        # Convert to C code
        output_dir = os.path.join(self.temp_dir, "mlp_output")
        os.makedirs(output_dir, exist_ok=True)
        codegen.convert(output_dir, "test_mlp", verbose=False, mem_pool=False)
        
        # Check that test_mlp.h was generated (not model.h)
        model_h_path = os.path.join(output_dir, "test_mlp.h")
        self.assertTrue(os.path.exists(model_h_path))
        
        # Check that test_mlp.bin was generated
        model_bin_path = os.path.join(output_dir, "test_mlp.bin")
        self.assertTrue(os.path.exists(model_bin_path))

    def test_convert_with_mem_pool(self):
        """Test that convert works with memory pooling enabled."""
        model = MLP(in_features=32, config={"Layers": [16, 10]})
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()
        
        codegen = MiCoCodeGen(model)
        example_input = torch.randn(1, 32)
        codegen.forward(example_input)
        
        # Convert with memory pooling
        output_dir = os.path.join(self.temp_dir, "mlp_mempool")
        os.makedirs(output_dir, exist_ok=True)
        codegen.convert(output_dir, "test_mlp_pool", verbose=False, mem_pool=True)
        
        # Check that files were generated (correct filenames)
        model_h_path = os.path.join(output_dir, "test_mlp_pool.h")
        self.assertTrue(os.path.exists(model_h_path))

    def test_benchmark_groups_dedupe_same_shape_calls(self):
        """Test benchmark grouping merges same-shape kernels with different layer names."""
        model = nn.Sequential(
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
        )
        model.eval()

        codegen = MiCoCodeGen(model)
        codegen.forward(torch.randn(1, 8))
        groups = codegen.get_benchmark_call_groups()

        group_counts = {group["function_name"]: group["count"] for group in groups}
        self.assertEqual(group_counts["MiCo_linear_f32"], 2)
        self.assertEqual(group_counts["MiCo_relu2d_f32"], 2)
        self.assertEqual(len(groups), 2)

    def test_benchmark_groups_split_different_shape_calls(self):
        """Test benchmark grouping keeps kernels with different shapes separate."""
        model = nn.Sequential(
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )
        model.eval()

        codegen = MiCoCodeGen(model)
        codegen.forward(torch.randn(1, 8))
        groups = codegen.get_benchmark_call_groups()

        linear_groups = [group for group in groups if group["function_name"] == "MiCo_linear_f32"]
        self.assertEqual(len(linear_groups), 2)
        self.assertEqual([group["count"] for group in linear_groups], [1, 1])

    def test_convert_with_benchmark_mode_generates_estimate(self):
        """Test benchmark mode emits unique-kernel timing and estimated runtime."""
        model = nn.Sequential(
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
        )
        model.eval()

        codegen = MiCoCodeGen(model)
        codegen.forward(torch.randn(1, 8))

        output_dir = os.path.join(self.temp_dir, "benchmark_output")
        os.makedirs(output_dir, exist_ok=True)
        codegen.convert(output_dir, "benchmark_model", verbose=False, mem_pool=False, benchmark_mode=True)

        model_h_path = os.path.join(output_dir, "benchmark_model.h")
        self.assertTrue(os.path.exists(model_h_path))
        with open(model_h_path) as f:
            model_h = f.read()

        self.assertIn('printf("Benchmark Mode: %d unique kernels, %d total occurrences\\n", 2, 4);', model_h)
        self.assertIn("occurrences=2", model_h)
        self.assertIn("Estimated Execution Time: %ld", model_h)
        self.assertNotIn('printf("Execution Time: %ld\\n", profile_time);', model_h)


class TestMiCoCodeGenModuleAccess(unittest.TestCase):
    """Test module access functionality."""
    
    def test_get_module_simple(self):
        """Test getting a module by name."""
        model = MLP(in_features=32, config={"Layers": [16, 10]})
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()
        
        codegen = MiCoCodeGen(model)
        
        # Get a module (this assumes the model has 'layers' attribute)
        if hasattr(model, 'layers'):
            module = codegen.get_module('layers')
            self.assertIsNotNone(module)


if __name__ == "__main__":
    unittest.main()
