#!/usr/bin/env python3
"""
Test suite for MiCoONNXGen class.
Tests ONNX export functionality including initialization,
per-layer bitwidth metadata export, and model re-loading.
"""

import torch
import torch.nn as nn
import sys
import os
import tempfile
import shutil
import json
import unittest

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import MLP, LeNet
from MiCoONNXGen import MiCoONNXGen
from MiCoUtils import fuse_model


class TestMiCoONNXGenInit(unittest.TestCase):
    """Test MiCoONNXGen initialization."""

    def test_init_mlp(self):
        """Test initialization with an MLP model."""
        model = MLP(in_features=32, config={"Layers": [16, 10]})
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()

        exporter = MiCoONNXGen(model)
        self.assertIsNotNone(exporter)
        self.assertIsNotNone(exporter.model)
        self.assertIsNotNone(exporter.graph)
        self.assertIsNotNone(exporter.gm)

    def test_init_lenet(self):
        """Test initialization with a LeNet model."""
        model = LeNet(1)
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()

        exporter = MiCoONNXGen(model)
        self.assertIsNotNone(exporter)


class TestMiCoONNXGenBitwidthCollection(unittest.TestCase):
    """Test bitwidth metadata collection."""

    def test_collect_uniform_bitwidths(self):
        """Test collecting uniform (all-8-bit) bitwidths."""
        model = MLP(in_features=32, config={"Layers": [16, 10]})
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()

        exporter = MiCoONNXGen(model)
        info = exporter._collect_bitwidth_info()

        self.assertGreater(len(info), 0)
        for layer_name, entry in info.items():
            self.assertEqual(entry["weight_bitwidth"], 8)
            self.assertEqual(entry["activation_bitwidth"], 8)
            self.assertIn("layer_type", entry)

    def test_collect_mixed_bitwidths(self):
        """Test collecting mixed-precision bitwidths."""
        model = MLP(in_features=32, config={"Layers": [16, 10]})
        weight_q = [4, 8]
        activation_q = [8, 4]
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()

        exporter = MiCoONNXGen(model)
        info = exporter._collect_bitwidth_info()

        self.assertEqual(len(info), 2)
        bitwidths = list(info.values())
        self.assertEqual(bitwidths[0]["weight_bitwidth"], 4)
        self.assertEqual(bitwidths[0]["activation_bitwidth"], 8)
        self.assertEqual(bitwidths[1]["weight_bitwidth"], 8)
        self.assertEqual(bitwidths[1]["activation_bitwidth"], 4)

    def test_collect_lenet_bitwidths(self):
        """Test collecting bitwidths for LeNet (Conv2d + Linear layers)."""
        model = LeNet(1)
        weight_q = [8, 6, 6, 4, 4]
        activation_q = [8, 8, 8, 8, 8]
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()

        exporter = MiCoONNXGen(model)
        info = exporter._collect_bitwidth_info()

        self.assertEqual(len(info), 5)
        wbits = [e["weight_bitwidth"] for e in info.values()]
        self.assertEqual(wbits, [8, 6, 6, 4, 4])

    def test_layer_types_lenet(self):
        """Test that layer types are correctly identified for LeNet."""
        model = LeNet(1)
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()

        exporter = MiCoONNXGen(model)
        info = exporter._collect_bitwidth_info()

        types = [e["layer_type"] for e in info.values()]
        # LeNet has 2 Conv2d layers followed by 3 Linear layers
        self.assertEqual(types.count("Conv2d"), 2)
        self.assertEqual(types.count("Linear"), 3)


class TestMiCoONNXGenExport(unittest.TestCase):
    """Test ONNX export and metadata persistence."""

    def setUp(self):
        torch.manual_seed(42)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_export_mlp_creates_file(self):
        """Test that export creates an .onnx file for MLP."""
        model = MLP(in_features=32, config={"Layers": [16, 10]})
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()

        exporter = MiCoONNXGen(model)
        path = exporter.export(self.temp_dir, "test_mlp", torch.randn(1, 32))

        self.assertTrue(os.path.exists(path))
        self.assertTrue(path.endswith(".onnx"))

    def test_export_lenet_creates_file(self):
        """Test that export creates an .onnx file for LeNet."""
        model = LeNet(1)
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()

        exporter = MiCoONNXGen(model)
        path = exporter.export(self.temp_dir, "test_lenet", torch.randn(1, 1, 28, 28))

        self.assertTrue(os.path.exists(path))

    def test_export_metadata_roundtrip(self):
        """Test that bitwidth metadata survives export -> load."""
        model = MLP(in_features=32, config={"Layers": [16, 10]})
        weight_q = [4, 8]
        activation_q = [8, 4]
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()

        exporter = MiCoONNXGen(model)
        path = exporter.export(self.temp_dir, "test_roundtrip", torch.randn(1, 32))

        # Reload metadata
        loaded_info = MiCoONNXGen.load_bitwidth_info(path)

        self.assertEqual(len(loaded_info), 2)
        values = list(loaded_info.values())
        self.assertEqual(values[0]["weight_bitwidth"], 4)
        self.assertEqual(values[0]["activation_bitwidth"], 8)
        self.assertEqual(values[1]["weight_bitwidth"], 8)
        self.assertEqual(values[1]["activation_bitwidth"], 4)

    def test_export_lenet_mixed_precision_roundtrip(self):
        """Test mixed-precision metadata roundtrip for LeNet."""
        model = LeNet(1)
        weight_q = [8, 6, 6, 4, 4]
        activation_q = [8, 8, 8, 8, 8]
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()

        exporter = MiCoONNXGen(model)
        path = exporter.export(self.temp_dir, "test_lenet_mpq", torch.randn(1, 1, 28, 28))

        loaded_info = MiCoONNXGen.load_bitwidth_info(path)

        self.assertEqual(len(loaded_info), 5)
        wbits = [e["weight_bitwidth"] for e in loaded_info.values()]
        self.assertEqual(wbits, [8, 6, 6, 4, 4])

    def test_export_custom_names(self):
        """Test export with custom input/output names."""
        model = MLP(in_features=32, config={"Layers": [16, 10]})
        weight_q = [8] * model.n_layers
        activation_q = [8] * model.n_layers
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()

        exporter = MiCoONNXGen(model)
        path = exporter.export(
            self.temp_dir,
            "test_custom_names",
            torch.randn(1, 32),
            input_names=["my_input"],
            output_names=["my_output"],
        )

        import onnx
        onnx_model = onnx.load(path)
        self.assertEqual(onnx_model.graph.input[0].name, "my_input")
        self.assertEqual(onnx_model.graph.output[0].name, "my_output")

    def test_export_creates_directory(self):
        """Test that export creates the output directory if needed."""
        nested_dir = os.path.join(self.temp_dir, "a", "b", "c")
        model = MLP(in_features=16, config={"Layers": [8]})
        weight_q = [8]
        activation_q = [8]
        model.set_qscheme([weight_q, activation_q])
        model = fuse_model(model)
        model.eval()

        exporter = MiCoONNXGen(model)
        path = exporter.export(nested_dir, "nested_model", torch.randn(1, 16))

        self.assertTrue(os.path.exists(path))

    def test_load_bitwidth_info_no_metadata(self):
        """Test load_bitwidth_info on a plain ONNX file without MiCo metadata."""
        # Export a plain model without MiCo metadata
        model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))
        model.eval()
        path = os.path.join(self.temp_dir, "plain.onnx")
        torch.onnx.export(model, torch.randn(1, 8), path, opset_version=17)

        loaded_info = MiCoONNXGen.load_bitwidth_info(path)
        self.assertEqual(loaded_info, {})


if __name__ == "__main__":
    unittest.main()
