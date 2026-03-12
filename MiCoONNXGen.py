import os
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn
import torch.fx

import onnx

from MiCoQLayers import BitQLayer, BitConv2d, BitConv1d, BitLinear
from MiCoCodeGen import MiCoTrace


class MiCoONNXGen(torch.fx.Interpreter):
    """
    ONNX exporter for mixed-precision quantized MiCo models.

    This class exports a PyTorch model to the ONNX format and attaches
    per-layer bitwidth metadata (weight and activation quantization types)
    so that downstream tools can reconstruct the mixed-precision configuration.

    Usage::

        from MiCoONNXGen import MiCoONNXGen
        from models import LeNet
        from MiCoUtils import fuse_model

        model = LeNet(1)
        model.set_qscheme([[8, 6, 6, 4, 4], [8, 8, 8, 8, 8]])
        model = fuse_model(model)
        model.eval()

        exporter = MiCoONNXGen(model)
        exporter.export("output", "lenet_mnist", torch.randn(1, 1, 28, 28))
    """

    def __init__(self, model: torch.nn.Module, log_level: int = logging.INFO):
        graph, gm = MiCoONNXGen._extract_graph_module(model)
        super().__init__(gm)

        self.model = model
        self.graph = graph
        self.gm = gm
        self.logger = logging.getLogger("MiCoONNXGen")
        self.logger.setLevel(log_level)

    # ------------------------------------------------------------------
    # Graph extraction (reuses MiCoTrace from MiCoCodeGen)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_graph_module(model: torch.nn.Module) -> Tuple[torch.fx.Graph, torch.fx.GraphModule]:
        """Trace the model and return (graph, graph_module)."""
        graph = MiCoTrace().trace(model)
        graph.lint()
        gm = torch.fx.GraphModule(model, graph)
        return graph, gm

    # ------------------------------------------------------------------
    # Collect per-layer quantization metadata
    # ------------------------------------------------------------------
    def _collect_bitwidth_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Walk the FX graph and collect bitwidth information for every
        quantized layer (``BitQLayer`` subclasses).

        Returns:
            A dictionary mapping layer names to their quantization
            parameters, e.g.::

                {
                    "layers.0": {"weight_bitwidth": 8, "activation_bitwidth": 8, "layer_type": "Conv2d"},
                    ...
                }
        """
        info: Dict[str, Dict[str, Any]] = {}

        for node in self.graph.nodes:
            if node.op != "call_module":
                continue

            module = self._get_module(node.target)
            if not isinstance(module, BitQLayer):
                continue

            entry: Dict[str, Any] = {
                "weight_bitwidth": int(module.qtype),
                "activation_bitwidth": int(module.act_q),
            }

            if isinstance(module, BitLinear):
                entry["layer_type"] = "Linear"
            elif isinstance(module, BitConv2d):
                entry["layer_type"] = "Conv2d"
            elif isinstance(module, BitConv1d):
                entry["layer_type"] = "Conv1d"
            else:
                entry["layer_type"] = type(module).__name__

            info[node.target] = entry

        return info

    def _get_module(self, target: str) -> torch.nn.Module:
        """Resolve a dotted target path to the actual sub-module."""
        parts = target.split(".")
        mod = self.model
        for part in parts:
            mod = getattr(mod, part)
        return mod

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def export(
        self,
        output_directory: str,
        model_name: str,
        example_input: torch.Tensor,
        *,
        opset_version: int = 18,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
    ) -> str:
        """
        Export the model to ONNX with per-layer bitwidth metadata.

        Args:
            output_directory: Directory where the ``.onnx`` file will be
                written.
            model_name: Base name for the output file (without extension).
            example_input: A representative input tensor used by
                ``torch.onnx.export`` for tracing.
            opset_version: ONNX opset version (default 18).
            input_names: Optional list of input names for the ONNX graph.
            output_names: Optional list of output names for the ONNX graph.

        Returns:
            The path to the written ``.onnx`` file.
        """
        os.makedirs(output_directory, exist_ok=True)
        onnx_path = os.path.join(output_directory, f"{model_name}.onnx")

        if input_names is None:
            input_names = ["input"]
        if output_names is None:
            output_names = ["output"]

        # 1. Collect bitwidth information from the traced graph
        bitwidth_info = self._collect_bitwidth_info()

        # 2. Export the model to ONNX via PyTorch
        self.model.eval()
        torch.onnx.export(
            self.model,
            example_input,
            onnx_path,
            opset_version=opset_version,
            input_names=input_names,
            output_names=output_names,
        )

        # 3. Re-load, attach metadata, and save
        onnx_model = onnx.load(onnx_path)

        # Add whole-model metadata with the full bitwidth map
        bitwidth_json = json.dumps(bitwidth_info)
        onnx_model.metadata_props.append(
            onnx.StringStringEntryProto(
                key="mico_bitwidth_info", value=bitwidth_json
            )
        )

        # Also add per-node metadata as ONNX node attributes where possible
        self._annotate_onnx_nodes(onnx_model, bitwidth_info)

        onnx.save(onnx_model, onnx_path)

        self.logger.info("ONNX model exported to %s", onnx_path)
        self.logger.info(
            "Per-layer bitwidth info (%d layers):\n%s",
            len(bitwidth_info),
            json.dumps(bitwidth_info, indent=2),
        )

        return onnx_path

    # ------------------------------------------------------------------
    # Helper: annotate ONNX graph nodes
    # ------------------------------------------------------------------
    @staticmethod
    def _annotate_onnx_nodes(
        onnx_model: "onnx.ModelProto",
        bitwidth_info: Dict[str, Dict[str, Any]],
    ) -> None:
        """
        Walk the ONNX graph and add ``weight_bitwidth`` /
        ``activation_bitwidth`` doc_string annotations to nodes whose
        names match the collected bitwidth info.

        Because ``torch.onnx.export`` flattens module hierarchy and may
        rename nodes, this uses a best-effort name-matching heuristic:
        a graph node is annotated if any key in *bitwidth_info* appears
        as a substring of the node's name or output name.
        """
        for node in onnx_model.graph.node:
            matched_key = MiCoONNXGen._match_node_to_layer(node, bitwidth_info)
            if matched_key is None:
                continue

            entry = bitwidth_info[matched_key]
            annotation = json.dumps(
                {
                    "mico_layer": matched_key,
                    "weight_bitwidth": entry["weight_bitwidth"],
                    "activation_bitwidth": entry["activation_bitwidth"],
                    "layer_type": entry.get("layer_type", ""),
                }
            )
            node.doc_string = annotation

    @staticmethod
    def _match_node_to_layer(
        node: "onnx.NodeProto",
        bitwidth_info: Dict[str, Dict[str, Any]],
    ) -> Optional[str]:
        """Return the bitwidth_info key that best matches *node*, or ``None``."""
        # Build a set of candidate strings from the node
        candidates = [node.name] + list(node.output)

        for key in bitwidth_info:
            # Normalise key: replace dots with underscores / slashes
            normalised_variants = [
                key,
                key.replace(".", "_"),
                key.replace(".", "/"),
            ]
            for candidate in candidates:
                if not candidate:
                    continue
                for variant in normalised_variants:
                    if variant in candidate:
                        return key
        return None

    # ------------------------------------------------------------------
    # Convenience: load and inspect metadata
    # ------------------------------------------------------------------
    @staticmethod
    def load_bitwidth_info(onnx_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Load an ONNX model and return the per-layer bitwidth info that
        was embedded by :meth:`export`.

        Args:
            onnx_path: Path to the ``.onnx`` file.

        Returns:
            The bitwidth info dictionary, or an empty dict if no
            metadata was found.
        """
        onnx_model = onnx.load(onnx_path)
        for prop in onnx_model.metadata_props:
            if prop.key == "mico_bitwidth_info":
                return json.loads(prop.value)
        return {}
