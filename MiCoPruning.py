"""Structured pruning utilities for MiCo models.

Channel pruning intentionally targets a small, explicit model class: a single
``nn.Sequential`` containing Conv2d/Linear layers and shape-preserving
activations/pooling.  Attention pruning uses a separate head/group adapter and
preserves the surrounding hidden dimension.  Both modes materialize smaller
dense modules so existing quantization and code-generation paths can consume
the result.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import torch
from torch import nn


class PruningError(ValueError):
    """Base exception for invalid pruning specifications or models."""


class UnsupportedPruningGraphError(PruningError):
    """Raised when the model topology is outside the structured MVP scope."""


@dataclass(frozen=True)
class CompressionScheme:
    """Serializable joint precision/pruning description.

    ``to_vector`` defines the append-only ``[W..., A..., P..., H...]`` encoding
    used by joint channel/head search, where ``H`` stores attention ratios.
    """

    weight_bits: tuple[int, ...]
    activation_bits: tuple[int, ...]
    pruning: PruningSpec | None = None
    attention_pruning: AttentionPruningSpec | None = None

    def to_vector(self) -> list[float | int]:
        vector: list[float | int] = list(self.weight_bits) + list(self.activation_bits)
        for spec in (self.pruning, self.attention_pruning):
            if spec is not None and spec.mode != "none":
                if isinstance(spec.ratios, Mapping):
                    raise PruningError("mapping ratios must be resolved before vector encoding")
                vector.extend(float(value) for value in spec.ratios)
        return vector


@dataclass(frozen=True)
class PruningSpec:
    """A layer-wise structured pruning request.

    ``ratios`` follows the order returned by :func:`discover_prunable_layers`,
    or can be a mapping keyed by qualified module path.  Missing mapping keys
    mean no pruning for that layer.  Ratios are target ratios; the realized
    ratio is recorded in :class:`LayerPruningResult` because small channel
    counts require integer rounding. ``importance='qweight'`` ranks using the
    effective quantized weight returned by ``BitQLayer.qweight()``.
    """

    ratios: Sequence[float] | Mapping[str, float] = ()
    mode: str = "structured"
    granularity: str = "channel"
    pattern: str = "l1"
    materialize: bool = True
    protect_last_output: bool = True
    importance: str = "weight"

    def __post_init__(self) -> None:
        if self.mode not in {"none", "structured"}:
            raise PruningError("mode must be 'none' or 'structured'")
        if self.importance not in {"weight", "qweight"}:
            raise PruningError("importance must be 'weight' or 'qweight'")
        if self.granularity != "channel":
            raise PruningError("only channel granularity is supported")
        if self.pattern != "l1":
            raise PruningError("only l1 magnitude pruning is supported")


@dataclass(frozen=True)
class LayerPruningResult:
    """Materialization details for one quantizable layer."""

    path: str
    layer_type: str
    target_ratio: float
    realized_ratio: float
    original_input_features: int
    remaining_input_features: int
    original_output_features: int
    remaining_output_features: int
    original_params: int
    remaining_params: int
    input_indices: tuple[int, ...]
    output_indices: tuple[int, ...]

    @property
    def mac_ratio(self) -> float:
        return (
            self.remaining_input_features / self.original_input_features
            * self.remaining_output_features / self.original_output_features
        )


@dataclass
class PruningResult:
    """Summary of a materialized pruning operation."""

    mode: str
    granularity: str
    pattern: str
    layers: list[LayerPruningResult]
    importance: str = "weight"

    @property
    def original_params(self) -> int:
        return sum(layer.original_params for layer in self.layers)

    @property
    def remaining_params(self) -> int:
        return sum(layer.remaining_params for layer in self.layers)

    @property
    def global_sparsity(self) -> float:
        if self.original_params == 0:
            return 0.0
        return 1.0 - self.remaining_params / self.original_params

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "granularity": self.granularity,
            "pattern": self.pattern,
            "original_params": self.original_params,
            "remaining_params": self.remaining_params,
            "global_sparsity": self.global_sparsity,
            "importance": self.importance,
            "layers": [asdict(layer) for layer in self.layers],
        }


@dataclass(frozen=True)
class _LayerPlan:
    path: str
    module: nn.Module
    target_ratio: float
    input_indices: tuple[int, ...]
    output_indices: tuple[int, ...]


@dataclass(frozen=True)
class PruningPlan:
    """Validated, deterministic plan that can be applied to a model copy."""

    container_path: str
    layers: tuple[_LayerPlan, ...]
    importance: str = "weight"


def _is_quantized_layer(module: nn.Module) -> bool:
    # Import lazily to keep MiCoPruning independent from MiCoModel imports.
    try:
        from MiCoQLayers import BitQLayer
    except ImportError:
        return False
    return isinstance(module, BitQLayer)


def discover_prunable_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Return supported and explicitly unsupported quantizable layers.

    The order matches the recursive traversal used by MiCo's quantization
    utilities.  Conv1d is included so callers receive a clear unsupported
    error instead of silently misaligning a ratio list.
    """

    layers = []
    for path, module in model.named_modules():
        if not path:
            continue
        if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
            layers.append((path, module))
    return layers


def _validate_ratio(value: float, path: str) -> float:
    try:
        ratio = float(value)
    except (TypeError, ValueError) as exc:
        raise PruningError(f"pruning ratio for {path} must be numeric") from exc
    if not math.isfinite(ratio) or not 0.0 <= ratio < 1.0:
        raise PruningError(
            f"pruning ratio for {path} must be finite and in [0, 1), got {value!r}"
        )
    return ratio


def _resolve_ratios(
    spec: PruningSpec, layers: Sequence[tuple[str, nn.Module]]
) -> list[float]:
    paths = [path for path, _ in layers]
    if isinstance(spec.ratios, Mapping):
        unknown = sorted(set(spec.ratios) - set(paths))
        if unknown:
            raise PruningError(f"unknown pruning layer paths: {unknown}")
        values = [spec.ratios.get(path, 0.0) for path in paths]
    else:
        values = list(spec.ratios)
        if len(values) == 0:
            values = [0.0] * len(paths)
        elif len(values) != len(paths):
            raise PruningError(
                f"expected {len(paths)} pruning ratios, got {len(values)}"
            )
    return [_validate_ratio(value, path) for path, value in zip(paths, values)]


def _module_parent(model: nn.Module, path: str) -> tuple[nn.Module, str]:
    pieces = path.split(".")
    child_name = pieces.pop()
    parent = model
    for piece in pieces:
        if piece.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
            parent = parent[int(piece)]
        else:
            parent = getattr(parent, piece)
    return parent, child_name


def _get_module(model: nn.Module, path: str) -> nn.Module:
    if not path:
        return model
    current = model
    for piece in path.split("."):
        if piece.isdigit() and isinstance(current, (nn.Sequential, nn.ModuleList)):
            current = current[int(piece)]
        else:
            current = getattr(current, piece)
    return current


def _path_parent(path: str) -> str:
    return path.rsplit(".", 1)[0] if "." in path else ""


def _layer_size(module: nn.Module) -> tuple[int, int, str]:
    if isinstance(module, nn.Linear):
        return module.in_features, module.out_features, "Linear"
    if isinstance(module, nn.Conv2d):
        if module.groups != 1:
            raise UnsupportedPruningGraphError(
                f"{type(module).__name__} at grouped layer cannot be pruned by the MVP "
                f"(groups={module.groups}); use a dependency-aware block pruner"
            )
        return module.in_channels, module.out_channels, "Conv2d"
    if isinstance(module, nn.Conv1d):
        raise UnsupportedPruningGraphError(
            "Conv1d structured pruning is not supported by the MVP"
        )
    raise UnsupportedPruningGraphError(
        f"unsupported quantizable layer type {type(module).__name__}"
    )


def _importance_weight(module: nn.Module, source: str) -> torch.Tensor:
    if source == "weight":
        return module.weight.detach()
    if source != "qweight":
        raise PruningError(f"unsupported pruning importance source: {source}")
    if getattr(module, "_mico_no_quantize", False):
        # W32A32 is an intentional FP sentinel, so its effective Qweight is
        # simply the original weight.
        return module.weight.detach()
    if not _is_quantized_layer(module) or not hasattr(module, "qweight"):
        raise PruningError(
            "importance='qweight' requires MiCo quantized layers; "
            "run QuantizationPass before PruningPass"
        )
    with torch.no_grad():
        # qweight() is the dequantized tensor used by the PTQ forward path.
        return module.qweight().detach()


def _importance_indices(
    module: nn.Module, keep: int, source: str = "weight"
) -> tuple[int, ...]:
    weight = _importance_weight(module, source)
    reduce_dims = tuple(range(1, weight.ndim))
    importance = weight.abs().sum(dim=reduce_dims)
    # Python sorting makes the tie-break on the original index explicit.
    ranked = sorted(
        range(int(importance.numel())),
        key=lambda index: (-float(importance[index].item()), index),
    )
    return tuple(sorted(ranked[:keep]))


def _expand_conv_output_to_linear(
    output_indices: Sequence[int],
    original_channels: int,
    linear_input_features: int,
) -> tuple[int, ...]:
    if linear_input_features % original_channels != 0:
        raise UnsupportedPruningGraphError(
            "Conv2d-to-Linear flatten mapping is ambiguous: linear input features "
            f"{linear_input_features} is not divisible by channels {original_channels}"
        )
    spatial_size = linear_input_features // original_channels
    return tuple(
        channel * spatial_size + offset
        for channel in output_indices
        for offset in range(spatial_size)
    )


def build_pruning_plan(model: nn.Module, spec: PruningSpec) -> PruningPlan:
    """Validate a structured MVP request and produce a deterministic plan."""

    layers = discover_prunable_layers(model)
    if not layers:
        raise PruningError("no Conv2d/Conv1d/Linear layers found for pruning")
    ratios = _resolve_ratios(spec, layers)
    parents = {_path_parent(path) for path, _ in layers}
    if len(parents) != 1:
        raise UnsupportedPruningGraphError(
            "structured MVP requires all quantizable layers in one nn.Sequential; "
            f"found containers {sorted(parents)}"
        )
    container_path = next(iter(parents))
    container = _get_module(model, container_path)
    if not isinstance(container, nn.Sequential):
        raise UnsupportedPruningGraphError(
            "structured MVP requires one nn.Sequential quantizable container"
        )

    supported_passthrough = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.Dropout,
        nn.Dropout2d,
        nn.Flatten,
        nn.Identity,
        nn.ReLU,
        nn.ReLU6,
        nn.AvgPool1d,
        nn.AvgPool2d,
        nn.AdaptiveAvgPool1d,
        nn.AdaptiveAvgPool2d,
        nn.MaxPool1d,
        nn.MaxPool2d,
    )
    quantizable_names = {path.rsplit(".", 1)[1] for path, _ in layers}
    for child_name, child in container.named_children():
        if child_name not in quantizable_names and not isinstance(child, supported_passthrough):
            raise UnsupportedPruningGraphError(
                f"unsupported operation {type(child).__name__} at "
                f"{container_path + '.' if container_path else ''}{child_name}; "
                "branching or shape-changing graphs require a dependency-aware pruner"
            )

    plans: list[_LayerPlan] = []
    previous_output: tuple[int, ...] | None = None
    previous_module: nn.Module | None = None
    previous_output_size = None
    for index, ((path, module), ratio) in enumerate(zip(layers, ratios)):
        input_size, output_size, layer_type = _layer_size(module)
        if index == len(layers) - 1 and spec.protect_last_output and ratio != 0.0:
            raise PruningError(
                f"last layer {path} is protected by default; set its ratio to 0"
            )

        if previous_output is None:
            input_indices = tuple(range(input_size))
        elif isinstance(previous_module, nn.Conv2d) and isinstance(module, nn.Conv2d):
            if input_size != previous_output_size:
                raise UnsupportedPruningGraphError(
                    f"channel dependency mismatch from {path}: expected input "
                    f"{previous_output_size}, got {input_size}"
                )
            input_indices = previous_output
        elif isinstance(previous_module, nn.Conv2d) and isinstance(module, nn.Linear):
            input_indices = _expand_conv_output_to_linear(
                previous_output, previous_output_size, input_size
            )
        elif isinstance(previous_module, nn.Linear) and isinstance(module, nn.Linear):
            if input_size != previous_output_size:
                raise UnsupportedPruningGraphError(
                    f"feature dependency mismatch from {path}: expected input "
                    f"{previous_output_size}, got {input_size}"
                )
            input_indices = previous_output
        else:
            raise UnsupportedPruningGraphError(
                f"unsupported dependency into {path} from "
                f"{type(previous_module).__name__ if previous_module else 'input'}"
            )

        keep = max(1, int(math.ceil((1.0 - ratio) * output_size)))
        output_indices = _importance_indices(module, keep, spec.importance)
        plans.append(
            _LayerPlan(
                path=path,
                module=module,
                target_ratio=ratio,
                input_indices=tuple(input_indices),
                output_indices=output_indices,
            )
        )
        previous_output = output_indices
        previous_module = module
        previous_output_size = output_size

    return PruningPlan(
        container_path=container_path,
        layers=tuple(plans),
        importance=spec.importance,
    )


def _copy_quantized_state(old_module: nn.Module, new_module: nn.Module) -> nn.Module:
    """Carry quantizer configuration while invalidating shape-dependent caches."""
    for name in (
        "qtype",
        "act_q",
        "qat",
        "use_norm",
        "haramard",
        "group_size",
        "qforward",
    ):
        if hasattr(old_module, name) and hasattr(new_module, name):
            setattr(new_module, name, getattr(old_module, name))
    # QAT BayesianBits quantizers may be registered modules/objects.  Preserve
    # them when present; their forward path remains valid for sliced weights.
    for name in ("bb_weight_quantizer", "bb_act_quantizer"):
        if hasattr(old_module, name) and hasattr(new_module, name):
            setattr(new_module, name, getattr(old_module, name))
    if hasattr(new_module, "qw"):
        new_module.qw = None
    if hasattr(new_module, "qw_scale"):
        new_module.qw_scale = None
    if hasattr(new_module, "macs"):
        new_module.macs = None
    if hasattr(new_module, "layer_features"):
        if isinstance(new_module, nn.Linear):
            new_module.layer_features = [new_module.in_features, new_module.out_features]
        new_module.in_w = getattr(old_module, "in_w", None)
        new_module.in_h = getattr(old_module, "in_h", None)
    if getattr(new_module, "qforward", False) and hasattr(new_module, "save_qweight"):
        new_module.save_qweight()
    return new_module


def _rebuild_linear(
    module: nn.Linear,
    input_indices: Sequence[int],
    output_indices: Sequence[int],
) -> nn.Linear:
    try:
        from MiCoQLayers import BitLinear
    except ImportError:  # pragma: no cover - normal package imports this module
        BitLinear = ()

    if isinstance(module, BitLinear):
        new_module = BitLinear(
            len(input_indices),
            len(output_indices),
            bias=module.bias is not None,
            qat=module.qat,
            use_norm=module.use_norm,
            hadamard=module.haramard,
            group_size=module.group_size,
            qtype=module.qtype,
            act_q=module.act_q,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
    else:
        new_module = nn.Linear(
            len(input_indices),
            len(output_indices),
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
    input_index = torch.tensor(input_indices, device=module.weight.device)
    output_index = torch.tensor(output_indices, device=module.weight.device)
    with torch.no_grad():
        new_module.weight.copy_(module.weight.index_select(0, output_index).index_select(1, input_index))
        if module.bias is not None:
            new_module.bias.copy_(module.bias.index_select(0, output_index))
    new_module.weight.requires_grad_(module.weight.requires_grad)
    if module.bias is not None:
        new_module.bias.requires_grad_(module.bias.requires_grad)
    if isinstance(module, BitLinear):
        _copy_quantized_state(module, new_module)
    elif getattr(module, "_mico_no_quantize", False):
        new_module._mico_no_quantize = True
        new_module._mico_weight_bits = getattr(module, "_mico_weight_bits", 32)
        new_module._mico_activation_bits = getattr(module, "_mico_activation_bits", 32)
    new_module.train(module.training)
    return new_module


def _rebuild_conv2d(
    module: nn.Conv2d,
    input_indices: Sequence[int],
    output_indices: Sequence[int],
) -> nn.Conv2d:
    try:
        from MiCoQLayers import BitConv2d
    except ImportError:  # pragma: no cover - normal package imports this module
        BitConv2d = ()

    if isinstance(module, BitConv2d):
        new_module = BitConv2d(
            len(input_indices),
            len(output_indices),
            module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=1,
            bias=module.bias is not None,
            qat=module.qat,
            use_norm=module.use_norm,
            qtype=module.qtype,
            act_q=module.act_q,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
    else:
        new_module = nn.Conv2d(
            len(input_indices),
            len(output_indices),
            module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=1,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
    input_index = torch.tensor(input_indices, device=module.weight.device)
    output_index = torch.tensor(output_indices, device=module.weight.device)
    with torch.no_grad():
        new_module.weight.copy_(module.weight.index_select(0, output_index).index_select(1, input_index))
        if module.bias is not None:
            new_module.bias.copy_(module.bias.index_select(0, output_index))
    new_module.weight.requires_grad_(module.weight.requires_grad)
    if module.bias is not None:
        new_module.bias.requires_grad_(module.bias.requires_grad)
    if isinstance(module, BitConv2d):
        _copy_quantized_state(module, new_module)
    elif getattr(module, "_mico_no_quantize", False):
        new_module._mico_no_quantize = True
        new_module._mico_weight_bits = getattr(module, "_mico_weight_bits", 32)
        new_module._mico_activation_bits = getattr(module, "_mico_activation_bits", 32)
    new_module.train(module.training)
    return new_module


def _rebuild_batch_norm(
    module: nn.modules.batchnorm._BatchNorm,
    indices: Sequence[int],
) -> nn.modules.batchnorm._BatchNorm:
    reference = next(module.parameters(), None)
    if reference is None:
        reference = next(module.buffers(), torch.empty(0))
    kwargs = {
        "eps": module.eps,
        "momentum": module.momentum,
        "affine": module.affine,
        "track_running_stats": module.track_running_stats,
        "device": reference.device,
        "dtype": reference.dtype,
    }
    new_module = type(module)(len(indices), **kwargs)
    index = torch.tensor(indices, device=reference.device)
    with torch.no_grad():
        if module.affine:
            new_module.weight.copy_(module.weight.index_select(0, index))
            new_module.bias.copy_(module.bias.index_select(0, index))
        if module.track_running_stats:
            new_module.running_mean.copy_(module.running_mean.index_select(0, index))
            new_module.running_var.copy_(module.running_var.index_select(0, index))
            new_module.num_batches_tracked.copy_(module.num_batches_tracked)
    new_module.train(module.training)
    return new_module


def _materialize_plan(model: nn.Module, plan: PruningPlan) -> PruningResult:
    layer_by_child = {
        path.rsplit(".", 1)[1]: layer for path, layer in ((item.path, item) for item in plan.layers)
    }
    container = _get_module(model, plan.container_path)
    for child_name, original_child in list(container.named_children()):
        layer = layer_by_child.get(child_name)
        if layer is not None:
            if isinstance(layer.module, nn.Linear):
                replacement = _rebuild_linear(
                    layer.module, layer.input_indices, layer.output_indices
                )
            elif isinstance(layer.module, nn.Conv2d):
                replacement = _rebuild_conv2d(
                    layer.module, layer.input_indices, layer.output_indices
                )
            else:
                raise UnsupportedPruningGraphError(
                    f"cannot materialize {type(layer.module).__name__} at {layer.path}"
                )
            setattr(container, child_name, replacement)
            continue

        if isinstance(original_child, (nn.BatchNorm1d, nn.BatchNorm2d)):
            previous = None
            child_index = list(container._modules).index(child_name)
            for prior_name in list(container._modules)[:child_index]:
                prior_layer = layer_by_child.get(prior_name)
                if prior_layer is not None:
                    previous = prior_layer
            if previous is not None:
                previous_features = (
                    previous.module.out_channels
                    if isinstance(previous.module, nn.Conv2d)
                    else previous.module.out_features
                )
            if previous is not None and original_child.num_features == previous_features:
                setattr(
                    container,
                    child_name,
                    _rebuild_batch_norm(original_child, previous.output_indices),
                )

    result_layers = []
    for layer in plan.layers:
        input_size, output_size, layer_type = _layer_size(layer.module)
        result_layers.append(
            LayerPruningResult(
                path=layer.path,
                layer_type=layer_type,
                target_ratio=layer.target_ratio,
                realized_ratio=1.0 - len(layer.output_indices) / output_size,
                original_input_features=input_size,
                remaining_input_features=len(layer.input_indices),
                original_output_features=output_size,
                remaining_output_features=len(layer.output_indices),
                original_params=int(layer.module.weight.numel()),
                remaining_params=int(
                    len(layer.input_indices) * len(layer.output_indices)
                    * (layer.module.weight.numel() // (input_size * output_size))
                ),
                input_indices=layer.input_indices,
                output_indices=layer.output_indices,
            )
        )
    return PruningResult(
        mode="structured",
        granularity="channel",
        pattern="l1",
        layers=result_layers,
        importance=plan.importance,
    )


def apply_pruning(
    model: nn.Module,
    spec: PruningSpec,
    *,
    inplace: bool = False,
) -> tuple[nn.Module, PruningResult]:
    """Apply structured pruning and return ``(model, result)``.

    The plan is built from the supplied model. By default a deep copy is
    materialized, which is important when evaluating multiple candidates. Set
    ``spec.importance='qweight'`` when the supplied model is already quantized;
    channel ranking then follows its effective quantized weights.
    """

    if not spec.materialize and spec.mode != "none":
        raise PruningError(
            "mask-only pruning is reserved for a later MVP; use materialize=True"
        )
    if spec.mode == "none":
        result = PruningResult(
            "none", spec.granularity, spec.pattern, [], importance=spec.importance
        )
        target = model if inplace else copy.deepcopy(model)
        target._mico_pruning_result = result
        return target, result
    plan = build_pruning_plan(model, spec)
    target = model if inplace else copy.deepcopy(model)
    result = _materialize_plan(target, plan)
    target.n_layers = len(discover_prunable_layers(target))
    target._mico_pruning_result = result
    return target, result


def sparsity_statistics(
    model: nn.Module, result: PruningResult | None = None
) -> dict:
    """Return serializable pruning statistics for a model."""

    if result is None:
        result = getattr(model, "_mico_pruning_result", None)
    if result is None:
        raise PruningError("model has no MiCoPruning result")
    return result.to_dict()


def estimate_effective_macs(
    result: PruningResult, dense_macs: Sequence[float]
) -> float:
    """Estimate structured MACs from dense per-layer MACs and shape ratios."""

    if len(dense_macs) != len(result.layers):
        raise PruningError(
            f"expected {len(result.layers)} dense MAC values, got {len(dense_macs)}"
        )
    return float(
        sum(float(macs) * layer.mac_ratio for macs, layer in zip(dense_macs, result.layers))
    )


__all__ = [
    "CompressionScheme",
    "LayerPruningResult",
    "PruningError",
    "PruningPlan",
    "PruningResult",
    "PruningSpec",
    "StructuredPruner",
    "UnsupportedPruningGraphError",
    "apply_pruning",
    "build_pruning_plan",
    "discover_prunable_layers",
    "estimate_effective_macs",
    "sparsity_statistics",
]


class StructuredPruner:
    """Small object-oriented facade for callers that prefer a pruner object."""

    def __init__(self, spec: PruningSpec):
        self.spec = spec

    def apply(self, model: nn.Module, *, inplace: bool = False):
        return apply_pruning(model, self.spec, inplace=inplace)


@dataclass(frozen=True)
class AttentionPruningSpec:
    """Layer-wise structured attention head/group pruning request.

    Ratios follow :func:`discover_attention_layers`.  For grouped-query
    attention, one pruning unit is a KV group and all query heads sharing that
    KV head are removed together.  The hidden dimension remains unchanged, so
    residual connections and downstream projections stay valid. Set
    ``importance='qweight'`` to rank heads/groups from quantized Q/K/V weights.
    """

    ratios: Sequence[float] | Mapping[str, float] = ()
    mode: str = "heads"
    granularity: str = "head"
    pattern: str = "l1"
    materialize: bool = True
    importance: str = "weight"

    def __post_init__(self) -> None:
        if self.mode not in {"none", "heads"}:
            raise PruningError("attention pruning mode must be 'none' or 'heads'")
        if self.importance not in {"weight", "qweight"}:
            raise PruningError("importance must be 'weight' or 'qweight'")
        if self.granularity != "head":
            raise PruningError("only attention head granularity is supported")
        if self.pattern != "l1":
            raise PruningError("only l1 magnitude attention pruning is supported")


@dataclass(frozen=True)
class AttentionLayerPruningResult:
    """Materialization details for one attention module."""

    path: str
    q_path: str
    k_path: str
    v_path: str
    output_path: str
    target_ratio: float
    realized_ratio: float
    original_heads: int
    remaining_heads: int
    original_kv_heads: int
    remaining_kv_heads: int
    head_dim: int
    original_params: int
    remaining_params: int
    query_heads: tuple[int, ...]
    kv_heads: tuple[int, ...]

    @property
    def head_ratio(self) -> float:
        return self.remaining_kv_heads / self.original_kv_heads

    @property
    def mac_ratio(self) -> float:
        # Q/K/V projections and output projection have different dimensions for
        # GQA; this aggregate is useful for reporting only.
        return self.remaining_heads / self.original_heads


@dataclass
class AttentionPruningResult:
    """Summary of materialized attention head pruning."""

    mode: str
    granularity: str
    pattern: str
    layers: list[AttentionLayerPruningResult]
    importance: str = "weight"

    @property
    def original_params(self) -> int:
        return sum(layer.original_params for layer in self.layers)

    @property
    def remaining_params(self) -> int:
        return sum(layer.remaining_params for layer in self.layers)

    @property
    def global_sparsity(self) -> float:
        if self.original_params == 0:
            return 0.0
        return 1.0 - self.remaining_params / self.original_params

    @property
    def original_heads(self) -> int:
        return sum(layer.original_kv_heads for layer in self.layers)

    @property
    def remaining_heads(self) -> int:
        return sum(layer.remaining_kv_heads for layer in self.layers)

    @property
    def head_sparsity(self) -> float:
        if self.original_heads == 0:
            return 0.0
        return 1.0 - self.remaining_heads / self.original_heads

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "granularity": self.granularity,
            "pattern": self.pattern,
            "original_params": self.original_params,
            "remaining_params": self.remaining_params,
            "global_sparsity": self.global_sparsity,
            "original_heads": self.original_heads,
            "remaining_heads": self.remaining_heads,
            "head_sparsity": self.head_sparsity,
            "importance": self.importance,
            "layers": [asdict(layer) for layer in self.layers],
        }


@dataclass(frozen=True)
class _AttentionLayout:
    path: str
    module: nn.Module
    q_name: str
    k_name: str
    v_name: str
    output_name: str
    head_name: str
    kv_head_name: str | None
    q_gain_name: str | None
    view_dim_name: str | None
    q_heads: int
    kv_heads: int
    head_dim: int


@dataclass(frozen=True)
class AttentionPruningPlan:
    """Validated attention plan, useful for cost-aware search integrations."""

    layers: tuple[AttentionLayerPruningResult, ...]
    importance: str = "weight"


def _attention_layout(path: str, module: nn.Module) -> _AttentionLayout | None:
    """Describe the custom attention modules shipped by MiCo.

    The adapter is attribute based so the pruning implementation does not
    import model modules (and therefore does not create import cycles).
    """

    if all(hasattr(module, name) for name in ("q", "k", "v")):
        if hasattr(module, "o"):
            output_name = "o"
        elif hasattr(module, "proj"):
            output_name = "proj"
        else:
            return None
        q_name, k_name, v_name = "q", "k", "v"
    elif all(hasattr(module, name) for name in ("wq", "wk", "wv", "wo")):
        q_name, k_name, v_name, output_name = "wq", "wk", "wv", "wo"
    elif all(hasattr(module, name) for name in ("c_q", "c_k", "c_v")):
        if hasattr(module, "c_proj"):
            output_name = "c_proj"
        elif hasattr(module, "proj"):
            output_name = "proj"
        else:
            return None
        q_name, k_name, v_name = "c_q", "c_k", "c_v"
    else:
        return None

    if hasattr(module, "num_heads"):
        head_name = "num_heads"
    elif hasattr(module, "n_heads"):
        head_name = "n_heads"
    elif hasattr(module, "head"):
        head_name = "head"
    elif hasattr(module, "n_local_heads"):
        head_name = "n_local_heads"
    else:
        return None

    if hasattr(module, "num_kv_heads"):
        kv_head_name = "num_kv_heads"
    elif hasattr(module, "n_local_kv_heads"):
        kv_head_name = "n_local_kv_heads"
    else:
        kv_head_name = None

    q_heads = int(getattr(module, head_name))
    kv_heads = int(getattr(module, kv_head_name)) if kv_head_name else q_heads
    head_dim_value = getattr(module, "head_dim", None)
    view_dim_name = None
    if head_dim_value is None:
        view_dim_name = "feats" if hasattr(module, "feats") else "dim" if hasattr(module, "dim") else None
        feature_dim = getattr(module, view_dim_name, None) if view_dim_name else None
        q_candidate = getattr(module, q_name)
        if feature_dim is None and isinstance(q_candidate, nn.Linear):
            feature_dim = q_candidate.out_features
        head_dim_value = (
            int(feature_dim) // q_heads if feature_dim is not None and int(feature_dim) % q_heads == 0 else 0
        )
    head_dim = int(head_dim_value)
    if head_dim <= 0:
        return None
    if q_heads <= 0 or kv_heads <= 0 or q_heads % kv_heads != 0:
        raise UnsupportedPruningGraphError(
            f"attention module {path} has invalid head layout: "
            f"query={q_heads}, kv={kv_heads}"
        )
    q_module = getattr(module, q_name)
    k_module = getattr(module, k_name)
    v_module = getattr(module, v_name)
    output_module = getattr(module, output_name)
    if not all(isinstance(item, nn.Linear) for item in (q_module, k_module, v_module, output_module)):
        raise UnsupportedPruningGraphError(
            f"attention module {path} must use nn.Linear Q/K/V/output projections"
        )
    if q_module.out_features != q_heads * head_dim:
        raise UnsupportedPruningGraphError(f"Q projection shape is incompatible at {path}")
    if k_module.out_features != kv_heads * head_dim or v_module.out_features != kv_heads * head_dim:
        raise UnsupportedPruningGraphError(f"K/V projection shape is incompatible at {path}")
    if output_module.in_features != q_heads * head_dim:
        raise UnsupportedPruningGraphError(f"output projection shape is incompatible at {path}")
    q_gain_name = "q_gain" if hasattr(module, "q_gain") else None
    if q_gain_name and getattr(module, q_gain_name).numel() != q_heads:
        raise UnsupportedPruningGraphError(f"q_gain shape is incompatible at {path}")
    return _AttentionLayout(
        path=path,
        module=module,
        q_name=q_name,
        k_name=k_name,
        v_name=v_name,
        output_name=output_name,
        head_name=head_name,
        kv_head_name=kv_head_name,
        q_gain_name=q_gain_name,
        view_dim_name=view_dim_name,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
    )


def discover_attention_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Return supported custom multi-head attention modules in traversal order."""

    result = []
    for path, module in model.named_modules():
        if not path:
            continue
        if _attention_layout(path, module) is not None:
            result.append((path, module))
    return result


def _resolve_attention_ratios(
    spec: AttentionPruningSpec,
    layers: Sequence[tuple[str, nn.Module]],
) -> list[float]:
    paths = [path for path, _ in layers]
    if isinstance(spec.ratios, Mapping):
        unknown = sorted(set(spec.ratios) - set(paths))
        if unknown:
            raise PruningError(f"unknown attention layer paths: {unknown}")
        values = [spec.ratios.get(path, 0.0) for path in paths]
    else:
        values = list(spec.ratios)
        if not values:
            values = [0.0] * len(paths)
        elif len(values) != len(paths):
            raise PruningError(
                f"expected {len(paths)} attention pruning ratios, got {len(values)}"
            )
    return [_validate_ratio(value, path) for path, value in zip(paths, values)]


def _head_feature_indices(heads: Sequence[int], head_dim: int) -> tuple[int, ...]:
    return tuple(head * head_dim + offset for head in heads for offset in range(head_dim))


def _head_group_scores(
    layout: _AttentionLayout, source: str = "weight"
) -> list[float]:
    q_module = getattr(layout.module, layout.q_name)
    k_module = getattr(layout.module, layout.k_name)
    v_module = getattr(layout.module, layout.v_name)
    q_weight = _importance_weight(q_module, source)
    k_weight = _importance_weight(k_module, source)
    v_weight = _importance_weight(v_module, source)
    q_scores = q_weight.abs().sum(dim=1)
    k_scores = k_weight.abs().sum(dim=1)
    v_scores = v_weight.abs().sum(dim=1)
    group_size = layout.q_heads // layout.kv_heads
    scores = []
    for group in range(layout.kv_heads):
        q_heads = range(group * group_size, (group + 1) * group_size)
        q_indices = _head_feature_indices(tuple(q_heads), layout.head_dim)
        kv_indices = _head_feature_indices((group,), layout.head_dim)
        score = q_scores[list(q_indices)].sum() + k_scores[list(kv_indices)].sum()
        score = score + v_scores[list(kv_indices)].sum()
        scores.append(float(score.item()))
    return scores


def build_attention_pruning_plan(
    model: nn.Module, spec: AttentionPruningSpec
) -> AttentionPruningPlan:
    """Build a deterministic head/group plan without changing ``model``."""

    layers = discover_attention_layers(model)
    if not layers:
        raise PruningError("no supported custom attention layers found for pruning")
    if not spec.materialize and spec.mode != "none":
        raise PruningError("mask-only attention pruning is not supported; use materialize=True")
    ratios = _resolve_attention_ratios(spec, layers)
    results = []
    for (path, module), ratio in zip(layers, ratios):
        layout = _attention_layout(path, module)
        if layout is None:  # pragma: no cover - discovery already filtered it
            continue
        keep_groups = max(1, int(math.ceil((1.0 - ratio) * layout.kv_heads)))
        scores = _head_group_scores(layout, spec.importance)
        ranked = sorted(range(layout.kv_heads), key=lambda index: (-scores[index], index))
        kept_kv = tuple(sorted(ranked[:keep_groups]))
        group_size = layout.q_heads // layout.kv_heads
        kept_q = tuple(
            q_head
            for group in kept_kv
            for q_head in range(group * group_size, (group + 1) * group_size)
        )
        q_module = getattr(module, layout.q_name)
        k_module = getattr(module, layout.k_name)
        v_module = getattr(module, layout.v_name)
        output_module = getattr(module, layout.output_name)
        original_params = sum(
            int(item.weight.numel()) + (int(item.bias.numel()) if item.bias is not None else 0)
            for item in (q_module, k_module, v_module, output_module)
        )
        remaining_params = (
            len(kept_q) * layout.head_dim * q_module.in_features
            + len(kept_kv) * layout.head_dim * k_module.in_features
            + len(kept_kv) * layout.head_dim * v_module.in_features
            + output_module.out_features * len(kept_q) * layout.head_dim
            + (int(q_module.bias.numel()) * len(kept_q) / layout.q_heads if q_module.bias is not None else 0)
            + (int(k_module.bias.numel()) * len(kept_kv) / layout.kv_heads if k_module.bias is not None else 0)
            + (int(v_module.bias.numel()) * len(kept_kv) / layout.kv_heads if v_module.bias is not None else 0)
            + (int(output_module.bias.numel()) if output_module.bias is not None else 0)
        )
        results.append(
            AttentionLayerPruningResult(
                path=path,
                q_path=f"{path}.{layout.q_name}",
                k_path=f"{path}.{layout.k_name}",
                v_path=f"{path}.{layout.v_name}",
                output_path=f"{path}.{layout.output_name}",
                target_ratio=ratio,
                realized_ratio=1.0 - keep_groups / layout.kv_heads,
                original_heads=layout.q_heads,
                remaining_heads=len(kept_q),
                original_kv_heads=layout.kv_heads,
                remaining_kv_heads=len(kept_kv),
                head_dim=layout.head_dim,
                original_params=original_params,
                remaining_params=int(remaining_params),
                query_heads=kept_q,
                kv_heads=kept_kv,
            )
        )
    return AttentionPruningPlan(
        layers=tuple(results),
        importance=spec.importance,
    )


def _materialize_attention_plan(
    model: nn.Module, plan: AttentionPruningPlan
) -> AttentionPruningResult:
    result_layers = []
    for planned in plan.layers:
        module = _get_module(model, planned.path)
        layout = _attention_layout(planned.path, module)
        if layout is None:
            raise UnsupportedPruningGraphError(
                f"attention module disappeared while materializing {planned.path}"
            )
        q_indices = _head_feature_indices(planned.query_heads, layout.head_dim)
        kv_indices = _head_feature_indices(planned.kv_heads, layout.head_dim)
        q_module = getattr(module, layout.q_name)
        k_module = getattr(module, layout.k_name)
        v_module = getattr(module, layout.v_name)
        output_module = getattr(module, layout.output_name)
        setattr(module, layout.q_name, _rebuild_linear(q_module, range(q_module.in_features), q_indices))
        setattr(module, layout.k_name, _rebuild_linear(k_module, range(k_module.in_features), kv_indices))
        setattr(module, layout.v_name, _rebuild_linear(v_module, range(v_module.in_features), kv_indices))
        setattr(module, layout.output_name, _rebuild_linear(output_module, q_indices, range(output_module.out_features)))
        setattr(module, layout.head_name, len(planned.query_heads))
        if layout.view_dim_name is not None:
            # ViT's custom attention derives head_dim from ``feats / head``.
            # Keep the original per-head width while shrinking the packed QKV width.
            setattr(module, layout.view_dim_name, len(planned.query_heads) * layout.head_dim)
        if layout.kv_head_name is not None:
            setattr(module, layout.kv_head_name, len(planned.kv_heads))
        if layout.q_gain_name is not None:
            q_gain = getattr(module, layout.q_gain_name)
            index = torch.tensor(planned.query_heads, device=q_gain.device)
            new_gain = nn.Parameter(q_gain.detach().index_select(0, index), requires_grad=q_gain.requires_grad)
            setattr(module, layout.q_gain_name, new_gain)
        result_layers.append(planned)
    return AttentionPruningResult(
        mode="heads",
        granularity="head",
        pattern="l1",
        layers=result_layers,
        importance=plan.importance,
    )


def apply_attention_pruning(
    model: nn.Module,
    spec: AttentionPruningSpec,
    *,
    inplace: bool = False,
) -> tuple[nn.Module, AttentionPruningResult]:
    """Apply structured attention head/group pruning and return ``(model, result)``."""

    if spec.mode == "none":
        result = AttentionPruningResult(
            "none", spec.granularity, spec.pattern, [], importance=spec.importance
        )
        target = model if inplace else copy.deepcopy(model)
        target._mico_attention_pruning_result = result
        return target, result
    plan = build_attention_pruning_plan(model, spec)
    target = model if inplace else copy.deepcopy(model)
    result = _materialize_attention_plan(target, plan)
    target._mico_attention_pruning_result = result
    return target, result


def attention_pruning_statistics(
    model: nn.Module, result: AttentionPruningResult | None = None
) -> dict:
    """Return serializable attention pruning statistics for a model."""

    if result is None:
        result = getattr(model, "_mico_attention_pruning_result", None)
    if result is None:
        raise PruningError("model has no MiCo attention pruning result")
    return result.to_dict()


__all__.extend([
    "AttentionLayerPruningResult",
    "AttentionPruningPlan",
    "AttentionPruningResult",
    "AttentionPruningSpec",
    "apply_attention_pruning",
    "attention_pruning_statistics",
    "build_attention_pruning_plan",
    "discover_attention_layers",
])
