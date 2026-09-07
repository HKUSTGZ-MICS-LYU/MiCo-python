"""Composable model compression process passes.

Each pass mutates the supplied model in place and returns a pass-specific
result.  ``CompressionPipeline`` is deliberately small: ordering is explicit,
and new transformation passes can be inserted without changing the model API.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable

from MiCoPruning import (
    AttentionPruningSpec,
    CompressionScheme,
    PruningSpec,
)


class MiCoProcessPass:
    """Base interface for an in-place MiCo model transformation."""

    name = "process"

    def apply(self, model):
        raise NotImplementedError

    def __call__(self, model):
        return self.apply(model)


@dataclass(frozen=True)
class QuantizationPass(MiCoProcessPass):
    """Apply generic layer and optional attention precision settings."""

    qscheme: Any = None
    qkwargs: dict | None = None
    attention_qscheme: dict | None = None
    attention_qkwargs: dict | None = None
    name = "quantization"

    def apply(self, model):
        if self.qscheme is not None:
            model.set_qscheme(self.qscheme, **(self.qkwargs or {}))
        if self.attention_qscheme is not None:
            model.set_attn_qscheme(
                self.attention_qscheme,
                **(self.attention_qkwargs or {}),
            )
        return None


@dataclass(frozen=True)
class PruningPass(MiCoProcessPass):
    """Apply ordinary structured channel pruning."""

    spec: PruningSpec
    name = "pruning"

    def apply(self, model):
        return model.set_pruning(self.spec)


@dataclass(frozen=True)
class AttentionPruningPass(MiCoProcessPass):
    """Apply structured attention head or GQA KV-group pruning."""

    spec: AttentionPruningSpec
    name = "attention_pruning"

    def apply(self, model):
        return model.set_attention_pruning(self.spec)


class CompressionPipeline:
    """Run an ordered list of composable compression passes."""

    def __init__(self, passes: Iterable[MiCoProcessPass] = ()):
        self.passes = tuple(passes)
        for process_pass in self.passes:
            if not isinstance(process_pass, MiCoProcessPass):
                raise TypeError(
                    "CompressionPipeline entries must inherit MiCoProcessPass"
                )

    def apply(self, model) -> dict[str, Any]:
        results = {}
        for process_pass in self.passes:
            result = process_pass.apply(model)
            if result is not None:
                results[process_pass.name] = result
        return results

    def __call__(self, model) -> dict[str, Any]:
        return self.apply(model)


def normalize_compression_order(order: str) -> str:
    aliases = {
        "prune_then_quantize": "prune_then_quantize",
        "prune_quantize": "prune_then_quantize",
        "prune->quantize": "prune_then_quantize",
        "quantize_then_prune": "quantize_then_prune",
        "quantize_prune": "quantize_then_prune",
        "quantize->prune": "quantize_then_prune",
    }
    try:
        return aliases[str(order).lower().replace(" ", "_")]
    except KeyError as exc:
        raise ValueError(
            "order must be 'prune_then_quantize' or 'quantize_then_prune'"
        ) from exc


def build_compression_pipeline(
    *,
    qscheme=None,
    pscheme=None,
    pruning_mode="none",
    materialize=True,
    attention_pscheme=None,
    attention_pruning_mode="none",
    attention_materialize=True,
    order="prune_then_quantize",
    attention_qscheme=None,
    attention_qkwargs=None,
    qkwargs=None,
):
    """Build a standard ordered pass pipeline from public compression args."""

    order = normalize_compression_order(order)
    scheme_pruning = None
    scheme_attention_pruning = None
    if isinstance(qscheme, CompressionScheme):
        scheme_pruning = qscheme.pruning
        scheme_attention_pruning = qscheme.attention_pruning
        qscheme = [list(qscheme.weight_bits), list(qscheme.activation_bits)]
    if pscheme is None and scheme_pruning is not None:
        pscheme = scheme_pruning
    if attention_pscheme is None and scheme_attention_pruning is not None:
        attention_pscheme = scheme_attention_pruning
    if pscheme is not None and pruning_mode == "none":
        pruning_mode = "structured"
    if attention_pscheme is not None and attention_pruning_mode == "none":
        attention_pruning_mode = "heads"

    has_channel_pruning = pscheme is not None or pruning_mode != "none"
    has_attention_pruning = (
        attention_pscheme is not None or attention_pruning_mode != "none"
    )
    if order == "quantize_then_prune" and (
        has_channel_pruning or has_attention_pruning
    ) and qscheme is None:
        raise ValueError(
            "qscheme is required for order='quantize_then_prune' when pruning is enabled"
        )

    # The order controls the importance tensor as well as pass placement:
    # quantize->prune ranks channels/heads using the effective Qweight.
    importance = "qweight" if order == "quantize_then_prune" else "weight"
    pruning_passes = []
    if has_channel_pruning:
        channel_spec = (
            PruningSpec(
                ratios=() if pscheme is None else pscheme,
                mode=pruning_mode,
                materialize=materialize,
                importance=importance,
            )
            if not isinstance(pscheme, PruningSpec)
            else replace(pscheme, importance=importance)
        )
        pruning_passes.append(PruningPass(channel_spec))
    if has_attention_pruning:
        attention_spec = (
            AttentionPruningSpec(
                ratios=() if attention_pscheme is None else attention_pscheme,
                mode=attention_pruning_mode,
                materialize=attention_materialize,
                importance=importance,
            )
            if not isinstance(attention_pscheme, AttentionPruningSpec)
            else replace(attention_pscheme, importance=importance)
        )
        pruning_passes.append(AttentionPruningPass(attention_spec))

    quantization_pass = QuantizationPass(
        qscheme=qscheme,
        qkwargs=qkwargs,
        attention_qscheme=attention_qscheme,
        attention_qkwargs=attention_qkwargs,
    )
    if order == "prune_then_quantize":
        passes = pruning_passes + [quantization_pass]
    else:
        passes = [quantization_pass] + pruning_passes
    return CompressionPipeline(passes)


def apply_process_passes(model, passes: Iterable[MiCoProcessPass]):
    """Run a caller-provided pass list and return named pass results."""

    return CompressionPipeline(passes).apply(model)


__all__ = [
    "AttentionPruningPass",
    "CompressionPipeline",
    "MiCoProcessPass",
    "PruningPass",
    "QuantizationPass",
    "apply_process_passes",
    "build_compression_pipeline",
    "normalize_compression_order",
]
