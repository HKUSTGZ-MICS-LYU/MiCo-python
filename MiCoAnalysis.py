from MiCoModel import MiCoModel
from MiCoPruning import (
    PruningError,
    attention_pruning_statistics as _attention_pruning_statistics,
    sparsity_statistics,
)

import torch
import numpy as np

def per_layer_weight_hist(model: MiCoModel, bins = 10):
    qlayers = model.get_qlayers()
    hists = []
    for qlayer in qlayers:
        weights = qlayer.weight.data.cpu().numpy().flatten()
        hist = np.histogram(weights, bins=bins)[0]
        hists.append(hist)
    hists = np.array(hists)
    return hists

def per_layer_weight_l2_error(model: MiCoModel, qscheme):
    model.set_qscheme(qscheme)
    qlayers = model.get_qlayers()
    errors = []
    for qlayer in qlayers:
        weights = qlayer.weight.data
        qweights = qlayer.weight_quant(weights)
        l2_norm = torch.norm(qweights - weights, p=2).item()
        errors.append(l2_norm)
    errors = np.array(errors)
    return errors

def per_layer_macs(model: MiCoModel, test_input: torch.Tensor = None):
    qlayers = model.get_qlayers()
    if test_input is not None:
        with torch.no_grad():
            model.forward(test_input)
    macs = []
    for qlayer in qlayers:
        macs.append(qlayer.get_mac())
    macs = np.array(macs)
    return macs

def per_layer_weight_num(model: MiCoModel):
    qlayers = model.get_qlayers()
    weights = []
    for qlayer in qlayers:
        weights.append(qlayer.get_params())
    weights = np.array(weights)
    return weights

def per_layer_weight_sparsity(model: MiCoModel, eps = 1e-3):
    qlayers = model.get_qlayers()
    sparsities = []
    for qlayer in qlayers:
        weight = qlayer.weight.data.cpu().numpy()
        non_zero_count = np.count_nonzero(np.abs(weight) > eps)
        total_count = weight.size
        sparsity = 1 - (non_zero_count / total_count)
        sparsities.append(sparsity)
    sparsities = np.array(sparsities)
    return sparsities


def pruning_statistics(model: MiCoModel) -> dict:
    """Return structural pruning statistics attached by ``MiCoPruning``."""
    try:
        return sparsity_statistics(model)
    except PruningError:
        return {
            "mode": "none",
            "granularity": "channel",
            "pattern": "l1",
            "original_params": 0,
            "remaining_params": 0,
            "global_sparsity": 0.0,
            "layers": [],
        }


def attention_pruning_statistics(model: MiCoModel) -> dict:
    """Return structural attention head/group pruning statistics."""
    try:
        return _attention_pruning_statistics(model)
    except PruningError:
        return {
            "mode": "none",
            "granularity": "head",
            "pattern": "l1",
            "original_params": 0,
            "remaining_params": 0,
            "global_sparsity": 0.0,
            "original_heads": 0,
            "remaining_heads": 0,
            "head_sparsity": 0.0,
            "layers": [],
        }


def per_layer_attention_head_sparsity(model: MiCoModel) -> np.ndarray:
    """Return realized KV-group sparsity for each attention module."""
    stats = attention_pruning_statistics(model)
    return np.asarray(
        [layer["realized_ratio"] for layer in stats["layers"]],
        dtype=float,
    )


def per_layer_structural_sparsity(model: MiCoModel) -> np.ndarray:
    """Return realized output-channel sparsity for each pruned layer."""
    stats = pruning_statistics(model)
    return np.asarray(
        [layer["realized_ratio"] for layer in stats["layers"]],
        dtype=float,
    )
