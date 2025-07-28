from MiCoModel import MiCoModel

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