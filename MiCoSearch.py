import torch
import torch.nn as nn
import torch.nn.utils.fusion as fusion
import torch.nn.functional as F

import numpy as np

from matplotlib import pyplot as plt

import os
import copy
import json
import random
import itertools 
import warnings

from tqdm import tqdm

from MiCoUtils import (
    list_qlayers,
    replace_quantize_layers,
    set_to_qforward
)
from MiCoModel import MiCoModel

WQ_TYPES   = [8, 4, 2, 1.58, 1]
AQ_TYPES   = [8, 4, 2, 1.58, 1]

RAND_SPACE_LIMIT = 1000_000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore")

class MiCoSearch:

    def __init__(self, model: MiCoModel, 
                 epochs, train_loader, test_loader, 
                 pretrained_model,
                 wq_types = WQ_TYPES, aq_types = AQ_TYPES,
                 seed=0,
                 lr=0.0001):

        self.wq_types = sorted(wq_types, reverse=True)
        self.aq_types = sorted(aq_types, reverse=True)
        self.layer_q = list(itertools.product(wq_types, aq_types))

        random.seed(seed)
        np.random.seed(seed)

        self.layers = list_qlayers(model)
        self.n_layers = len(self.layers)
        self.space_size = len(self.layer_q) ** self.n_layers
        assert self.n_layers > 0, "No QLayer found! Please convert the model first."
        self.layer_macs = [layer.get_mac() for layer in self.layers]
        self.layer_params = [layer.get_params() for layer in self.layers]

        self.model = model
        self.epochs = epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.pretrained_model = pretrained_model
        self.lr = lr

        print("Model:", model._get_name())
        print("Number of QLayers: ", self.n_layers)
        print("Total MACs: ", np.sum(self.layer_macs))
        print("Total Params: ", np.sum(self.layer_params))

        # Estimate Total Space Size: ((Q_a*Q_w)^L)
        print("Total Space Size = ", self.space_size)
        return

    def get_search_bound(self):
        lb = [0] * self.n_layers * 2 # WQ + AQ
        ub = [len(self.wq_types)] * self.n_layers + [len(self.aq_types)] * self.n_layers
        return lb, ub

    def rand_sample(self, n_sample = 1):
        total_space_size = len(self.layer_q) ** self.n_layers
        if total_space_size >= RAND_SPACE_LIMIT:
            return self.np_rand_sample(n_sample)
        total_space = list(itertools.product(self.layer_q, repeat=self.n_layers))
        samples = random.sample(total_space, n_sample)
        samples = [list(tuple(zip(*qscheme))) for qscheme in samples]
        samples = [[list(qscheme[0]), list(qscheme[1])] for qscheme in samples]
        return samples
    
    def np_rand_sample(self, n_sample = 1):
        lb, ub = self.get_search_bound()
        vec = np.random.randint(lb, ub, size=(n_sample, 2*self.n_layers))
        samples = [self.vec_to_sample(self.map_cat_to_q(v)) for v in vec]
        return samples

    def np_uniform_sample(self, n_sample = 1):
        lb, ub = self.get_search_bound()
        vec = np.random.uniform(lb, ub, size=(n_sample, 2*self.n_layers))
        vec = vec.astype(int)
        samples = [self.vec_to_sample(self.map_cat_to_q(v)) for v in vec]
        return samples
    
    def fixed_grid_sample(self, n_sample = 1):
        assert n_sample >= len(self.wq_types), "we need at least 3 samples to cover cases"
        samples = []

        # Unified Quantization Data
        for i in range(len(self.wq_types)):
            samples.append([[sorted(self.wq_types)[i]] * self.n_layers, 
                             [sorted(self.aq_types)[i]] * self.n_layers])
        
        constr_q = [q for q in self.layer_q if (q[0] <= q[1])]
        remain = n_sample - len(samples)
        n_qlayers = self.n_layers // remain + 1
        all_layers = []
        while remain > 0:
            sample = [[sorted(self.wq_types)[-1]] * self.n_layers,
                      [sorted(self.aq_types)[-1]] * self.n_layers]
            # Select N Layers
            if len(all_layers) == 0:
                all_layers = list(range(self.n_layers))
            
            selected_layers = random.sample(all_layers, 
                                            min(n_qlayers, len(all_layers)))
            # Remove Selected Layers                
            for l_idx in selected_layers:
                q = random.choice(constr_q)
                sample[0][l_idx] = q[0]
                sample[1][l_idx] = q[1]
                all_layers.remove(l_idx)
            if sample in samples:
                continue
            samples.append(sample)
            remain -= 1
        return samples

    def map_cat_to_q(self, cat):
        wq = cat[:self.n_layers]
        aq = cat[self.n_layers:]
        return [self.wq_types[w] for w in wq] + [self.aq_types[a] for a in aq]

    def sample_to_vec(self, qscheme):
        wq = qscheme[0]
        aq = qscheme[1]
        return wq + aq
    
    def vec_to_sample(self, vec):
        wq = vec[:self.n_layers]
        aq = vec[self.n_layers:]
        return [wq, aq]

    def tuple_to_sample(self, tup):
        sample = list(zip(*tup))
        sample = [list(sample[0]), list(sample[1])]
        return sample

    def sort_space_by_bops(self, space=None, use_max_q=True):
        print("Sorting Space by BOPs...")
        if space is None: # Use the complete space
            total_space = list(itertools.product(self.layer_q, repeat=self.n_layers))
            total_space = [list(tuple(zip(*qscheme))) for qscheme in total_space]
            total_space = [[list(qscheme[0]), list(qscheme[1])] for qscheme in total_space]
            total_space = sorted(total_space, key=lambda x: self.get_bops(x, use_max_q))
        else:
            total_space = sorted(space, key=lambda x: self.get_bops(x, use_max_q))
        return total_space

    def sort_space_by_size(self, space=None):
        print("Sorting Space by Size...")
        if space is None: # Use the complete space
            total_space = list(itertools.product(self.layer_q, repeat=self.n_layers))
            total_space = [list(tuple(zip(*qscheme))) for qscheme in total_space]
            total_space = [[list(qscheme[0]), list(qscheme[1])] for qscheme in total_space]
            total_space = sorted(total_space, key=lambda x: self.get_model_size(x, use_max_q=True))
        else:
            total_space = sorted(space, key=lambda x: self.get_model_size(x, use_max_q=True))
        return total_space

    def get_bops(self, qscheme, use_max_q = True):
        bops = 0
        wq = np.array(qscheme[0])
        aq = np.array(qscheme[1])
        if use_max_q:
            bops = np.dot(np.max([wq, aq], axis=0)**2 + wq + aq, self.layer_macs)
        else:
            bops = np.dot(wq*aq + wq + aq, self.layer_macs)
        return bops
    def get_model_size(self, qscheme):
        wq = np.array(qscheme[0])
        return np.dot(wq, self.layer_params)
    
    def get_mem_loads(self, qscheme):
        wq = np.array(qscheme[0])
        aq = np.array(qscheme[1])
        return np.dot(wq + aq, self.layer_macs)
    
    def get_qat_accuracy(self, qscheme, verbose=False, epochs=None):
        if epochs is None:
            epochs = self.epochs
        wq = copy.deepcopy(qscheme[0])
        wq = [f"{int(w)}b" if w != 1.58 else "1.5b" for w in wq]
        aq = copy.deepcopy(qscheme[1])
        checkpoint = torch.load(self.pretrained_model)
        self.model.load_state_dict(checkpoint)
        replace_quantize_layers(self.model, wq, aq, quant_aware=True)
        res = self.model.train_loop(epochs,
                            self.train_loader, self.test_loader, 
                            verbose=verbose, lr=self.lr)
        if verbose:
            print("Model Results: ", res)
        return res["TestAcc"]
    
    def get_ptq_accuracy(self, qscheme, verbose=False):
        wq = copy.deepcopy(qscheme[0])
        wq = [f"{int(w)}b" if w != 1.58 else "1.5b" for w in wq]
        aq = copy.deepcopy(qscheme[1])
        checkpoint = torch.load(self.pretrained_model)
        self.model.load_state_dict(checkpoint)
        replace_quantize_layers(self.model, wq, aq, quant_aware=False)
        set_to_qforward(self.model)
        res = self.model.test(self.test_loader)
        if verbose:
            print("Model Results: ", res)
        return res["TestAcc"]
    
    def get_scheme_str(self, qscheme):
        wq = qscheme[0]
        aq = qscheme[1]
        name = "-".join([f"w{w}a{a}" for w, a in zip(wq, aq)])
        return name
    
    def eval_scheme(self, qscheme, verbose = False, ptq = False, epochs=None):

        print("QScheme:", qscheme)
        BOPs = self.get_bops(qscheme)
        print("BOPs:", BOPs)
        OldBOPs = self.get_bops(qscheme, False)
        Size = self.get_model_size(qscheme)
        print("Model Size:", Size)
        if ptq:
            Accuracy = self.get_ptq_accuracy(qscheme, verbose)
        else:
            Accuracy = self.get_qat_accuracy(qscheme, verbose, epochs)
        print("Accuracy:", Accuracy)

        return {"MaxBOPs": float(BOPs), 
                "BOPs": float(OldBOPs), 
                "Size": float(Size), 
                "Accuracy": Accuracy}
    
    def get_subspace(self, space_size, constr_bops = None, 
                     roi_range : float = 1.0, min_space_size = 1,
                     use_max_q = False):
        if self.space_size < space_size:
            subspace = self.sort_space_by_bops(use_max_q=use_max_q)
        else:
            print(f"Space Size is too large! Uniformly Sampling {space_size} samples...")
            subspace = self.np_uniform_sample(space_size)
            subspace = self.sort_space_by_bops(use_max_q=use_max_q, space=subspace)
        if constr_bops is not None:
            constr_subspace = []
            for sample in subspace:
                bops = self.get_bops(sample, use_max_q=use_max_q)
                if (bops <= constr_bops) and (bops > constr_bops * (1-roi_range)):
                    constr_subspace.append(sample)
            
            while len(constr_subspace) < min_space_size:
                subspace = self.np_uniform_sample(space_size)
                for sample in subspace:
                    bops = self.get_bops(sample, use_max_q=use_max_q)
                    if (bops <= constr_bops) and (bops > constr_bops * (1-roi_range)):
                        constr_subspace.append(sample)
                        if len(constr_subspace) >= space_size:
                            break
            subspace = constr_subspace
            print("Constrained Space Size:", len(subspace))
        return subspace
    
    def dist_to_roi(self, sample, constr_bops, roi_range = 0.1):
        max_bops = self.get_bops(sample, use_max_q=True)
        lb = constr_bops * (1-roi_range)
        ub = constr_bops
        max_bops_dist = 0.0
        if (max_bops <= ub) and (max_bops >= lb):
            max_bops_dist = 0.0
        elif max_bops < lb:
            max_bops_dist =  lb - max_bops
        else:
            max_bops_dist = max_bops - ub
        return max_bops_dist

    def get_subspace_exhaustive(self, constr_bops,
                                roi_range : float = 0.2,
                                use_max_q = False):
        # NOTE: This method is only suitable for small search space
        constr_q = [q for q in self.layer_q if q[0] <= q[1]]
        subspace = []
        for scheme in itertools.product(constr_q, repeat=self.n_layers):
            sample = self.tuple_to_sample(scheme)
            bops = self.get_bops(sample, use_max_q)
            if bops <= constr_bops and bops >= constr_bops * (1-roi_range):
                subspace.append(sample)
        print("Constrained Space Size:", len(subspace))
        return subspace
    
    def get_subspace_rand(self, constr_bops, SAMPLE_SIZE = 1000,
                                roi_range : float = 0.2,
                                use_max_q = False):
        # NOTE: This method is only suitable for small search space
        constr_q = [q for q in self.layer_q if q[0] <= q[1]]
        subspace = []
        while len(subspace) < SAMPLE_SIZE:
            sample = random.choices(constr_q, k=self.n_layers)
            bops = self.get_bops(sample, use_max_q)
            if bops <= constr_bops and bops >= constr_bops * (1-roi_range):
                subspace.append(sample)
        return subspace

    def get_subspace_genetic(self, constr_bops, 
                            roi_range : float = 0.2, min_space_size = 1,
                            use_max_q = False, max_gen = 100, prune = True):
        subspace = []
        # Constrain AQ >= WQ
        constr_q = [q for q in self.layer_q if q[0] <= q[1]] if prune else self.layer_q
        # Insert Uniform Base Samples
        for q in constr_q:
            sample = [q] * self.n_layers
            sample = self.tuple_to_sample(sample)
            subspace.append(sample)
        constr_space = []
        for sample in subspace:
            bops = self.get_bops(sample, use_max_q)
            if bops <= constr_bops:
                constr_space.append(sample)
        if len(constr_space) <= 1:
            print(f"Warning: Constr is too tight! Only {len(constr_space)} Base Solution(s)!")
        # Initial Population
        while len(subspace) < min_space_size:
            sample = random.choices(constr_q, k=self.n_layers)
            sample = self.tuple_to_sample(sample)
            if sample in subspace:
                continue
            if self.get_bops(sample, use_max_q) >= constr_bops*(1-roi_range):
                subspace.append(sample)
        assert(len(subspace) > 1), "Initial Population is too small!"
        gen = 0
        while True:
            # Get Generation
            while len(subspace) < min_space_size*2:
                pair = random.sample(subspace, 2)

                # Cross Over
                sample = copy.deepcopy(pair[0])
                for i in range(self.n_layers):
                    rand_prob = self.layer_macs[i] / max(self.layer_macs)
                    if random.random() < rand_prob:
                        sample[0][i] = pair[1][0][i]
                        sample[1][i] = pair[1][1][i]

                # Mutation
                for i in range(self.n_layers):
                    if random.random() < 0.1:
                        q = random.choice(constr_q)
                        sample[0][i] = q[0]
                        sample[1][i] = q[1]
                if sample in subspace:
                    continue
                subspace.append(sample)
            # Ranking
            subspace = sorted(
                subspace, 
                key=lambda x: self.dist_to_roi(x, constr_bops, roi_range)
                )
            subspace = subspace[:min_space_size]
            if self.dist_to_roi(subspace[-1], constr_bops, 
                                roi_range) == 0:
                # Complete
                break
            gen += 1
            if gen > max_gen:
                print("Warning: Genetic Search Exceeds Max Generation!")
                for i in range(min_space_size):
                    if self.dist_to_roi(subspace[i], 
                        constr_bops, roi_range) > 0:
                        end_idx = i
                        break
                subspace = subspace[:end_idx]
                break
        return subspace

    def per_layer_analysis(self, qbit = 4):
        data = {}
        scheme = [[8] * self.n_layers, [8] * self.n_layers]
        print("QScheme:", scheme)
        name = self.get_scheme_str(scheme)
        print("QScheme Name:", name)
        data[name] = self.eval_scheme(scheme, verbose=True, ptq=True)
        for i in range(self.n_layers):
            scheme = [[8] * self.n_layers, [8] * self.n_layers]
            scheme[0][i] = qbit
            scheme[1][i] = qbit
            print("QScheme:", scheme)
            name = self.get_scheme_str(scheme)
            print("QScheme Name:", name)
            data[name] = self.eval_scheme(scheme, verbose=True, ptq=True)
            print(data[name])
        return data

    def per_layer_l2_analysis(self):
        data = {}
        idx = 0
        print("Per Layer L2 Analysis...")
        for layer in tqdm(self.layers):
            layer_l2_error = {}
            for qbit in self.wq_types:
                layer.qtype = f"{qbit}b"
                weights = layer.weight.data
                qweights = layer.weight_quant(weights)
                l2_norm = torch.norm(qweights - weights, p=2).item()
                layer_l2_error[qbit] = l2_norm
            data[f"{idx}"] = layer_l2_error
            idx += 1
        return data
    
    def per_layer_act_analysis(self, qbit = 4):
        data = {}
        idx = 0
        for i in range(self.n_layers):
            scheme = [[8] * self.n_layers, [8] * self.n_layers]
            scheme[1][i] = qbit
            print("QScheme:", scheme)
            name = self.get_scheme_str(scheme)
            print("QScheme Name:", name)
            data[name] = self.eval_scheme(scheme, verbose=True, ptq=True)
            print(data[name])
            idx += 1
        return data
    
    def per_layer_distribution_analysis(self, save_path = "layer_distribution.pdf"):
        data = {}
        print("Per Layer Distribution Analysis...")
        plt.subplots(self.n_layers, 1, figsize=(8, 6*self.n_layers))
        for i in range(self.n_layers):
            weights = self.layers[i].weight.data
            absmean = torch.mean(torch.abs(weights)).item()
            dist = torch.histc(weights, bins=256, 
                               min=weights.min().item(), 
                               max=weights.max().item())
            line = np.linspace(weights.min().item(), weights.max().item(), 256)
            plt.subplot(self.n_layers, 1, i+1)
            plt.vlines( absmean, 0, dist.max().item(), color="red")
            plt.vlines(-absmean, 0, dist.max().item(), color="red")
            plt.plot(line, dist.cpu().numpy(), color="blue", label="Original")
            plt.title(f"Layer {i}")
            scheme = [[8] * self.n_layers, [8] * self.n_layers]
            scheme[0][i] = 1
            scheme[1][i] = 1
            self.get_qat_accuracy(scheme, verbose=True)

            weights = self.layers[i].weight.data
            dist = torch.histc(weights, bins=256, 
                               min=weights.min().item(), 
                               max=weights.max().item())
            line = np.linspace(weights.min().item(), weights.max().item(), 256)
            plt.plot(line, dist.cpu().numpy(), color="red", label="QAT-ed")
            plt.title(f"Layer {i}")
        plt.savefig(save_path)
        return data