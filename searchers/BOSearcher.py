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

from torch.quasirandom import SobolEngine

from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.acquisition import LogExpectedImprovement

from gpytorch.constraints import Interval
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

from MiCoSearch import MiCoSearch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_RESTARTS = 10

class MiCoBOSearcher:
    def __init__(self, search: MiCoSearch) -> None:
        self.mpq = search
        self.layer_macs = self.mpq.layer_macs
        self.layer_params = self.mpq.layer_params
        self.n_layers = self.mpq.n_layers
        return
    
    def search(self, search_budget, data_path:str, 
            constr_bops=None, constr_size=None,
            n_init : int = None, use_max_q = False,
            ptq = False, SAMPLE_MIN  = 1000):
        if n_init is None:
            n_init = search_budget // 2
        assert n_init < search_budget, "n_init should be less than search_budget"

        data = {}
        trace = []
        if os.path.exists(data_path):
            with open(data_path, "r") as f:
                data = json.load(f)
        
        # Initial Samples
        samples = self.mpq.fixed_grid_sample(n_init)
        X_data = []
        Y_data = []
        for qscheme in samples:
            name = self.mpq.get_scheme_str(qscheme)
            if name not in data:
                # Shorter Re-training for BO Search
                data[name] = self.mpq.eval_scheme(qscheme, ptq=ptq, 
                                                epochs=self.mpq.epochs//5)
                with open(data_path, "w") as f:
                    json.dump(data, f, indent=2)
            Y_data.append(data[name]["Accuracy"])
            X_data.append(self.mpq.sample_to_vec(qscheme))
            trace.append((name, data[name]))
            search_budget -= 1
        print("-" * 32)
        print("Start Regressive Search...")
        while search_budget > 0:
            print("Search Budget:", search_budget)
            # Fit Regressor
            print("Fitting Regressor...")
            X_tensor = torch.tensor([X_data], dtype=torch.float64, device=DEVICE) / 8 # Normalize X
            Y_tensor = torch.tensor([Y_data], dtype=torch.float64, device=DEVICE).unsqueeze(-1)

            regressor = SingleTaskGP(X_tensor, Y_tensor, 
                            covar_module=ScaleKernel(MaternKernel(
                                nu=2.5, lengthscale_constraint=Interval(0.005, 4.0)
                            )))

            mll = ExactMarginalLogLikelihood(regressor.likelihood, regressor)
            fit_gpytorch_mll(mll)

            best_Y = max(Y_data[n_init:]) if len(Y_data) > n_init else max(Y_data)
            acq = LogExpectedImprovement(regressor, best_f=best_Y)

            qscheme = None
            acc_pred = 0.0
            print("Optimizing Acquisition Function...")
            samples = []
            for _ in range(NUM_RESTARTS):
                samples += self.mpq.get_subspace_genetic(constr_bops, 
                                              min_space_size=SAMPLE_MIN,
                                              use_max_q=use_max_q,
                                              prune=not ptq)
            vecs = np.array([self.mpq.sample_to_vec(sample) for sample in samples])
            vec_tensor = torch.tensor(vecs, dtype=torch.float64, device=DEVICE) / 8 # Normalize X
            acq_vals = acq(
                vec_tensor.unsqueeze(-2)
            )
            acq_args = torch.argmax(acq_vals).item()
            qscheme = samples[acq_args]

            acc_pred = regressor.posterior(vec_tensor[acq_args].unsqueeze(0)).mean.item()

            name = self.mpq.get_scheme_str(qscheme)
            print("QScheme Name:", name)
            if name in data:
                print("Already Evaluated")
            else:
                # Shorter Re-training for BO Search
                data[name] = self.mpq.eval_scheme(qscheme, ptq=ptq, 
                                                epochs=self.mpq.epochs//5)
                with open(data_path, "w") as f:
                    json.dump(data, f, indent=2)
            print(data[name])
            acc_real = data[name]["Accuracy"]

            # Get GP Accuracy
            all_pred = regressor.posterior(X_tensor).mean
            err_pred = nn.functional.mse_loss(all_pred, Y_tensor).item()
            print("Model Error: ", err_pred)
            print(f"Predicted: {acc_pred:.4f}, Real: {acc_real:.4f}",
                  f"Error: {acc_pred - acc_real:.4f}")

            print("-" * 32)
            Y_data.append(data[name]["Accuracy"])
            X_data.append(self.mpq.sample_to_vec(qscheme))
            search_budget -= 1
            trace.append((name, data[name]))

        bo_X, bo_Y = X_data[n_init:], Y_data[n_init:]
        best_idx = np.argmax(bo_Y)
        best_scheme = self.mpq.vec_to_sample(bo_X[best_idx])
        print("Best Accuracy:", bo_Y[best_idx], 
              "Best Scheme:", best_scheme)
        # QAT on Best Scheme
        res = self.mpq.eval_scheme(best_scheme, verbose=True, ptq=ptq)
        best_name = self.mpq.get_scheme_str(best_scheme)
        print("Final Results: ", res)
        return {best_name: res}, trace