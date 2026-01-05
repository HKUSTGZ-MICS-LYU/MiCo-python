import torch
import torch.nn as nn

from tqdm import tqdm
import numpy as np

from matplotlib import pyplot as plt

from MiCoEval import MiCoEval
from searchers.QSearcher import QSearcher
from DimTransform import DimTransform

from searchers.SearchUtils import (
    random_sample, random_sample_min_max, 
    grid_sample, near_constr_sample
)

from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.acquisition import LogExpectedImprovement, UpperConfidenceBound

from gpytorch.constraints import Interval
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood


from xgboost import XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor

from searchers.Ensemble.Models import MPGPEnsemble, RFModel

class MiCoSearcher(QSearcher):

    NUM_SAMPLES = 1000
    constr_value : float = None

    def __init__(self, evaluator: MiCoEval, 
                 n_inits: int = 10, 
                 qtypes: list = [4,5,6,7,8],
                 dim_trans: DimTransform = None) -> None:
        
        super().__init__(evaluator, n_inits, qtypes)
        
        # self.n_inits = n_inits
        # self.evaluator = evaluator
        self.dims = self.dim
        self.qtypes = qtypes
        self.roi = 0.2 # Start with 20% ROI
        self.dim_transform = dim_trans
        if self.dim_transform is not None:
            assert self.dim_transform.out_dim == self.n_layers * 2
            self.dims = self.dim_transform.in_dim
        print("Searcher Input Dim:", self.dims)
        return
    
    def constr(self, scheme: list):
        if self.dim_transform is not None:
            scheme = self.dim_transform(scheme)
        return self.evaluator.constr(scheme)
    
    def eval(self, scheme: list):
        if self.dim_transform is not None:
            scheme = self.dim_transform(scheme)
        return self.evaluator.eval(scheme)

    def sample(self, n_samples: int):

        # TODO: Extract High Quality Sampled Schemes
        initial_pop = []

        return near_constr_sample(n_samples=n_samples,
                                 qtypes=self.qtypes,
                                 dims=self.dims,
                                 constr_func=self.constr,
                                 constr_value=self.constr_value,
                                 roi=self.roi,
                                 layer_macs=self.evaluator.layer_macs,
                                 initial_pop=initial_pop)
    
    def initial(self, n_samples: int):
        # return near_constr_sample(n_samples=n_samples,
        #                          qtypes=self.qtypes,
        #                          dims=self.dims,
        #                          constr_func=self.evaluator.constr,
        #                          constr_value=self.constr_value)
        return grid_sample(n_samples, self.qtypes, self.dims)
        # return random_sample_min_max(n_samples, self.qtypes, self.dims)
    
    def select(self, X, constr_value):
        constrs = []
        for x in X:
            constrs.append(self.constr(x))
        constrs = np.array(constrs) <= constr_value
        X = np.array(X)
        X = X[constrs].tolist()
        return X

    def search(self, n_iter: int, target: str, 
               constr: str = None, 
               constr_value = None):

        self.evaluator.set_eval(target)
        if constr:
            self.evaluator.set_constraint(constr)
            self.constr_value = constr_value
        
        # Initialize the search space
        sampled_X = self.initial(self.n_inits)
        sampled_y = []
        for x in sampled_X:
            print("Initial Scheme:", x)
            y = self.eval(x)
            print("Initial Result:", y)
            sampled_y.append(y)
        sampled_y = np.array(sampled_y)

        final_x = None
        final_y = None
        for i in range(n_iter):
            
            self.roi = 0.2 + 0.3 * (i / n_iter)
            print("ROI:", self.roi)

            X = []
            if constr:
                timeout_count = 0
                while len(X) == 0:
                    X = self.sample(self.NUM_SAMPLES)
                    X = self.select(X, constr_value)
                    timeout_count += 1
                    if timeout_count > 10:
                        raise ValueError("Cannot find any feasible solution. Please consider relaxing the constraint.")
            else:
                X = self.sample(self.NUM_SAMPLES)

            # best_x, best_y = self.bayes_opt(sampled_X, sampled_y, X)
            # best_x, best_y = self.xgb_opt(sampled_X, sampled_y, X)
            best_x, best_y = self.rf_opt(sampled_X, sampled_y, X)
            # best_x, best_y = self.bayes_ensemble_opt(sampled_X, sampled_y, X)
            
            print("Predicted Best Scheme:", best_x)
            print("Predicted Best Result:", best_y)
            eval_y = self.eval(best_x)
            print("Actual Best Result:", eval_y)
            if (final_y is None) or (eval_y > final_y):
                final_x = best_x
                final_y = eval_y

            sampled_X = np.vstack([sampled_X, best_x])
            sampled_y = np.append(sampled_y, eval_y)
            
            self.best_trace.append(final_y)

        print("Optimization Ends...")
        print("Best Scheme:", final_x)
        print("Best Result:", final_y)
        if constr:
            constr_x = self.constr(final_x)
            print(f"Constraint Value: {constr_x} ({constr_x/constr_value:.5f})")
        return final_x, final_y

    def bayes_opt(self, sampled_X, sampled_y, X):

        X_tensor = torch.tensor(sampled_X, dtype=torch.float)
        Y_tensor = torch.tensor(sampled_y, dtype=torch.float).unsqueeze(-1)

        gpr = SingleTaskGP(X_tensor, Y_tensor, 
                            covar_module=ScaleKernel(MaternKernel(
                                nu=2.5, lengthscale_constraint=Interval(0.005, 4.0)
                            )))
        mll = ExactMarginalLogLikelihood(gpr.likelihood, gpr)
        fit_gpytorch_mll(mll)

        max_Y = max(Y_tensor)
        acq = LogExpectedImprovement(gpr, best_f=max_Y)
        acq_val = acq(torch.tensor(X, dtype=torch.float).unsqueeze(-2))
        best = torch.argmax(acq_val)
        best_x = X[best]
        best_y = gpr.posterior(
                torch.tensor([best_x], dtype=torch.float)).mean.item()
            
        return best_x, best_y
    
    def bayes_ensemble_opt(self, sampled_X, sampled_y, X):

        X_tensor = torch.tensor(sampled_X, dtype=torch.float)
        Y_tensor = torch.tensor(sampled_y, dtype=torch.float).unsqueeze(-1)

        model = MPGPEnsemble(X_tensor, Y_tensor)

        max_Y = max(Y_tensor)
        acq = LogExpectedImprovement(model, best_f=max_Y)
        acq_val = acq(torch.tensor(X, dtype=torch.float).unsqueeze(-2))
        best = torch.argmax(acq_val)
        best_x = X[best]
        best_x_tensor = torch.tensor(best_x, dtype=torch.float).view(1, -1)
        best_y = model.posterior(best_x_tensor).mean.item()
            
        return best_x, best_y
    
    def xgb_opt(self, sampled_X, sampled_y, X):
        xgb = XGBRegressor()

        xgb.fit(sampled_X, sampled_y)
        y_pred = xgb.predict(X)
        best_idx = np.argmax(y_pred)
        best_x = X[best_idx]
        best_y = y_pred[best_idx]

        return best_x, best_y
    
    def rf_opt(self, sampled_X, sampled_y, X):
        
        if self.dims > 10:
            # Prevent overfitting for high-dimensional data
            rf = RandomForestRegressor(
                n_estimators=250, max_depth=15, max_features="sqrt"
            )
        else:
            # Use default parameters for low-dimensional data
            rf = RandomForestRegressor()

        rf.fit(sampled_X, sampled_y)
        y_pred = rf.predict(X)
        best_idx = np.argmax(y_pred)
        best_x = X[best_idx]
        best_y = y_pred[best_idx]

        return best_x, best_y