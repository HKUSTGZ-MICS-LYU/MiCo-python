import torch
import torch.nn as nn

from tqdm import tqdm
import numpy as np

from matplotlib import pyplot as plt

from MiCoEval import MiCoEval
from searchers.QSearcher import QSearcher

from searchers.SearchUtils import (
    random_sample, grid_sample
)

from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.acquisition import LogExpectedImprovement, UpperConfidenceBound

from gpytorch.constraints import Interval
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

class MiCoSearcher(QSearcher):

    NUM_SAMPLES = 10000

    def __init__(self, evaluator: MiCoEval, 
                 n_inits: int = 10, 
                 qtypes: list = [4,5,6,7,8]) -> None:
        
        super().__init__(evaluator, n_inits, qtypes)
        
        # self.n_inits = n_inits
        # self.evaluator = evaluator
        self.n_layers = evaluator.model.n_layers
        self.dims = self.n_layers * 2
        self.qtypes = qtypes

        return
    
    def sample(self, n_samples: int):
        return random_sample(n_samples, self.qtypes, self.dims)
    
    def initial(self, n_samples: int):
        return grid_sample(n_samples, self.qtypes, self.dims)
    
    def select(self, X, constr_value):
        constrs = []
        for x in X:
            constrs.append(self.evaluator.constr(x))
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
        
        # Initialize the search space
        sampled_X = self.initial(self.n_inits)
        sampled_y = []
        for x in sampled_X:
            print("Initial Scheme:", x)
            y = self.evaluator.eval(x)
            print("Initial Result:", y)
            sampled_y.append(y)
        sampled_y = np.array(sampled_y)

        final_x = None
        final_y = None
        for _ in range(n_iter):
            
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

            acq_val = acq(torch.tensor(X, dtype=torch.float).unsqueeze(-2))
            best = torch.argmax(acq_val)
            best_x = X[best]
            best_y = gpr.posterior(
                torch.tensor([best_x], dtype=torch.float)).mean.item()
            
            print("Predicted Best Scheme:", best_x)
            print("Predicted Best Result:", best_y)
            eval_y = self.evaluator.eval(best_x)
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
            constr_x = self.evaluator.constr(final_x)
            print(f"Constraint Value: {constr_x} ({constr_x/constr_value:.2f})")
        return final_x, final_y