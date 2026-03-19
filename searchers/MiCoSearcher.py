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
from sklearn.ensemble import RandomForestRegressor

class MiCoSearcher(QSearcher):

    NUM_SAMPLES_NC = 1000 # Near-Constrained Sampling
    NUM_SAMPLES_RAND = 5000 # Random Sampling
    constr_value : float = None

    def __init__(self, evaluator: MiCoEval, 
                 n_inits: int = 10, 
                 qtypes: list = [4,5,6,7,8],
                 regressor: str = "ensmble",
                 initial_method: str = "orth",
                 sample_method: str = "near-constr",
                 feature_en: bool = True,
                 ucb : bool = True,
                 roi: float = 0.1,
                 max_roi: float = 0.3,
                 rf_n_estimators: int = None,
                 rf_max_depth: int = None,
                 rf_max_features = None,
                 rf_min_samples_split = None,
                 rf_min_samples_leaf = None,
                 rf_random_state: int = None,
                 dim_trans: DimTransform = None) -> None:
        
        super().__init__(evaluator, n_inits, qtypes)
        
        self.regressor = regressor
        self.initial_method = initial_method
        self.sample_method = sample_method
        self.feature_en = feature_en

        self.NUM_SAMPLES = {
            "random": self.NUM_SAMPLES_RAND,
            "near-constr": self.NUM_SAMPLES_NC
        }[self.sample_method]

        self.regressor_dict = {
            "rf": self.regress_opt,
            "xgb": self.regress_opt,
            "bayes": self.bayes_opt,
            "ensmble": self.ensmble_opt,
        }
        if ucb:
            self.regressor_dict["rf"] = self.rf_ucb

        assert regressor in self.regressor_dict, f"Regressor {regressor} not supported."

        self.dims = self.dim
        self.qtypes = qtypes
        self.roi_start = roi
        self.max_roi = max_roi
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_depth = rf_max_depth
        self.rf_max_features = rf_max_features
        self.rf_min_samples_split = rf_min_samples_split
        self.rf_min_samples_leaf = rf_min_samples_leaf
        self.rf_random_state = rf_random_state
        self.dim_transform = dim_trans
        if self.dim_transform is not None:
            assert self.dim_transform.out_dim == self.n_layers * 2
            self.dims = self.dim_transform.in_dim
        print("Searcher Input Dim:", self.dims)
        self.sampled_X = None
        self.sampled_y = None

        self.max_value = self.constr([max(qtypes)] * self.dims)

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
        if self.sample_method == "random":
            return random_sample(n_samples, self.qtypes, self.dims)
        elif self.sample_method == "near-constr":
            # Extract High Quality Sampled Schemes
            new_X = np.array(self.sampled_X[self.n_inits:])
            new_Y = np.array(self.sampled_y[self.n_inits:])
            topk = max(1, len(new_Y) // 4)
            topk_indices = np.argsort(-new_Y)[:topk]
            initial_pop = new_X[topk_indices].tolist()

            return near_constr_sample(n_samples=n_samples,
                                     qtypes=self.qtypes,
                                     dims=self.dims,
                                     constr_func=self.constr,
                                     constr_value=self.constr_value,
                                     max_value=self.max_value,
                                     roi=self.roi,
                                     initial_pop=initial_pop)
        else:
            raise ValueError(f"Sample method {self.sample_method} not supported.")

    def initial(self, n_samples: int):
        if self.initial_method == "random":
            return random_sample_min_max(n_samples, self.qtypes, self.dims)
        elif self.initial_method == "orth":
            return grid_sample(n_samples, self.qtypes, self.dims)
        else:
            raise ValueError(f"Initial method {self.initial_method} not supported.")
    
    def select(self, X, constr_value):
        # Filter out the schemes that do not satisfy the constraint
        constrs = []
        for x in X:
            constrs.append(self.constr(x))
        constrs = np.array(constrs) <= constr_value
        X = np.array(X)
        X = X[constrs].tolist()
        return X
    
    def optimize(self, X):
        score = self.regressor_dict[self.regressor](X)

        sort_score = np.argsort(-score)  # Descending order

        # Convert sampled_X to list for reliable containment check
        sampled_X_list = self.sampled_X
        if isinstance(self.sampled_X, np.ndarray):
            sampled_X_list = self.sampled_X.tolist()

        topn = 0
        best_idx = sort_score[topn]
        best_x = X[best_idx]
        
        while best_x in sampled_X_list:
            topn += 1
            if topn >= len(X):
                print("All candidate schemes have been sampled!")
                topn = 0 # Fallback to best score (even if sampled)
                break
            best_idx = sort_score[topn]
            best_x = X[best_idx]
        best_y = score[best_idx]

        print("Predicted Best Scheme:", best_x)
        print("Predicted Score:", best_y)
        return best_x, best_y

    def search(self, n_iter: int, target: str, 
               constr: str = None, 
               constr_value = None):

        self.evaluator.set_eval(target)
        if constr:
            self.evaluator.set_constraint(constr)
            self.constr_value = constr_value
        
        # Initialize the search space
        self.sampled_X = self.initial(self.n_inits)
        self.sampled_y = []
        for x in self.sampled_X:
            print("Initial Scheme:", x)
            y = self.eval(x)
            print("Initial Result:", y)
            self.sampled_y.append(y)
        self.sampled_y = np.array(self.sampled_y)

        final_x = None
        final_y = None
        for i in range(n_iter):
            
            self.roi = self.roi_start + (self.max_roi - self.roi_start) * (i / n_iter)
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

            best_x, pred_y = self.optimize(X)
            eval_y = self.eval(best_x)
            print("Actual Best Result:", eval_y)
            if (final_y is None) or (eval_y > final_y):
                final_x = best_x
                final_y = eval_y

            self.sampled_X = np.vstack([self.sampled_X, best_x])
            self.sampled_y = np.append(self.sampled_y, eval_y)
            
            self.best_trace.append(final_y)

        print("Optimization Ends...")
        print("Best Scheme:", final_x)
        print("Best Result:", final_y)
        if constr:
            constr_x = self.constr(final_x)
            print(f"Constraint Value: {constr_x} ({constr_x/constr_value:.5f})")
        return final_x, final_y

    def feature_expand(self, X):
        # Compose Weight and Activation Bitwidths
        X = np.array(X)
        WQ = X[:, :self.n_layers]
        AQ = X[:, self.n_layers:]
        MACS = np.array(self.evaluator.layer_macs) / np.sum(self.evaluator.layer_macs)
        
        # First/Last Layer WQ/AQ
        # FST_WQ = WQ[:, 0:1]
        # FST_AQ = AQ[:, 0:1]
        # LST_WQ = WQ[:, -1:]
        # LST_AQ = AQ[:, -1:]

        # Expand to [WQ, AQ, WQ*MACS, AQ*MACS]
        expanded_X = np.hstack([WQ, AQ, WQ * MACS, AQ * MACS])

        return expanded_X

    def bayes_opt(self, X):

        train_X = self.feature_expand(self.sampled_X) if self.feature_en else self.sampled_X

        X_tensor = torch.tensor(train_X, dtype=torch.float)
        Y_tensor = torch.tensor(self.sampled_y, dtype=torch.float).unsqueeze(-1)

        gpr = SingleTaskGP(X_tensor, Y_tensor, 
                            covar_module=ScaleKernel(MaternKernel(
                                nu=2.5, lengthscale_constraint=Interval(0.005, 4.0)
                            )))
        mll = ExactMarginalLogLikelihood(gpr.likelihood, gpr)
        fit_gpytorch_mll(mll)

        X = self.feature_expand(X) if self.feature_en else X
        max_Y = max(Y_tensor)
        # acq = LogExpectedImprovement(gpr, best_f=max_Y)
        acq = UpperConfidenceBound(gpr, beta=0.1)
        acq_val = acq(torch.tensor(X, dtype=torch.float).unsqueeze(-2))

        # best = torch.argmax(acq_val)
        # best_x = X[best]
        # best_y = gpr.posterior(
        #         torch.tensor([best_x], dtype=torch.float)).mean.item()
            
        return acq_val.detach().cpu().numpy()

    def get_regressor(self):
        if self.regressor == "xgb":
            if self.dims > 20:
                # Prevent overfitting for high-dimensional data
                model = XGBRegressor(
                    n_estimators=250, max_depth=15, colsample_bytree=0.5
                )
            else:
                # Use constrained parameters for low-dimensional data to prevent overfitting
                model = XGBRegressor()
            return model
        elif self.regressor == "rf":

            rf_kwargs = {}
            sample_num = len(self.sampled_X)
            if self.dims > 20:
                rf_kwargs["max_features"] = "sqrt"
                rf_kwargs["n_estimators"] = 250
                rf_kwargs["max_depth"] = 15

            rf_kwargs["random_state"] = 42
            rf_kwargs["oob_score"] = True

            if self.rf_n_estimators is not None:
                rf_kwargs["n_estimators"] = self.rf_n_estimators
            if self.rf_max_depth is not None:
                rf_kwargs["max_depth"] = self.rf_max_depth
            if self.rf_max_features is not None:
                rf_kwargs["max_features"] = self.rf_max_features
            if self.rf_min_samples_split is not None:
                rf_kwargs["min_samples_split"] = self.rf_min_samples_split
            if self.rf_min_samples_leaf is not None:
                rf_kwargs["min_samples_leaf"] = self.rf_min_samples_leaf
            if self.rf_random_state is not None:
                rf_kwargs["random_state"] = self.rf_random_state
            model = RandomForestRegressor(**rf_kwargs)
            return model
        else:
            raise ValueError(f"Regressor {self.regressor} not supported.")

    def rf_ucb(self, X):
        model = self.get_regressor()

        train_X = self.feature_expand(self.sampled_X) if self.feature_en else self.sampled_X
        pred_X = self.feature_expand(X) if self.feature_en else X

        model.fit(train_X, self.sampled_y)

        if not hasattr(model, "estimators_") or len(model.estimators_) == 0:
            return model.predict(pred_X)

        tree_preds = np.array([est.predict(pred_X) for est in model.estimators_])
        pred_mean = np.mean(tree_preds, axis=0)
        pred_std = np.std(tree_preds, axis=0)

        ucb_beta = 0.5 * self.n_inits / len(self.sampled_X)
        return pred_mean + ucb_beta * pred_std

    def regress_opt(self, X):

        model = self.get_regressor()

        train_X = self.feature_expand(self.sampled_X) if self.feature_en else self.sampled_X
        pred_X = self.feature_expand(X) if self.feature_en else X

        model.fit(train_X, self.sampled_y)
        y_pred = model.predict(pred_X)

        return y_pred

    def ensmble_opt(self, X):
        # Voting Ensemble - combine predictions from multiple models via rank voting
        regressors = ["rf", "xgb", "bayes"]
        scores = []

        for regressor in regressors:
            self.regressor = regressor
            score = self.regressor_dict[regressor](X)
            scores.append(score)

        self.regressor = "ensmble"  # Reset the regressor

        # Rank-based voting: convert scores to ranks and average
        scores = np.array(scores)
        n_models, n_samples = scores.shape

        # Rank each model's predictions (higher score = higher rank)
        ranks = np.zeros_like(scores)
        for i in range(n_models):
            ranks[i] = np.argsort(np.argsort(-scores[i]))  # Descending rank

        # Average ranks across models
        avg_ranks = np.mean(ranks, axis=0)

        # Convert back to scores (higher average rank = better)
        return -avg_ranks  # Negate so higher = better for argmax
        