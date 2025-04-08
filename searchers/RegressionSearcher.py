import numpy as np

from matplotlib import pyplot as plt

from MiCoEval import MiCoEval
from searchers.QSearcher import QSearcher
from searchers.SearchUtils import (
    random_sample, random_sample_min_max, grid_sample
)

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

class RegressionSearcher(QSearcher):

    NUM_SAMPLES = 1000

    def __init__(self, evaluator: MiCoEval, 
                 n_inits: int = 10, 
                 qtypes: list = [4,5,6,7,8],
                 model_type: str = "xgb") -> None:
        
        super().__init__(evaluator, n_inits, qtypes)

        self.n_layers = evaluator.model.n_layers
        self.dims = self.n_layers * 2
        if model_type == "gp":
            self.model = GaussianProcessRegressor()
        elif model_type == "rf":
            self.model = RandomForestRegressor()
        elif model_type == "xgb":
            self.model = XGBRegressor()
        else:
            raise ValueError("Unknown Model Type")

        self.best_trace = []

        return
    
    def sample(self, n_samples: int):
        return random_sample(n_samples, self.qtypes, self.dims)
    
    def initial(self, n_samples: int):
        # return grid_sample(n_samples, self.qtypes, self.dims)
        return random_sample_min_max(n_samples, self.qtypes, self.dims)
    
    def select(self, X, constr_value):
        X = [x for x in X if self.evaluator.constr(x) <= constr_value]
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

            self.model.fit(sampled_X, sampled_y)
            X = []

            if constr:
                while len(X) == 0:
                    X = self.sample(self.NUM_SAMPLES)
                    X = self.select(X, constr_value)
            else:
                X = self.sample(self.NUM_SAMPLES)

            y = self.model.predict(X)
            best = np.argmax(y)
            best_x = X[best]
            best_y = y[best]
            
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