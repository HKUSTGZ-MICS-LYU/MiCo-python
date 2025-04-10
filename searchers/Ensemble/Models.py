from typing import List, Optional, Union

import numpy as np

import torch

from botorch import fit_gpytorch_mll
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.ensemble import EnsembleModel
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model import Model, ModelList
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.posteriors.posterior import Posterior
from botorch.posteriors.torch import TorchPosterior
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.arc_kernel import ArcKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from sklearn.ensemble import RandomForestRegressor
from torch import Tensor, distributions, nn
from xgboost import XGBRegressor


class RFModel(EnsembleModel):
    rf: List[RandomForestRegressor]
    num_samples: int
    _num_outputs: int

    def __init__(self, train_X : Tensor, train_Y : Tensor,
                  num_samples: int = 100) -> None:
        super(RFModel, self).__init__()
        self._num_outputs = train_Y.shape[-1]
        self.num_samples = num_samples
        self.rf = [
            RandomForestRegressor(n_estimators=num_samples)
            for _ in range(self._num_outputs)
        ]
        self.fit(train_X, train_Y)
        return

    def fit(self, X: Tensor, y: Tensor) -> None:
        for i in range(self._num_outputs):
            self.rf[i].fit(X.cpu().numpy(), y[:, i].cpu().numpy())

    def forward(self, X: Tensor) -> Tensor:
        # Get numpy array but preserve dimensions needed for sklearn
        X_np = X.detach().cpu().numpy()
        
        # Ensure X_np is 2D for sklearn, with shape [n_samples, n_features]
        if len(X_np.shape) == 3:
            # Input has batched shape [batch, candidates, features]
            # Reshape to 2D for sklearn
            batch_size, n_candidates, n_features = X_np.shape
            X_np_2d = X_np.reshape(batch_size * n_candidates, n_features)
        else:
            # Input is already 2D [samples, features] or reshape if needed
            if len(X_np.shape) == 1:
                # Single sample with shape [features], reshape to [1, features]
                X_np_2d = X_np.reshape(1, -1)
            else:
                X_np_2d = X_np
        
        # Now predict with the properly shaped input
        y_pred = np.stack([
            np.array([tree.predict(X_np_2d) for tree in rf.estimators_])
            for rf in self.rf
        ], axis=-1)
        
        samples = torch.from_numpy(y_pred).to(X)
        
        if len(X.shape) == 3:
            samples = samples.reshape(X.shape[0], self.num_samples, -1, self._num_outputs)
        
        return samples

class AveragePosterior(Posterior):
    def __init__(self, posteriors: List[Posterior]) -> None:
        self.posteriors = posteriors
        self.mean = torch.stack([p.mean for p in self.posteriors], dim=-1).mean(dim=-1)
        self.variance = torch.stack([p.variance for p in self.posteriors], dim=-1).mean(dim=-1)
        return

    @property
    def device(self) -> torch.device:
        return self.posteriors[0].device
    
    @property
    def dtype(self) -> torch.dtype:
        return self.posteriors[0].dtype

    def rsample(self, sample_shape = None):
        samples = torch.stack([p.rsample(sample_shape) for p in self.posteriors], dim=-1)
        return samples.mean(dim=-1)

class RFGPEnsemble(Model):
    _num_outputs: int

    def __init__(self, train_X: Tensor, train_Y: Tensor):
        super(RFGPEnsemble, self).__init__()
        self._num_outputs = train_Y.shape[-1]

        # First model is a Random Forest
        self.rf = RFModel(train_X, train_Y)

        # Second model is a GP
        covar = ScaleKernel(MaternKernel())
        self.gp = SingleTaskGP(train_X, train_Y, covar_module=covar,
            input_transform=Normalize(d=train_X.shape[-1]),
            outcome_transform=Standardize(m=train_Y.shape[-1]))
        self.mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_mll(self.mll)

        return
    
    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[list[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        posterior_transform: Optional[PosteriorTransform] = None,
    ) -> AveragePosterior:
        if output_indices:
            X = X[..., output_indices]

        rf_pos = self.rf.posterior(X)
        gp_pos = self.gp.posterior(X)
        posterior = AveragePosterior([rf_pos, gp_pos])

        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior


def transform_input(X: Tensor) -> Tensor:
    n_layers = X.shape[-1] // 2
    if len(X.shape) == 2:
        X_wq = X[:, :n_layers]
        X_aq = X[:, n_layers:2*n_layers]
        new_Xw = torch.zeros(X.shape[0], n_layers-1)
        new_Xa = torch.zeros(X.shape[0], n_layers-1)

        # Multiply adjacent bits in the input
        for i in range(n_layers - 1):
            new_Xw[:, i] = X_wq[:, i] * X_wq[:, i + 1]
            new_Xa[:, i] = X_aq[:, i] * X_aq[:, i + 1]
        new_X = torch.cat((new_Xw, new_Xa), dim=-1)
    elif len(X.shape) == 3:
        X_wq = X[:, :, :n_layers]
        X_aq = X[:, :, n_layers:2*n_layers]
        new_Xw = torch.zeros(X.shape[0], X.shape[1], n_layers-1)
        new_Xa = torch.zeros(X.shape[0], X.shape[1], n_layers-1)

        # Multiply adjacent bits in the input
        for i in range(n_layers - 1):
            new_Xw[:, :, i] = X_wq[:, :, i] * X_wq[:, :, i + 1]
            new_Xa[:, :, i] = X_aq[:, :, i] * X_aq[:, :, i + 1]
        new_X = torch.cat((new_Xw, new_Xa), dim=-1)
    else:
        raise ValueError("Input tensor must be either 2D or 3D.")
    return new_X

class MPGPEnsemble(Model):
    _num_outputs: int

    def __init__(self, train_X: Tensor, train_Y: Tensor):
        super(MPGPEnsemble, self).__init__()
        self._num_outputs = train_Y.shape[-1]

        # First model is a Random Forest
        self.rf = RFModel(train_X, train_Y)

        # Second model is a GP
        covar = ScaleKernel(MaternKernel())
        self.gp1 = SingleTaskGP(train_X, train_Y, covar_module=covar)
        self.mll1 = ExactMarginalLogLikelihood(self.gp1.likelihood, self.gp1)
        fit_gpytorch_mll(self.mll1)

        # Third model is a GP with Transformed Input

        # Multiply adjacent bits in the input
        X_trans = transform_input(train_X)

        covar2 = ScaleKernel(MaternKernel())
        self.gp2 = SingleTaskGP(X_trans, train_Y, covar_module=covar2)
        self.mll2 = ExactMarginalLogLikelihood(self.gp2.likelihood, self.gp2)
        fit_gpytorch_mll(self.mll2)

        return
    
    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[list[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        posterior_transform: Optional[PosteriorTransform] = None,
    ) -> AveragePosterior:
        if output_indices:
            X = X[..., output_indices]

        rf_pos = self.rf.posterior(X)
        gp1_pos = self.gp1.posterior(X)
        gp2_pos = self.gp2.posterior(transform_input(X))
        posterior = AveragePosterior([rf_pos, gp1_pos, gp2_pos])

        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior