import torch
from torch import nn

from MiCoModel import MiCoModel


class HARMLP(MiCoModel):
    def __init__(self, input_dim: int = 561, hidden=None, n_classes: int = 6, dropout: float = 0.2) -> None:
        super().__init__()
        if hidden is None:
            hidden = (256, 128, 64)

        layers = []
        in_dim = input_dim
        for width in hidden:
            layers.extend([nn.Linear(in_dim, width), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = width
        layers.append(nn.Linear(in_dim, n_classes))

        self.layers = nn.Sequential(*layers)
        self.n_layers = len(hidden) + 1
        self.default_dataset = "UCI_HAR"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        return self.layers(x)
