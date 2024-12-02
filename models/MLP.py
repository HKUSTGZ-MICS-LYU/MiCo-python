import torch
import json
import numpy as np
from torch import nn
from tqdm import tqdm

from MiCoModel import MiCoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(MiCoModel):
    def __init__(self, in_features: int, config:dict) -> None:
        super(MLP, self).__init__()

        self.config = config
        self.in_features = in_features

        layers = []
        in_shape = in_features
        for layer_width in config["Layers"]:
            layers.append(nn.Linear(in_shape, layer_width, bias=False))
            layers.append(nn.ReLU())
            in_shape = layer_width
        layers.pop() # remove last ReLU
        self.layers = nn.Sequential(*layers)
        self.n_layers = len(config["Layers"])
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1) # anyway flatten the input first
        x = self.layers(x)
        return x