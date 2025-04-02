import torch
import json
import numpy as np
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from MiCoModel import MiCoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CmsisCNN(MiCoModel):
    def __init__(self, in_channels: int) -> None:
        super(CmsisCNN, self).__init__()

        self.in_channels = in_channels
        self.default_dataset = "CIFAR10"
        layers = [
            nn.Conv2d(in_channels, 32, (5,5), padding=2),
            nn.ReLU(),
            nn.MaxPool2d((2,2),2),
            nn.Conv2d(32, 32, (5,5), padding=2),
            nn.ReLU(),
            nn.MaxPool2d((2,2),2),
            nn.Conv2d(32, 64, (5,5), padding=2),
            nn.ReLU(),
            nn.MaxPool2d((2,2),2),
            nn.Flatten(),
            nn.Linear(1024, 10),
        ]
        self.layers = nn.Sequential(*layers)
        self.n_layers = 4 # 3 conv2d + 1 linear
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x