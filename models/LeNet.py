import torch
import json
import numpy as np
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from MiCoModel import MiCoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LeNet(MiCoModel):
    def __init__(self, in_channels: int) -> None:
        super(LeNet, self).__init__()

        self.in_channels = in_channels

        layers = [
            nn.Conv2d(in_channels, 6, (5,5), padding=2, bias=False),
            nn.ReLU(),
            nn.AvgPool2d((2,2), stride=2),
            nn.Conv2d(6, 16, (5,5), bias=False),
            nn.ReLU(),
            nn.AvgPool2d((2,2), stride=2),
            nn.Flatten(),
            nn.Linear(16*5*5, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, 32, bias=False),
            nn.ReLU(),
            nn.Linear(32, 10, bias=False)
        ]
        self.layers = nn.Sequential(*layers)
        self.n_layers = 5 # 2 conv2d + 3 linear
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x
    

class LeNetBN(MiCoModel):
    def __init__(self, in_channels: int) -> None:
        super(LeNetBN, self).__init__()

        self.in_channels = in_channels

        layers = [
            nn.Conv2d(in_channels, 6, (5,5), padding=2, bias=False),
            nn.ReLU(),
            nn.AvgPool2d((2,2), stride=2),
            nn.Conv2d(6, 16, (5,5)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d((2,2), stride=2),
            nn.Flatten(),
            nn.Linear(16*5*5, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        ]
        self.layers = nn.Sequential(*layers)
        self.n_layers = 5 # 2 conv2d + 3 linear
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x