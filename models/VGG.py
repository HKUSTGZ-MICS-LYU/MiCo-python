import torch
import json
import numpy as np
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from MiCoModel import MiCoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VGG(MiCoModel):
    def __init__(self, in_channels: int, num_class = 100) -> None:
        super(VGG, self).__init__()

        self.in_channels = in_channels
        self.num_block = 3
        features = [
            # Block 1
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Block 4
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        classifier = [
            # Classifier
            nn.Flatten(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        ]
        layers = features + classifier
        self.layers = nn.Sequential(*layers)
        self.n_layers = 7 # 4 Conv + 3 FC
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x