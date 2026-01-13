import torch
from torch import nn

from MiCoModel import MiCoModel

# https://arxiv.org/pdf/1610.00087.pdf

class M5(MiCoModel):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.layers = [
            nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride),
            nn.BatchNorm1d(n_channel),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(n_channel, n_channel, kernel_size=3),
            nn.BatchNorm1d(n_channel),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3),
            nn.BatchNorm1d(2 * n_channel),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3),
            nn.BatchNorm1d(2 * n_channel),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(2 * n_channel, n_output)
        ]
        self.layers = nn.Sequential(*self.layers)
        self.n_layers = len(self.get_qlayers())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x