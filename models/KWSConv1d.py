import torch
from torch import nn

from MiCoModel import MiCoModel
from MiCoQLayers import BitConv1d


class KWSConv1d(MiCoModel):
    def __init__(self, n_classes: int = 12, input_length: int = 16000):
        super().__init__()
        self.n_classes = n_classes
        self.input_length = input_length
        self.default_dataset = "SPEECHCOMMANDS"

        self.features = nn.Sequential(
            BitConv1d(1, 32, kernel_size=9, stride=2, padding=4, act_q=8, qtype=8),
            nn.ReLU(),
            nn.MaxPool1d(4),
            BitConv1d(32, 64, kernel_size=9, stride=2, padding=4, act_q=8, qtype=8),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )

        # Compute flattened dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_length)
            flat_dim = self.features(dummy).shape[-1] * 64

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

        self.n_layers = 4  # two conv + two linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
