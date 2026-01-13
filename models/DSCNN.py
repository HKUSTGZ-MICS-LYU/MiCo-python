import torch
import json
import numpy as np
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from MiCoModel import MiCoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DSCNN(MiCoModel):

    n_filters = 64
    n_blocks = 4
    default_dataset = "SPEECHCOMMANDS_2D"

    def __init__(self, in_channels: int = 1, n_classes = 35, input_size = [40, 81]) -> None:
        super(DSCNN, self).__init__()

        self.in_channels = in_channels
        self.input_size = input_size
        self.flat_size = (self.input_size[0] // 2, self.input_size[1] // 2)
        self.final_size = self.n_filters * self.flat_size[0] * self.flat_size[1]
        self.default_dataset = "SPEECHCOMMANDS_2D"
        input_conv = [
            # Input Conv
            nn.Conv2d(in_channels, self.n_filters, kernel_size=(10,4), 
                      stride=(2,2), padding=2),
            nn.BatchNorm2d(self.n_filters),
            nn.ReLU(),
            nn.Dropout(0.2)
        ]

        depthwise_blocks = []

        for _ in range(self.n_blocks):
            depthwise_blocks += [
                # Depthwise Separable Conv Block
                nn.Conv2d(self.n_filters, self.n_filters, kernel_size=(3,3),
                          groups=self.n_filters, padding=1),
                nn.BatchNorm2d(self.n_filters),
                nn.ReLU(),
                nn.Conv2d(self.n_filters, self.n_filters, kernel_size=(1,1)),
                nn.BatchNorm2d(self.n_filters),
                nn.ReLU(),
            ]

        output_conv = [
            nn.Dropout(0.4),
            nn.AdaptiveAvgPool2d((self.flat_size[0], self.flat_size[1])),
            nn.Flatten(),
            nn.Linear(self.final_size, n_classes),
        ]

        layers = input_conv + depthwise_blocks + output_conv

        self.layers = nn.Sequential(*layers)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x
    

# if __name__ == "__main__":

#     model = DSCNN(in_channels=1, n_classes=35)
#     x = torch.randn((1,1,40,81))
#     y = model(x)
#     print(y.shape)

