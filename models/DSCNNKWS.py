import torch
from torch import nn

from MiCoModel import MiCoModel
from MiCoQLayers import BitConv1d


class DSCNNKWS(MiCoModel):
    def __init__(self, n_classes: int = 12, input_length: int = 16000):
        super().__init__()
        self.n_classes = n_classes
        self.input_length = input_length
        self.default_dataset = "SPEECHCOMMANDS"

        def dw_pw(in_ch, out_ch, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                BitConv1d(
                    in_ch,
                    in_ch,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=in_ch,
                    qtype=8,
                    act_q=8,
                ),
                nn.ReLU(),
                BitConv1d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, qtype=8, act_q=8),
                nn.ReLU(),
            )

        self.features = nn.Sequential(
            BitConv1d(1, 64, kernel_size=10, stride=2, padding=4, qtype=8, act_q=8),
            nn.ReLU(),
            dw_pw(64, 64),
            dw_pw(64, 64),
            dw_pw(64, 64),
            nn.AvgPool1d(kernel_size=4, stride=4),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_length)
            feat = self.features(dummy)
            flat_dim = feat.shape[1] * feat.shape[2]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

        self.n_layers = 8  # conv + three dw/pw + two linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
