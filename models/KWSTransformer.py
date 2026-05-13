import torch
from torch import nn

from MiCoModel import MiCoModel
from models.CCT import TransformerClassifier


class MFCCTokenizer(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        embedding_dim: int = 96,
        n_conv_layers: int = 2,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        pooling_kernel_size=(2, 2),
        pooling_stride=(2, 2),
        pooling_padding=(0, 0),
    ):
        super().__init__()

        channels = [in_channels] + [embedding_dim] * n_conv_layers
        layers = []
        for i in range(n_conv_layers):
            layers.extend(
                [
                    nn.Conv2d(
                        channels[i],
                        channels[i + 1],
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=False,
                    ),
                    nn.BatchNorm2d(channels[i + 1]),
                    nn.ReLU(),
                    nn.MaxPool2d(
                        kernel_size=pooling_kernel_size,
                        stride=pooling_stride,
                        padding=pooling_padding,
                    ),
                ]
            )

        self.layers = nn.Sequential(*layers)
        self.flatten = nn.Flatten(2, 3)

    def sequence_length(self, input_size):
        with torch.no_grad():
            x = torch.zeros(1, 1, input_size[0], input_size[1])
            return self.forward(x).shape[1]

    def forward(self, x):
        x = self.layers(x)
        return self.flatten(x).transpose(1, 2)


class KWSTransformer(MiCoModel):
    def __init__(
        self,
        n_classes: int = 35,
        input_size=(64, 81),
        embedding_dim: int = 96,
        n_conv_layers: int = 2,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        stochastic_depth: float = 0.0,
        positional_embedding: str = "learnable",
    ):
        super().__init__()
        self.default_dataset = "SPEECHCOMMANDS_2D"
        self.input_size = tuple(input_size)

        self.tokenizer = MFCCTokenizer(
            in_channels=1,
            embedding_dim=embedding_dim,
            n_conv_layers=n_conv_layers,
        )
        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(self.input_size),
            embedding_dim=embedding_dim,
            seq_pool=True,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth=stochastic_depth,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=n_classes,
            positional_embedding=positional_embedding,
        )
        self.n_layers = len(self.get_qlayers())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tokenizer(x)
        return self.classifier(x)


def tiny_kws_transformer(n_classes: int = 35):
    return KWSTransformer(
        n_classes=n_classes,
        embedding_dim=64,
        n_conv_layers=2,
        num_layers=2,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.1,
        attention_dropout=0.1,
    )
