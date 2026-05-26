import torch
from torch import nn

from MiCoModel import MiCoModel
from models.ViT import TransformerEncoder


class KWTPatchEmbedding(nn.Module):
    def __init__(
        self,
        input_res=(64, 81),
        patch_res=(64, 1),
        channels: int = 1,
        dim: int = 64,
    ):
        super().__init__()
        self.input_res = tuple(input_res)
        self.patch_res = tuple(patch_res)
        self.channels = channels

        if self.input_res[0] % self.patch_res[0] != 0:
            raise ValueError("input_res[0] must be divisible by patch_res[0]")
        if self.input_res[1] % self.patch_res[1] != 0:
            raise ValueError("input_res[1] must be divisible by patch_res[1]")

        self.grid_size = (
            self.input_res[0] // self.patch_res[0],
            self.input_res[1] // self.patch_res[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        patch_dim = channels * self.patch_res[0] * self.patch_res[1]
        self.proj = nn.Linear(patch_dim, dim)

    def forward(self, x):
        p_h, p_w = self.patch_res
        x = x.unfold(2, p_h, p_h).unfold(3, p_w, p_w)
        x = x.permute(0, 2, 3, 4, 5, 1)
        x = x.reshape(x.size(0), self.num_patches, -1)
        return self.proj(x)


class KWT(MiCoModel):
    """
    Keyword Transformer adapted to MiCo.

    This implementation keeps the KWT patch-token layout for MFCC inputs and
    reuses the MiCo ViT TransformerEncoder, so attention goes through
    MiCoMisc.AttentionScore and can be found by MiCo attention quantization.
    """
    def __init__(
        self,
        input_res=(64, 81),
        patch_res=(64, 1),
        num_classes: int = 35,
        dim: int = 64,
        depth: int = 12,
        heads: int = 1,
        mlp_dim: int = 256,
        pool: str = "cls",
        channels: int = 1,
        dim_head=None,
        dropout: float = 0.0,
        emb_dropout: float = 0.1,
        pre_norm: bool = True,
        **kwargs,
    ):
        super().__init__()
        if pool not in {"cls", "mean"}:
            raise ValueError("pool must be either 'cls' or 'mean'")
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        if dim_head is not None and dim_head * heads != dim:
            raise ValueError("The MiCo ViT encoder requires dim_head * heads == dim")

        self.default_dataset = "SPEECHCOMMANDS_2D"
        self.input_res = tuple(input_res)
        self.patch_res = tuple(patch_res)
        self.num_classes = num_classes
        self.pool = pool
        self.pre_norm = pre_norm

        self.to_patch_embedding = KWTPatchEmbedding(
            input_res=input_res,
            patch_res=patch_res,
            channels=channels,
            dim=dim,
        )
        num_patches = self.to_patch_embedding.num_patches
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = nn.Sequential(
            *[
                TransformerEncoder(
                    feats=dim,
                    mlp_hidden=mlp_dim,
                    head=heads,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
        )
        self.n_layers = len(self.get_qlayers())

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = self.cls_token.repeat(b, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, : n + 1]
        x = self.dropout(x)

        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        return self.mlp_head(x)


def KWT1(n_classes: int = 35):
    return KWT(
        input_res=(64, 81),
        patch_res=(64, 1),
        num_classes=n_classes,
        mlp_dim=256,
        dim=64,
        heads=1,
        depth=12,
        dropout=0.0,
        emb_dropout=0.1,
        pre_norm=True,
    )


def KWT2(n_classes: int = 35):
    return KWT(
        input_res=(64, 81),
        patch_res=(64, 1),
        num_classes=n_classes,
        mlp_dim=512,
        dim=128,
        heads=2,
        depth=12,
        dropout=0.0,
        emb_dropout=0.1,
        pre_norm=True,
    )


def KWT3(n_classes: int = 35):
    return KWT(
        input_res=(64, 81),
        patch_res=(64, 1),
        num_classes=n_classes,
        mlp_dim=768,
        dim=192,
        heads=3,
        depth=12,
        dropout=0.0,
        emb_dropout=0.1,
        pre_norm=True,
    )


def tiny_kwt(n_classes: int = 35):
    return KWT(
        input_res=(64, 81),
        patch_res=(64, 1),
        num_classes=n_classes,
        mlp_dim=128,
        dim=64,
        heads=4,
        depth=2,
        dropout=0.1,
        emb_dropout=0.1,
        pre_norm=True,
    )


def kwt_from_name(model_name: str, n_classes: int = 35):
    models = {
        "kwt": tiny_kwt,
        "tiny_kwt": tiny_kwt,
        "kwt-1": KWT1,
        "kwt-2": KWT2,
        "kwt-3": KWT3,
    }
    if model_name not in models:
        raise ValueError(
            f"Unsupported model_name {model_name}; must be one of {list(models.keys())}"
        )
    return models[model_name](n_classes=n_classes)
