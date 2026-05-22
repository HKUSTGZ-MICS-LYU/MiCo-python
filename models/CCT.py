import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear, Dropout, LayerNorm, Identity, Parameter, init

from MiCoModel import MiCoModel, MiCoFunc
from MiCoMisc import AttentionScore, LinearAttention

"""
https://github.com/SHI-Labs/Compact-Transformers
"""

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Attention(Module):
    """
    Obtained from timm: github.com:rwightman/pytorch-image-models
    """

    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** 0.5

        self.q = Linear(dim, dim, bias=False)
        self.k = Linear(dim, dim, bias=False)
        self.v = Linear(dim, dim, bias=False)
        self.attn_score = AttentionScore(self.scale)
        self.attn_drop = Dropout(attention_dropout)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(projection_dropout)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        x = self.attn_score(q, k, v).flatten(2)
        x = self.attn_drop(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerEncoderLayer(Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and timm.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.pre_norm = LayerNorm(d_model)
        self.self_attn = Attention(dim=d_model, num_heads=nhead,
                                   attention_dropout=attention_dropout, projection_dropout=dropout)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout1 = Dropout(dropout)
        self.norm1 = LayerNorm(d_model)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.dropout2 = Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else Identity()

        self.activation = nn.GELU()

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src

class TransformerClassifier(Module):
    def __init__(self,
                 seq_pool=True,
                 embedding_dim=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 dropout=0.1,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 positional_embedding='learnable',
                 sequence_length=None):
        super().__init__()
        positional_embedding = positional_embedding if \
            positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.seq_pool = seq_pool
        self.num_tokens = 0

        assert sequence_length is not None or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        if not seq_pool:
            sequence_length += 1
            self.class_emb = Parameter(torch.zeros(1, 1, self.embedding_dim),
                                       requires_grad=True)
            self.num_tokens = 1
        else:
            self.attention_pool = Linear(self.embedding_dim, 1)

        if positional_embedding != 'none':
            if positional_embedding == 'learnable':
                self.positional_emb = Parameter(torch.zeros(1, sequence_length, embedding_dim),
                                                requires_grad=True)
                init.trunc_normal_(self.positional_emb, std=0.2)
            else:
                self.positional_emb = Parameter(self.sinusoidal_embedding(sequence_length, embedding_dim),
                                                requires_grad=False)
        else:
            self.positional_emb = None

        self.dropout = Dropout(p=dropout)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, num_layers)]
        self.blocks = ModuleList([
            TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                    dim_feedforward=dim_feedforward, dropout=dropout,
                                    attention_dropout=attention_dropout, drop_path_rate=dpr[i])
            for i in range(num_layers)])
        self.norm = LayerNorm(embedding_dim)

        self.fc = Linear(embedding_dim, num_classes)
        self.apply(self.init_weight)

    def forward(self, x):
        if self.positional_emb is None and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)

        if not self.seq_pool:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        if self.positional_emb is not None:
            x += self.positional_emb

        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.seq_pool:
            attn = F.softmax(self.attention_pool(x).transpose(-1, -2), dim=-1)
            x = torch.einsum("bkn,bnd->bd", attn, x)
        else:
            x = x[:, 0]

        x = self.fc(x)
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, Linear):
            init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, Linear) and m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)

class Tokenizer(nn.Module):
    def __init__(self,
                 kernel_size, stride, padding,
                 pooling_kernel_size=3, pooling_stride=2, pooling_padding=1,
                 n_conv_layers=1,
                 n_input_channels=3,
                 n_output_channels=64,
                 in_planes=64,
                 activation=None,
                 max_pool=True,
                 conv_bias=False):
        super(Tokenizer, self).__init__()

        n_filter_list = [n_input_channels] + \
                        [in_planes for _ in range(n_conv_layers - 1)] + \
                        [n_output_channels]

        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(n_filter_list[i], n_filter_list[i + 1],
                          kernel_size=(kernel_size, kernel_size),
                          stride=(stride, stride),
                          padding=(padding, padding), bias=conv_bias),
                nn.Identity() if activation is None else activation(),
                nn.MaxPool2d(kernel_size=pooling_kernel_size,
                             stride=pooling_stride,
                             padding=pooling_padding) if max_pool else nn.Identity()
            )
                for i in range(n_conv_layers)
            ])

        self.flattener = nn.Flatten(2, 3)
        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, height=224, width=224):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def forward(self, x):
        return self.flattener(self.conv_layers(x)).transpose(-2, -1)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


class TextTokenizer(nn.Module):
    def __init__(self,
                 kernel_size, stride, padding,
                 pooling_kernel_size=3, pooling_stride=2, pooling_padding=1,
                 embedding_dim=300,
                 n_output_channels=128,
                 activation=None,
                 max_pool=True,
                 *args, **kwargs):
        super(TextTokenizer, self).__init__()

        self.max_pool = max_pool
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, n_output_channels,
                      kernel_size=(kernel_size, embedding_dim),
                      stride=(stride, 1),
                      padding=(padding, 0), bias=False),
            nn.Identity() if activation is None else activation(),
            nn.MaxPool2d(
                kernel_size=(pooling_kernel_size, 1),
                stride=(pooling_stride, 1),
                padding=(pooling_padding, 0)
            ) if max_pool else nn.Identity()
        )

        self.apply(self.init_weight)

    def seq_len(self, seq_len=32, embed_dim=300):
        return self.forward(torch.zeros((1, seq_len, embed_dim)))[0].shape[1]

    def forward_mask(self, mask):
        new_mask = mask.unsqueeze(1).float()
        cnn_weight = torch.ones(
            (1, 1, self.conv_layers[0].kernel_size[0]),
            device=mask.device,
            dtype=torch.float)
        new_mask = F.conv1d(
            new_mask, cnn_weight, None,
            self.conv_layers[0].stride[0], self.conv_layers[0].padding[0], 1, 1)
        if self.max_pool:
            new_mask = F.max_pool1d(
                new_mask, self.conv_layers[2].kernel_size[0],
                self.conv_layers[2].stride[0], self.conv_layers[2].padding[0], 1, False, False)
        new_mask = new_mask.squeeze(1)
        new_mask = (new_mask > 0)
        return new_mask

    def forward(self, x, mask=None):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.transpose(1, 3).squeeze(1)
        if mask is not None:
            mask = self.forward_mask(mask).unsqueeze(-1).float()
            x = x * mask
        return x, mask

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)

class CCT(MiCoModel):
    def __init__(self,
                 img_size=224,
                 embedding_dim=768,
                 n_input_channels=3,
                 n_conv_layers=1,
                 kernel_size=7,
                 stride=2,
                 padding=3,
                 pooling_kernel_size=3,
                 pooling_stride=2,
                 pooling_padding=1,
                 dropout=0.,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 num_layers=14,
                 num_heads=6,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 positional_embedding='learnable',
                 *args, **kwargs):
        super(CCT, self).__init__()

        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   pooling_kernel_size=pooling_kernel_size,
                                   pooling_stride=pooling_stride,
                                   pooling_padding=pooling_padding,
                                   max_pool=True,
                                   activation=nn.ReLU,
                                   n_conv_layers=n_conv_layers,
                                   conv_bias=False)

        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_size,
                                                           width=img_size),
            embedding_dim=embedding_dim,
            seq_pool=True,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth=stochastic_depth,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            positional_embedding=positional_embedding
        )
        self.n_layers = len(self.get_qlayers())

    def forward(self, x):
        x = self.tokenizer(x)
        return self.classifier(x)


def cct_2(n_classes=10):
    model = CCT(
        img_size=32, embedding_dim=128, kernel_size=3, n_conv_layers=2,
        num_layers=2, num_heads=2, mlp_ratio=1, num_classes=n_classes
    )
    model.default_dataset = "CIFAR10"
    return model

def cct_7(n_classes=100):
    model = CCT(
        img_size=32, embedding_dim=256, kernel_size=3, stride=1, padding=1,
        n_conv_layers=1, num_layers=7, num_heads=4, mlp_ratio=2, num_classes=n_classes
    )
    model.default_dataset = "CIFAR100"
    return model

# if __name__ == "__main__":
#     model = TinyCCT()
#     print(sum(p.numel() for p in model.parameters()))
