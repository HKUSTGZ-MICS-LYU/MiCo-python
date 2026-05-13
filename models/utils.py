# Collection of some shared kernels/modules

import torch
import torch.nn as nn
import torch.nn.functional as F

from MiCoModel import MiCoModel, MiCoFunc

class AttentionScore(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = float(scale)
        self.MiCo_func = MiCoFunc(
            "MiCo_ViT_attention_{dtype}",
            params=[self.scale]
        )

    def forward(self, q, k, v):
        score = F.softmax(torch.einsum("bhif, bhjf->bhij", q, k) / self.scale, dim=-1)
        return torch.einsum("bhij, bhjf->bihf", score, v)
    

class LinearAttentionScore(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.MiCo_func = MiCoFunc(
            "MiCo_linear_attention_{dtype}",
            params=[self.eps]
        )

    def forward(self, q, k, v):
        q = F.elu(q) + 1
        k = F.elu(k) + 1

        context = torch.einsum('bhnd,bhnm->bhdm', k, v)
        k_sum = k.sum(dim=2)

        num = torch.einsum('bhnd,bhdm->bnhm', q, context)
        den = torch.einsum('bhnd,bhd->bnh', q, k_sum).unsqueeze(-1)

        return num / (den + self.eps)


class LinearAttention(nn.Module):
    """
    Drop-in linear-attention replacement for CCT.Attention.

    Input/output contract:
      x: [batch, tokens, dim] -> [batch, tokens, dim]
    """

    def __init__(self, dim, num_heads=8, attention_dropout=0.1,
                 projection_dropout=0.1, eps=1e-6):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads

        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.attn_score = LinearAttentionScore(eps=eps)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)

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
