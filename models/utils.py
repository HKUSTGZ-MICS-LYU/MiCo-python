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
    


class LinearAttention(nn.Module):

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.MiCo_func = MiCoFunc(
            "MiCo_linear_attention_{dtype}",
        )
        
    def forward(self, q, k, v):
        # Apply a non-negative feature map (e.g., ELU + 1)
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        # Compute the "context" matrix (K^T * V)
        # k: [B, H, N, D_k], v: [B, H, N, D_v]
        # context: [B, H, D_k, D_v]
        context = torch.einsum('bhnd,bhnm->bhdm', k, v)
        
        # Compute the denominator (normalizer)
        k_sum = k.sum(dim=2) # [B, H, D_k]
        
        # Compute the final attention output
        # q: [B, H, N, D_k]
        num = torch.einsum('bhnd,bhdm->bhnm', q, context)
        den = torch.einsum('bhnd,bhd->bhn', q, k_sum).unsqueeze(-1)
        
        return num / (den + self.eps)