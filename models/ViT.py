import torch
import json
import numpy as np
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from MiCoModel import MiCoModel, MiCoFunc


class Im2Word(nn.Module):
    def __init__(self, img_size:int=32, patch:int=8):
        super(Im2Word, self).__init__()
        self.patch = patch
        self.patch_size = img_size//self.patch
        self.MiCo_func = MiCoFunc(
            "MiCo_im2word",
            params=[patch]
        )

    def forward(self, x):
        """
        (b, c, h, w) -> (b, n, f)
        """
        out = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).permute(0,2,3,4,5,1)
        out = out.reshape(x.size(0), self.patch**2 ,-1)
        return out


class AttentionScore(nn.Module):
    def __init__(self, scale: float):
        super(AttentionScore, self).__init__()
        self.scale = float(scale)
        self.MiCo_func = MiCoFunc(
            "MiCo_ViT_attention_{dtype}",
            params=[self.scale]
        )

    def forward(self, q, k, v):
        score = F.softmax(torch.einsum("bhif, bhjf->bhij", q, k) / self.scale, dim=-1)
        out = torch.einsum("bhij, bhjf->bihf", score, v)
        return out

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

'''
Modified from https://github.com/omihub777/ViT-CIFAR
'''
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feats:int, head:int=8, dropout:float=0.):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = self.feats**0.5

        self.q = nn.Linear(feats, feats)
        self.k = nn.Linear(feats, feats)
        self.v = nn.Linear(feats, feats)

        self.o = nn.Linear(feats, feats)
        self.dropout = nn.Dropout(dropout)
        self.attn_score = AttentionScore(self.sqrt_d)
        # self.attn_score = LinearAttention()

    def forward(self, x):
        b, n, f = x.size()
        q = self.q(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        k = self.k(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        v = self.v(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)

        attn = self.attn_score(q, k, v) #(b,n,h,f//h)
        o = self.dropout(self.o(attn.flatten(2)))
        return o

class TransformerEncoder(nn.Module):
    def __init__(self, feats:int, mlp_hidden:int, head:int=8, dropout:float=0.):
        super(TransformerEncoder, self).__init__()
        self.la1 = nn.LayerNorm(feats)
        self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout)
        self.la2 = nn.LayerNorm(feats)
        self.mlp = nn.Sequential(
            nn.Linear(feats, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, feats),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.msa(self.la1(x)) + x
        out = self.mlp(self.la2(out)) + out
        return out

class ViT(MiCoModel):
    def __init__(self, 
                in_c:int=3, 
                num_classes:int=10, 
                img_size:int=32, 
                patch:int=8, 
                dropout:float=0., 
                num_layers:int=7, 
                hidden:int=384, 
                mlp_hidden:int=384*4, 
                head:int=8, 
                is_cls_token:bool=True):
        
        super(ViT, self).__init__()

        self.default_dataset = "cifar100"

        self.patch = patch # number of patches in one row(or col)
        self.is_cls_token = is_cls_token
        self.patch_size = img_size//self.patch

        f = (img_size//self.patch)**2*3 # 48 # patch vec length
        num_tokens = (self.patch**2)+1 if self.is_cls_token else (self.patch**2)

        self.im2word = Im2Word(img_size=img_size, patch=patch)

        self.emb = nn.Linear(f, hidden) # (b, n, f)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden)) if is_cls_token else None
        self.pos_emb = nn.Parameter(torch.randn(1,num_tokens, hidden))

        enc_list = [
            TransformerEncoder(hidden,
                               mlp_hidden=mlp_hidden, 
                               dropout=dropout, 
                               head=head)
            for _ in range(num_layers)]
        
        self.enc = nn.Sequential(*enc_list)
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_classes) # for cls_token
        )

        self.n_layers = len(self.get_qlayers())

    def forward(self, x):
        out = self.im2word(x)  # (b, n, f)
        out = self.emb(out)
        if self.is_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0),1,1), out],dim=1)
        out = out + self.pos_emb
        out = self.enc(out)
        if self.is_cls_token:
            out = out[:,0]
        else:
            out = out.mean(1)
        out = self.fc(out)
        return out
    
    
def TinyViT1M(n_classes=10):
    return ViT(in_c=3, 
               num_classes=n_classes, 
               img_size=32, patch=8, dropout=0.1, 
               num_layers=8, hidden=128, 
               mlp_hidden=256, head=4, 
               is_cls_token=False)


def ViTNAS(args: dict):
    '''
    args: {
        "num_layers": int,  # Number of transformer layers
        "hidden": int,      # Hidden dimension size
        "mlp_hidden": int,  # MLP hidden dimension size
        "head": int,        # Number of attention heads
        "dropout": float    # Dropout rate
    }
    '''
    return ViT(
        in_c=3, num_classes=100, img_size=32, patch=8,
        num_layers=args["num_layers"], 
        hidden=args["hidden"], 
        mlp_hidden=args["mlp_hidden"],
        head=args["head"],
        dropout=args["dropout"],
        is_cls_token=True
    )
