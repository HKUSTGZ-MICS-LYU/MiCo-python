import torch
import torch.nn as nn

from MiCoModel import MiCoModel
from models.utils import LinearAttentionScore

"""
Work in Progress: Linear Transformers for TinyML,
M. Scherer, C. Cioflan, M. Magno, and L. Benini, 
doi: 10.23919/DATE58400.2024.10546828.
"""

# ================== Convolutional Linear Cross-Attention (CLCA) ==================
class CLCA(nn.Module):
    """
    Convolutional Linear Cross-Attention.
    Uses depthwise + pointwise convolutions to produce Q and a shared branch for K and V.
    """
    def __init__(self, in_dim, out_dim, head_dim, kernel_size=5, stride=1, eps=1e-6):
        super().__init__()
        self.out_dim = out_dim
        self.head_dim = head_dim
        assert out_dim % head_dim == 0, "out_dim must be divisible by head_dim"
        self.num_heads = out_dim // head_dim

        # Q branch
        self.dw_conv_q = nn.Conv1d(
            in_dim, in_dim, kernel_size, stride=stride,
            padding=kernel_size//2, groups=in_dim, bias=True
        )
        self.pw_conv_q = nn.Conv1d(in_dim, out_dim, 1, bias=True)

        # Shared K, V depthwise branch with separate pointwise projections.
        # This keeps the model FX/codegen friendly by avoiding Tensor.chunk().
        self.dw_conv_kv = nn.Conv1d(
            in_dim, in_dim, kernel_size, stride=stride,
            padding=kernel_size//2, groups=in_dim, bias=True
        )
        self.pw_conv_k = nn.Conv1d(in_dim, out_dim, 1, bias=True)
        self.pw_conv_v = nn.Conv1d(in_dim, out_dim, 1, bias=True)
        self.attn_score = LinearAttentionScore(eps=eps)

    def forward(self, x):
        B, C, S = x.shape

        # Q
        q = self.dw_conv_q(x)
        q = self.pw_conv_q(q)                              # (B, out_dim, S_out)
        q = q.view(B, self.num_heads, self.head_dim, -1).transpose(2, 3)  # (B, H, S_out, D)

        # K, V
        kv = self.dw_conv_kv(x)
        k = self.pw_conv_k(kv)                              # (B, out_dim, S_out)
        v = self.pw_conv_v(kv)                              # (B, out_dim, S_out)
        k = k.view(B, self.num_heads, self.head_dim, -1).transpose(2, 3)
        v = v.view(B, self.num_heads, self.head_dim, -1).transpose(2, 3)

        attn_out = self.attn_score(q, k, v)                 # (B, S_out, H, D)
        attn_out = attn_out.flatten(2).transpose(1, 2)      # (B, out_dim, S_out)

        return attn_out


# ================== Encoder block (Pre-Norm + residual) ==================
class EncoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, head_dim, mlp_dim, stride=1, kernel_size=5):
        super().__init__()
        self.clca = CLCA(in_dim, out_dim, head_dim, kernel_size, stride)
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(out_dim)

        self.ffn = nn.Sequential(
            nn.Conv1d(out_dim, mlp_dim, 1),
            nn.GELU(),
            nn.Conv1d(mlp_dim, out_dim, 1),
        )

        self.use_projection = (in_dim != out_dim) or (stride > 1)
        if self.use_projection:
            self.proj = nn.Conv1d(in_dim, out_dim, 1, bias=True)
            self.pool = nn.AvgPool1d(
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            )
        else:
            self.proj = nn.Identity()
            self.pool = nn.Identity()

    def forward(self, x):
        # CLCA path
        normed = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        attn_out = self.clca(normed)

        # Residual connection (project + pool)
        shortcut = self.pool(self.proj(x))      # now matches attn_out's length
        attn_out = attn_out + shortcut

        # FFN path
        normed = self.norm2(attn_out.transpose(1, 2)).transpose(1, 2)
        ffn_out = self.ffn(normed)
        out = attn_out + ffn_out

        return out


# ================== Full WaveFormer model ==================
class WaveFormer(MiCoModel):
    """
    WaveFormer: Linear Transformer for TinyML keyword spotting.
    """
    def __init__(self, encoders_config, num_classes, in_channels=1):
        super().__init__()
        self.default_dataset = "SPEECHCOMMANDS"
        self.num_classes = num_classes
        self.encoders = nn.ModuleList()
        prev_dim = in_channels
        for cfg in encoders_config:
            self.encoders.append(
                EncoderBlock(
                    in_dim=prev_dim,
                    out_dim=cfg['dim'],
                    head_dim=cfg['head_dim'],
                    mlp_dim=cfg['mlp_dim'],
                    stride=cfg.get('stride', 1),
                    kernel_size=cfg.get('kernel_size', 5),
                )
            )
            prev_dim = cfg['dim']

        self.final_dim = prev_dim
        self.classifier = nn.Linear(self.final_dim, num_classes)
        self.n_layers = len(self.get_qlayers())

    def forward(self, x):
        # x: (B, C, L) raw waveform
        for enc in self.encoders:
            x = enc(x)                     # (B, dim, seq_len)

        # Global average pooling over the sequence dimension
        x = x.mean(dim=-1)                 # (B, dim)
        x = self.classifier(x)
        return x

def tiny_waveformer(n_classes: int = 35):
    return WaveFormer(_tiny_waveformer_config(), num_classes=n_classes)


def _tiny_waveformer_config():
    return [
        {"dim": 16,  "head_dim": 4, "mlp_dim": 32,  "stride": 4},
        {"dim": 32,  "head_dim": 4, "mlp_dim": 64,  "stride": 4},
        {"dim": 64,  "head_dim": 8, "mlp_dim": 128, "stride": 4},
        {"dim": 128, "head_dim": 8, "mlp_dim": 160, "stride": 2},
    ]

# Example usage
if __name__ == '__main__':
    # Assume 1 second of 16 kHz audio
    dummy = torch.randn(2, 1, 16000)
    model = tiny_waveformer(12)
    out = model(dummy)
    print(out.shape)  # expected: (2, 12)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
