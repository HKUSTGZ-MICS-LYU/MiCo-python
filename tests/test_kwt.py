import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import KWT, tiny_kwt


BATCH_SIZE = 2
NUM_CLASSES = 12
INPUT_SIZE = (64, 81)


def test_kwt_forward_shape():
    model = KWT(
        input_res=INPUT_SIZE,
        patch_res=(64, 1),
        num_classes=NUM_CLASSES,
        dim=64,
        depth=2,
        heads=4,
        mlp_dim=128,
    )
    x = torch.randn(BATCH_SIZE, 1, *INPUT_SIZE)
    y = model(x)
    assert y.shape == (BATCH_SIZE, NUM_CLASSES)
    assert model.n_attn_layers == 2


def test_tiny_kwt_int8_qscheme_forward_shape():
    model = tiny_kwt(n_classes=NUM_CLASSES)
    model.set_qscheme([[8] * model.n_layers, [8] * model.n_layers])
    x = torch.randn(BATCH_SIZE, 1, *INPUT_SIZE)
    y = model(x)
    assert y.shape == (BATCH_SIZE, NUM_CLASSES)
