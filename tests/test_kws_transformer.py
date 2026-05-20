import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import KWSTransformer, tiny_kws_transformer

BATCH_SIZE = 2
NUM_CLASSES = 12
INPUT_SIZE = (64, 81)


def test_kws_transformer_forward_shape():
    model = KWSTransformer(n_classes=NUM_CLASSES, input_size=INPUT_SIZE)
    x = torch.randn(BATCH_SIZE, 1, *INPUT_SIZE)
    y = model(x)
    assert y.shape == (BATCH_SIZE, NUM_CLASSES)


def test_tiny_kws_transformer_int8_qscheme_forward_shape():
    model = tiny_kws_transformer(n_classes=NUM_CLASSES)
    model.set_qscheme([[8] * model.n_layers, [8] * model.n_layers])
    x = torch.randn(BATCH_SIZE, 1, *INPUT_SIZE)
    y = model(x)
    assert y.shape == (BATCH_SIZE, NUM_CLASSES)
