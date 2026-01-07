import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import DSCNNKWS

BATCH_SIZE = 2
INPUT_LEN = 16000
NUM_CLASSES = 12


def test_kws_ds_cnn_forward_shape():
    model = DSCNNKWS(n_classes=NUM_CLASSES, input_length=INPUT_LEN)
    x = torch.randn(BATCH_SIZE, 1, INPUT_LEN)
    y = model(x)
    assert y.shape == (BATCH_SIZE, NUM_CLASSES)
