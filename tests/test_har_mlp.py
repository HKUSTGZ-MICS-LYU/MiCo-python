import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import HARMLP

BATCH_SIZE = 4
INPUT_DIM = 561
NUM_CLASSES = 6


def test_har_mlp_forward_shape():
    model = HARMLP()
    dummy_input = torch.randn(BATCH_SIZE, INPUT_DIM)
    output = model(dummy_input)
    assert output.shape == (BATCH_SIZE, NUM_CLASSES)
