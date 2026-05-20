import os
import sys
import tempfile

import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from MiCoCodeGen import MiCoCodeGen
from models import tiny_waveformer

BATCH_SIZE = 2
INPUT_LEN = 512
NUM_CLASSES = 35


def _model():
    model = tiny_waveformer(n_classes=NUM_CLASSES)
    model.eval()
    return model


def test_waveformer_forward_shape_35class():
    model = _model()
    x = torch.randn(BATCH_SIZE, 1, INPUT_LEN)
    y = model(x)
    assert y.shape == (BATCH_SIZE, NUM_CLASSES)


def test_waveformer_speechcommands_shaped_training_step():
    torch.manual_seed(42)
    model = tiny_waveformer(n_classes=NUM_CLASSES)
    model.train()

    x = torch.randn(4, 1, INPUT_LEN)
    labels = torch.randint(0, NUM_CLASSES, (4,))
    loader = DataLoader(TensorDataset(x, labels), batch_size=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    batch_x, batch_y = next(iter(loader))

    optimizer.zero_grad()
    logits = model(batch_x)
    loss = criterion(logits, batch_y)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert logits.shape == (2, NUM_CLASSES)


def test_waveformer_codegen_emits_linear_attention():
    model = _model()
    codegen = MiCoCodeGen(model, align_to=32)
    codegen.forward(torch.randn(1, 1, INPUT_LEN))

    with tempfile.TemporaryDirectory() as tmpdir:
        codegen.convert(tmpdir, "waveformer_test", verbose=False)
        header = os.path.join(tmpdir, "waveformer_test.h")
        with open(header, "r", encoding="utf-8") as f:
            generated = f.read()

    assert "MiCo_linear_attention_f32" in generated
    assert "MiCo_mean2d_f32" in generated
