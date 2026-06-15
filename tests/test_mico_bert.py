#!/usr/bin/env python3
"""Tests for native MiCo BERT integration."""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MiCoDatasets import local_mlm_text
from models.BERT import MicroBERTLocal


class TestMiCoBERT(unittest.TestCase):
    def test_local_mlm_dataset_shapes(self):
        train_loader, _ = local_mlm_text(batch_size=2, max_seq_len=32, num_works=0)
        input_ids, labels = next(iter(train_loader))

        self.assertEqual(tuple(input_ids.shape), (2, 32))
        self.assertEqual(tuple(labels.shape), (2, 32))
        self.assertTrue(labels.ne(-100).any())

    def test_forward_and_loss(self):
        model = MicroBERTLocal()
        train_loader, _ = local_mlm_text(batch_size=2, max_seq_len=model.params.max_seq_len, num_works=0)
        input_ids, labels = next(iter(train_loader))

        logits = model(input_ids, labels)

        self.assertEqual(tuple(logits.shape), (2, model.params.max_seq_len, model.params.vocab_size))
        self.assertIsNotNone(model.last_loss)
        self.assertTrue(torch.isfinite(model.last_loss))

    def test_int8_qscheme_eval(self):
        model = MicroBERTLocal()
        _, test_loader = local_mlm_text(batch_size=2, max_seq_len=model.params.max_seq_len, num_works=0)
        model.set_qscheme([[8] * model.n_layers, [8] * model.n_layers])

        # Move model to the same device that model.test() will use
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        results = model.test(test_loader, n_eval_batches=1)

        self.assertIn("TestLoss", results)
        self.assertIn("TestAcc", results)
        self.assertTrue(torch.isfinite(torch.tensor(results["TestLoss"])))


if __name__ == "__main__":
    unittest.main()
