import unittest

import torch

from MiCoQLayers import BitLinear


class GroupwiseBitLinearExportTest(unittest.TestCase):
    def test_groupwise_export_returns_scale_tensor_data(self):
        layer = BitLinear(8, 4, group_size=4, qtype=8, act_q=8, bias=False)
        layer.save_qweight()

        exported = layer.export_qweight()

        self.assertEqual(exported["LayerType"], "Linear")
        self.assertEqual(exported["GroupSize"], 4)
        scale = torch.tensor(exported["Scale"])
        self.assertEqual(scale.shape, layer.qw_scale.shape)
        self.assertEqual(exported["Weight"], layer.qw.cpu().tolist())

    def test_non_groupwise_export_keeps_scalar_scale(self):
        layer = BitLinear(8, 4, group_size=1, qtype=8, act_q=8, bias=False)
        layer.save_qweight()

        exported = layer.export_qweight()

        self.assertIsInstance(exported["Scale"], (float, int))
        self.assertEqual(exported["GroupSize"], 1)


if __name__ == "__main__":
    unittest.main()
