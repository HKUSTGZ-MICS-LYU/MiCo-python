import unittest

import torch

from MiCoQLayers import weight_quant, weight_quantnb_group


class TestWeightQuantization(unittest.TestCase):
    def test_grouped_one_bit_matches_ungrouped_for_single_group(self):
        w = torch.tensor([[-2.0, -1.0, 0.0, 1.0]])

        u_ref, scale_ref = weight_quant(w, 1)
        u_group, scale_group = weight_quantnb_group(w, 1, dim=1, group_size=4)

        torch.testing.assert_close(u_group, u_ref)
        torch.testing.assert_close(scale_group, torch.full_like(w, scale_ref))
        torch.testing.assert_close(u_group * scale_group, u_ref * scale_ref)

    def test_grouped_one_point_five_bit_matches_ungrouped_for_single_group(self):
        w = torch.tensor([[-2.0, -1.0, 0.0, 1.0]])

        u_ref, scale_ref = weight_quant(w, 1.5)
        u_group, scale_group = weight_quantnb_group(w, 1.5, dim=1, group_size=4)

        torch.testing.assert_close(u_group, u_ref)
        torch.testing.assert_close(scale_group, torch.full_like(w, scale_ref))
        torch.testing.assert_close(u_group * scale_group, u_ref * scale_ref)

    def test_grouped_one_bit_supports_compact_scales(self):
        w = torch.tensor([[-2.0, -1.0, 0.0, 1.0]])

        u_group, scale_group = weight_quantnb_group(
            w,
            1,
            dim=1,
            group_size=2,
            return_expanded=False,
        )

        self.assertEqual(u_group.shape, w.shape)
        self.assertEqual(scale_group.shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
