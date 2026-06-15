import unittest

import torch

from MiCoQLayers import activation_nquant, activation_nquant_2d, activation_nquant_group


class TestActivationQuantization(unittest.TestCase):
    def test_one_bit_zero_handling_is_consistent(self):
        x = torch.tensor([[-1.0, 0.0, 1.0]])

        y_linear = activation_nquant(x, 1)
        y_conv2d = activation_nquant_2d(x.reshape(1, 1, 1, 3), 1).reshape_as(x)
        y_group = activation_nquant_group(
            torch.tensor([[-1.0, 0.0, 1.0, 0.0]]),
            1,
            group_size=4,
        )

        self.assertFalse(torch.any(y_linear == 0.0))
        self.assertFalse(torch.any(y_conv2d == 0.0))
        self.assertFalse(torch.any(y_group == 0.0))
        torch.testing.assert_close(y_conv2d, y_linear)

    def test_one_point_five_bit_keeps_ternary_zero_level(self):
        x = torch.tensor([[-1.0, 0.0, 1.0]])

        y_linear = activation_nquant(x, 1.5)
        y_conv2d = activation_nquant_2d(x.reshape(1, 1, 1, 3), 1.5).reshape_as(x)

        self.assertTrue(torch.any(y_linear == 0.0))
        torch.testing.assert_close(y_conv2d, y_linear)


if __name__ == "__main__":
    unittest.main()
