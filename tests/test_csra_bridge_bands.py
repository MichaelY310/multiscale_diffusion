import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'2d'/'attention_analysis'))
from csra_prefix_probe_math import split_error


class CellBandTests(unittest.TestCase):
    def test_child_cells(self):
        x = torch.tensor([1., 3., 7., 9.]).reshape(1, 4, 1)
        low, high = split_error(x)
        torch.testing.assert_close(low, torch.full_like(x, 5.))
        torch.testing.assert_close(high, x-5.)

    def test_projection_and_oracle(self):
        g = torch.Generator().manual_seed(401)
        hidden, true = [torch.randn(3, 256, 7, generator=g) for _ in range(2)]
        lo, hi = split_error(true-hidden)
        torch.testing.assert_close(hidden+lo+hi, true)
        torch.testing.assert_close(split_error(lo)[0], lo)
        torch.testing.assert_close(split_error(hi)[0], torch.zeros_like(hi), atol=3e-7, rtol=0)
        torch.testing.assert_close((lo*hi).sum(), torch.tensor(0.), atol=1e-4, rtol=0)
        torch.testing.assert_close((true-hidden).square().sum(), lo.square().sum()+hi.square().sum())

    def test_invalid_grid(self):
        with self.assertRaises(ValueError): split_error(torch.zeros(1, 9, 2))


if __name__ == '__main__': unittest.main()
