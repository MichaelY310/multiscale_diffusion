import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'2d'/'attention_analysis'))
from diagnose_csra_trained_prefix import output_bands, scores


class TrainedPrefixDiagnosticTests(unittest.TestCase):
    def test_orthogonal_cell_energy(self):
        torch.manual_seed(43)
        x = torch.randn(3, 4, 8, 8)
        low, high = output_bands(x)
        torch.testing.assert_close(low+high, x)
        self.assertLess(float((low*high).mean().abs()), 1e-7)
        torch.testing.assert_close(x.square().mean(), low.square().mean()+high.square().mean())

    def test_constant_and_checkerboard(self):
        const = torch.ones(1, 4, 8, 8)
        checker = torch.tensor([[1., -1.], [-1., 1.]]).repeat(1, 4, 4, 4)
        self.assertEqual(float(output_bands(const)[1].abs().max()), 0.)
        self.assertEqual(float(output_bands(checker)[0].abs().max()), 0.)

    def test_zero_and_variance_separated(self):
        x = torch.randn(2, 8, 8, 8)
        self.assertEqual(float(scores(x, x)['main_mse'].sum()), 0.)
        y = x.clone(); y[:, 4:] += 1.
        s = scores(y, x)
        self.assertEqual(float(s['main_mse'].sum()), 0.)
        torch.testing.assert_close(s['variance_mse'], torch.ones(2))


if __name__ == '__main__': unittest.main()
