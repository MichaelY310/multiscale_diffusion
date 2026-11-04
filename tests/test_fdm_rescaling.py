import unittest

import torch

from diffusion.fdm_rescaling import rescale_alpha_sigma, spatial_dimension_ratio


class FDMRescalingTest(unittest.TestCase):
    def test_spatial_dimension_ratio_is_cumulative(self):
        self.assertEqual(spatial_dimension_ratio(32, 32), 1.0)
        self.assertEqual(spatial_dimension_ratio(32, 16), 4.0)
        self.assertEqual(spatial_dimension_ratio(32, 8), 16.0)

    def test_sp_matches_fdm_formula_and_multiplies_snr(self):
        alpha = torch.tensor([0.6], dtype=torch.float64)
        sigma = torch.tensor([0.8], dtype=torch.float64)
        out_alpha, out_sigma = rescale_alpha_sigma(alpha, sigma, 4.0, "sp")
        self.assertTrue(torch.equal(out_alpha, alpha))
        self.assertTrue(torch.allclose(out_sigma, sigma / 2.0))
        old_snr = alpha.square() / sigma.square()
        new_snr = out_alpha.square() / out_sigma.square()
        self.assertTrue(torch.allclose(new_snr, 4.0 * old_snr))

    def test_vp_preserves_variance_and_multiplies_snr(self):
        alpha = torch.tensor([0.6], dtype=torch.float64)
        sigma = torch.tensor([0.8], dtype=torch.float64)
        out_alpha, out_sigma = rescale_alpha_sigma(alpha, sigma, 4.0, "vp")
        self.assertTrue(
            torch.allclose(
                out_alpha.square() + out_sigma.square(), torch.ones_like(alpha)
            )
        )
        old_snr = alpha.square() / sigma.square()
        new_snr = out_alpha.square() / out_sigma.square()
        self.assertTrue(torch.allclose(new_snr, 4.0 * old_snr))

    def test_dimension_ratio_one_is_identity(self):
        alpha = torch.tensor([0.2, 0.7])
        sigma = torch.sqrt(1.0 - alpha.square())
        for mode in ("none", "sp", "vp"):
            with self.subTest(mode=mode):
                out_alpha, out_sigma = rescale_alpha_sigma(alpha, sigma, 1.0, mode)
                self.assertTrue(torch.equal(out_alpha, alpha))
                self.assertTrue(torch.equal(out_sigma, sigma))

    def test_invalid_dimension_ratio_rejected(self):
        with self.assertRaises(ValueError):
            rescale_alpha_sigma(torch.ones(1), torch.ones(1), 0.5, "sp")


if __name__ == "__main__":
    unittest.main()
