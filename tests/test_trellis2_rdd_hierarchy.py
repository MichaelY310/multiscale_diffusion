import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "RDD_DiT_3D"))

from trellis2_rdd.hierarchy import build_masked_latent_pyramid, project_detail  # noqa: E402


class Trellis2RDDHierarchyTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        dense_coords = torch.cartesian_prod(torch.arange(8), torch.arange(8), torch.arange(8))
        keep = torch.rand(len(dense_coords)) < 0.18
        self.coords = dense_coords[keep] * 4 + torch.randint(0, 4, (int(keep.sum()), 3))
        self.coords = torch.unique(self.coords, dim=0)
        self.feats = torch.randn(len(self.coords), 5)

    def test_each_detail_has_zero_mean_per_active_sibling_group(self):
        pyramid = build_masked_latent_pyramid(self.coords, self.feats)
        for transition in pyramid.transitions:
            projected = project_detail(transition.child.logical_coords, transition.detail)
            torch.testing.assert_close(projected, transition.detail, atol=1e-6, rtol=0)

    def test_each_transition_reconstructs_child_exactly(self):
        pyramid = build_masked_latent_pyramid(self.coords, self.feats)
        for transition in pyramid.transitions:
            torch.testing.assert_close(
                transition.upsampled + transition.detail,
                transition.child.feats,
                atol=1e-6,
                rtol=0,
            )

    def test_detail_projection_is_idempotent(self):
        pyramid = build_masked_latent_pyramid(self.coords, self.feats)
        child = pyramid.levels[-1]
        once = project_detail(child.logical_coords, child.feats)
        twice = project_detail(child.logical_coords, once)
        torch.testing.assert_close(once, twice, atol=1e-6, rtol=0)

    def test_recursive_pooling_weights_active_children_not_fine_token_count(self):
        # At 16^3, the first active child contains two 32^3 tokens and the
        # second contains one.  The 8^3 parent must average the two child
        # latents (1 and 10) to 5.5, not all three fine tokens to 4.
        coords = torch.tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        feats = torch.tensor([[0.0], [2.0], [10.0]])
        pyramid = build_masked_latent_pyramid(coords, feats)
        self.assertEqual(pyramid.levels[1].feats[:, 0].tolist(), [1.0, 10.0])
        torch.testing.assert_close(pyramid.levels[0].feats, torch.tensor([[5.5]]))
        transition = pyramid.transitions[0]
        torch.testing.assert_close(
            project_detail(transition.child.logical_coords, transition.detail),
            transition.detail,
            atol=1e-6,
            rtol=0,
        )


if __name__ == "__main__":
    unittest.main()
