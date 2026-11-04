import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "RDD_DiT_3D"))

from slat_pyramid import (  # noqa: E402
    SparseGrid,
    decompose_level,
    decompose_masked_level,
    dense_from_sparse,
    expand_all_children,
    sparse_from_dense,
    masked_projector,
    zero_filled_downsample,
)


def _grid(coords, feats, resolution=4):
    coords = torch.tensor([[0, *coord] for coord in coords], dtype=torch.long)
    feats = torch.tensor(feats, dtype=torch.float32).reshape(len(coords), -1)
    return SparseGrid(coords, feats, resolution)


class SparsePyramidTest(unittest.TestCase):
    def test_missing_children_are_zeros_not_active_mean(self):
        fine = _grid([(0, 0, 0)], [[8.0]], resolution=2)
        coarse = zero_filled_downsample(fine)
        self.assertEqual(coarse.num_tokens, 1)
        torch.testing.assert_close(coarse.feats, torch.tensor([[1.0]]))

    def test_exact_candidate_reconstruction_and_inactive_cancellation(self):
        fine = _grid([(0, 0, 0), (1, 1, 1)], [[8.0], [4.0]], resolution=2)
        transition = decompose_level(fine)
        reconstructed = transition.candidate.feats + transition.residual.feats
        torch.testing.assert_close(reconstructed, transition.target.feats)
        self.assertEqual(transition.candidate.num_tokens, 8)
        self.assertEqual(int(transition.target_active.sum()), 2)
        self.assertTrue(bool(torch.all(transition.residual.feats[~transition.target_active] < 0)))

    def test_sibling_residuals_sum_to_zero(self):
        fine = _grid([(0, 0, 0), (0, 1, 0), (3, 3, 3)], [[1.0], [3.0], [7.0]], resolution=4)
        transition = decompose_level(fine)
        parent = transition.residual.coords.clone()
        parent[:, 1:] //= 2
        unique, inverse = torch.unique(parent, dim=0, return_inverse=True)
        sums = torch.zeros((unique.shape[0], fine.channels))
        sums.index_add_(0, inverse, transition.residual.feats)
        torch.testing.assert_close(sums, torch.zeros_like(sums), atol=1e-6, rtol=0)

    def test_sparse_pool_matches_dense_avg_pool3d(self):
        torch.manual_seed(0)
        dense = torch.randn(2, 4, 4, 4, 3)
        active = torch.rand(2, 4, 4, 4) > 0.65
        dense = dense * active[..., None]
        sparse = sparse_from_dense(dense)
        coarse = zero_filled_downsample(sparse)
        got = dense_from_sparse(coarse, batch_size=2)
        expected = torch.nn.functional.avg_pool3d(
            dense.permute(0, 4, 1, 2, 3), kernel_size=2, stride=2
        ).permute(0, 2, 3, 4, 1)
        torch.testing.assert_close(got, expected)

    def test_projector_is_idempotent_on_full_candidate_support(self):
        fine = _grid([(0, 0, 0), (0, 1, 1), (2, 2, 2)], [[2.0], [6.0], [4.0]], resolution=4)
        projected = expand_all_children(zero_filled_downsample(fine))
        projected_twice = expand_all_children(zero_filled_downsample(projected))
        torch.testing.assert_close(
            dense_from_sparse(projected), dense_from_sparse(projected_twice), atol=1e-6, rtol=0
        )

    def test_known_support_projector_is_idempotent(self):
        fine = _grid(
            [(0, 0, 0), (0, 1, 1), (2, 2, 2), (2, 3, 2)],
            [[2.0], [6.0], [4.0], [8.0]],
            resolution=4,
        )
        once = masked_projector(fine)
        twice = masked_projector(once)
        torch.testing.assert_close(once.feats, twice.feats, atol=1e-6, rtol=0)

    def test_known_support_detail_has_zero_active_sibling_mean(self):
        fine = _grid(
            [(0, 0, 0), (0, 1, 1), (2, 2, 2), (2, 3, 2)],
            [[2.0], [6.0], [4.0], [8.0]],
            resolution=4,
        )
        transition = decompose_masked_level(fine)
        parent = transition.residual.coords.clone()
        parent[:, 1:] //= 2
        unique, inverse = torch.unique(parent, dim=0, return_inverse=True)
        sums = torch.zeros((unique.shape[0], fine.channels))
        sums.index_add_(0, inverse, transition.residual.feats)
        torch.testing.assert_close(sums, torch.zeros_like(sums), atol=1e-6, rtol=0)


if __name__ == "__main__":
    unittest.main()
