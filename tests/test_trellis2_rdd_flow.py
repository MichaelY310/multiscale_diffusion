import os
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
TRELLIS = Path(os.environ.get("TRELLIS2_ROOT", ROOT / "third_party" / "TRELLIS.2"))
if TRELLIS.is_dir():
    sys.path.insert(0, str(TRELLIS))
sys.path.insert(0, str(ROOT / "RDD_DiT_3D"))

try:
    from trellis2.modules import sparse as sp
    from trellis2_rdd.hierarchy import project_detail
    from trellis2_rdd.rdd_sampler import SparseRDDDetailEulerCfgSampler
    from trellis2_rdd.rdd_sparse_flow_matching import SparseRDDDetailFlowMatchingTrainer

    IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - local Windows env lacks TRELLIS/PyTorch
    IMPORT_ERROR = exc


@unittest.skipIf(IMPORT_ERROR is not None, f"TRELLIS.2 unavailable: {IMPORT_ERROR}")
class Trellis2RDDFlowTest(unittest.TestCase):
    @staticmethod
    def sparse_batch():
        coords0 = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [16, 16, 16], [17, 16, 16]],
            dtype=torch.int32,
        )
        coords1 = torch.tensor(
            [[4, 4, 4], [5, 4, 4], [6, 4, 4], [24, 24, 24]], dtype=torch.int32
        )
        feats0 = torch.arange(len(coords0) * 3, dtype=torch.float32).reshape(-1, 3) / 10
        feats1 = torch.arange(len(coords1) * 3, dtype=torch.float32).reshape(-1, 3) / 7
        coords0 = torch.cat((torch.zeros((len(coords0), 1), dtype=torch.int32), coords0), 1)
        coords1 = torch.cat((torch.ones((len(coords1), 1), dtype=torch.int32), coords1), 1)
        return sp.SparseTensor.from_tensor_list([feats0, feats1], [coords0, coords1])

    @staticmethod
    def trainer_for_stage(stage: int):
        trainer = object.__new__(SparseRDDDetailFlowMatchingTrainer)
        trainer.rdd_resolutions = (8, 16, 32)
        trainer.rdd_fine_resolution = 32
        trainer.rdd_time_boundaries = torch.tensor([0.0, 0.5, 0.8, 1.0], dtype=torch.float64)
        trainer.rdd_stage_probabilities = torch.nn.functional.one_hot(
            torch.tensor(stage), 3
        ).to(torch.float64)
        trainer.sigma_min = 1e-5
        trainer.sample_t = lambda batch: torch.full((batch,), 0.5)
        return trainer

    def test_every_stage_builds_a_valid_batched_sparse_state(self):
        x0 = self.sparse_batch()
        for stage in range(3):
            trainer = self.trainer_for_stage(stage)
            xt, target, global_t, stage_ids, logical = trainer._make_training_state(x0)
            self.assertEqual(xt.shape[0], 2)
            self.assertEqual(xt.shape, target.shape)
            self.assertTrue(torch.isfinite(xt.feats).all())
            self.assertTrue(torch.isfinite(target.feats).all())
            self.assertTrue(torch.all(stage_ids == stage))
            if stage > 0:
                for values, coords in zip(target.to_tensor_list()[0], logical):
                    torch.testing.assert_close(
                        project_detail(coords, values), values, atol=2e-5, rtol=0
                    )
            expected = (0.9, 0.65, 0.25)[stage]
            torch.testing.assert_close(global_t, torch.full((2,), expected), atol=1e-6, rtol=0)

    def test_sampler_runs_all_stages_with_official_sparse_tensor_api(self):
        class ZeroModel:
            def __call__(self, x, t, cond, **kwargs):
                return x.replace(torch.zeros_like(x.feats))

        fine_coords = self.sparse_batch().to_tensor_list()[1][0][:, 1:]
        sampler = SparseRDDDetailEulerCfgSampler()
        result = sampler.sample(
            ZeroModel(),
            fine_coords,
            cond=None,
            neg_cond=None,
            channels=3,
            steps=(1, 1, 1),
            guidance_strength=1.0,
            verbose=False,
        )
        self.assertEqual(result.samples.feats.shape, (len(fine_coords), 3))
        self.assertTrue(torch.isfinite(result.samples.feats).all())


if __name__ == "__main__":
    unittest.main()
