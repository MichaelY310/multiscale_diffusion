import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2d" / "attention_analysis"))
from probe_csra_joint_stage_isolation import StageSwap, TraceRollout
from sample_csra_joint_preview import JointRollout


class FakeStage:
    def __init__(self, value):
        self.value = value
        self.seen_cache = None

    def stage(self, x, t, y, cache=None, capture=False):
        self.seen_cache = cache
        result = torch.cat([x, x], dim=1) + self.value
        if cache is not None:
            result = result + cache[13]
        return result, {13: torch.tensor(float(self.value))} if capture else {}


class StageIsolationTests(unittest.TestCase):
    def test_same_arm_matches_existing_joint_rollout(self):
        a, b = StageSwap(FakeStage(1), FakeStage(1)), JointRollout(FakeStage(1))
        for size, time in [(8,999), (8,600), (16,599), (16,300), (32,299), (32,0)]:
            x, t, y = torch.zeros(2,4,size,size), torch.full((2,), time), torch.ones(2, dtype=torch.long)
            torch.testing.assert_close(a.forward(x,t,y), b.forward(x,t,y), atol=0, rtol=0)

    def test_swap_uses_selected_source_and_fine(self):
        coarse, fine = FakeStage(2), FakeStage(5)
        wrapper = StageSwap(coarse, fine)
        y = torch.zeros(2, dtype=torch.long)
        wrapper.forward(torch.zeros(2,4,16,16), torch.full((2,),300), y)
        output = wrapper.forward(torch.zeros(2,4,32,32), torch.full((2,),299), y)
        self.assertTrue(torch.all(output == 7))
        self.assertEqual(fine.seen_cache[13].item(), 2)

    def test_missing_cache_fails_but_dense_bypass_works(self):
        x, t, y = torch.zeros(2,4,32,32), torch.zeros(2,dtype=torch.long), torch.zeros(2,dtype=torch.long)
        with self.assertRaises(RuntimeError):
            StageSwap(FakeStage(2), FakeStage(5)).forward(x,t,y)
        self.assertTrue(torch.all(StageSwap(FakeStage(2), FakeStage(5), False).forward(x,t,y) == 5))

    def test_trace_checks_identical_state_and_does_not_change_output(self):
        raw = StageSwap(FakeStage(2), FakeStage(5), False)
        teacher = StageSwap(FakeStage(2), FakeStage(5), False)
        traced = TraceRollout(raw, teacher)
        x, t, y = torch.zeros(2,4,8,8), torch.full((2,),999), torch.zeros(2,dtype=torch.long)
        torch.testing.assert_close(traced.forward(x,t,y), raw.forward(x,t,y), atol=0, rtol=0)
        self.assertEqual(traced.teacher_checks[0]["main_mse"], 0)
        self.assertEqual(traced.counts[8], 1)


if __name__ == "__main__":
    unittest.main()
