import copy
import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2d" / "attention_analysis"))
from test_csra_output_distillation import TinyTeacher
from train_rdd_csra_adapter import CSRAAdapterBank
from csra_joint_distillation import JointCSRAStudent, PreservedCoarseStudent


class JointTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(222)
        self.model = TinyTeacher()
        self.adapter = CSRAAdapterBank([1, 3], 16, 4, "fine_context_v2")
        # Real pilot loads TRAINED adapters; zero-init low_up would intentionally
        # block the source gradient at initialization, unrelated to graph wiring.
        for module in self.adapter.adapters.values():
            torch.nn.init.normal_(module.low_up.weight, std=.1)
        self.batch = {"x": torch.randn(2,4,4,4), "t": torch.tensor([10,20]),
                      "labels": torch.tensor([1,2]), "source16_x": torch.randn(2,4,2,2),
                      "source16_t": torch.tensor([100,100]), "source8_x": torch.randn(2,4,2,2),
                      "source8_t": torch.tensor([200,200])}

    def test_full_backbone_gradients_and_live_source(self):
        net = JointCSRAStudent(self.model, self.adapter)
        self.batch["source16_x"].requires_grad_(True)
        fine, middle, coarse = net(self.batch)
        # Fine loss alone must reach the source branch (not merely anchor loss).
        fine.square().mean().backward()
        self.assertGreater(float(self.batch["source16_x"].grad.abs().sum()), 0)
        for layer in [1,3]:
            self.assertGreater(float(net.backbone.blocks[layer].linear.weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is not None for p in net.parameters()))

    def test_frozen_control_value_parity_and_no_backbone_grads(self):
        a = JointCSRAStudent(copy.deepcopy(self.model), copy.deepcopy(self.adapter), True, False)
        b = JointCSRAStudent(copy.deepcopy(self.model), copy.deepcopy(self.adapter), False, False)
        for u, v in zip(a(self.batch), b(self.batch)):
            # Trainability can choose a different CPU GEMM implementation.
            torch.testing.assert_close(u, v, atol=1e-6, rtol=1e-6)
        b(self.batch)[0].square().mean().backward()
        self.assertTrue(all(p.grad is None for p in b.backbone.parameters()))
        self.assertTrue(any(p.grad is not None for p in b.adapter.parameters()))

    def test_checkpoint_gradient_parity(self):
        a = JointCSRAStudent(copy.deepcopy(self.model), copy.deepcopy(self.adapter), True, True)
        b = JointCSRAStudent(copy.deepcopy(self.model), copy.deepcopy(self.adapter), True, False)
        ya, yb = a(self.batch), b(self.batch)
        for u,v in zip(ya,yb):
            torch.testing.assert_close(u,v,atol=0,rtol=0)
        sum(x.square().mean() for x in ya).backward()
        sum(x.square().mean() for x in yb).backward()
        for pa,pb in zip(a.parameters(),b.parameters()):
            torch.testing.assert_close(pa.grad,pb.grad,atol=1e-6,rtol=1e-5)

    def test_preserved_source_trains_only_executed_fine_weights(self):
        net = PreservedCoarseStudent(self.model, self.adapter, True, False)
        batch = {**self.batch, "coarse": {l: torch.randn(2,1,16) for l in [1,3]}}
        net(batch).square().mean().backward()
        self.assertTrue(all(p.grad is not None for p in net.parameters() if p.requires_grad))
        self.assertTrue(all(p.grad is None for l in [1,3] for p in net.backbone.blocks[l].parameters()))
        self.assertTrue(all(p.grad is not None for l in [0,2] for p in net.backbone.blocks[l].parameters()))

    def test_preserved_source_forward_matches_same_fine_stage(self):
        a = PreservedCoarseStudent(copy.deepcopy(self.model), copy.deepcopy(self.adapter), False, False)
        b = JointCSRAStudent(copy.deepcopy(self.model), copy.deepcopy(self.adapter), False, False)
        batch = {**self.batch, "coarse": {l: torch.randn(2,1,16) for l in [1,3]}}
        torch.testing.assert_close(a(batch), b.stage(batch["x"],batch["t"],batch["labels"],cache=batch["coarse"])[0],atol=0,rtol=0)

    def test_trained_zero_control_never_reads_source_and_matches_explicit_zeros(self):
        a = PreservedCoarseStudent(copy.deepcopy(self.model), copy.deepcopy(self.adapter), True, False, "zero")
        b = PreservedCoarseStudent(copy.deepcopy(self.model), copy.deepcopy(self.adapter), True, False)
        batch = {**self.batch, "coarse": {l: None for l in [1,3]}}
        zeros = {**self.batch, "coarse": {l: torch.zeros(2,1,16) for l in [1,3]}}
        ya, yb = a(batch), b(zeros)
        torch.testing.assert_close(ya,yb,atol=0,rtol=0)
        ya.square().mean().backward()
        yb.square().mean().backward()
        self.assertTrue(all(p.grad is not None for p in a.parameters() if p.requires_grad))
        self.assertEqual(float(a.adapter.adapters['1'].low_down.weight.grad.abs().sum()),0.)
        # With zero source and zero-initialized time shift, low_up's input
        # is zero too: its weight gradient is legitimately zero. Test the
        # complete gradients against the explicit-zero reference instead of
        # incorrectly requiring every source-branch gradient to be nonzero.
        for pa, pb in zip(a.parameters(), b.parameters()):
            if pa.requires_grad:
                torch.testing.assert_close(pa.grad,pb.grad,atol=1e-6,rtol=1e-5)

    def test_unknown_preserved_source_is_rejected(self):
        with self.assertRaises(ValueError):
            PreservedCoarseStudent(self.model,self.adapter,coarse_source="other")


if __name__ == "__main__":
    unittest.main()
