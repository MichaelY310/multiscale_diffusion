"""Small CPU tests for the new training graph, independent of checkpoints."""
import copy
import sys
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2d" / "attention_analysis"))
from csra_output_distillation import CSRAOutputStudent, prediction_loss
from train_rdd_csra_adapter import CSRAAdapterBank


class Embed(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 16, 1)

    def forward(self, x):
        return self.conv(x).flatten(2).transpose(1, 2)


class Label(nn.Embedding):
    def forward(self, labels, training):
        return super().forward(labels)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 16)

    def forward(self, h, c, cos, sin):
        return h + torch.tanh(self.linear(h + c[:, None]))


class Head(nn.Linear):
    def forward(self, h, c):
        return super().forward(h + c[:, None])


class TinyTeacher(nn.Module):
    patch_size = 1

    def __init__(self):
        super().__init__()
        self.x_embedder = Embed()
        self.t_embedder = nn.Embedding(300, 16)
        self.y_embedder = Label(10, 16)
        self.blocks = nn.ModuleList([Block() for _ in range(5)])
        self.final_layer = Head(16, 8)

    def get_rope(self, gh, gw, device):
        return torch.ones(gh * gw, 1, device=device), torch.zeros(gh * gw, 1, device=device)

    def unpatchify(self, output, gh, gw):
        return output.transpose(1, 2).reshape(output.shape[0], 8, gh, gw)


class OutputDistillationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.teacher = TinyTeacher().eval().requires_grad_(False)
        self.adapter = CSRAAdapterBank([1, 3], 16, 4, "fine_context_v2")
        self.x = torch.randn(2, 4, 4, 4)
        self.t = torch.tensor([10, 200])
        self.labels = torch.tensor([1, 2])
        self.coarse = {k: torch.randn(2, 4, 16) for k in [1, 3]}

    def test_gradients_reach_early_and_late_adapter_not_teacher(self):
        student = CSRAOutputStudent(self.teacher, self.adapter)
        output = student(self.x, self.t, self.labels, self.coarse)
        loss, _ = prediction_loss(output, torch.randn_like(output))
        loss.backward()
        for k in [1, 3]:
            self.assertGreater(float(self.adapter.adapters[str(k)].low_up.weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in self.teacher.parameters()))
        self.assertEqual(set(student.state_dict()), {"adapter." + k for k in self.adapter.state_dict()})

    def test_checkpointing_preserves_output_and_gradient(self):
        other = copy.deepcopy(self.adapter)
        a = CSRAOutputStudent(self.teacher, self.adapter, True)
        b = CSRAOutputStudent(self.teacher, other, False)
        ya = a(self.x, self.t, self.labels, self.coarse)
        yb = b(self.x, self.t, self.labels, self.coarse)
        torch.testing.assert_close(ya, yb, rtol=0, atol=0)
        ya.square().mean().backward()
        yb.square().mean().backward()
        for pa, pb in zip(self.adapter.parameters(), other.parameters()):
            torch.testing.assert_close(pa.grad, pb.grad, rtol=1e-6, atol=1e-7)

    def test_trainable_backbone_rejected(self):
        with self.assertRaises(ValueError):
            CSRAOutputStudent(TinyTeacher(), self.adapter)

    def test_loss_separates_variance(self):
        target = torch.zeros(1, 8, 2, 2)
        candidate = target.clone()
        candidate[:, 4:] = 2
        loss, metrics = prediction_loss(candidate, target, .05)
        self.assertAlmostEqual(float(loss), .2)
        self.assertEqual(float(metrics["main_mse"]), 0)


if __name__ == "__main__":
    unittest.main()
