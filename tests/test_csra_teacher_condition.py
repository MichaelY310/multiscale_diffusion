"""Compatibility and gradient tests for the optional current-condition branch."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2d" / "attention_analysis"))
from csra_runtime import predict_adapter
from csra_output_distillation import CSRAOutputStudent
from probe_rdd_vs_dit_block_reuse import manual_forward
from train_rdd_csra_adapter import CSRAAdapterBank, initialize_adapter_weights, upsample_tokens
from test_csra_output_distillation import TinyTeacher


class TeacherConditionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(810)
        self.v2 = CSRAAdapterBank([1, 3], 16, 4, "fine_context_v2")
        with torch.no_grad():
            for p in self.v2.parameters():
                p.copy_(.1 * torch.randn_like(p))
        self.payload = {"layers": [1, 3], "hidden_size": 16, "rank": 4,
                        "adapter_architecture": "fine_context_v2", "adapter": self.v2.state_dict()}
        self.v3 = CSRAAdapterBank([1, 3], 16, 4, "teacher_condition_v3")
        initialize_adapter_weights(self.v3, self.payload)
        self.coarse = torch.randn(2, 16, 16)
        self.hidden = torch.randn(2, 16, 16)
        self.t = torch.tensor([10, 200])
        self.c = torch.randn(2, 16)

    def test_zero_condition_gate_preserves_trained_v2_exactly(self):
        a = predict_adapter(self.v2.adapters["1"], self.coarse, self.hidden, self.t, "fp32")
        b = predict_adapter(self.v3.adapters["1"], self.coarse, self.hidden, self.t, "fp32", self.c)
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_missing_condition_is_not_silently_ignored(self):
        with self.assertRaises(ValueError):
            predict_adapter(self.v3.adapters["1"], self.coarse, self.hidden, self.t, "fp32")

    def test_gate_receives_gradients_and_changes_with_condition(self):
        branch = self.v3.adapters["1"]
        value = predict_adapter(branch, self.coarse, self.hidden, self.t, "fp32", self.c)
        value.square().mean().backward()
        self.assertGreater(float(branch.condition_mlp[-1].weight.grad.abs().sum()), 0)
        with torch.no_grad():
            branch.condition_mlp[-1].weight.normal_(0, .1)
        a = predict_adapter(branch, self.coarse, self.hidden, self.t, "fp32", self.c)
        b = predict_adapter(branch, self.coarse, self.hidden, self.t, "fp32", -self.c)
        self.assertGreater(float((a-b).abs().max()), 0)

    def test_old_callbacks_and_new_conditioned_callbacks_match(self):
        teacher = TinyTeacher().eval().requires_grad_(False)
        x, y = torch.randn(2, 4, 4, 4), torch.tensor([1, 2])
        cache = {l: self.coarse for l in [1, 3]}
        plain = {l: (lambda h, l=l: predict_adapter(self.v2.adapters[str(l)],
                  cache[l], h, self.t, "fp32")) for l in [1, 3]}
        conditioned = {l: (lambda h, c, l=l: predict_adapter(self.v3.adapters[str(l)],
                        cache[l], h, self.t, "fp32", c)) for l in [1, 3]}
        a, _ = manual_forward(teacher, x, self.t, y, multires=True, replacements=plain)
        b, _ = manual_forward(teacher, x, self.t, y, multires=True,
                              replacements=conditioned, conditioned_replacements=True)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        student = CSRAOutputStudent(teacher, self.v3)
        z = student(x, self.t, y, cache)
        torch.testing.assert_close(a, z, rtol=0, atol=0)
        z.square().mean().backward()
        for l in [1, 3]:
            grad = self.v3.adapters[str(l)].condition_mlp[-1].weight.grad
            self.assertGreater(float(grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_conversion_rejects_missing_old_key_and_v1_direct_upgrade(self):
        malformed = dict(self.payload, adapter=dict(self.payload["adapter"]))
        del malformed["adapter"]["adapters.1.fine_up.weight"]
        with self.assertRaises(ValueError):
            initialize_adapter_weights(self.v3, malformed)
        with self.assertRaises(ValueError):
            initialize_adapter_weights(self.v3, dict(self.payload, adapter_architecture="legacy_v1"))


if __name__ == "__main__":
    unittest.main()
