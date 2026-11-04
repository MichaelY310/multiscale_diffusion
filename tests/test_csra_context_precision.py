"""CPU unit checks; CUDA smoke testing is a separate deployment step."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2d" / "attention_analysis"))
from csra_runtime import predict_adapter
from train_rdd_csra_adapter import CSRAAdapterBank, initialize_adapter_weights


class CSRAContextTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.v1 = CSRAAdapterBank([1], 16, 4)
        with torch.no_grad():
            for parameter in self.v1.parameters():
                parameter.copy_(torch.randn_like(parameter) * 0.1)
        self.payload = {"layers": [1], "rank": 4, "hidden_size": 16,
                        "adapter": self.v1.state_dict()}
        self.coarse = torch.randn(2, 16, 16)
        self.hidden = torch.randn(2, 16, 16)
        self.time = torch.tensor([60, 200])

    def test_zero_output_new_branch_preserves_v1(self):
        v2 = CSRAAdapterBank([1], 16, 4, "fine_context_v2")
        initialize_adapter_weights(v2, self.payload)
        self.assertTrue(torch.equal(self.v1.adapters["1"](self.coarse, self.hidden, self.time),
                                    v2.adapters["1"](self.coarse, self.hidden, self.time)))

    def test_v2_missing_input_branch_receives_gradients(self):
        v2 = CSRAAdapterBank([1], 16, 4, "fine_context_v2")
        initialize_adapter_weights(v2, self.payload)
        result = v2.adapters["1"](self.coarse, self.hidden, self.time)
        result.square().mean().backward()
        self.assertGreater(float(v2.adapters["1"].fine_up.weight.grad.abs().sum()), 0)

    def test_precision_policy_is_local_and_default_is_legacy(self):
        adapter = self.v1.adapters["1"]
        reference = adapter(self.coarse, self.hidden, self.time)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            legacy = predict_adapter(adapter, self.coarse, self.hidden, self.time)
            fp32 = predict_adapter(adapter, self.coarse, self.hidden, self.time, "fp32")
            self.assertEqual(torch.mm(torch.ones(2, 2), torch.ones(2, 2)).dtype, torch.bfloat16)
        self.assertEqual(legacy.dtype, torch.bfloat16)
        self.assertEqual(fp32.dtype, torch.float32)
        self.assertTrue(torch.equal(reference, fp32))

    def test_mask_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            initialize_adapter_weights(CSRAAdapterBank([2], 16, 4), self.payload)

    def test_rank_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            initialize_adapter_weights(CSRAAdapterBank([1], 16, 8), self.payload)

    def test_missing_existing_weight_rejected(self):
        malformed = dict(self.payload, adapter=dict(self.payload["adapter"]))
        del malformed["adapter"]["adapters.1.low_up.weight"]
        with self.assertRaises(ValueError):
            initialize_adapter_weights(CSRAAdapterBank([1], 16, 4), malformed)


if __name__ == "__main__":
    unittest.main()
