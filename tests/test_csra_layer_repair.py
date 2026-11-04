"""Diagnostic controls must not silently change mask, cache, or conditioning."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2d" / "attention_analysis"))
from probe_csra_layer_repair import replacement_callbacks, prediction_metrics, residual_metrics
from train_rdd_csra_adapter import CSRAAdapterBank
from csra_runtime import predict_adapter


class LayerRepairTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(22)
        self.adapter = CSRAAdapterBank([1,3], 16, 4, "teacher_condition_v3").eval()
        for module in self.adapter.adapters.values():
            torch.nn.init.normal_(module.low_up.weight)
        self.source = {k:torch.randn(2,16,16) for k in [1,3]}
        self.h = torch.randn(2,16,16)
        self.t = torch.tensor([42.,42.])
        self.c = torch.randn(2,16)

    def test_paired_equals_runtime_and_restores_exactly_one_layer(self):
        callbacks = replacement_callbacks(self.adapter,self.source,self.t,restore=1)
        self.assertEqual(set(callbacks),{3})
        expected = predict_adapter(self.adapter.adapters["3"],self.source[3],self.h,self.t,
                                   "fp32",condition=self.c)
        torch.testing.assert_close(callbacks[3](self.h,self.c),expected,rtol=0,atol=0)

    def test_controls_preserve_source_and_use_intended_tensor(self):
        before = {k:v.clone() for k,v in self.source.items()}
        for mode in ["zero","permuted"]:
            callbacks = replacement_callbacks(self.adapter,self.source,self.t,cache_mode=mode)
            for k in self.source:
                source = torch.zeros_like(self.source[k]) if mode=="zero" else self.source[k].roll(1,0)
                expected = predict_adapter(self.adapter.adapters[str(k)],source,self.h,self.t,
                                           "fp32",condition=self.c)
                torch.testing.assert_close(callbacks[k](self.h,self.c),expected,rtol=0,atol=0)
                torch.testing.assert_close(self.source[k],before[k],rtol=0,atol=0)

    def test_unknown_controls_fail(self):
        with self.assertRaises(ValueError):
            replacement_callbacks(self.adapter,self.source,self.t,restore=9)
        with self.assertRaises(ValueError):
            replacement_callbacks(self.adapter,self.source,self.t,cache_mode="oops")
        with self.assertRaises(ValueError):
            replacement_callbacks(self.adapter,{k:v[:1] for k,v in self.source.items()},self.t[:1],
                                  cache_mode="permuted")

    def test_error_energy_decomposes_and_identical_prediction_zero(self):
        y = torch.randn_like(self.h)
        result = residual_metrics(self.h,y)
        self.assertAlmostEqual(result["mse"],result["low_mse"]+result["high_mse"],places=5)
        x = torch.randn(2,8,4,4)
        self.assertEqual(prediction_metrics(x,x)["main_mse"],0)


if __name__ == "__main__":
    unittest.main()
