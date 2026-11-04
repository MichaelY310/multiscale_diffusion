"""CPU-only contract/gradient checks. Does not validate speed or video quality."""
import json
from pathlib import Path
import unittest
import torch
from committed_band_process import (split_mean_detail,merge_mean_detail,
    detail_noise_like,conditional_noisy_detail,sample_three_committed_stages,
    codec_distillation_loss)
from progressive_renderer import TemporalInnovationRenderer


class Checks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        torch.set_num_threads(2)

    def test_band_reconstruction_and_parent_commit(self):
        x=torch.randn(1,16,8,3,5,dtype=torch.float64)
        c,h=split_mean_detail(x)
        torch.testing.assert_close(merge_mean_detail(c,h),x,rtol=0,atol=1e-15)
        other_h=torch.randn_like(h)*10
        got,_=split_mean_detail(merge_mean_detail(c,other_h))
        # Large detail cancellation has scale-dependent FP64 roundoff.
        bound=8*torch.finfo(x.dtype).eps*float(other_h.abs().max())
        torch.testing.assert_close(got,c,rtol=0,atol=bound)

    def test_birth_noise_has_correct_nullspace_covariance(self):
        parent=torch.zeros(1,1,1,1,100000)
        g=detail_noise_like(parent,torch.Generator().manual_seed(98))
        born=merge_mean_detail(parent,g)
        low,_=split_mean_detail(born)
        self.assertEqual(float(low.abs().max()),0)
        self.assertAlmostEqual(float(g.var()),.5,delta=.01)
        self.assertAlmostEqual(float((born[:,:,0]*born[:,:,1]).mean()),-.5,delta=.01)
        clean_h=torch.randn(1,16,2,2,2)
        torch.testing.assert_close(conditional_noisy_detail(clean_h,0,torch.Generator()),clean_h)

    def test_three_calls_seed_and_training_gradient(self):
        weight=torch.tensor(.2,requires_grad=True)
        calls=[]
        def root(noise,sigma):
            calls.append(0)
            return noise*weight
        def detail(c,h,sigma,level):
            calls.append(level)
            return h*weight+c*.05
        result,stages=sample_three_committed_stages((1,16,2,2,2),root,detail,seed=3)
        self.assertEqual(calls,[0,1,2])
        self.assertEqual(result.shape,(1,16,8,2,2))
        self.assertEqual(result.device.type,'cpu')
        for before,after in zip(stages,stages[1:]):
            pooled,_=split_mean_detail(after.clean)
            torch.testing.assert_close(pooled,before.clean,rtol=1e-5,atol=1e-7)
        repeat,_=sample_three_committed_stages((1,16,2,2,2),root,detail,seed=3)
        torch.testing.assert_close(result,repeat,rtol=0,atol=0)
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(weight.grad))
        self.assertGreater(float(weight.grad.abs()),0)

    def test_renderer_forecast_control_and_backward(self):
        model=TemporalInnovationRenderer(width=8,depth=1,forecast_correction=True)
        c=torch.randn(1,16,2,2,2)
        h=torch.randn_like(c,requires_grad=True)
        reference=c+torch.randn_like(c)*.1
        # Native parent length4T-3=5; fine length8T-3=13.
        rgb=torch.rand(1,3,5,16,16)*1.4-.7
        pred=model(c,h,rgb,reference)
        split=model.finish(rgb,model.prepare_detail(c,h,reference))
        torch.testing.assert_close(pred,split,rtol=0,atol=0)
        self.assertEqual(pred.shape,(1,3,13,16,16))
        self.assertTrue(torch.isfinite(pred).all())
        losses=codec_distillation_loss(pred,torch.zeros_like(pred))
        losses['total'].backward()
        self.assertGreater(float(h.grad.abs().sum()),0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        committed=model(c,h.detach(),rgb,c)
        self.assertGreater(float((committed-pred.detach()).detach().abs().max()),0)

    def test_bad_shapes_rejected(self):
        with self.assertRaises(ValueError):
            split_mean_detail(torch.zeros(1,16,3,2,2))
        with self.assertRaises(ValueError):
            merge_mean_detail(torch.zeros(1,16,2,2,2),torch.zeros(1,16,1,2,2))
        with self.assertRaises(ValueError):
            TemporalInnovationRenderer(forecast_correction=True).prepare_detail(
                torch.zeros(1,16,2,2,2),torch.zeros(1,16,2,2,2))


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    output=Path('eval_results/progressive_infra_20261003/ppr_cpu_checks.json')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(tests=result.testsRun,failures=len(result.failures),
        errors=len(result.errors),passed=result.wasSuccessful(),device='cpu',
        limitation='No CUDA execution, trained quality, timestep calibration, or speed validation'),indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
