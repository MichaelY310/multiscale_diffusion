"""CPU-only format/algebra checks. No CUDA initialization or latency tests."""
import json
from pathlib import Path
import unittest
import torch
from anchor_nvfp4_reference import *
from spatial_detail_basis import signs


class Checks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(74)
        torch.set_num_threads(2)

    def test_codebook_ties_signed_zero(self):
        values=torch.tensor([0.,.5,1,1.5,2,3,4,6])
        self.assertEqual(e2m1_codes(values).tolist(),list(range(8)))
        self.assertEqual(e2m1_codes(-values).tolist(),list(range(8,16)))
        self.assertEqual(e2m1_codes(torch.tensor([.25,.75,1.25,1.75,2.5,3.5,5])).tolist(),[0,2,2,4,4,6,6])
        self.assertEqual(e2m1_codes(torch.tensor([-100.,100.])).tolist(),[15,7])
        self.assertEqual(e2m1_codes(torch.tensor([1e10,-1e10,1e20,-1e20])).tolist(),[7,15,7,15])

    def test_packing_and_quantized_scales(self):
        x=torch.tensor([0.,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6]).reshape(1,-1)
        p=pack_nvfp4(x,1.)
        self.assertEqual(p.data.tolist(),[[16,50,84,118,152,186,220,254]])
        torch.testing.assert_close(unpack_nvfp4(p),x,rtol=0,atol=0)
        x=torch.randn(2,37)
        p=pack_nvfp4(x,.0317)
        self.assertEqual(tuple(p.data.shape),(2,24))
        self.assertEqual(tuple(p.scales.shape),(2,3))
        sf=p.scales.view(torch.float8_e4m3fn).float()*p.global_scale
        codes=e2m1_codes(F.pad(x,(0,11)).reshape(2,3,16)/sf.unsqueeze(-1))
        self.assertTrue(torch.equal(p.data, codes.flatten(-2)[:,::2] | (codes.flatten(-2)[:,1::2]<<4)))

    def test_zero_tiny_clip_and_nonfinite(self):
        p=pack_nvfp4(torch.zeros(2,19))
        self.assertEqual(int(p.data.sum()),0)
        self.assertEqual(int(p.scales.sum()),0)
        self.assertEqual(int(p.clipped.sum()),0)
        self.assertEqual(int(pack_nvfp4(-torch.zeros(1,16)).data.sum()),0)
        p=pack_nvfp4(torch.full((1,17),1e-20),1.)
        self.assertTrue(bool(torch.isfinite(unpack_nvfp4(p)).all()))
        p=pack_nvfp4(torch.full((1,16),10000.),1.)
        self.assertEqual(int(p.clipped.sum()),16)
        self.assertTrue(bool((unpack_nvfp4(p)==2688).all()))
        p=pack_nvfp4(torch.full((1,16),1e20),1.)
        self.assertTrue(bool((unpack_nvfp4(p)==2688).all()))
        p=pack_nvfp4(torch.ones(1,16),torch.finfo(torch.float32).tiny/1000)
        self.assertEqual(p.data[0,0].item(),119)
        p=pack_nvfp4(torch.ones(1,16),1e38)
        self.assertTrue(bool(torch.isfinite(unpack_nvfp4(p)).all()))
        with self.assertRaises(ValueError): pack_nvfp4(torch.tensor([[float('nan')]]))
        with self.assertRaises(ValueError): pack_nvfp4(torch.ones(1,16),0.)

    def test_fresh_gelu_and_mean_constraint(self):
        a=torch.randn(9,37).bfloat16()
        h=torch.randn(9,3,37).bfloat16()
        y=F.gelu(a.float().unsqueeze(1)+torch.einsum('pb,nbg->npg',signs(h.float()),h.float()))
        reference=torch.einsum('pb,npg->nbg',signs(y),y)/4
        got=fresh_detail_gelu(a,h)
        torch.testing.assert_close(got,reference,rtol=2e-5,atol=1e-6)
        # Quantized detail remains zero-mean when synthesized; not a quality claim.
        packed=fused_epilogue_reference(a,h)
        q=unpack_nvfp4(packed).reshape(9,3,37)
        children=torch.einsum('pb,nbg->npg',signs(q),q)
        self.assertLess(float(children.mean(1).abs().max()),2e-7)

    def test_same_operator_ordinary_current_mean_control(self):
        # Ordinary diffusion may compute its CURRENT mean, then run the SAME
        # fused epilogue. Its only extra operation is current_mean @ W1.
        c=torch.randn(5,16); h=torch.randn(5,3,16); w=torch.randn(16,64)*.1
        cached=c@w
        dynamic=c@w
        p=fused_epilogue_reference(cached,h@w,.01)
        ordinary=fused_epilogue_reference(dynamic,h@w,.01)
        self.assertTrue(torch.equal(p.data,ordinary.data))
        self.assertTrue(torch.equal(p.scales,ordinary.scales))


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    a=torch.randn(128,8960)*.5; h=torch.randn(128,3,8960)*.2
    high=fresh_detail_gelu(a,h)
    p=fused_epilogue_reference(a,h)
    q=unpack_nvfp4(p).reshape_as(high)
    audit=dict(tests=result.testsRun,passed=result.wasSuccessful(),device='cpu',
        isolated_synthetic_epilogue_relative_l2=float((q-high).norm()/high.norm()),
        clipped_values=int(p.clipped.sum()),values=high.numel(),
        budget=unique_cache_budget(),
        scope='CPU reference only; no CUDA launch, GEMM, full-model timing or generated quality.')
    path=Path('eval_results/progressive_infra_20261003/anchor_nvfp4_cpu_checks.json')
    path.write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
