"""No GPU use, no timing claims. Algebra and ordinary-packing counterexample."""
import json
from pathlib import Path
import unittest
import torch
from spatial_detail_basis import *


class Checks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(190)
        torch.set_num_threads(2)

    def test_reconstruct_and_three_pixel_example(self):
        x=torch.randn(1,16,2,8,12,dtype=torch.float64)
        c,h=split_spatial_mean_detail(x)
        torch.testing.assert_close(merge_spatial_mean_detail(c,h),x,rtol=0,atol=1e-15)
        torch.testing.assert_close(signs(x).T@signs(x),torch.eye(3,dtype=x.dtype)*4)
        self.assertEqual(4*10-(8+9+11),12)

    def test_innovation_covariance_and_fixed_mean(self):
        c=torch.zeros(1,1,1,1,100000)
        h=sample_spatial_detail_noise(c,torch.Generator().manual_seed(8))
        covariance=h.reshape(-1,3).T@h.reshape(-1,3)/100000
        torch.testing.assert_close(covariance,torch.eye(3)*.25,rtol=0,atol=.004)
        x=merge_spatial_mean_detail(c,h)
        got,_=split_spatial_mean_detail(x)
        self.assertLessEqual(float(got.abs().max()),4*torch.finfo(c.dtype).eps)
        cells=torch.stack((x[...,0::2,0::2],x[...,0::2,1::2],
                           x[...,1::2,0::2],x[...,1::2,1::2]),-1).reshape(-1,4)
        expected=torch.eye(4)-torch.ones(4,4)/4
        torch.testing.assert_close(cells.T@cells/len(cells),expected,rtol=0,atol=.01)

    def test_three_row_ffn_matches_dense_high_and_gradients(self):
        c=torch.randn(2,5,16,dtype=torch.float64,requires_grad=True)
        h=torch.randn(2,5,3,16,dtype=torch.float64,requires_grad=True)
        w1=torch.randn(16,64,dtype=torch.float64,requires_grad=True)*.2
        w2=torch.randn(64,16,dtype=torch.float64,requires_grad=True)*.2
        b1=torch.randn(64,dtype=torch.float64,requires_grad=True)
        b2=torch.randn(16,dtype=torch.float64,requires_grad=True)
        variables=(c,h,w1,w2,b1,b2)
        cached=cache_parent_preactivation(c,w1,b1)
        fast=high_band_ffn(h,cached,w1,w2)
        reference=dense_high_band_ffn_reference(c,h,w1,b1,w2,b2)
        torch.testing.assert_close(fast,reference,rtol=1e-12,atol=1e-12)
        gf=torch.autograd.grad(fast.square().sum(),variables[:-1],retain_graph=True)
        gr=torch.autograd.grad(reference.square().sum(),variables[:-1])
        for a,b in zip(gf,gr):
            torch.testing.assert_close(a,b,rtol=1e-11,atol=1e-11)

    def test_child_layernorm_breaks_cached_anchor_assumption(self):
        c=torch.randn(2,16,dtype=torch.float64)
        h=torch.randn(2,3,16,dtype=torch.float64)
        x=c.unsqueeze(-2)+torch.einsum('pb,...bd->...pd',signs(h),h)
        fine_mean=torch.nn.functional.layer_norm(x,(16,)).mean(-2)
        coarse=torch.nn.functional.layer_norm(c,(16,))
        self.assertGreater(float((fine_mean-coarse).norm()),.1)

    def test_ordinary_three_token_lossless_counter(self):
        # Four4C-band values -> three tokens, each containing LL+one detail.
        # With D>=8C, tall full-rank projection retains all four bands.
        # This shows3tokens alone is NOT inherently progressive-exclusive.
        channels=16
        band_width=4*channels
        projection=torch.linalg.qr(torch.randn(256,2*band_width,dtype=torch.float64)).Q
        bands=torch.randn(4,band_width,dtype=torch.float64)
        tokens=torch.stack([torch.cat((bands[0],bands[i]))@projection.T for i in (1,2,3)])
        recovered=tokens@projection
        self.assertEqual(int(torch.linalg.matrix_rank(projection)),128)
        torch.testing.assert_close(recovered[:,:band_width],bands[0].expand(3,-1),rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(recovered[:,band_width:],bands[1:],rtol=1e-12,atol=1e-12)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    path=Path('eval_results/progressive_infra_20261003/spatial_detail_cpu_checks.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(dict(tests=result.testsRun,passed=result.wasSuccessful(),
        failures=len(result.failures),errors=len(result.errors),device='cpu',
        scope='Algebra only. Not a complete conditional Wan, GPU kernel, trained quality or speed result.'),indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
