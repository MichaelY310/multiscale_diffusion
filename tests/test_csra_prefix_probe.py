import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'2d'/'attention_analysis'))
from csra_prefix_probe_math import select_records, coarse_lift, features, split_error, Moments, fit_ridge, predict


class PrefixProbeTests(unittest.TestCase):
    def test_disjoint_selection(self):
        files = [{'name':f'{s}_{i}.pt','split':s,'seed':offset+i} for s,offset,n in
                 [('train',100,12),('validation',200,4)] for i in range(n)]
        selected = select_records({'files':files})
        self.assertEqual([sum(r['role']==role for r in selected) for role in ('fit','calibration','validation')],[6,2,4])
        files[-1]['seed'] = 100
        with self.assertRaises(ValueError): select_records({'files':files})

    def test_spatial_controls_and_no_source(self):
        source = torch.arange(8.).reshape(2,4,1)
        lifted = coarse_lift(source,'coarse_aligned')
        torch.testing.assert_close(lifted[0,:,0],torch.tensor([0.,0.,1.,1.,0.,0.,1.,1.,2.,2.,3.,3.,2.,2.,3.,3.]))
        torch.testing.assert_close(coarse_lift(source,'coarse_wrong')[0],lifted[1])
        torch.testing.assert_close(coarse_lift(source,'coarse_mean').mean(1),source.mean(1))
        self.assertFalse(torch.equal(coarse_lift(source,'coarse_roll'),lifted))
        h0,c = torch.ones(2,16,1),torch.ones(2,1)
        self.assertEqual(features(h0,c,source,'fine_only').shape,(32,2))
        torch.testing.assert_close(features(h0,c,source,'fine_only'),features(h0,c,source*99,'fine_only'))

    def test_error_projection_adds(self):
        g = torch.Generator().manual_seed(31)
        error = torch.randn(3,16,8,generator=g)
        low, high = split_error(error)
        torch.testing.assert_close(error.square().mean(),low.square().mean()+high.square().mean())
        self.assertLess(abs(float((low*high).mean())),1e-7)

    def test_ridge_matches_direct_centered_solve(self):
        g = torch.Generator().manual_seed(78)
        x = torch.randn(50,6,generator=g)
        x[:,5] = 2.0
        y = x@torch.randn(6,3,generator=g) + .4
        m = Moments(6,3,'cpu')
        m.add(x[:17],y[:17]); m.add(x[17:],y[17:])
        fitted = fit_ridge(m.cpu(),.1)
        xd,yd = x.double(),y.double()
        mean = xd.mean(0); scale = xd.var(0,unbiased=False).clamp_min(1e-8).sqrt()
        z = (xd-mean)/scale
        w = torch.linalg.solve(z.T@z/50+.1*torch.eye(6),z.T@(yd-yd.mean(0))/50)
        expected = z@w+yd.mean(0)
        torch.testing.assert_close(predict(xd,fitted),expected,atol=2e-5,rtol=2e-5)
        self.assertLess(fitted['solve_relative_residual'],1e-12)


if __name__ == '__main__': unittest.main()
