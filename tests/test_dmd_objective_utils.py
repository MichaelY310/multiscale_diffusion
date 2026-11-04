import sys
from pathlib import Path
import unittest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'2d/attention_analysis'));sys.path.insert(0,str(ROOT/'DiT'))
from dmd_objective_utils import DenseScore,chunked_value_gradient,value_with_gradient,schedule,score_candidates
from dmd_cascade_reuse import CascadeDMD
from models_multires import DiTMultiRes


class Tests(unittest.TestCase):
    def test_chunked_gradient_exact(self):
        torch.manual_seed(61)
        frozen=torch.nn.Conv2d(3,5,3,padding=1).eval().requires_grad_(False)
        x=torch.randn(7,3,8,8,requires_grad=True);ref=torch.randn_like(x)
        def function(z,target):return (frozen(z).tanh()-frozen(target).tanh()).square().flatten(1).mean(1)
        expected=function(x,ref).mean();g,=torch.autograd.grad(expected,x)
        for chunk in (1,2,4,7):
            value,gradient=chunked_value_gradient(x,ref,function,chunk)
            proxy=value_with_gradient(x,value,gradient);actual,=torch.autograd.grad(proxy,x)
            torch.testing.assert_close(value,expected)
            torch.testing.assert_close(actual,g)

    def test_dense_score_parity_and_score_times(self):
        torch.manual_seed(61)
        base=DiTMultiRes(depth=4,hidden_size=32,num_heads=4,patch_size=2,num_classes=10,base_size=16).eval()
        with torch.no_grad():
            for p in base.parameters():p.add_(torch.randn_like(p)*.002)
        score=DenseScore(base).eval()
        for r,t in ((8,832),(16,400),(32,79)):
            x=torch.randn(2,4,r,r);times=torch.full((2,),t);y=torch.tensor([1,2])
            torch.testing.assert_close(score(x,times,y),base(x,times,y),rtol=0,atol=0)
        indices=score_candidates(score.core.diffusions)
        self.assertEqual([len(i) for i in indices],[380,300,280])

    def test_schedule(self):
        self.assertEqual(schedule(0,1000)['dmd'],0)
        self.assertAlmostEqual(schedule(1000,1000)['final_mse'],.1)
        self.assertEqual(schedule(30000,1000)['dmd'],1)
        self.assertEqual(schedule(30000,1000)['lpips'],.25)

if __name__=='__main__':unittest.main()
