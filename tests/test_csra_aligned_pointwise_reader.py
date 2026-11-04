import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'2d/attention_analysis'))
from csra_layout_student import SpanReader
from csra_aligned_pointwise_reader import AlignedPointwiseReader


class Tests(unittest.TestCase):
    def test_exact_gate0_without_any_attention(self):
        torch.manual_seed(77);h=8
        init=dict(weight=torch.randn(2*h,h),scale=torch.ones(2*h),mean=torch.zeros(2*h),output_mean=torch.zeros(h))
        full=SpanReader(h,2,init)
        with torch.no_grad():
            full.proj.weight[:,2*h:]=torch.eye(h)*.2
            for p in full.nonlinear.parameters():p.normal_(std=.1)
        small=AlignedPointwiseReader(full)
        x,c,m=torch.randn(2,16,h),torch.randn(2,h),torch.randn(2,4,h)
        hook=full.read_out.register_forward_hook(lambda module,args,out:torch.zeros_like(out))
        try:expected=full(x,c,m)
        finally:hook.remove()
        with patch('torch.nn.functional.scaled_dot_product_attention',side_effect=AssertionError('Attention unexpectedly executed')):
            actual=small(x,c,m)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        self.assertGreater((actual-small(x,c,m.roll(1,1))).abs().max().item(),0)
        self.assertNotIn('query.weight',small.state_dict())
        self.assertNotIn('key_value.weight',small.state_dict())


if __name__=='__main__':unittest.main()
