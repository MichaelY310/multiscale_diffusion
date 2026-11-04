import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"2d"/"attention_analysis"))
from probe_csra_capacity_floor import output_basis,projection_floor
from train_rdd_csra_adapter import CrossScaleResidualAdapter


class CapacityFloorTests(unittest.TestCase):
    def test_adapter_lies_in_output_subspace(self):
        torch.manual_seed(122)
        a=CrossScaleResidualAdapter(32,4,architecture="fine_context_v2").eval()
        for p in a.parameters():torch.nn.init.normal_(p,std=.1)
        u,b,rank=output_basis(a)
        h=torch.randn(2,16,32);coarse=torch.randn_like(h)
        y=a(coarse,h,torch.tensor([299.,40.]))
        self.assertLessEqual(rank,12)
        self.assertLess(projection_floor(y,u,b),1e-10)
        target=torch.randn_like(y)
        self.assertLessEqual(projection_floor(target,u,b),float((y-target).square().mean())+1e-6)

    def test_fixed_coarse_offset_can_preserve_out_of_basis_information(self):
        u=torch.eye(4)[:,:1];b=torch.zeros(4)
        target=torch.tensor([[[0.,1.,0.,0.]]])
        self.assertEqual(projection_floor(target,u,b),.25)
        self.assertEqual(projection_floor(target,u,b,offset=target),0.)


if __name__=="__main__":unittest.main()
