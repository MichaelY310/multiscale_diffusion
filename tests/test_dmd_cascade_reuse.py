import sys
from pathlib import Path
import unittest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'DiT'));sys.path.insert(0,str(ROOT/'2d/attention_analysis'));sys.path.insert(0,str(ROOT))
from models_multires import DiTMultiRes
from dmd_cascade_reuse import CascadeDMD
from diffusion.cross_scale_reuse_config import layer_masks


class Tests(unittest.TestCase):
    def model(self):
        torch.manual_seed(41)
        m=DiTMultiRes(depth=24,hidden_size=32,num_heads=4,patch_size=2,num_classes=10,base_size=16).eval()
        with torch.no_grad():
            for p in m.parameters():p.add_(torch.randn_like(p)*.003)
        return m

    def test_mapping(self):
        for f,counts in [('high_heavy',((8,20),(10,25))),('balanced',((16,16),(20,20))),('balanced_plus',((16,18),(20,23)))]:
            for depth,expected in zip((24,30),counts):
                masks=layer_masks(f,depth)
                self.assertEqual(tuple(len(masks[s]) for s in (1,2)),expected)
                for s in (1,2):self.assertFalse(set(masks[s]) & {0,1,depth-2,depth-1})

    def test_dense_stage_parity_and_boundary_gradients(self):
        m=self.model();net=CascadeDMD(m,'dense',False)
        y=torch.tensor([1]);x=torch.randn(1,4,16,16);t=torch.tensor([599])
        raw,_=net.stage(x,t,y,1)
        torch.testing.assert_close(raw,m(x,t,y),atol=0,rtol=0)
        for family in ('high_heavy','balanced','balanced_plus'):
            net=CascadeDMD(m,family,True)
            noises=[torch.randn(1,4,r,r,requires_grad=True) for r in (8,16,32)]
            result=net(noises,y);result['cleans'][-1].square().mean().backward()
            self.assertTrue(all(n.grad is not None and torch.isfinite(n.grad).all() and n.grad.abs().sum()>0 for n in noises))
            for i in set(net.masks[1]) & set(net.masks[2]):
                self.assertGreater(m.blocks[i].attn.qkv.weight.grad.abs().sum().item(),0)
            self.assertEqual(sum(p.numel() for p in net.parameters()),sum(p.numel() for p in m.parameters()))
            m.zero_grad(set_to_none=True)

if __name__=='__main__':unittest.main()
