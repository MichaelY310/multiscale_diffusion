"""CPU regression tests: capture parity, on-policy targets and cross-stage grads."""
import copy
import sys
from pathlib import Path
import unittest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'DiT'),str(ROOT/'2d/attention_analysis')]
from models_multires import DiTMultiRes
from dmd_cascade_reuse import CascadeDMD
from dmd_hidden_distillation import (HIDDEN_LAYERS, normalized_hidden_mse, pool_hidden,
                                     hidden_distillation_loss, hidden_config, hidden_weight_at)


class HiddenTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(1949)

    def model(self, family='high_heavy', checkpoint=True):
        m=DiTMultiRes(depth=24,hidden_size=32,num_heads=4,patch_size=2,num_classes=10,base_size=16).eval()
        with torch.no_grad():
            for p in m.parameters(): p.add_(torch.randn_like(p)*.003)
        return CascadeDMD(m,family,checkpoint)

    def inputs(self):
        return [torch.randn(1,4,r,r,requires_grad=True) for r in (8,16,32)],torch.tensor([2])

    def test_capture_preserves_outputs_rng_state_dict_and_gradients(self):
        plain=self.model(); tapped=copy.deepcopy(plain)
        noises,y=self.inputs()
        before=torch.get_rng_state().clone()
        old=plain(noises,y); new=tapped(noises,y,hidden_layers=HIDDEN_LAYERS)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertEqual(set(plain.state_dict()),set(tapped.state_dict()))
        for name in ('inputs','cleans','raws'):
            for x,z in zip(old[name],new[name]): torch.testing.assert_close(x,z,atol=0,rtol=0)
        old['cleans'][-1].square().mean().backward()
        new['cleans'][-1].square().mean().backward()
        for p,q in zip(plain.parameters(),tapped.parameters()):
            if p.grad is None: self.assertIsNone(q.grad)
            else: torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)

    def test_taps_are_actual_post_block_hidden(self):
        net=self.model('dense',False); recorded={}
        handles=[net.backbone.blocks[i].register_forward_hook(
            lambda module,args,out,i=i: recorded.update({i:out})) for i in (21,23)]
        x=torch.randn(1,4,32,32); y=torch.tensor([1]); taps={}
        net.clean_at(x,y,2,capture_layers=(21,23),hidden_out=taps)
        for h in handles:h.remove()
        for i in (21,23):torch.testing.assert_close(taps[i],recorded[i],atol=0,rtol=0)

    def test_hidden_loss_trains_producers_through_boundaries(self):
        student=self.model(); teacher=self.model('dense',False).requires_grad_(False)
        noises,y=self.inputs(); pred=student(noises,y,hidden_layers=HIDDEN_LAYERS)
        refs=[{},{},{}]
        with torch.no_grad():
            for stage in (1,2):
                teacher.clean_at(pred['inputs'][stage].detach(),y,stage,
                                 capture_layers=HIDDEN_LAYERS[stage],hidden_out=refs[stage])
        value,terms=hidden_distillation_loss(pred['hidden'],refs)
        value.backward()
        self.assertEqual(len(terms),3)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        for noise in noises:
            self.assertTrue(torch.isfinite(noise.grad).all())
            self.assertGreater(noise.grad.abs().sum().item(),0)
        for block in student.backbone.blocks:
            grad=block.attn.qkv.weight.grad
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(),0)

    def test_stop_gradient_normalization_and_zero_power(self):
        s=torch.full((2,4,3),2.,requires_grad=True); t=torch.ones_like(s,requires_grad=True)
        value=normalized_hidden_mse(s,t)
        self.assertEqual(value.item(),1.)
        value.backward(); self.assertIsNone(t.grad)
        self.assertGreater(s.grad.abs().sum().item(),0)
        self.assertTrue(torch.isfinite(normalized_hidden_mse(s,torch.zeros_like(s))))
        torch.testing.assert_close(normalized_hidden_mse(s*3,t*3),value)

    def test_lowpass_excludes_cell_internal_detail(self):
        base=torch.ones(1,4,1)
        checker=torch.tensor([1.,-1.,-1.,1.]).view(1,4,1)
        self.assertEqual(normalized_hidden_mse(pool_hidden(base+checker),pool_hidden(base)).item(),0.)
        self.assertGreater(normalized_hidden_mse(base+checker,base).item(),0.)
        with self.assertRaises(ValueError):pool_hidden(torch.ones(1,9,1),2)

    def test_ramp_and_default_compatibility(self):
        self.assertEqual(hidden_config(0,500),{'enabled':False})
        self.assertAlmostEqual(hidden_weight_at(250,.05,500),.025)
        self.assertAlmostEqual(hidden_weight_at(1000,.05,500),.05)
        self.assertAlmostEqual(hidden_weight_at(1,.05,0),.05)
        for invalid in (-1,float('nan')):
            with self.assertRaises(ValueError):hidden_config(invalid,500)
        with self.assertRaises(ValueError):hidden_config(.05,-1)


if __name__=='__main__':unittest.main()
