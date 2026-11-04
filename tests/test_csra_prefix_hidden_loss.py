import copy
import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'2d/attention_analysis'))
import test_csra_prefix_student as fixtures
from csra_prefix_hidden_loss import teacher_prefix_hidden,hidden_loss_per_image
from probe_rdd_vs_dit_block_reuse import manual_forward


class HiddenLossTests(unittest.TestCase):
    def test_optional_output_and_gradient(self):
        model,_=fixtures.PrefixStudentTests().make()
        batch=dict(x=torch.randn(2,4,8,8),t=torch.tensor([299,40]),labels=torch.tensor([1,2]),
                   coarse={1:torch.randn(2,4,16)})
        old=model(batch);new,hidden=model(batch,return_hidden=True)
        torch.testing.assert_close(old,new,atol=0,rtol=0)
        hidden_loss_per_image(hidden,torch.ones_like(hidden)).mean().backward()
        self.assertGreater(float(model.proj.weight.grad.abs().sum()),0.)
        self.assertIsNone(model.backbone.blocks[0].attn.qkv.weight.grad)

    def test_teacher_target_parity_and_frozen(self):
        model,_=fixtures.PrefixStudentTests().make()
        teacher=copy.deepcopy(model.backbone).eval().requires_grad_(False)
        batch=dict(x=torch.randn(2,4,8,8),t=torch.tensor([299,40]),labels=torch.tensor([1,2]))
        actual=teacher_prefix_hidden(teacher,batch,2)
        _,trace=manual_forward(teacher,batch['x'],batch['t'],batch['labels'],multires=True,capture_layers={1})
        torch.testing.assert_close(actual,trace[1]['output'],atol=0,rtol=0)
        self.assertFalse(actual.requires_grad)
        teacher.requires_grad_(True)
        with self.assertRaises(ValueError):teacher_prefix_hidden(teacher,batch,2)

    def test_normalization_and_detached_target(self):
        s=torch.ones(2,4,3,requires_grad=True);t=torch.full_like(s,2.,requires_grad=True)
        value=hidden_loss_per_image(s,t)
        torch.testing.assert_close(value,torch.full((2,),.25))
        torch.testing.assert_close(value,hidden_loss_per_image(3*s,3*t))
        value.mean().backward();self.assertIsNone(t.grad);self.assertIsNotNone(s.grad)


if __name__=='__main__':unittest.main()
