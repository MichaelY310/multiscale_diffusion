import copy
import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'2d'/'attention_analysis'))
from train_rdd_csra_adapter import DiTMultiRes_models
from models_multires import DiTMultiRes
from csra_prefix_student import (PrefixLiftStudent,ConditionalPrefixStudent,
                                check_prefix_initializer,prefix_student_class)
from csra_prefix_probe_math import predict


class PrefixStudentTests(unittest.TestCase):
    def test_rejects_initializer_for_different_target_depth(self):
        old={'version':'prefix_information_v1'}
        check_prefix_initializer(old,14)
        with self.assertRaises(ValueError): check_prefix_initializer(old,8)
        new={**old,'prefix_depth':8,'source_layer':13}
        check_prefix_initializer(new,8)
        with self.assertRaises(ValueError): check_prefix_initializer(new,14)
        with self.assertRaises(ValueError): check_prefix_initializer(new,8,7)

    def make(self):
        torch.manual_seed(481)
        m=DiTMultiRes(hidden_size=16,depth=4,num_heads=4,patch_size=2,num_classes=10,base_size=4)
        for p in m.parameters(): torch.nn.init.normal_(p,std=.04)
        fit={'mean':torch.randn(32).double(),'scale':torch.ones(32,dtype=torch.float64),
             'weight':torch.randn(32,16,dtype=torch.float64)*.05,'output_mean':torch.randn(16,dtype=torch.float64)*.05}
        s=PrefixLiftStudent(m,fit,'aligned',prefix_depth=2,source_layer=1,checkpoint_blocks=False)
        return s,fit

    def test_common_initial_function_and_gradients(self):
        aligned,_=self.make()
        zero=copy.deepcopy(aligned); zero.coarse_source='zero'
        batch={'x':torch.randn(2,4,8,8),'t':torch.tensor([299,40]),'labels':torch.tensor([1,2]),
               'coarse':{1:torch.randn(2,4,16)}}
        torch.testing.assert_close(aligned(batch),zero(batch),atol=0,rtol=0)
        aligned(batch).square().mean().backward()
        self.assertTrue(all(p.grad is not None for p in aligned.parameters() if p.requires_grad))
        self.assertTrue(all(p.grad is None for p in aligned.backbone.blocks[0].parameters()))
        self.assertGreater(float(aligned.proj.weight.grad[:,32:].abs().sum()),0.)

    def test_raw_affine_initialization_matches_standardized_probe(self):
        student,fit=self.make()
        h,c,source=torch.randn(2,16,16),torch.randn(2,16),torch.randn(2,4,16)
        x=torch.cat((h,c[:,None].expand_as(h)),-1).double()
        expected=h.double()+predict(x,fit)
        torch.testing.assert_close(student.lift(h,c,source).double(),expected,atol=2e-6,rtol=2e-6)

    def test_conditional_bridge_common_initial_function(self):
        old,fit=self.make()
        new=ConditionalPrefixStudent(copy.deepcopy(old.backbone),fit,'aligned',prefix_depth=2,
                                     source_layer=1,checkpoint_blocks=False)
        zero=copy.deepcopy(new); zero.coarse_source='zero'
        batch={'x':torch.randn(2,4,8,8),'t':torch.tensor([299,40]),'labels':torch.tensor([1,2]),
               'coarse':{1:torch.randn(2,4,16)}}
        torch.testing.assert_close(old(batch),new(batch),atol=0,rtol=0)
        torch.testing.assert_close(new(batch),zero(batch),atol=0,rtol=0)
        new(batch).square().mean().backward()
        self.assertTrue(all(p.grad is not None for p in new.parameters() if p.requires_grad))
        self.assertGreater(float(new.read_out.weight.grad.abs().sum()),0.)
        self.assertGreater(float(new.nonlinear[-1].weight.grad.abs().sum()),0.)

    def test_factory_is_opt_in(self):
        self.assertIs(prefix_student_class('linear'),PrefixLiftStudent)
        self.assertIs(prefix_student_class('conditional_attention'),ConditionalPrefixStudent)
        with self.assertRaises(ValueError): prefix_student_class('unknown')

    def test_six_block_prefix_preserves_suffix_and_common_initialization(self):
        old,fit=self.make()
        backbone=copy.deepcopy(old.backbone)
        backbone.blocks=torch.nn.ModuleList([copy.deepcopy(b) for b in list(backbone.blocks)*2])
        a=ConditionalPrefixStudent(backbone,fit,'aligned',prefix_depth=6,
                                   source_layer=1,checkpoint_blocks=False)
        z=copy.deepcopy(a);z.coarse_source='zero'
        batch=dict(x=torch.randn(2,4,8,8),t=torch.tensor([299,40]),labels=torch.tensor([1,2]),
                   coarse={1:torch.randn(2,4,16)})
        calls=[]
        handles=[b.register_forward_hook(lambda m,i,o,j=j:calls.append(j)) for j,b in enumerate(a.backbone.blocks)]
        out=a(batch)
        for handle in handles:handle.remove()
        self.assertEqual(calls,[6,7])
        torch.testing.assert_close(out,z(batch),atol=0,rtol=0)
        out.square().mean().backward()
        self.assertTrue(all(p.grad is None for b in a.backbone.blocks[:6] for p in b.parameters()))
        self.assertTrue(all(p.grad is not None for p in a.parameters() if p.requires_grad))


if __name__=='__main__':unittest.main()
