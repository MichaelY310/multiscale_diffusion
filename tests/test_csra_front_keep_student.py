import copy
import sys
import unittest
from argparse import Namespace
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'2d'/'attention_analysis'))
from csra_front_keep_student import FrontKeepStudent,FORMAT,check_resume,check_initializer
from csra_prefix_student import ConditionalPrefixStudent
from probe_csra_front_keep import common,layout
from train_rdd_csra_adapter import DiTMultiRes_models
from models_multires import DiTMultiRes


class FrontStudentTests(unittest.TestCase):
    def make(self):
        torch.manual_seed(923)
        backbone=DiTMultiRes(hidden_size=16,depth=4,num_heads=4,patch_size=2,num_classes=10,base_size=4)
        for p in backbone.parameters():torch.nn.init.normal_(p,std=.04)
        fit=dict(mean=torch.zeros(32,dtype=torch.float64),scale=torch.ones(32,dtype=torch.float64),
                 weight=torch.randn(32,16,dtype=torch.float64)*.01,output_mean=torch.zeros(16,dtype=torch.float64))
        model=FrontKeepStudent(backbone,fit,'aligned',prefix_depth=2,source_layer=1,checkpoint_blocks=False)
        batch=dict(x=torch.randn(2,4,8,8),t=torch.tensor([299,40]),labels=torch.tensor([1,2]),
                   coarse={1:torch.randn(2,4,16)})
        return model,batch,fit

    def test_calls_gradients_and_equal_budget(self):
        model,batch,fit=self.make()
        calls=[]
        handles=[b.register_forward_hook(lambda m,i,o,j=j:calls.append(j)) for j,b in enumerate(model.backbone.blocks)]
        pred=model(batch)
        for h in handles:h.remove()
        self.assertEqual(calls,[0,3])
        pred.square().mean().backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters() if p.requires_grad))
        self.assertTrue(all(p.grad is None for b in model.backbone.blocks[1:3] for p in b.parameters()))
        old=ConditionalPrefixStudent(copy.deepcopy(model.backbone),fit,'aligned',prefix_depth=2,source_layer=1)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),
                         sum(p.numel() for p in old.parameters() if p.requires_grad))

    def test_common_source_zero_initial_function_and_source_gradients(self):
        model,batch,_=self.make()
        zero=copy.deepcopy(model);zero.coarse_source='zero'
        torch.testing.assert_close(model(batch),zero(batch),atol=0,rtol=0)
        model(batch).square().mean().backward()
        self.assertGreater(float(model.proj.weight.grad[:,32:].abs().sum()),0.)
        self.assertGreater(float(model.read_out.weight.grad.abs().sum()),0.)

    def test_checkpointed_blocks_match(self):
        a,batch,_=self.make();b=copy.deepcopy(a);b.checkpoint_blocks=True
        ya,yb=a(batch),b(batch)
        torch.testing.assert_close(ya,yb,atol=0,rtol=0)
        ya.square().mean().backward();yb.square().mean().backward()
        for pa,pb in zip(a.parameters(),b.parameters()):
            if pa.requires_grad:torch.testing.assert_close(pa.grad,pb.grad,atol=0,rtol=0)

    def test_resume_and_initializer_guard(self):
        init={**common(),**layout(1),'fits':{'fine_only':{'weight':torch.zeros(2048,1024)}}}
        check_initializer(init)
        with self.assertRaises(ValueError):check_initializer({**init,**layout(0)})
        args=Namespace(base_ckpt='base',data_root='data',hidden_root='hidden',initializer='init',coarse_source='aligned',seed=7)
        payload=dict(format=FORMAT,front_keep=1,prefix_depth=6,source_layer=13,initializer_sha256='sha',args=vars(args))
        check_resume(payload,args,'sha')
        for key,value in [('front_keep',0),('format','csra_prefix_conditional_v8'),('initializer_sha256','other')]:
            with self.assertRaises(ValueError):check_resume({**payload,key:value},args,'sha')


if __name__=='__main__':unittest.main()
