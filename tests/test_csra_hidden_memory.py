import copy
import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'2d'/'attention_analysis'))
from train_rdd_csra_adapter import CSRAAdapterBank, initialize_adapter_weights
from csra_joint_distillation import PreservedCoarseStudent
from test_csra_output_distillation import TinyTeacher


class HiddenMemoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1861)
        self.old=CSRAAdapterBank([1,3],16,4,'fine_context_v2')
        for p in self.old.parameters():
            torch.nn.init.normal_(p,std=.05)
        self.new=CSRAAdapterBank([1,3],16,4,'hidden_memory_v6')
        initialize_adapter_weights(self.new,{'layers':[1,3],'hidden_size':16,'rank':4,
            'adapter_architecture':'fine_context_v2','adapter':self.old.state_dict()})

    def test_warmstart_exact_zero_source_even_with_nonzero_h(self):
        fine,h,t=torch.randn(2,16,16),torch.randn(2,4,16)*20,torch.tensor([0,299])
        for l in [1,3]:
            expected=self.old.adapters[str(l)](torch.zeros_like(fine),fine,t)
            actual=self.new.adapters[str(l)](h,fine,t)
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            self.assertEqual(self.new.adapters[str(l)].coarse_feature,'output')

    def test_hidden_is_used_only_through_memory_and_can_receive_gradient(self):
        module=self.new.adapters['1']
        torch.nn.init.normal_(module.memory_out.weight,std=.1)
        fine=torch.randn(2,16,16,requires_grad=True)
        h=torch.randn(2,4,16,requires_grad=True)
        module(h,fine,torch.tensor([299,84])).square().mean().backward()
        for p in (h,fine,module.memory_q.weight,module.memory_k.weight,module.memory_v.weight):
            self.assertIsNotNone(p.grad)
            self.assertGreater(float(p.grad.abs().sum()),0.)

    def test_full_student_initialized_equivalence_and_gradients(self):
        backbone=TinyTeacher()
        old=PreservedCoarseStudent(copy.deepcopy(backbone),self.old,True,False,'zero')
        new=PreservedCoarseStudent(copy.deepcopy(backbone),self.new,True,False,'aligned')
        batch={'x':torch.randn(2,4,4,4),'t':torch.tensor([0,299]),'labels':torch.tensor([1,2]),
               'coarse':{l:torch.randn(2,4,16) for l in [1,3]}}
        actual=new(batch)
        torch.testing.assert_close(actual,old(batch),atol=0,rtol=0)
        actual.square().mean().backward()
        self.assertTrue(all(p.grad is not None for p in new.parameters() if p.requires_grad))


if __name__=='__main__':
    unittest.main()
