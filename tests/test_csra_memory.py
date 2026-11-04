import copy
import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '2d' / 'attention_analysis'))
from train_rdd_csra_adapter import CSRAAdapterBank, initialize_adapter_weights, upsample_tokens
from csra_joint_distillation import PreservedCoarseStudent
from test_csra_output_distillation import TinyTeacher


class MemoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(813)
        self.old = CSRAAdapterBank([1,3], 16, 4, 'fine_context_v2')
        for p in self.old.parameters():
            torch.nn.init.normal_(p, std=.05)
        self.new = CSRAAdapterBank([1,3], 16, 4, 'coarse_memory_v5')
        initialize_adapter_weights(self.new, {'layers':[1,3], 'hidden_size':16,
            'rank':4, 'adapter_architecture':'fine_context_v2', 'adapter':self.old.state_dict()})

    def test_zero_readout_warmstart_preserves_v2_exactly(self):
        fine, coarse, t = torch.randn(2,16,16), torch.randn(2,4,16), torch.tensor([0,299])
        for layer in [1,3]:
            expected = self.old.adapters[str(layer)](upsample_tokens(coarse,4),fine,t)
            actual = self.new.adapters[str(layer)](coarse,fine,t)
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)

    def test_global_read_has_fine_query_and_native_source_gradients(self):
        module = self.new.adapters['1']
        torch.nn.init.normal_(module.memory_out.weight,std=.1)
        fine = torch.randn(2,16,16,requires_grad=True)
        coarse = torch.randn(2,4,16,requires_grad=True)
        module(coarse,fine,torch.tensor([80,200])).square().mean().backward()
        for p in (fine,coarse,module.memory_q.weight,module.memory_k.weight,module.memory_v.weight):
            self.assertIsNotNone(p.grad)
            self.assertGreater(float(p.grad.abs().sum()),0.)

    def test_rejects_upsampled_or_wrong_shape_memory(self):
        with self.assertRaises(ValueError):
            self.new.adapters['1'](torch.randn(2,16,16),torch.randn(2,16,16),torch.ones(2))

    def test_joint_native_cache_and_zero_control_gradient_paths(self):
        model = TinyTeacher()
        aligned = PreservedCoarseStudent(copy.deepcopy(model),copy.deepcopy(self.new),True,False,'aligned')
        zero = PreservedCoarseStudent(copy.deepcopy(model),copy.deepcopy(self.new),True,False,'zero')
        batch = {'x':torch.randn(2,4,4,4),'t':torch.tensor([10,20]),'labels':torch.tensor([1,2]),
                 'coarse':{l:torch.zeros(2,4,16) for l in [1,3]}}
        ya = aligned(batch)
        yz = zero({**batch,'coarse':{l:None for l in [1,3]}})
        torch.testing.assert_close(ya,yz,atol=0,rtol=0)
        for net, output in ((aligned,ya),(zero,yz)):
            output.square().mean().backward()
            self.assertTrue(all(p.grad is not None for p in net.parameters() if p.requires_grad))


if __name__ == '__main__':
    unittest.main()
