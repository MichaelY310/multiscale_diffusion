import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"2d"/"attention_analysis"))
from train_rdd_csra_adapter import CSRAAdapterBank, initialize_adapter_weights


class FullChannelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(482)
        self.v2=CSRAAdapterBank([1],16,4,"fine_context_v2")
        for name,p in self.v2.named_parameters():
            torch.nn.init.normal_(p,std=.1)
        self.v4=CSRAAdapterBank([1],16,4,"full_channel_v4")
        payload={"layers":[1],"hidden_size":16,"rank":4,"adapter_architecture":"fine_context_v2",
                 "adapter":self.v2.state_dict()}
        initialize_adapter_weights(self.v4,payload)
        self.coarse=torch.randn(2,16,16)
        self.fine=torch.randn(2,16,16)
        self.t=torch.tensor([30,290])

    def test_zero_warm_start_exact_output_and_input_gradient(self):
        a=self.fine.clone().requires_grad_(True)
        b=self.fine.clone().requires_grad_(True)
        ya=self.v2.adapters['1'](self.coarse,a,self.t)
        yb=self.v4.adapters['1'](self.coarse,b,self.t)
        torch.testing.assert_close(ya,yb,atol=0,rtol=0)
        ya.square().mean().backward(); yb.square().mean().backward()
        torch.testing.assert_close(a.grad,b.grad,atol=1e-7,rtol=1e-6)
        grad=self.v4.adapters['1'].diag_time_mlp[-1].weight.grad
        self.assertGreater(float(grad.abs().sum()),0)

    def test_full_channel_branch_is_exact_diagonal_affine(self):
        module=self.v4.adapters['1']
        with torch.no_grad():
            module.diag_time_mlp[-1].bias[:16].fill_(.25)
            module.diag_time_mlp[-1].bias[16:].fill_(.1)
        delta=module(self.coarse,self.fine,self.t)-self.v2.adapters['1'](self.coarse,self.fine,self.t)
        expected=.25*module.fine_norm(self.fine)+.1
        torch.testing.assert_close(delta,expected,atol=1e-6,rtol=1e-5)

    def test_old_checkpoint_keys_and_new_parameter_budget(self):
        self.assertFalse(any('diag_' in k for k in self.v2.state_dict()))
        extra=sum(p.numel() for p in self.v4.parameters())-sum(p.numel() for p in self.v2.parameters())
        self.assertEqual(extra,64*64+64+64*(2*16)+2*16)


if __name__=="__main__": unittest.main()
