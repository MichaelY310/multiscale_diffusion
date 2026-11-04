import copy
import sys
import unittest
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DistributedSampler
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'2d/attention_analysis'))
from csra_output_distillation import prediction_loss
from csra_jump_weighted_loss import TRAIN_TIMES, make_calibration, weighted_prediction_loss, weight_table
from audit_csra_jump_head_sensitivity import audit
from train_csra_prefix6_jump_loss import check_parent, check_optimizer, check_resume


class JumpWeightedLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cal = make_calibration(audit())

    def test_unit_weights_reproduce_old_loss_and_gradients(self):
        for batch in (1, 8):
            p = torch.randn(batch,8,8,8,requires_grad=True)
            q = p.detach().clone().requires_grad_(True)
            target = torch.randn_like(p)
            old, _ = prediction_loss(p,target)
            new, _ = weighted_prediction_loss(q,target,torch.ones(batch,2))
            old.backward();new.backward()
            torch.testing.assert_close(old,new,rtol=2e-7,atol=2e-7)
            torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)

    def test_isotropic_normalizer_common_expected_scale(self):
        self.assertEqual(self.cal['train_times'],list(TRAIN_TIMES))
        self.assertAlmostEqual(sum(.25*w['low']+.75*w['high'] for w in self.cal['weights'])/8,1.,places=12)
        table = weight_table(self.cal,'cpu')
        self.assertTrue(torch.isnan(table[1]).all())
        self.assertEqual(table[0].sum(),0.)

    def test_loss_matches_actual_jump_response_on_mixed_field(self):
        table = weight_table(self.cal,'cpu')
        gains = {r['t']:r for r in self.cal['gains']}
        p = torch.randn(8,8,8,8,requires_grad=True)
        target = torch.randn_like(p)
        error = p[:,:4]-target[:,:4]
        low = F.avg_pool2d(error,2).repeat_interleave(2,-2).repeat_interleave(2,-1)
        gl = torch.tensor([gains[t]['low_gain'] for t in TRAIN_TIMES])[:,None,None,None]
        gh = torch.tensor([gains[t]['high_gain'] for t in TRAIN_TIMES])[:,None,None,None]
        actual = (gl*low+gh*(error-low)).square().mean()/self.cal['normalizer']
        actual += .05*(p[:,4:]-target[:,4:]).square().mean()
        loss,_ = weighted_prediction_loss(p,target,table[torch.tensor(TRAIN_TIMES)])
        torch.testing.assert_close(loss,actual,atol=5e-7,rtol=5e-7)
        loss.backward();self.assertTrue(torch.isfinite(p.grad).all())

    def test_final_identity_only_trains_variance(self):
        p = torch.randn(2,8,8,8,requires_grad=True)
        target = torch.randn_like(p)
        weights = torch.zeros(2,2,requires_grad=True)
        loss,_ = weighted_prediction_loss(p,target,weights)
        torch.testing.assert_close(loss,.05*(p[:,4:]-target[:,4:]).square().mean(),atol=0,rtol=0)
        loss.backward()
        self.assertEqual(p.grad[:,:4].abs().sum(),0.)
        self.assertGreater(float(p.grad[:,4:].abs().sum()),0.)
        self.assertIsNone(weights.grad)

    def test_invalid_weights_and_shapes_rejected(self):
        p = torch.zeros(2,8,8,8)
        for w in (torch.ones(2,3),torch.full((2,2),-1.),torch.full((2,2),float('nan'))):
            with self.assertRaises(ValueError):weighted_prediction_loss(p,p,w)
        with self.assertRaises(ValueError):
            weighted_prediction_loss(p[:,:,:7],p[:,:,:7],torch.ones(2,2))

    def test_bad_sampler_calibration_rejected(self):
        for key in ('low_gain','high_gain'):
            audit_data = {'rows':copy.deepcopy(self.cal['gains'])}
            audit_data['rows'][-1][key] = 1.
            with self.assertRaises(ValueError):make_calibration(audit_data)


class JumpResumeTests(unittest.TestCase):
    def parent(self):
        return dict(step=1600,prefix_depth=6,source_layer=13,format='csra_prefix_conditional_v8',args=dict(
            coarse_source='aligned',bridge_kind='conditional_attention',seed=20279000,
            base_ckpt='checkpoints/RDD-Unified-L/0400000.pt',data_root='eval_results/csra_rollout_data4k_20260912',
            hidden_root='eval_results/csra_hidden_sources_20260913',
            initializer='eval_results/csra_prefix6_information_20260913/selected_ridge.pt'))

    def test_parent_and_resume_reject_crossed_arms(self):
        parent = self.parent();check_parent(parent,'aligned')
        with self.assertRaises(ValueError):check_parent(parent,'zero')
        bad = copy.deepcopy(parent);bad['step']=1500
        with self.assertRaises(ValueError):check_parent(bad,'aligned')
        e = dict(loss_mode='jump',parent_step=1600)
        ck = dict(parent,jump_supervision=e,initializer_sha256='correct')
        check_resume(ck,e,parent['args'],'correct')
        for bad_e in (dict(e,loss_mode='old'),dict(e,parent_step=800)):
            with self.assertRaises(ValueError):check_resume(ck,bad_e,parent['args'],'correct')

    def test_optimizer_states_and_hyperparameters_preserved(self):
        params = [torch.nn.Parameter(torch.ones(2)) for _ in range(2)]
        opt = torch.optim.AdamW([{'params':[params[0]],'lr':1e-4},{'params':[params[1]],'lr':1e-5}],weight_decay=.01)
        sum(p.square().sum() for p in params).backward();opt.step()
        saved = copy.deepcopy(opt.state_dict())
        opt.load_state_dict(saved);check_optimizer(opt)
        for state in opt.state.values():self.assertEqual(state['step'],1.)
        opt.param_groups[0]['lr']=1e-3
        with self.assertRaises(ValueError):check_optimizer(opt)

    def test_four_rank_restored_data_offset(self):
        epoch,offset=divmod(1600,128)
        self.assertEqual((epoch,offset),(12,64))
        seen=[]
        for rank in range(4):
            sampler=DistributedSampler(range(32768),4,rank,seed=20279000,drop_last=True)
            sampler.set_epoch(epoch);indices=list(iter(sampler))
            resumed=indices[offset*64:(offset+1)*64]
            replay=list(iter(sampler))[64*64:65*64]
            self.assertEqual(resumed,replay);seen+=resumed
        self.assertEqual(len(set(seen)),256)


if __name__=='__main__':unittest.main()
