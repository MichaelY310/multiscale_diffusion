"""CPU operator/gradient tests, no samples or CUDA initialization."""
import json
from pathlib import Path
import unittest
import torch
from root_local_attention_reference import *


class Checks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(91); torch.set_num_threads(2)

    def data(self):
        return [torch.randn(*s,dtype=torch.float64,requires_grad=True) for s in
                [(2,3,18,5),(2,3,18,5),(2,3,18,7),(2,3,11,5),(2,3,11,7)]]

    def test_output_gradients_shared_softmax(self):
        values=self.data(); tiles=spatial_tiles((2,3,3),2,2)
        bias=torch.tensor([-.7,.2,-2.],dtype=torch.float64,requires_grad=True)
        dense=joint_dense(*values,tiles,bias=bias)
        stream=joint_streaming(*values,tiles,bias=bias,chunk=3)
        torch.testing.assert_close(stream,dense,rtol=1e-12,atol=1e-12)
        weight=torch.randn_like(dense)
        gd=torch.autograd.grad((dense*weight).sum(),values+[bias],retain_graph=True)
        gs=torch.autograd.grad((stream*weight).sum(),values+[bias])
        for a,b in zip(gd,gs): torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-11)

    def test_masks_extreme_logits_all_invalid(self):
        values=self.data(); tiles=spatial_tiles((2,3,3),2,2)
        values[0]=values[0]*100
        valid=torch.ones(2,18,dtype=torch.bool); valid[0,::2]=False; valid[1]=False
        root_valid=torch.ones(2,11,dtype=torch.bool); root_valid[1]=False
        a=joint_dense(*values,tiles,bias=10000.,valid=valid,root_valid=root_valid)
        b=joint_streaming(*values,tiles,bias=10000.,valid=valid,root_valid=root_valid,chunk=4)
        torch.testing.assert_close(a,b,rtol=1e-11,atol=1e-11)
        self.assertTrue(torch.isfinite(b).all()); self.assertEqual(float(b[1].detach().abs().sum()),0.)

    def test_bank_mass_initialization(self):
        q=torch.zeros(1,1,4,2,dtype=torch.float64); k=q.clone(); v=torch.zeros_like(q)
        rk=torch.zeros(1,1,12,2,dtype=torch.float64); rv=torch.ones_like(rk)
        out=joint_streaming(q,k,v,rk,rv,(torch.arange(4),),bias=bank_bias(12,4,.2),chunk=3)
        torch.testing.assert_close(out,torch.full_like(out,.2),rtol=1e-12,atol=1e-12)

    def test_original_coarse_addition(self):
        values=self.data(); tiles=spatial_tiles((2,3,3),2,2)
        y=joint_streaming(*values,tiles,chunk=4)
        coarse=torch.randn_like(y); gate=torch.randn_like(y)
        total=joint_streaming(*values,tiles,chunk=4,coarse=coarse,gate=gate)
        torch.testing.assert_close(total,y+coarse*gate)
        valid=torch.ones(2,18,dtype=torch.bool); valid[:,0]=False
        masked=joint_streaming(*values,tiles,chunk=4,coarse=coarse,gate=gate,valid=valid)
        self.assertEqual(float(masked[:,:,0].detach().abs().sum()),0.)

    def test_one_feature_bank_and_current_control(self):
        root=torch.randn(2,11,8,dtype=torch.float64,requires_grad=True)
        wk=torch.randn(8,15,dtype=torch.float64,requires_grad=True)
        wv=torch.randn(8,21,dtype=torch.float64,requires_grad=True)
        bk,bv=layer_bank(root,wk,wv)
        (bk.sum()+bv.sum()).backward()
        self.assertGreater(float(root.grad.abs().sum()),0.)
        values=self.data(); tiles=spatial_tiles((2,3,3),2,2)
        # Same bank values -> same operator, regardless of provenance label.
        ancestral=joint_streaming(*values,tiles,chunk=5)
        current=joint_streaming(*values[:3],values[3].clone(),values[4].clone(),tiles,chunk=5)
        torch.testing.assert_close(ancestral,current,rtol=0,atol=0)

    def test_invalid_tiling(self):
        values=self.data()
        with self.assertRaises(ValueError): joint_streaming(*values,(torch.arange(17),))
        with self.assertRaises(ValueError): joint_streaming(*values,(torch.arange(18),),chunk=0)
        with self.assertRaises(ValueError): joint_streaming(*values,())
        with self.assertRaises(ValueError): layer_bank(torch.randn(2,3,5),torch.randn(6,7),torch.randn(5,7))
        with self.assertRaises(ValueError): Budget(tokens=17).calculate()
        with self.assertRaises(ValueError): Budget(density=float('nan')).calculate()


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report=dict(tests=result.testsRun,passed=result.wasSuccessful(),gpu_run=False,
                trained_quality_measured=False,measured_speedup=None,
                scope='CPU attention/reference gradients only. No Wan integration, GPU execution or training validated.',
                settings={str(r):Budget(root_slots=r).calculate() for r in (3840,960)},
                bias3840=bank_bias(3840,1024,.2),bias960=bank_bias(960,1024,.2),
                provenance='One final ROOT-generating feature; not clean-image encoder KV; layer projection is charged',
                missing=['Common physical RoPE adapter','Actual tile-major Wan integration and native coarse branch',
                         'Triton forward GPU compilation/execution and efficient backward',
                         'Teacher/student fine-stage training','Distributed measurements'])
    report['backward_scope']='Autograd reference only; retains score chunks. No memory-efficient fused/recomputing backward.'
    out=Path('eval_results/progressive_infra_20261003'); out.mkdir(parents=True,exist_ok=True)
    (out/'root_local_cpu_checks.json').write_text(json.dumps(report,indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
