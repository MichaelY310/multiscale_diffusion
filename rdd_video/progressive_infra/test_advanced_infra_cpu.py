"""CPU-only geometry/cost/quantization-contract checks, not speed evidence."""
import json
from dataclasses import replace
from pathlib import Path
import unittest
import torch
from advanced_infra_plan import *


class Checks(unittest.TestCase):
    def test_payload_and_strong_ordinary_control(self):
        b=CSPBudget().calculate()
        self.assertAlmostEqual(b['native_remote_sent_bytes_per_rank_layer']/2**20,168.75)
        self.assertAlmostEqual(b['c_current_summary_remote_sent_bytes_per_rank_layer']/2**20,4.21875)
        self.assertAlmostEqual(b['c_per_layer_payload_reduction'],.975)
        self.assertAlmostEqual(b['root_feature_logical_delivery_bytes_once']/2**20,33.75)
        self.assertAlmostEqual(b['ordinary_current_bank_remote_sent_bytes_per_rank_layer']/2**20,21.09375)
        self.assertEqual(b['ordinary_n64_reuse_summary_extra_bank_bytes'],0)
        self.assertIsNone(b['measured_speedup'])

    def test_one_rank_and_array_contract(self):
        b=CSPBudget(ranks=1).calculate()
        self.assertEqual(b['native_remote_sent_bytes_per_rank_layer'],0)
        self.assertEqual(b['root_feature_logical_delivery_bytes_once'],0)
        no_gate=CSPBudget(native_arrays=4).calculate()
        self.assertAlmostEqual(no_gate['native_remote_sent_bytes_per_rank_layer']/2**20,135.)
        self.assertGreater(b['root_projection_flops_all_ranks'],0)

    def test_windows_are_balanced_and_no_cube_straddles(self):
        shape=(16,48,80); groups=balanced_window_owners(shape)
        flat=[x for group in groups for x in group]
        self.assertEqual(sorted(flat),list(range(61440)))
        self.assertTrue(all(len(group)==15360 for group in groups))
        rank_of={value:rank for rank,group in enumerate(groups) for value in group}
        t,h,w=shape
        for z in range(0,t,4):
            for y in range(0,h,4):
                for x in range(0,w,4):
                    owner={rank_of[(zz*h+yy)*w+xx]
                           for zz in range(z,z+4) for yy in range(y,y+4) for xx in range(x,x+4)}
                    self.assertEqual(len(owner),1)

    def test_bank_key_changes_with_time_layer_and_producer(self):
        key=BankVersion('req1-root','weights1','none',3,522,'physical-center-v1','bf16')
        for kwargs in ({'timestep':757},{'layer':4},{'producer_handle':'req2-root'},
                       {'model_revision':'weights2'},{'lora_revision':'lora1'},
                       {'rope_layout':'different'},{'precision_layout':'nvfp4'}):
            self.assertNotEqual(key,replace(key,**kwargs))

    def test_token_and_pair_and_memory_caps(self):
        budget=research_budget()
        self.assertEqual([x['batch'] for x in budget['stage_capacity_envelope']],[16,4,1])
        self.assertEqual(batch_capacity(10,100,1000,150,99),1)
        self.assertEqual(batch_capacity(10,100,1000,100000,2),2)
        self.assertEqual(batch_capacity(10,100,5,10000,2),0)

    def test_per_bank_centering_must_restore_relative_mass(self):
        # Independent local/root K centering is NOT softmax-invariant.
        q=torch.tensor([[1.,2.]],dtype=torch.float64)
        local=torch.tensor([[1.,0.],[2.,0.]],dtype=torch.float64)
        root=torch.tensor([[0.,3.],[0.,4.]],dtype=torch.float64)
        scale=1/math.sqrt(2); bias=-.7
        logits=torch.cat((q@local.T*scale,q@root.T*scale+bias),-1)
        ml,mr=local.mean(0),root.mean(0)
        centered=torch.cat((q@(local-ml).T*scale,q@(root-mr).T*scale+bias),-1)
        restored=centered+torch.cat(((q@ml*scale).expand(1,2),(q@mr*scale).expand(1,2)),-1)
        torch.testing.assert_close(logits,restored,rtol=1e-14,atol=1e-14)
        self.assertGreater(float((logits.softmax(-1)-centered.softmax(-1)).abs().max()),.1)

    def test_invalid_contracts(self):
        for kwargs in ({'tokens':True},{'ranks':0},{'tokens':61441},{'root_slots':61441}):
            with self.assertRaises(ValueError):CSPBudget(**kwargs).calculate()
        for args in (((16,48,80),2,4),((16,48,80),8,8)):
            with self.assertRaises(ValueError):balanced_window_owners(*args)
        with self.assertRaises(ValueError): batch_capacity(10,0,100,100,1)
        with self.assertRaises(ValueError):BankVersion('a','b','none',0,float('nan'),'r','bf16')


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    report=research_budget()
    report.update(tests=result.testsRun,passed=result.wasSuccessful(),
                  scope='CPU formulas/geometry/cache identities/centering only; no SP or serving runtime.')
    out=Path('eval_results/progressive_infra_20261003')
    (out/'advanced_infra_cpu_checks.json').write_text(json.dumps(report,indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
