import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'2d/attention_analysis'))
from summarize_csra_jump_loss_2x2 import check_checkpoint, same_reference, interaction, PARENT_SHA, FIELDS


class FactorialSummaryTests(unittest.TestCase):
    def config(self):
        return dict(step=2400,checkpoint='checkpoints/RDD-CSRA-prefix6-jump-loss-jump-aligned-L400K-20260914/latest.pt',
            args=dict(coarse_source='aligned',jump_loss_mode='jump',bridge_kind='conditional_attention',prefix_depth=6,seed=20279000),
            jump_supervision=dict(name='prefix6_jump_loss_2x2_v1',loss_mode='jump',parent_step=1600,
                fixed_endpoint=2400,extra_steps=800,parent_sha256=PARENT_SHA['aligned'],calibration_sha256='cal',
                per_gpu_batch=64,world_size=4,accumulation=1))

    def test_fixed_latest_contract(self):
        c=self.config();check_checkpoint(c,'jump','aligned','cal')
        for key,value in (('checkpoint',c['checkpoint'].replace('latest.pt','best.pt')),('step',2300)):
            bad=copy.deepcopy(c);bad[key]=value
            with self.assertRaises(ValueError):check_checkpoint(bad,'jump','aligned','cal')
        with self.assertRaises(ValueError):check_checkpoint(c,'old','aligned','cal')
        with self.assertRaises(ValueError):check_checkpoint(c,'jump','zero','cal')
        with self.assertRaises(ValueError):check_checkpoint(c,'jump','aligned','wrong')

    def test_unmatched_baseline_rejected(self):
        rows=[dict(image=0,image_sha256='a',target=3)]
        g={k:dict(original_dense=copy.deepcopy(rows)) for k in ('parent','old','jump')}
        same_reference(g)
        g['jump']['original_dense'][0]['image_sha256']='b'
        with self.assertRaises(ValueError):same_reference(g)

    def test_interaction_is_paired_difference_of_differences(self):
        def row(value):return {f:value for f in FIELDS}
        g=dict(old=dict(aligned=[row(1),row(1)],zero=[row(0),row(0)]),
               jump=dict(aligned=[row(1),row(1)],zero=[row(1),row(1)]))
        result=interaction(g,repetitions=100)
        for field in FIELDS:
            self.assertEqual(result[field]['mean_difference'],-1.)
            self.assertEqual(result[field]['paired_bootstrap_95'],[-1.,-1.])


if __name__=='__main__':unittest.main()
