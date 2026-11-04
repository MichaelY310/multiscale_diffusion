import sys
from pathlib import Path
import unittest
from types import SimpleNamespace
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'2d/attention_analysis'))
from aggressive_direct_reuse import MASKS
from rdd_video.attention_analysis.one_call_cascade_controller import OneCallCascadeController


class AggressiveReuseTest(unittest.TestCase):
    def test_masks_keep_end_blocks(self):
        for key,n in [('reuse12',12),('reuse18',18)]:
            self.assertEqual(len(MASKS[key]),n)
            self.assertEqual(len(set(MASKS[key])),n)
            self.assertTrue(all(0<i<23 for i in MASKS[key]))

    def make_controller(self,optimized):
        c=OneCallCascadeController.__new__(OneCallCascadeController)
        c.model=SimpleNamespace(patch_size=(1,2,2),blocks=[None])
        c.lifetime_optimized=optimized
        c.cross_stage_reuse_layers=frozenset({0});c.cross_stage_reuse_layers_by_stage={}
        c.cross_stage_reuse_stages=frozenset({1,2})
        c.cross_stage_reuse_scale=1.;c.cross_stage_reuse_scale_by_stage={}
        c.cross_stage_reuse_layer_coefficients={};c.cross_stage_reuse_layer_coefficients_by_stage={}
        c.cross_stage_reuse_precompose=False;c.cross_stage_reuse_resize='nearest'
        c.cross_stage_precomposed_cache={};c.attention_mode='vsa';c.routes={}
        c.cross_stage_residual_cache={(0,0):((2,2,2),torch.arange(6.).reshape(1,2,3))}
        c.stats={'cross_stage_block_reuse_hits':0}
        c.seen_stages=set();c.shape_steps={}
        return c

    def test_consumed_cache_eviction_is_value_preserving(self):
        old,new=self.make_controller(False),self.make_controller(True)
        for stage,shape,n in [(1,(2,4,4),8),(2,(2,8,8),32)]:
            h=torch.randn(1,n,3);outputs=[]
            for c in (old,new):
                c.begin_model_forward(stage,shape)
                outputs.append(c._dispatch_block(0,None,h,torch.empty(0),torch.empty(0),
                                                 (torch.empty(0),torch.empty(0)),n))
            torch.testing.assert_close(*outputs,atol=0,rtol=0)
        self.assertEqual(new.cross_stage_residual_cache,{})
        self.assertGreater(len(old.cross_stage_residual_cache),0)
        with self.assertRaises(ValueError): new.begin_model_forward(2,(2,8,8))


if __name__=='__main__': unittest.main()
