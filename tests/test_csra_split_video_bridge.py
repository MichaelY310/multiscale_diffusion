"""Scope and reset tests, independent of GPU or FastVideo checkpoint."""
import sys
from pathlib import Path
import unittest
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rdd_video'))
from test_csra_prefix_video_bridge import FakeModel, FakeBlock
from attention_analysis.csra_split_video_bridge import VideoSpanSamplerProxy


class FakeDeepModel(FakeModel):
    def __init__(self):
        super().__init__()
        self.blocks=nn.ModuleList([FakeBlock() for _ in range(14)])


class CountingReader(nn.Module):
    def __init__(self):
        super().__init__(); self.marker=nn.Parameter(torch.zeros(())); self.inputs=[]
    def forward(self,h,c,memory,sg,tg):
        self.inputs.append(float(h.mean()))
        assert float(memory.mean())==14. and c.shape==(1,8)
        return h+3.


class Tests(unittest.TestCase):
    def test_split_real_intermediate_blocks_and_no_leak(self):
        readers=[CountingReader(),CountingReader()]
        p=VideoSpanSamplerProxy(FakeDeepModel(),FakeDeepModel(),readers)
        for variant in ('baseline','split3_3','baseline','split3_3'):
            p.reset(variant)
            for stage,shape in enumerate(((1,1,1),(1,2,2),(1,4,4))):
                p.begin_model_forward(stage,shape)
                out=p(hidden_states=torch.zeros(1,shape[0]*shape[1]*shape[2],8),
                    encoder_hidden_states=torch.zeros(1,2,8),timestep=torch.zeros(1))
                torch.testing.assert_close(out,torch.full_like(out,14),atol=0,rtol=0)
            audit=p.audit()
            self.assertEqual(audit['bridge_calls'],2 if variant=='split3_3' else 0)
            self.assertIsNone(p.memory)
        self.assertEqual(readers[0].inputs,[0.,0.])
        self.assertEqual(readers[1].inputs,[6.,6.])

    def test_prefix_counts_and_order_validation(self):
        p=VideoSpanSamplerProxy(FakeDeepModel(),FakeDeepModel(),[CountingReader(),CountingReader()])
        p.reset('prefix6')
        with self.assertRaises(ValueError):p.begin_model_forward(2,(1,4,4))
        for stage in range(3):
            p.begin_model_forward(stage,(1,1,1))
            p(hidden_states=torch.zeros(1,1,8),encoder_hidden_states=torch.zeros(1,2,8),timestep=torch.zeros(1))
        self.assertEqual(p.audit()['executed'],[14,14,8])
        self.assertEqual(p.audit()['bridge_calls'],1)
        with self.assertRaises(ValueError):p.reset('unknown')


if __name__=='__main__':unittest.main()
