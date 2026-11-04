"""Small CPU tests for layout, prefix scope and unchanged native execution."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rdd_video'))
from attention_analysis.csra_prefix_video_bridge import (
    lift_token_grid, VideoConditionalPrefixBridge, VideoPrefixSamplerProxy)


class FakeCondition(nn.Module):
    def forward(self, t, context):
        return (torch.ones((len(t), 8)), None, context, None)


class FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = SimpleNamespace(attn_impl=SimpleNamespace(forward=lambda *a: None))

    def forward(self, h, context, temb, freqs, length):
        return h + 1


class FakeModel(nn.Module):
    patch_size = (1, 1, 1)
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([FakeBlock() for _ in range(4)])
        self.condition_embedder = FakeCondition()

    def forward(self, hidden_states, encoder_hidden_states, timestep):
        c = self.condition_embedder(timestep, encoder_hidden_states)[0]
        h = hidden_states
        for block in self.blocks:
            h = block(h, encoder_hidden_states, c, None, h.shape[1])
        return h


class BridgeTests(unittest.TestCase):
    def test_t_h_w_layout(self):
        x = torch.arange(8.).reshape(1, 8, 1)
        actual = lift_token_grid(x, (2,2,2), (4,4,4))
        expected = x.reshape(1,2,2,2,1).repeat_interleave(2,1).repeat_interleave(2,2).repeat_interleave(2,3).reshape(1,64,1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_reject_nonnested_or_mislabeled_grid(self):
        for src, dst in [((1,2,2),(1,3,4)), ((2,2,2),(4,4,4))]:
            with self.assertRaises(ValueError): lift_token_grid(torch.zeros(1,4,8),src,dst)

    def test_zero_nonlinear_matches_affine(self):
        m = VideoConditionalPrefixBridge(8, 2)
        for layer in (m.read_out, m.nonlinear[-1]):
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)
        h, c, memory = torch.randn(2,16,8), torch.randn(2,8), torch.randn(2,4,8)
        expected = h + m.proj(torch.cat((h,c[:,None].expand(-1,16,-1),
            lift_token_grid(memory,(1,2,2),(1,4,4))),-1))
        torch.testing.assert_close(m(h,c,memory,(1,2,2),(1,4,4)), expected, rtol=0, atol=0)

    def test_native_memory_affects_output(self):
        torch.manual_seed(7)
        m = VideoConditionalPrefixBridge(8,2)
        h,c,memory=torch.randn(1,16,8),torch.randn(1,8),torch.randn(1,4,8)
        a=m(h,c,memory,(1,2,2),(1,4,4))
        b=m(h,c,torch.zeros_like(memory),(1,2,2),(1,4,4))
        self.assertTrue(torch.isfinite(a).all())
        self.assertGreater((a-b).abs().max().item(), 1e-3)

    def test_three_nfe_scope_and_reset(self):
        original,fine=FakeModel(),FakeModel()
        m=VideoPrefixSamplerProxy(original,fine,VideoConditionalPrefixBridge(8,2),source_layer=1)
        for depth in (0,2,0):
            m.reset(depth)
            for stage,shape in enumerate(((1,1,1),(1,2,2),(1,4,4))):
                m.begin_model_forward(stage,shape)
                out=m(hidden_states=torch.zeros(1,shape[0]*shape[1]*shape[2],8),
                      encoder_hidden_states=torch.zeros(1,2,8),timestep=torch.zeros(1))
                if stage < 2 or depth == 0:
                    torch.testing.assert_close(out,torch.full_like(out,4),rtol=0,atol=0)
            self.assertEqual(m.audit()['executed'],[4,4,4-depth])
            self.assertIsNone(m.memory)

    def test_fail_missing_stage_and_shared_weights(self):
        a=FakeModel()
        with self.assertRaises(ValueError):
            VideoPrefixSamplerProxy(a,a,VideoConditionalPrefixBridge(8,2),source_layer=1)
        m=VideoPrefixSamplerProxy(a,FakeModel(),VideoConditionalPrefixBridge(8,2),source_layer=1)
        with self.assertRaises(ValueError): m.begin_model_forward(2,(1,4,4))


if __name__ == '__main__': unittest.main()
