import copy
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rdd_video'))
from attention_analysis.csra_fine_neighbor import FineSpatialNeighbor
from attention_analysis.csra_neighbor_video_bridge import VideoNeighborPrefixBridge
from attention_analysis.csra_prefix_video_bridge import VideoPrefixSamplerProxy
from test_csra_prefix_video_bridge import FakeModel


class NeighborTests(unittest.TestCase):
    def test_zero_output_equivalence_and_branch_really_executes(self):
        for dtype in (torch.float32, torch.bfloat16):
            module = FineSpatialNeighbor(8, 4)
            calls = []
            hook = module.spatial.register_forward_hook(lambda m, i, o: calls.append(o.shape))
            hidden = torch.randn(2, 30, 8).to(dtype)
            actual = module(hidden, (2, 3, 5))
            hook.remove()
            self.assertTrue(torch.equal(hidden, actual))
            self.assertEqual(actual.dtype, dtype)
            self.assertEqual(calls, [torch.Size([4, 4, 3, 5])])

    def test_gradients_zero_init_then_nonzero(self):
        module = FineSpatialNeighbor(8, 4)
        hidden = torch.randn(2, 30, 8, requires_grad=True)
        loss = (module(hidden, (2, 3, 5)) - torch.randn_like(hidden)).square().mean()
        loss.backward()
        self.assertGreater(module.up.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is not None for p in module.parameters()))
        # Zero-output init blocks first-step gradients to the earlier branch
        # weights; after the output learns, those weights must receive signal.
        self.assertEqual(module.down.weight.grad.abs().sum().item(), 0)
        with torch.no_grad(): module.up.weight.normal_(std=.1)
        module.zero_grad(set_to_none=True)
        module(hidden, (2, 3, 5)).square().mean().backward()
        for p in module.parameters():
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(p.grad.abs().sum().item(), 0)

    def test_explicit_grid_and_no_frame_or_row_wrap(self):
        module = FineSpatialNeighbor(8, 4)
        with torch.no_grad(): module.up.weight.normal_(std=.1)
        h = torch.randn(1, 2 * 3 * 5, 8)
        first = module(h, (2, 3, 5))
        changed = h.clone(); changed[:, 0] += torch.arange(8.)
        second = module(changed, (2, 3, 5))
        delta = (first-second).abs().sum(-1)[0]
        self.assertGreater(delta[:2].sum().item(), 0)
        expected_untouched = [2,3,4,7,8,9,10,11,12,13,14] + list(range(15,30))
        self.assertEqual(delta[expected_untouched].max().item(), 0)
        for grid in ((3,5), (2,3,4), (0,3,5), (2.,3,5)):
            with self.assertRaises(ValueError): module(h, grid)

    def test_video_front1_scope_and_zero_neighbor_parity(self):
        bridge = VideoNeighborPrefixBridge(8, 2, 4)
        a = VideoPrefixSamplerProxy(FakeModel(), FakeModel(), bridge, source_layer=1, front_keep=1)
        outputs = []
        for enabled in (False, True):
            bridge.neighbor_enabled = enabled
            a.reset(2)
            # Same real inputs in both runs, no reference to flattened frame order.
            torch.manual_seed(99)
            for stage, shape in enumerate(((1,1,1),(1,2,2),(1,4,4))):
                a.begin_model_forward(stage, shape)
                value = a(hidden_states=torch.ones(1, shape[0]*shape[1]*shape[2], 8),
                          encoder_hidden_states=torch.zeros(1,2,8), timestep=torch.zeros(1))
            self.assertEqual(a.audit()['skipped'], [1,2])
            self.assertEqual(a.audit()['executed'], [4,4,2])
            outputs.append(value)
        torch.testing.assert_close(*outputs, atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA AMP test')
    def test_cuda_fp32_and_bf16_gradient(self):
        torch.cuda.set_per_process_memory_fraction(.1)
        for dtype in (torch.float32, torch.bfloat16):
            module = FineSpatialNeighbor(32, 8).cuda()
            h = torch.randn(2, 30, 32, device='cuda', dtype=dtype, requires_grad=True)
            out = module(h, (2,3,5))
            torch.testing.assert_close(out, h, atol=0, rtol=0)
            out.float().square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()))


if __name__ == '__main__': unittest.main()
