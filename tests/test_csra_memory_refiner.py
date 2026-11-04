import copy
import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rdd_video'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '2d/attention_analysis'))
from attention_analysis.csra_memory_refiner import pool_token_grid, CoarseMemoryRefiner
from attention_analysis.csra_refined_video_bridge import VideoRefinedPrefixBridge
from csra_neighbor_student import NeighborFrontStudent
from csra_refined_student import RefinedFrontStudent
from test_csra_front_keep_student import FrontStudentTests


class MemoryRefinerTests(unittest.TestCase):
    def test_nested_grid_pooling(self):
        x = torch.arange(2 * 4 * 8.).reshape(1, 64, 1)
        y = pool_token_grid(x, (2, 4, 8), (1, 2, 4))
        reference = torch.nn.functional.avg_pool3d(x.reshape(1, 1, 2, 4, 8), 2).flatten().reshape(1, 8, 1)
        torch.testing.assert_close(y, reference, atol=0, rtol=0)
        with self.assertRaises(ValueError): pool_token_grid(x, (2, 4, 8), (1, 3, 4))

    def test_zero_init_real_execution_and_gradient_scope(self):
        m = CoarseMemoryRefiner(8, 2, 2)
        memory = torch.randn(2, 4, 8, requires_grad=True)
        fine, c = torch.randn(2, 16, 8), torch.randn(2, 8)
        saved = memory.detach().clone()
        calls = []
        handles = [b.register_forward_hook(lambda *_: calls.append(True)) for b in m.blocks]
        y = m(memory, fine, c, (1, 2, 2), (1, 4, 4))
        torch.testing.assert_close(y, memory, atol=0, rtol=0)
        y.square().mean().backward()
        self.assertEqual(len(calls), 2)
        self.assertGreater(m.output.weight.grad.abs().sum().item(), 0)
        for p in m.parameters(): self.assertIsNotNone(p.grad)
        torch.testing.assert_close(memory.detach(), saved, atol=0, rtol=0)
        for h in handles: h.remove()
        with torch.no_grad(): m.output.weight.normal_(std=.1)
        a = m(memory, fine, c, (1, 2, 2), (1, 4, 4))
        b = m(memory, fine + torch.randn_like(fine), c, (1, 2, 2), (1, 4, 4))
        self.assertGreater((a - b).abs().max().item(), 1e-5)
        with torch.no_grad(): m.blocks[0].condition[-1].weight.normal_(std=.1)
        d = m(memory, fine, c + 1, (1, 2, 2), (1, 4, 4))
        e = m(memory, fine, c, (1, 2, 2), (1, 4, 4))
        self.assertGreater((d - e).abs().max().item(), 1e-5)

    def test_2d_parent_parity_and_exact_skips(self):
        parent, batch, fit = FrontStudentTests().make()
        old = NeighborFrontStudent(copy.deepcopy(parent.backbone), fit, 'aligned', prefix_depth=2,
                                  source_layer=1, checkpoint_blocks=False, neighbor_enabled=False, neighbor_rank=4)
        old.initialize_from_front_parent(parent.state_dict())
        new = RefinedFrontStudent(copy.deepcopy(parent.backbone), fit, 'aligned', prefix_depth=2,
                                 source_layer=1, checkpoint_blocks=False, neighbor_enabled=False, neighbor_rank=4)
        new.initialize_from_neighbor_parent(old.state_dict())
        torch.testing.assert_close(new(batch), old(batch), atol=0, rtol=0)
        calls = []
        handles = [b.register_forward_hook(lambda m, i, o, j=j: calls.append(j)) for j, b in enumerate(new.backbone.blocks)]
        new(batch).square().mean().backward()
        self.assertEqual(calls, [0, 3])
        for n, p in new.named_parameters():
            if p.requires_grad: self.assertIsNotNone(p.grad, n)
        for h in handles: h.remove()
        with self.assertRaises(RuntimeError): new.load_state_dict(old.state_dict())

    def test_video_parent_zero_init_spatial_and_temporal(self):
        m = VideoRefinedPrefixBridge(8, 2)
        for coarse, fine in [((2, 2, 2), (2, 4, 4)), ((1, 2, 2), (2, 4, 4))]:
            memory = torch.randn(1, coarse[0] * coarse[1] * coarse[2], 8)
            h, c = torch.randn(1, 32, 8), torch.randn(1, 8)
            m.refiner_enabled = False
            a = m(h, c, memory, coarse, fine)
            m.refiner_enabled = True
            torch.testing.assert_close(m(h, c, memory, coarse, fine), a, atol=0, rtol=0)


if __name__ == '__main__': unittest.main()
