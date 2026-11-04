import copy
import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '2d/attention_analysis'))
from csra_neighbor_student import NeighborFrontStudent, FORMAT
from test_csra_front_keep_student import FrontStudentTests


class StudentNeighborTests(unittest.TestCase):
    def test_explicit_parent_loading_zero_equivalence_and_scope(self):
        parent, batch, fit = FrontStudentTests().make()
        for enabled in (False, True):
            model = NeighborFrontStudent(copy.deepcopy(parent.backbone), fit, 'aligned',
                prefix_depth=2, source_layer=1, checkpoint_blocks=False,
                neighbor_enabled=enabled, neighbor_rank=4)
            model.initialize_from_front_parent(parent.state_dict())
            with torch.no_grad():
                torch.testing.assert_close(model(batch), parent(batch), atol=0, rtol=0)
            calls = []
            handles = [b.register_forward_hook(lambda m, i, o, j=j: calls.append(j))
                       for j, b in enumerate(model.backbone.blocks)]
            model(batch).square().mean().backward()
            for handle in handles: handle.remove()
            self.assertEqual(calls, [0, 3])
            for name, p in model.named_parameters():
                if p.requires_grad: self.assertIsNotNone(p.grad, name)
            self.assertEqual(FORMAT, 'csra_front1_skip6_spatial_neighbor_v1')
            with self.assertRaises(RuntimeError): model.load_state_dict(parent.state_dict())
            with self.assertRaises(ValueError): model.initialize_from_front_parent(model.state_dict())

    def test_checkpointed_suffix_gradients(self):
        parent, batch, fit = FrontStudentTests().make()
        a = NeighborFrontStudent(parent.backbone, fit, 'aligned', prefix_depth=2, source_layer=1,
                                checkpoint_blocks=False, neighbor_rank=4)
        with torch.no_grad(): a.neighbor.up.weight.normal_(std=.02)
        b = copy.deepcopy(a); b.checkpoint_blocks = True
        ya, yb = a(batch), b(batch)
        torch.testing.assert_close(ya, yb, atol=0, rtol=0)
        ya.square().mean().backward(); yb.square().mean().backward()
        for pa, pb in zip(a.parameters(), b.parameters()):
            if pa.requires_grad: torch.testing.assert_close(pa.grad, pb.grad, atol=0, rtol=0)


if __name__ == '__main__': unittest.main()
