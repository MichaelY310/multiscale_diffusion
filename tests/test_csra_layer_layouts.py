import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'2d/attention_analysis'))
from csra_layer_layouts import LAYOUTS,layout


class LayoutTests(unittest.TestCase):
    def test_budget(self):
        for name in LAYOUTS:
            spec=layout(name)
            self.assertEqual(len(spec['skipped_layers']),6)
            self.assertEqual(len(spec['executed_layers']),18)
            self.assertEqual(sorted(spec['skipped_layers']+spec['executed_layers']),list(range(24)))
        self.assertEqual(layout('split3_3')['readers'],2)

    def test_reader_parity_old_math_and_gradients(self):
        try: import torch
        except ImportError: self.skipTest('Requires torch')
        from torch import nn
        from types import SimpleNamespace
        from csra_layout_student import SpanReader
        from csra_prefix_student import ConditionalPrefixStudent
        class Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.x_embedder=SimpleNamespace(proj=SimpleNamespace(out_channels=16))
                self.blocks=nn.ModuleList([nn.Identity() for _ in range(24)])
                for block in self.blocks: block.attn=SimpleNamespace(num_heads=4)
        torch.manual_seed(42)
        fit=dict(weight=torch.randn(32,16)*.01,scale=torch.rand(32)+.1,
                 mean=torch.randn(32),output_mean=torch.randn(16))
        for arm in ('aligned','zero'):
            old=ConditionalPrefixStudent(Backbone(),fit,arm,prefix_depth=6)
            new=SpanReader(16,4,fit,arm)
            new.load_state_dict({k:v for k,v in old.state_dict().items() if not k.startswith('backbone.')})
            h,c,s=torch.randn(2,64,16),torch.randn(2,16),torch.randn(2,16,16)
            torch.testing.assert_close(new(h,c,s),old.lift(h,c,s),atol=0,rtol=0)
            new(h,c,s).square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in new.parameters()))


if __name__=='__main__': unittest.main()
