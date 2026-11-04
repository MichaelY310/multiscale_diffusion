import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'DiT'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'2d/attention_analysis'))
from models_multires import DiTMultiRes
from direct_reuse_fullmodel import DirectReuseStudent, lift


class DirectReuseTest(unittest.TestCase):
    def test_gradients_and_no_parameters_added(self):
        torch.manual_seed(29)
        model = DiTMultiRes(depth=4, hidden_size=64, num_heads=4, patch_size=2,
                           num_classes=10, base_size=16).float().eval()
        # DiT initialization otherwise has zero output/modulation projections.
        for p in model.parameters():
            if p.requires_grad:
                with torch.no_grad(): p.add_(torch.randn_like(p)*.015)
        net = DirectReuseStudent(model, [1, 2], True)
        self.assertEqual(sum(p.numel() for p in net.parameters()), sum(p.numel() for p in model.parameters()))
        x = torch.randn(2, 4, 16, 16, requires_grad=True)
        y = torch.tensor([1, 2]); t = torch.full((2,), 300.)
        plain, _ = net.stage(x, t, y)
        torch.testing.assert_close(plain, model(x, t, y), atol=0, rtol=0)
        _, cache = net.stage(x, t, y, capture=True)
        fine, _ = net.stage(torch.randn(2, 4, 32, 32), t-100, y, cache=cache)
        fine.square().mean().backward()
        self.assertGreater(x.grad.abs().sum().item(), 0.)
        # Fine loss must reach blocks which were SKIPPED in fine through coarse.
        for l in [1, 2]:
            self.assertGreater(model.blocks[l].attn.qkv.weight.grad.abs().sum().item(), 0.)
        with torch.no_grad():
            live = net.stage(torch.randn(2,4,16,16), t, y, capture=True)[1]
            for d in live.values(): self.assertEqual(lift(d,16,16).shape, (2,256,64))


if __name__ == '__main__': unittest.main()
