"""Three actual AdamW updates: 2D no-drift training versus zero-scaled drift.

Small CPU math/storage scope; no native DiT/ImageNet quality claim.
"""
from typing import Any
import copy
import json
import os
from pathlib import Path
import sys


def main() -> None:
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU only')
    w = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(w / 'multiscale_diffusion'))
    import torch
    from diffusion import create_diffusion_with_drift
    torch.manual_seed(69)
    torch.set_num_threads(4)

    class Predictor(torch.nn.Module):

        def __init__(self: Any) -> None:
            super().__init__()
            self.layer = torch.nn.Conv2d(4, 8, 1)

        def forward(self: Any, x: Any, t: Any) -> Any:
            return self.layer(x)

    reference = Predictor()
    scaled = copy.deepcopy(reference)
    optimizers = [torch.optim.AdamW(m.parameters(), lr=1e-4) for m in (reference, scaled)]
    checks = []
    for stage, (a, b, size) in enumerate(zip((0, 300, 600), (300, 600, 1000), (32, 16, 8), strict=False)):
        absent = create_diffusion_with_drift('', c=None, T_c=a, T_c_plus_1=b)
        zero = create_diffusion_with_drift('', c=stage if stage < 2 else None, T_c=a, T_c_plus_1=b)
        zero.drift_scale_multiplier = 0.
        x = torch.randn(2, 4, size, size)
        noise = torch.randn_like(x)
        t = torch.tensor([0, b - a - 1])
        losses = []
        for model, opt, diffusion in zip((reference, scaled), optimizers, (absent, zero), strict=False):
            opt.zero_grad()
            loss = diffusion.training_losses(model, x, t, noise=noise)['loss'].mean()
            loss.backward()
            opt.step()
            losses.append(loss.detach())
        equal = torch.equal(losses[0], losses[1]) and all(
            torch.equal(p, q) for p, q in zip(reference.parameters(), scaled.parameters(), strict=False))
        checks.append(dict(stage=stage, loss=float(losses[0]), all_weights_bitwise_equal=equal))
    assert all(c['all_weights_bitwise_equal'] for c in checks)
    assert not torch.cuda.is_initialized()
    (w / 'gpu_checks_20261006/IMAGE_NO_DRIFT_TRAINING_CPU.json').write_text(
        json.dumps(dict(
            status='passed',
            checks=checks,
            cpu_only=True,
            optimizer_updates=3,
            synthetic_component_input=True,
            scope=
            'Existing 2D no-drift train objective, loss and gradient updates; full DiT/ImageNet entry still requires native assets'
        ),
                   indent=2))


if __name__ == '__main__':
    main()
