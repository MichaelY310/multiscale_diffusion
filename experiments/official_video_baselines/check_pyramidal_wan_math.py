"""Independent scalar-equation, derivative and distribution checks for PW."""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

import pyramidal_wan_paper_math as paper


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(702)
    # Exact published dimensions are audited without allocating dense tensors.
    shapes = [(81, 448, 832), (41, 224, 416), (21, 112, 208)]
    assert [(1 + (t - 1) // 4, h // 8, w // 8) for t, h, w in shapes] == [
        (21, 56, 104), (11, 28, 52), (6, 14, 26)]
    pixel = torch.randn(2, 3, 33, 64, 128, dtype=torch.float64)
    pp = paper.pixel_pyramid(pixel)
    assert [x.shape[2:] for x in pp] == [(33, 64, 128), (17, 32, 64), (9, 16, 32)]
    clean = [torch.randn(2, 16, 9, 8, 16, dtype=torch.float64),
             torch.randn(2, 16, 5, 4, 8, dtype=torch.float64),
             torch.randn(2, 16, 3, 2, 4, dtype=torch.float64)]
    noise = torch.randn_like(clean[0]); errors = []
    for i in range(3):
        natural = torch.tensor([.37, .61], dtype=torch.float64)
        natural = paper.NATURAL_CLEAN[i] + natural * (paper.NATURAL_NOISY[i] - paper.NATURAL_CLEAN[i])
        prepared = paper.prepare(clean, noise, natural, i)
        # Independent equations use slices and scalar mappings, without calling
        # the tested endpoint/global mapping helpers.
        for first, sl in [(True, slice(0, 1)), (False, slice(1, None))]:
            w = (2 if first else math.sqrt(8)) ** i
            c, n = paper.NATURAL_CLEAN[i], paper.NATURAL_NOISY[i]
            sc, sn = c / (w - (w - 1) * c), n / (w - (w - 1) * n)
            sg = (natural / (w - (w - 1) * natural)).view(2, 1, 1, 1, 1)
            eps = noise[:, :, sl]
            for _ in range(i):
                eps = F.avg_pool3d(eps, (1 if first else 2, 2, 2)) * (2 if first else math.sqrt(8))
            xc = clean[i][:, :, sl]
            yc = (1 - sc) * xc + sc * eps
            yn = eps if i == 2 else (1 - sn) * F.interpolate(
                clean[i + 1][:, :, sl], scale_factor=(1 if first else 2, 2, 2), mode='nearest') + sn * eps
            expected = yc + (sg - sc) / (sn - sc) * (yn - yc)
            derivative = (yn - yc) / (sn - sc)
            errors.append(float((expected - prepared.noisy[:, :, sl]).abs().max()))
            errors.append(float((derivative - prepared.target[:, :, sl]).abs().max()))
            # Finite-difference in GLOBAL sigma confirms the velocity target.
            h = 1e-5
            plus = yc + (sg + h - sc) / (sn - sc) * (yn - yc)
            minus = yc + (sg - h - sc) / (sn - sc) * (yn - yc)
            assert torch.allclose((plus - minus) / (2 * h), derivative, atol=1e-9, rtol=1e-9)
        assert torch.allclose(paper.to_natural(paper.to_global(natural, i), i), natural, atol=1e-14)
    assert max(errors) < 1e-13
    covariance_errors = []
    for i in (1, 2):
        # Constant coarse clean enables direct checks of first/tail marginal
        # variance and neighbor covariance after Haar transition.
        coarse_shape = (40000, 1, 3, 2, 2)
        coarse = torch.randn(coarse_shape, dtype=torch.float64)
        nat = torch.tensor(paper.NATURAL_CLEAN[i], dtype=torch.float64)
        sigma = paper.sigma_tensor(coarse, nat, i)
        state = sigma * coarse
        innovation = torch.randn_like(paper.up(state))
        high = innovation - paper.up(paper.down(innovation))
        assert paper.down(high).abs().max() < 1e-14
        out = paper.transition(state, i, innovation)
        target_sigma = paper.sigma_tensor(out, nat, i - 1)
        z = out / target_sigma
        for first, sl in [(True, slice(0, 1)), (False, slice(1, None))]:
            values = z[:, :, sl]
            variance_error = abs(float(values.square().mean()) - 1)
            neighbor_covariance = abs(float((values[..., 0::2] * values[..., 1::2]).mean()))
            assert variance_error < .015 and neighbor_covariance < .015
            covariance_errors.append({'stage': i, 'first': first, 'variance_error': variance_error,
                                      'neighbor_covariance': neighbor_covariance})
    args.output.write_text(json.dumps({'status': 'passed', 'cuda_used': False,
        'scope': 'published spatiotemporal geometry/equations; no model/optimizer or quality',
        'max_scalar_equation_error': max(errors), 'haar_distribution_checks': covariance_errors,
        'paper_rgb_and_latent_shapes': shapes, 'first_frame_2d_rest_3d': True,
        'published_boundaries': [0, .5858, .9412, 1]}, indent=2))
    print('PyramidalWan paper math checks passed')


if __name__ == '__main__':
    main()
