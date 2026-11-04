"""PyramidalWan arXiv:2601.04792v1, Eqs.1–10/23–41 and Appendix A3.

Paper indices are used: 0=full, 1=middle, 2=coarse. The first latent frame
scales in 2D (omega=2); the remaining frames scale in 3D (omega=sqrt(8)).
This is independent of the old spatial-only controlled transplant.
"""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

NATURAL_CLEAN = (0.0, 0.5858, 0.9412)
NATURAL_NOISY = (0.5858, 0.9412, 1.0)


def _check(x):
    if x.ndim != 5 or x.shape[2] < 2:
        raise ValueError('Expected B,C,T,H,W with a separate first frame and a nonempty tail')


def down(x):
    """One Haar/average-pooling level; first frame does not pool in time."""
    _check(x)
    if (x.shape[2] - 1) % 2 or x.shape[3] % 2 or x.shape[4] % 2:
        raise ValueError('Tail length and spatial axes must be divisible by two')
    return torch.cat([F.avg_pool3d(x[:, :, :1], (1, 2, 2)),
                      F.avg_pool3d(x[:, :, 1:], (2, 2, 2))], dim=2)


def up(x):
    """Nearest-neighbor synthesis with separate first-frame handling."""
    _check(x)
    return torch.cat([F.interpolate(x[:, :, :1], scale_factor=(1, 2, 2), mode='nearest'),
                      F.interpolate(x[:, :, 1:], scale_factor=2, mode='nearest')], dim=2)


def frame_constants(x, first, rest):
    """Broadcast one value per sample independently to first and tail frames."""
    a = torch.as_tensor(first, dtype=x.dtype, device=x.device).reshape(-1, 1, 1, 1, 1)
    b = torch.as_tensor(rest, dtype=x.dtype, device=x.device).reshape(-1, 1, 1, 1, 1)
    return torch.cat([a.expand(-1, 1, 1, 1, 1),
                      b.expand(-1, 1, x.shape[2] - 1, 1, 1)], dim=2)


def omega(i, first=False):
    if i not in (0, 1, 2):
        raise ValueError('PyramidalWan has exactly three stages')
    return (2.0 if first else math.sqrt(8.0)) ** i


def to_global(natural, i, first=False):
    """Inverse of the recursively applied Eq.39."""
    scale = omega(i, first)
    return natural / (scale - (scale - 1) * natural)


def to_natural(global_sigma, i, first=False):
    scale = omega(i, first)
    return scale * global_sigma / (1 + (scale - 1) * global_sigma)


def sigma_tensor(x, natural, i):
    return frame_constants(x, to_global(natural, i, True), to_global(natural, i, False))


def noise_at_stage(full_noise, i):
    """Eq.10 shared full-resolution Gaussian, downsampled with unit variance."""
    result = full_noise
    for _ in range(i):
        result = down(result)
        result = result * frame_constants(result, 2.0, math.sqrt(8.0))
    return result


@dataclass
class Prepared:
    noisy: torch.Tensor
    target: torch.Tensor
    sigma: torch.Tensor
    cleaner_sigma: torch.Tensor
    noisy_sigma: torch.Tensor
    natural: torch.Tensor
    stage: int


def prepare(clean_pyramid, full_noise, natural, i):
    """Eqs.1–4/7; derivative is w.r.t. GLOBAL sigma, not natural time/rho."""
    clean = clean_pyramid[i]
    noise = noise_at_stage(full_noise, i)
    if clean.shape != noise.shape:
        raise ValueError('RGB-before-VAE pyramid and pooled noise shapes differ')
    if torch.any(natural < NATURAL_CLEAN[i]) or torch.any(natural > NATURAL_NOISY[i]):
        raise ValueError('Natural noise level lies outside this stage')
    sc = sigma_tensor(clean, clean.new_tensor(NATURAL_CLEAN[i]), i)
    sn = sigma_tensor(clean, clean.new_tensor(NATURAL_NOISY[i]), i)
    sigma = sigma_tensor(clean, natural, i)
    yc = (1 - sc) * clean + sc * noise
    yn = noise if i == 2 else (1 - sn) * up(clean_pyramid[i + 1]) + sn * noise
    rho = (sigma - sc) / (sn - sc)
    return Prepared((1 - rho) * yc + rho * yn, (yn - yc) / (sn - sc),
                    sigma, sc, sn, natural, i)


def teacher_boundary(full_prediction, i):
    """Eq.10 dimensional alignment using Eq.25/29 amplitude normalization.

The main text suppresses the resampling in Eq.10. After average pooling,
scale by 1+(omega**i-1)*sigma_c to obtain the student's GLOBAL-noise law.
Both first and remaining frames use their respective scale.
"""
    value = full_prediction
    for _ in range(i):
        value = down(value)
    natural = value.new_tensor(NATURAL_CLEAN[i])
    first = 1 + (omega(i, True) - 1) * to_global(natural, i, True)
    rest = 1 + (omega(i, False) - 1) * to_global(natural, i, False)
    return value * frame_constants(value, first, rest)


def transition(coarse_state, coarse_i, full_noise):
    """Eq.38 Haar high-frequency filling; no jitter or IID replacement.

    full_noise has the shape of the next finer stage and is freshly IID.
    The first frame has four spatial children; tail cells have eight children.
    """
    if coarse_i not in (1, 2):
        raise ValueError('Transition is from stage 2→1 or 1→0')
    lifted = up(coarse_state)
    if full_noise.shape != lifted.shape:
        raise ValueError('Incorrect next-stage innovation shape')
    high = full_noise - up(down(full_noise))
    natural = coarse_state.new_tensor(NATURAL_CLEAN[coarse_i])
    coarse_sigma = frame_constants(lifted, to_global(natural, coarse_i, True),
                                   to_global(natural, coarse_i, False))
    scale = frame_constants(lifted, 2.0, math.sqrt(8.0))
    return (lifted + scale * coarse_sigma * high) / (1 + (scale - 1) * coarse_sigma)


def pixel_pyramid(video):
    """Definition②: resize RGB before each separate VAE encode.

    Trilinear interpolation with align_corners=False is an explicit choice;
    the paper does not disclose its exact RGB-resize options.
    """
    _check(video)
    result = [video]
    for _ in range(2):
        x = result[-1]
        if (x.shape[2] - 1) % 2 or x.shape[3] % 2 or x.shape[4] % 2:
            raise ValueError('RGB pyramid requires divisible temporal tail/spatial shapes')
        first = F.interpolate(x[:, :, :1], size=(1, x.shape[3] // 2, x.shape[4] // 2),
                              mode='trilinear', align_corners=False)
        rest = F.interpolate(x[:, :, 1:], size=((x.shape[2] - 1) // 2, x.shape[3] // 2,
                                               x.shape[4] // 2), mode='trilinear', align_corners=False)
        result.append(torch.cat([first, rest], dim=2))
    return result
