"""Distribution-preserving compatibility for the official singular block noise.

At gamma=1/3, C=(1+gamma)I-gamma*11^T is positive semidefinite,
with a zero DC eigenvalue. No diagonal jitter or gamma change is allowed.
The factor below has AA^T=C; RNG samples differ from a Cholesky sampler.
"""
import math


def block_noise_factor(gamma, torch):
    if not 0 <= gamma <= 1/3:
        raise ValueError("Official block covariance requires 0 <= gamma <= 1/3")
    dc = torch.ones(4, 4, dtype=torch.float64) / 4
    return math.sqrt(1 + gamma) * (torch.eye(4, dtype=torch.float64) - dc) + math.sqrt(max(0, 1 - 3*gamma)) * dc


def sample_psd_block_noise(gamma, bs, ch, temp, height, width):
    import torch
    from einops import rearrange
    factor = block_noise_factor(gamma, torch).float()
    count = bs * ch * temp * (height // 2) * (width // 2)
    noise = torch.randn(count, 4) @ factor.T
    return rearrange(noise, '(b c t h w) (p q) -> b c t (h p) (w q)',
                     b=bs, c=ch, t=temp, h=height//2, w=width//2, p=2, q=2)
