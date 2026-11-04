"""Small, independently testable helpers; no change to old samplers/checkpoints."""
import torch
from torch import nn
from dmd_cascade_reuse import CascadeDMD


class DenseScore(nn.Module):
    """Original dense DiT forward, with activation checkpointing for the critic."""
    def __init__(self, backbone):
        super().__init__()
        self.core = CascadeDMD(backbone, 'dense', True)

    def forward(self, x, t, y):
        return self.core.stage(x, t, y, 0)[0]


def value_with_gradient(x, value, gradient):
    """Chain rule only: return value, but backpropagate a precomputed dL/dx."""
    linear = (x.float() * gradient.detach().float()).sum()
    return linear - linear.detach() + value.detach()


def chunked_value_gradient(prediction, target, function, chunk_size):
    """Exact mean sample loss and its latent gradient without retaining decoder graphs.

    function must return one loss per sample, with NO cross-sample operations.
    All chunks participate; this is not subsampling or gradient accumulation of
    the generator batch. Detached gradients are exact first-order chain rule.
    """
    gradient = torch.zeros_like(prediction, dtype=torch.float32)
    value = prediction.new_zeros((), dtype=torch.float32)
    for start in range(0, len(prediction), chunk_size):
        stop = min(start + chunk_size, len(prediction))
        z = prediction[start:stop].detach().float().requires_grad_(True)
        losses = function(z, target[start:stop].detach())
        if losses.numel() != stop - start:
            raise ValueError('perceptual function must return per-sample losses')
        partial = losses.float().sum() / len(prediction)
        g, = torch.autograd.grad(partial, z)
        if not torch.isfinite(g).all() or not torch.isfinite(partial):
            raise FloatingPointError('Nonfinite perceptual loss/latent gradient')
        gradient[start:stop] = g
        value += partial.detach()
    return value, gradient


def schedule(update, ramp_steps):
    ramp = min(1., max(0., update / max(1, ramp_steps)))
    return dict(dmd=ramp, final_mse=1. - .9 * ramp,
                onpolicy=.25 * (1. - ramp), lpips=.25 * ramp,
                coarse_mse=.5, middle_mse=.5, variance=.01)


def score_candidates(diffusions):
    """diffusions are coarse-first; count-weighting preserves uniform global time."""
    candidates = []
    for diffusion in diffusions:
        mapping = torch.tensor(diffusion.timestep_map)
        candidates.append(((mapping >= 20) & (mapping < 980)).nonzero().flatten())
    if sum(len(c) for c in candidates) != 960:
        raise ValueError('Score phases must partition global t=20..979')
    return candidates


def choose_phase(candidates, step, seed):
    # Stateless and identical across ranks; independent of the shared paired-noise stream.
    rng = torch.Generator().manual_seed(seed + step * 104729)
    index = int(torch.randint(sum(len(c) for c in candidates), (1,), generator=rng))
    for stage, values in enumerate(candidates):
        if index < len(values):
            return stage
        index -= len(values)
    raise AssertionError('unreachable')
