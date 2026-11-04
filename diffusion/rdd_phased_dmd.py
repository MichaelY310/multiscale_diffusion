"""Minimal RDD adaptation of the original DMD distribution-matching loss.

The equations in this file use the names and the exact forward expression from
``GaussianDiffusionWithDrift.q_sample``.  There is no selectable path
correction, preservation target, gradient clipping, or DMD2 update rule here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


Tensor = torch.Tensor


@dataclass(frozen=True)
class RDDPhase:
    """One inference call, ordered from coarse to fine."""

    phase_index: int
    diffusion_index: int
    resolution: int


def sampling_phases(resolutions_finest_first: Sequence[int]) -> tuple[RDDPhase, ...]:
    """Return phases in inference order (coarse -> fine)."""

    resolutions = tuple(int(value) for value in resolutions_finest_first)
    if not resolutions or any(value <= 0 for value in resolutions):
        raise ValueError("resolutions must contain positive integers")
    if any(first <= second for first, second in zip(resolutions, resolutions[1:])):
        raise ValueError("resolutions must be strictly decreasing (finest -> coarsest)")
    return tuple(
        RDDPhase(phase_index=phase, diffusion_index=index, resolution=resolutions[index])
        for phase, index in enumerate(range(len(resolutions) - 1, -1, -1))
    )


def projection(x: Tensor, scale: int) -> Tensor:
    """Apply ``P=UD``: average pool by ``scale``, then nearest-repeat."""

    scale = int(scale)
    if scale <= 0:
        raise ValueError("projection scale must be positive")
    if scale == 1:
        return x
    if x.shape[-2] % scale or x.shape[-1] % scale:
        raise ValueError(f"tensor shape {tuple(x.shape)} is not divisible by scale={scale}")
    pooled = F.avg_pool2d(x, kernel_size=scale, stride=scale)
    return pooled.repeat_interleave(scale, dim=-2).repeat_interleave(scale, dim=-1)


def _extract(values, timesteps: Tensor, x: Tensor) -> Tensor:
    table = torch.as_tensor(values, device=x.device, dtype=torch.float32)
    result = table[timesteps.long()]
    return result.reshape(result.shape[0], *([1] * (x.ndim - 1))).to(dtype=x.dtype)


def global_timesteps(diffusion, local_timesteps: Tensor) -> Tensor:
    """Map a respaced/local stage index to the model's global timestep."""

    mapping = torch.as_tensor(
        diffusion.timestep_map, device=local_timesteps.device, dtype=torch.long
    )
    return mapping[local_timesteps.long()]


def drift_fraction(diffusion, local_timesteps: Tensor, x: Tensor) -> Tensor:
    """Return RDD's linear ``lambda_t`` including the configured multiplier."""

    if diffusion.c is None:
        return torch.zeros(
            (local_timesteps.shape[0], *([1] * (x.ndim - 1))),
            device=x.device,
            dtype=x.dtype,
        )
    global_t = global_timesteps(diffusion, local_timesteps).to(dtype=x.dtype)
    fraction = (global_t - float(diffusion.T_c)) / float(
        diffusion.T_c_plus_1 - diffusion.T_c
    )
    fraction = fraction * float(getattr(diffusion, "drift_scale_multiplier", 1.0))
    return fraction.reshape(fraction.shape[0], *([1] * (x.ndim - 1)))


def apply_resolution_drift(
    diffusion, value: Tensor, local_timesteps: Tensor
) -> Tensor:
    """Apply exactly ``value + drift_fraction * (down_up(value) - value)``.

    This is the clean-signal expression inside the brackets in the repository's
    RDD ``q_sample``.  It is also its own transpose because average-pool followed
    by nearest-repeat is a symmetric projection under the ordinary tensor inner
    product.
    """

    if diffusion.c is None:
        return value
    scale = int(getattr(diffusion, "sample_scale", 2))
    fraction = drift_fraction(diffusion, local_timesteps, value)
    return value + fraction * (projection(value, scale) - value)


def rdd_renoise(
    diffusion,
    clean: Tensor,
    local_timesteps: Tensor,
    noise: Tensor | None = None,
) -> Tensor:
    """Sample the exact stage-local RDD forward marginal from ``clean``."""

    if noise is None:
        noise = torch.randn_like(clean)
    if noise.shape != clean.shape:
        raise ValueError("noise must have the same shape as clean")
    drift = None
    if diffusion.c is not None:
        scale = int(getattr(diffusion, "sample_scale", 2))
        drift = projection(clean, scale) - clean
    return diffusion.q_sample(clean, local_timesteps, d=drift, noise=noise)


def fresh_transition(
    clean_coarse: Tensor,
    target_resolution: int,
    diffusion,
    noise: Tensor | None = None,
) -> Tensor:
    """RDD's established clean-upsample plus fresh-noise boundary transition."""

    target_resolution = int(target_resolution)
    current = int(clean_coarse.shape[-1])
    if clean_coarse.shape[-2] != current or target_resolution % current:
        raise ValueError("noise filling requires square integer-ratio resolutions")
    scale = target_resolution // current
    upsampled = clean_coarse.repeat_interleave(scale, dim=-2).repeat_interleave(
        scale, dim=-1
    )
    top = torch.full(
        (upsampled.shape[0],),
        int(diffusion.num_timesteps) - 1,
        device=upsampled.device,
        dtype=torch.long,
    )
    # P(upsampled)==upsampled at a matched boundary, hence the RDD drift is zero.
    if noise is None:
        noise = torch.randn_like(upsampled)
    if noise.shape != upsampled.shape:
        raise ValueError("boundary noise must match the upsampled clean tensor")
    return diffusion.q_sample(upsampled, top, noise=noise)


def downsample_clean(clean: Tensor, target_resolution: int) -> Tensor:
    """Average-pool a final clean latent to a teacher stage resolution.

    Unified RDD is trained on exactly this average-pooled clean target at its
    lower-resolution stages.  Keeping this operation in the autograd graph is
    essential when DMD scores the *final* three-call generator distribution.
    """

    target_resolution = int(target_resolution)
    height, width = int(clean.shape[-2]), int(clean.shape[-1])
    if height != width or target_resolution <= 0 or height % target_resolution:
        raise ValueError(
            f"cannot average-pool clean latent {height}x{width} to "
            f"{target_resolution}x{target_resolution}"
        )
    scale = height // target_resolution
    if scale == 1:
        return clean
    return F.avg_pool2d(clean, kernel_size=scale, stride=scale)


def rdd_marginal_ddim_update(
    diffusion,
    current: Tensor,
    predicted_clean: Tensor,
    local_t: Tensor,
    local_s: Tensor,
) -> Tensor:
    r"""Deterministically couple two exact RDD forward marginals.

    RDD's stage-local marginal is

    ``x_t = alpha_t A_t x_0 + sigma_t epsilon``.

    The deterministic coupling keeps the same inferred ``epsilon`` while
    moving to ``s``:

    ``epsilon_hat = (x_t - alpha_t A_t x0_hat) / sigma_t``
    ``x_s = alpha_s A_s x0_hat + sigma_s epsilon_hat``.

    With an exact clean prediction this reproduces the target RDD marginal
    sample exactly.  With drift disabled it is the ordinary eta=0 DDIM update.
    It is deliberately *not* the DDPM posterior mean: dropping the random term
    from a Bayesian posterior contracts variance and is not an ODE/DDIM path.
    """

    if current.shape != predicted_clean.shape:
        raise ValueError("current and predicted_clean must have identical shapes")
    if local_t.shape != local_s.shape or local_t.ndim != 1:
        raise ValueError("local_t and local_s must be matching rank-one tensors")
    if torch.equal(local_t, local_s):
        return current

    alpha_t = _extract(diffusion.sqrt_alphas_cumprod, local_t, current)
    sigma_t = _extract(diffusion.sqrt_one_minus_alphas_cumprod, local_t, current)
    alpha_s = _extract(diffusion.sqrt_alphas_cumprod, local_s, current)
    sigma_s = _extract(diffusion.sqrt_one_minus_alphas_cumprod, local_s, current)
    clean_mean_t = apply_resolution_drift(diffusion, predicted_clean, local_t)
    clean_mean_s = apply_resolution_drift(diffusion, predicted_clean, local_s)
    inferred_noise = (current - alpha_t * clean_mean_t) / sigma_t.clamp_min(1e-12)
    return alpha_s * clean_mean_s + sigma_s * inferred_noise


@torch.no_grad()
def rdd_marginal_ddim_sample_loop(
    diffusion,
    model,
    shape,
    *,
    noise: Tensor,
    model_kwargs: dict,
    clip_denoised: bool = False,
) -> dict[str, Tensor]:
    """Run the deterministic marginal-preserving RDD/DDIM coupling."""

    if tuple(noise.shape) != tuple(shape):
        raise ValueError("noise shape does not match requested sample shape")
    state = noise
    final = None
    indices = list(range(int(diffusion.num_timesteps)))[::-1]
    for index, timestep in enumerate(indices):
        local_t = torch.full(
            (shape[0],), timestep, device=state.device, dtype=torch.long
        )
        next_timestep = indices[index + 1] if index + 1 < len(indices) else 0
        local_s = torch.full_like(local_t, next_timestep)
        output = diffusion.p_mean_variance(
            model,
            state,
            local_t,
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
        )
        predicted_clean = output["pred_xstart"]
        state = rdd_marginal_ddim_update(
            diffusion, state, predicted_clean, local_t, local_s
        )
        final = {"sample": state, "pred_xstart": predicted_clean}
    if final is None:
        raise RuntimeError("RDD marginal-DDIM loop had no timesteps")
    return final


def rdd_dmd_gradient(
    diffusion,
    generated_clean: Tensor,
    teacher_clean: Tensor,
    fake_clean: Tensor,
    local_timesteps: Tensor,
) -> Tensor:
    """Return the original-DMD Eq. 7/8 gradient for the exact RDD kernel.

    ``teacher_clean`` and ``fake_clean`` are the two posterior-clean means used
    by DMD.  Starting from the repository's literal RDD forward mean, the score
    difference applies the bracketed resolution-drift expression once.  The
    chain rule through ``q_sample`` applies its transpose once more.  DMD Eq. 8
    cancels the noise variance and one signal-scale factor, leaving exactly one
    ``sqrt_alphas_cumprod[t]`` factor.

    When drift is disabled this reduces to the VP form of the original DMD
    Eq. 7/8.  When the signal scale is one it reduces to DMD Algorithm 2.
    """

    clean_prediction_difference = fake_clean - teacher_clean
    score_mean_difference = apply_resolution_drift(
        diffusion, clean_prediction_difference, local_timesteps
    )
    clean_space_difference = apply_resolution_drift(
        diffusion, score_mean_difference, local_timesteps
    )
    signal_scale = _extract(
        diffusion.sqrt_alphas_cumprod,
        local_timesteps,
        clean_space_difference,
    )

    reduce_dims = tuple(range(1, generated_clean.ndim))
    normalizer = (generated_clean - teacher_clean).abs().mean(
        dim=reduce_dims, keepdim=True
    )
    return (signal_scale * clean_space_difference / normalizer).detach()


def dmd_surrogate_loss(generated_clean: Tensor, gradient: Tensor) -> Tensor:
    """DMD Algorithm 2's surrogate loss, written literally."""

    if generated_clean.shape != gradient.shape:
        raise ValueError("generated_clean and gradient must have identical shapes")
    target = (generated_clean - gradient).detach()
    return 0.5 * F.mse_loss(generated_clean.float(), target.float())
