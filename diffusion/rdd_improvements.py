"""Optional, backward-compatible training and sampling improvements for RDD.

The helpers in this module are deliberately model-agnostic.  Importing the
module does not change an existing training or sampling path; callers must opt
in explicitly.  This is important because the repository contains historical
RDD checkpoints whose numerical behavior must remain reproducible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def diffusion_snr(diffusion, timesteps: torch.Tensor) -> torch.Tensor:
    """Return the local-stage SNR alpha_bar / (1 - alpha_bar).

    ``SpacedDiffusionWithDrift`` reconstructs a local beta schedule for each
    RDD stage.  Therefore its ``alphas_cumprod`` is already the correct SNR for
    the conditional stage objective and ``timesteps`` indexes it directly.
    """

    alpha_bar = torch.as_tensor(
        diffusion.alphas_cumprod,
        device=timesteps.device,
        dtype=torch.float32,
    )[timesteps.long()]
    return alpha_bar / (1.0 - alpha_bar).clamp_min(1e-12)


def diffusion_loss_weights(
    diffusion,
    timesteps: torch.Tensor,
    scheme: str = "none",
    *,
    min_snr_gamma: float = 5.0,
    p2_gamma: float = 1.0,
    p2_k: float = 1.0,
) -> torch.Tensor:
    """Compute per-example weights for an epsilon-prediction MSE objective.

    Supported schemes:

    * ``none``: legacy RDD objective.
    * ``min-snr``: min(SNR, gamma) / SNR (Hang et al., ICCV 2023).
    * ``p2``: (k + SNR)^(-gamma) (Choi et al., CVPR 2022).

    RDD diffusion currently predicts epsilon.  The formulas for x0- or
    v-prediction are different, so this function rejects unsupported model
    mean types instead of silently applying the wrong objective.
    """

    normalized = scheme.lower().replace("_", "-")
    if normalized == "none":
        return torch.ones_like(timesteps, dtype=torch.float32)

    mean_type = getattr(getattr(diffusion, "model_mean_type", None), "name", None)
    if mean_type != "EPSILON":
        raise ValueError(
            f"{scheme} weighting is implemented for epsilon prediction, got {mean_type!r}"
        )

    snr = diffusion_snr(diffusion, timesteps)
    if normalized in {"min-snr", "minsnr"}:
        if min_snr_gamma <= 0:
            raise ValueError("min_snr_gamma must be positive")
        return torch.minimum(snr, torch.full_like(snr, min_snr_gamma)) / snr.clamp_min(1e-12)
    if normalized == "p2":
        if p2_k <= 0 or p2_gamma < 0:
            raise ValueError("p2_k must be positive and p2_gamma must be non-negative")
        return (p2_k + snr).pow(-p2_gamma)
    raise ValueError(f"Unknown diffusion loss-weighting scheme: {scheme}")


def weighted_diffusion_loss(
    terms: Dict[str, torch.Tensor],
    diffusion,
    timesteps: torch.Tensor,
    scheme: str = "none",
    *,
    min_snr_gamma: float = 5.0,
    p2_gamma: float = 1.0,
    p2_k: float = 1.0,
    importance_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-example loss and the applied objective weight.

    For learned-variance models only the denoising MSE is reweighted.  The VB
    term remains unchanged, matching the intended hybrid objective rather than
    accidentally attenuating variance learning at high-SNR timesteps.
    ``importance_weights`` is the unbiased correction returned by OpenAI's
    loss-second-moment timestep sampler and is applied to the full objective.
    """

    weights = diffusion_loss_weights(
        diffusion,
        timesteps,
        scheme,
        min_snr_gamma=min_snr_gamma,
        p2_gamma=p2_gamma,
        p2_k=p2_k,
    ).to(dtype=terms["loss"].dtype)

    if scheme.lower().replace("_", "-") == "none":
        loss = terms["loss"]
    elif "mse" not in terms:
        raise ValueError(f"{scheme} requires an MSE diffusion objective")
    else:
        loss = terms["mse"] * weights
        if "vb" in terms:
            loss = loss + terms["vb"]

    if importance_weights is not None:
        loss = loss * importance_weights.to(device=loss.device, dtype=loss.dtype)
    return loss, weights


def sample_fm_time(
    batch_size: int,
    device: torch.device,
    *,
    sampler: str = "uniform",
    mean: float = 0.0,
    std: float = 1.0,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Sample normalized local FM time in [0, 1].

    ``logit-normal`` samples sigmoid(N(mean, std)), a common continuous-time
    flow-matching recipe.  Stage boundaries remain unchanged; this only
    changes density *inside* each RDD stage, so the coarse/middle/fine budget
    stays comparable to the uniform baseline.
    """

    normalized = sampler.lower().replace("_", "-")
    if normalized == "uniform":
        tau = torch.rand(batch_size, device=device)
    elif normalized in {"logit-normal", "logitnormal", "lognormal"}:
        if std <= 0:
            raise ValueError("FM logit-normal std must be positive")
        tau = torch.sigmoid(torch.randn(batch_size, device=device) * std + mean)
    else:
        raise ValueError(f"Unknown FM time sampler: {sampler}")
    return tau.clamp(eps, 1.0 - eps)


@dataclass(frozen=True)
class GuidanceConfig:
    """Time-dependent guidance configuration.

    Time follows the diffusion convention: 0 is the clean endpoint and 1 is
    the highest-noise endpoint.  ``low=0, high=1, schedule='constant'`` is the
    legacy all-step CFG behavior.
    """

    scale: float = 1.0
    low: float = 0.0
    high: float = 1.0
    schedule: str = "constant"
    schedule_power: float = 1.0
    min_scale: float = 1.0

    def validate(self) -> None:
        if self.scale < 1.0 or self.min_scale < 1.0:
            raise ValueError("guidance scales must be >= 1")
        if not (0.0 <= self.low <= self.high <= 1.0):
            raise ValueError("guidance interval must satisfy 0 <= low <= high <= 1")
        if self.schedule_power <= 0:
            raise ValueError("guidance schedule_power must be positive")
        if self.schedule not in {"constant", "linear", "cosine"}:
            raise ValueError(f"Unknown guidance schedule: {self.schedule}")

    def scales(self, normalized_t: torch.Tensor) -> torch.Tensor:
        """Return a per-example CFG/autoguidance scale."""

        self.validate()
        t = normalized_t.float().clamp(0.0, 1.0)
        active = (t >= self.low) & (t <= self.high)
        if self.schedule == "constant" or self.high == self.low:
            scheduled = torch.full_like(t, self.scale)
        else:
            # Sampling travels high -> low.  Ramp guidance from min_scale at
            # the high-noise edge toward scale at the low-noise edge.
            progress = ((self.high - t) / (self.high - self.low)).clamp(0.0, 1.0)
            progress = progress.pow(self.schedule_power)
            if self.schedule == "cosine":
                progress = 0.5 - 0.5 * torch.cos(math.pi * progress)
            scheduled = self.min_scale + (self.scale - self.min_scale) * progress
        return torch.where(active, scheduled, torch.ones_like(scheduled))


def apply_cfg_to_model_output(
    model_output: torch.Tensor,
    scales: torch.Tensor,
    *,
    guided_channels: int = 3,
) -> torch.Tensor:
    """Apply per-example CFG to a [conditional; unconditional] model batch.

    The default of three guided channels exactly matches the legacy DiT code in
    this repository.  Remaining channels (including learned variance and an
    optional drift head) are copied from the conditional branch.
    """

    cond, uncond = model_output.chunk(2, dim=0)
    channels = min(guided_channels, cond.shape[1])
    view_shape = (scales.shape[0],) + (1,) * (cond.ndim - 1)
    guided = uncond[:, :channels] + scales.view(view_shape) * (
        cond[:, :channels] - uncond[:, :channels]
    )
    combined = torch.cat([guided, cond[:, channels:]], dim=1)
    return torch.cat([combined, combined], dim=0)


def apply_autoguidance_to_model_output(
    good_output: torch.Tensor,
    guide_output: torch.Tensor,
    scales: torch.Tensor,
    *,
    guided_channels: Optional[int] = None,
) -> torch.Tensor:
    """Apply Autoguidance: guide.lerp(good, scale).

    ``guided_channels=None`` guides all prediction channels.  For RDD models
    with learned variance, callers should normally pass the latent channel
    count (4) so that variance remains from the good model.
    """

    if good_output.shape != guide_output.shape:
        raise ValueError(
            f"Autoguidance models must have identical outputs, got "
            f"{tuple(good_output.shape)} and {tuple(guide_output.shape)}"
        )
    channels = good_output.shape[1] if guided_channels is None else guided_channels
    channels = min(channels, good_output.shape[1])
    view_shape = (scales.shape[0],) + (1,) * (good_output.ndim - 1)
    guided = guide_output[:, :channels] + scales.view(view_shape) * (
        good_output[:, :channels] - guide_output[:, :channels]
    )
    return torch.cat([guided, good_output[:, channels:]], dim=1)


def pool_teacher_tokens(tokens: torch.Tensor, target_tokens: int) -> torch.Tensor:
    """Average-pool square ViT patch tokens to an RDD stage token grid."""

    source_side = int(round(math.sqrt(tokens.shape[1])))
    target_side = int(round(math.sqrt(target_tokens)))
    if source_side * source_side != tokens.shape[1]:
        raise ValueError(f"Teacher token count is not square: {tokens.shape[1]}")
    if target_side * target_side != target_tokens:
        raise ValueError(f"RDD token count is not square: {target_tokens}")
    features = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], source_side, source_side)
    features = F.adaptive_avg_pool2d(features, (target_side, target_side))
    return features.flatten(2).transpose(1, 2)


def representation_alignment_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Per-example negative cosine alignment loss used by REPA."""

    if student.shape != teacher.shape:
        raise ValueError(
            f"Student/teacher feature shapes differ: {tuple(student.shape)} vs {tuple(teacher.shape)}"
        )
    student = F.normalize(student.float(), dim=-1)
    teacher = F.normalize(teacher.float(), dim=-1)
    return -(student * teacher).sum(dim=-1).mean(dim=-1)
