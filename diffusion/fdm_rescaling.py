"""Signal-preserving (SP) and variance-preserving (VP) rescaling from f-DM.

The f-DM paper rescales the signal/noise coefficients for an *entire* lower-
dimensional stage.  The functions in this module implement those coefficient
identities exactly.  They do not, by themselves, turn an already-trained RDD
checkpoint into an f-DM-rescaled model; a sampler may use them as an explicitly
labelled inference-only probe, while a faithful model must use the rescaled
forward process during training as well.
"""

from __future__ import annotations

from typing import Tuple

import torch


VALID_FDM_RESCALINGS = ("none", "sp", "vp")


def validate_fdm_rescaling(mode: str) -> str:
    """Normalize and validate an f-DM rescaling name."""
    normalized = str(mode).strip().lower()
    if normalized not in VALID_FDM_RESCALINGS:
        raise ValueError(
            f"fdm rescaling must be one of {VALID_FDM_RESCALINGS}, got {mode!r}"
        )
    return normalized


def spatial_dimension_ratio(finest_size: int, stage_size: int) -> float:
    """Return f-DM's cumulative dimension ratio for square spatial tensors.

    Channels are unchanged, so reducing each spatial side from 32 to 16 changes
    dimensionality by 4, while reducing 32 to 8 changes it by 16.
    """
    finest_size = int(finest_size)
    stage_size = int(stage_size)
    if finest_size <= 0 or stage_size <= 0:
        raise ValueError("spatial sizes must be positive")
    if finest_size % stage_size != 0:
        raise ValueError(
            f"finest_size={finest_size} must be divisible by stage_size={stage_size}"
        )
    scale = finest_size // stage_size
    return float(scale * scale)


def rescale_alpha_sigma(
    alpha: torch.Tensor,
    sigma: torch.Tensor,
    dimension_ratio: float,
    mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the exact f-DM SP or VP coefficient rescaling.

    Let ``d = M_fine / M_stage`` be the cumulative dimensionality reduction.
    f-DM first uses the signal-preserving coefficients

        alpha_SP = alpha,       sigma_SP = sigma / sqrt(d).

    Its variance-preserving version jointly normalizes those two coefficients:

        (alpha_VP, sigma_VP) = (alpha_SP, sigma_SP)
                               / sqrt(alpha_SP^2 + sigma_SP^2).

    Consequently both variants multiply SNR by ``d``; VP additionally preserves
    ``alpha_VP**2 + sigma_VP**2 == 1`` (when the input schedule is VP).
    """
    mode = validate_fdm_rescaling(mode)
    d = float(dimension_ratio)
    if d < 1.0:
        raise ValueError(f"dimension_ratio must be >= 1, got {d}")
    if alpha.shape != sigma.shape:
        raise ValueError(
            f"alpha and sigma shapes must match, got {alpha.shape} and {sigma.shape}"
        )
    if mode == "none" or d == 1.0:
        return alpha, sigma

    sigma_sp = sigma / d**0.5
    if mode == "sp":
        return alpha, sigma_sp

    normalization = torch.sqrt(alpha.square() + sigma_sp.square())
    tiny = torch.finfo(normalization.dtype).tiny
    normalization = normalization.clamp_min(tiny)
    return alpha / normalization, sigma_sp / normalization
