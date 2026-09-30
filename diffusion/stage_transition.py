"""Utilities for coupling Gaussian noise across resolution stages."""

from __future__ import annotations

import torch
from torch import Tensor


def lift_noise_with_nullspace_2d(
    coarse_noise: Tensor,
    target_size: int,
    *,
    fresh_noise: Tensor | None = None,
) -> Tensor:
    """Lift coarse white noise and fill only the new fine-grid directions.

    Nearest-neighbour repetition divided by the spatial scale is an isometric
    embedding from an HxW grid into an (sH)x(sW) grid.  The orthogonal
    complement of that block-constant subspace is filled with an independent
    fine-resolution Gaussian.  Consequently, if ``coarse_noise`` and
    ``fresh_noise`` are standard Gaussian, the returned tensor is also a
    standard Gaussian, while its coarse (block-sum/scale) projection is exactly
    ``coarse_noise``.

    This is the covariance-preserving version of "noise filling".  It is not
    the same coupling as redrawing an independent fine-resolution Gaussian.
    """

    if coarse_noise.ndim != 4:
        raise ValueError(
            f"coarse_noise must have shape [N,C,H,W], got {tuple(coarse_noise.shape)}"
        )
    source_h, source_w = coarse_noise.shape[-2:]
    target_size = int(target_size)
    if source_h != source_w or target_size <= source_h or target_size % source_h:
        raise ValueError(
            "noise filling requires square nested grids, got "
            f"{source_h}x{source_w} -> {target_size}x{target_size}"
        )
    scale = target_size // source_h

    # U / sqrt(scale**2) = U / scale is an isometric 2-D lift.
    lifted = coarse_noise.repeat_interleave(scale, dim=-2).repeat_interleave(
        scale, dim=-1
    ) / float(scale)

    if fresh_noise is None:
        fresh_noise = torch.randn_like(lifted)
    if fresh_noise.shape != lifted.shape:
        raise ValueError(
            f"fresh_noise has shape {tuple(fresh_noise.shape)}, expected {tuple(lifted.shape)}"
        )

    # Orthogonal projector onto block-constant tensors is repeat(avg_pool(.));
    # writing it as reshape+mean avoids depending on torch.nn.functional here.
    n, channels = fresh_noise.shape[:2]
    pooled = fresh_noise.reshape(
        n, channels, source_h, scale, source_w, scale
    ).mean(dim=(3, 5))
    projected = pooled.repeat_interleave(scale, dim=-2).repeat_interleave(
        scale, dim=-1
    )
    null_noise = fresh_noise - projected
    return lifted + null_noise


def project_noise_to_coarse_2d(fine_noise: Tensor, target_size: int) -> Tensor:
    """Apply the adjoint of the isometric lift used above.

    This is block averaging multiplied by the spatial scale (equivalently,
    block sum divided by the scale), not ordinary image average pooling.
    """

    if fine_noise.ndim != 4:
        raise ValueError(
            f"fine_noise must have shape [N,C,H,W], got {tuple(fine_noise.shape)}"
        )
    source_h, source_w = fine_noise.shape[-2:]
    target_size = int(target_size)
    if source_h != source_w or target_size <= 0 or source_h % target_size:
        raise ValueError(
            f"cannot project {source_h}x{source_w} noise to {target_size}x{target_size}"
        )
    scale = source_h // target_size
    n, channels = fine_noise.shape[:2]
    pooled = fine_noise.reshape(
        n, channels, target_size, scale, target_size, scale
    ).mean(dim=(3, 5))
    return pooled * float(scale)
