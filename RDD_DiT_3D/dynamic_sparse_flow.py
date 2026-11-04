"""Flow-matching objective and dynamic-pruning sampler for sparse RDD-3D."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from dynamic_sparse_representation import (
    DynamicSparseStage,
    dilate_keep_mask,
    expand_eight,
    full_grid,
    project_complete_sibling_detail,
    recover_trellis_features,
    select_predicted_nonempty,
)


@dataclass(frozen=True)
class StageTrainingState:
    state: Tensor
    target_velocity: Tensor
    global_time: Tensor
    band_width: float


@dataclass(frozen=True)
class DynamicSampleStage:
    resolution: int
    candidate_tokens: int
    retained_tokens: int
    mean_nonempty_probability: float
    mass_mean: float


@dataclass(frozen=True)
class DynamicSample:
    coords: Tensor
    representation: Tensor
    trellis_features: Tensor
    stages: tuple[DynamicSampleStage, ...]
    retained_coords_by_stage: tuple[Tensor, ...]
    retained_representations_by_stage: tuple[Tensor, ...]
    candidate_coords_by_stage: tuple[Tensor, ...]
    candidate_representations_by_stage: tuple[Tensor, ...]
    nonempty_probabilities_by_stage: tuple[Tensor, ...]
    masses_by_stage: tuple[Tensor, ...]


def time_boundaries(time_fractions: tuple[float, ...]) -> np.ndarray:
    """Return clean->noise boundaries; fractions are ordered fine->coarse."""

    fractions = np.asarray(time_fractions, dtype=np.float64)
    if np.any(fractions <= 0):
        raise ValueError("time fractions must be positive")
    fractions = fractions / fractions.sum()
    return np.concatenate(([0.0], np.cumsum(fractions)))


def stage_band(stage_index: int, num_stages: int, boundaries: np.ndarray) -> tuple[float, float]:
    band = num_stages - 1 - stage_index
    return float(boundaries[band]), float(boundaries[band + 1])


def fm_data_time_boundaries(time_fractions: tuple[float, ...]) -> np.ndarray:
    """Return noise->data FM boundaries for coarse->fine generation.

    ``time_fractions`` keeps the repository-wide resolution order
    (fine->coarse).  RDD-FM data time runs in the opposite spatial order, so a
    300/300/400 fine/middle/coarse allocation becomes coarse/middle/fine
    intervals 400/300/300 and boundaries ``[0, .4, .7, 1]``.
    """

    fractions = np.asarray(time_fractions, dtype=np.float64)
    if np.any(fractions <= 0):
        raise ValueError("time fractions must be positive")
    fractions = fractions / fractions.sum()
    return np.concatenate(([0.0], np.cumsum(fractions[::-1])))


def fm_data_stage_band(
    stage_index: int, num_stages: int, boundaries: np.ndarray
) -> tuple[float, float]:
    """Return one coarse->fine stage interval in FM noise->data time."""

    if not 0 <= stage_index < num_stages:
        raise IndexError(stage_index)
    return float(boundaries[stage_index]), float(boundaries[stage_index + 1])


def make_stage_training_state(
    stage: DynamicSparseStage,
    *,
    stage_index: int,
    num_stages: int,
    local_p: Tensor,
    time_fractions: tuple[float, ...],
    sigma_min: float = 1e-5,
) -> StageTrainingState:
    boundaries = time_boundaries(time_fractions)
    low, high = stage_band(stage_index, num_stages, boundaries)
    width = high - low
    p = local_p.to(device=stage.target_rep.device, dtype=stage.target_rep.dtype).reshape(())
    sigma = sigma_min + (1.0 - sigma_min) * p
    if stage.parent_rep is None:
        noise = torch.randn_like(stage.target_rep)
        state = (1.0 - p) * stage.target_rep + sigma * noise
        local_velocity = (1.0 - sigma_min) * noise - stage.target_rep
    else:
        assert stage.detail is not None
        noise = project_complete_sibling_detail(
            stage.logical_coords, torch.randn_like(stage.target_rep)
        )
        state = stage.parent_rep + (1.0 - p) * stage.detail + sigma * noise
        local_velocity = (1.0 - sigma_min) * noise - stage.detail
    return StageTrainingState(
        state=state,
        target_velocity=local_velocity / width,
        global_time=torch.as_tensor(low + width * float(p), device=stage.target_rep.device),
        band_width=width,
    )


def make_rdd_fm_training_state(
    stage: DynamicSparseStage,
    *,
    stage_index: int,
    num_stages: int,
    local_p: Tensor,
    time_fractions: tuple[float, ...],
    train_eps: float = 1e-5,
) -> StageTrainingState:
    """Build the exact 3-D analogue of the 2-D RDD-FM training path.

    The network is conditioned only on global FM data time ``t``.  ``local_p``
    is merely a uniform random number used to sample that global interval.

    For a stage ``[s_k, s_{k+1}]`` this implements the same equations as
    :mod:`diffusion.flow_matching_with_drift`::

      ``x_t = t*x_1 + (1-t)*eps + b(t)*d``
      ``u_t = x_1 - eps + b'(t)*d``

    where ``d = UP(DOWN(x_1)) - x_1``.  In the sparse hierarchy the exact
    upsampled pooled representation is already stored as ``parent_rep``, so
    ``d = parent_rep - target_rep = -detail`` without a dense voxel grid.
    """

    boundaries = fm_data_time_boundaries(time_fractions)
    low, high = fm_data_stage_band(stage_index, num_stages, boundaries)
    width = high - low
    if train_eps < 0 or 2.0 * train_eps >= width:
        raise ValueError("train_eps must be non-negative and smaller than half a stage")

    raw_p = local_p.to(device=stage.target_rep.device, dtype=stage.target_rep.dtype).reshape(())
    if bool((raw_p < 0) | (raw_p > 1)):
        raise ValueError("local_p must lie in [0, 1]")
    p_margin = train_eps / width
    p = p_margin + raw_p * (1.0 - 2.0 * p_margin)
    t = torch.as_tensor(low, device=stage.target_rep.device, dtype=stage.target_rep.dtype)
    t = t + width * p

    noise = torch.randn_like(stage.target_rep)
    drift = None if stage.parent_rep is None else stage.parent_rep - stage.target_rep
    state = t * stage.target_rep + (1.0 - t) * noise
    target_velocity = stage.target_rep - noise
    if drift is not None:
        drift_coefficient = t * (high - t) / width
        drift_derivative = (high - 2.0 * t) / width
        state = state + drift_coefficient * drift
        target_velocity = target_velocity + drift_derivative * drift

    return StageTrainingState(
        state=state,
        target_velocity=target_velocity,
        global_time=t,
        band_width=width,
    )


@torch.no_grad()
def sample_dynamic_sparse_rdd(
    model,
    *,
    resolutions: tuple[int, ...],
    steps: tuple[int, ...],
    time_fractions: tuple[float, ...],
    probability_thresholds: tuple[float, ...],
    mass_thresholds: tuple[float, ...],
    condition: Tensor | None = None,
    state_channels: int = 33,
    sigma_min: float = 1e-5,
    seed: int = 0,
    min_keep: int = 1,
    solver: str = "euler",
    pruning_mode: str = "or",
    force_keep_resolutions: tuple[int, ...] = (),
    halo_radii: tuple[int, ...] | None = None,
    device: torch.device | str = "cuda",
) -> DynamicSample:
    if not (
        len(resolutions)
        == len(steps)
        == len(time_fractions)
        == len(probability_thresholds)
        == len(mass_thresholds)
    ):
        raise ValueError("all stage settings must have identical lengths")
    if any(b != 2 * a for a, b in zip(resolutions, resolutions[1:])):
        raise ValueError("resolutions must form a dyadic hierarchy")
    if solver not in ("euler", "heun"):
        raise ValueError("solver must be euler or heun")
    if halo_radii is None:
        halo_radii = tuple(0 for _ in resolutions)
    if len(halo_radii) != len(resolutions):
        raise ValueError("halo_radii must match resolutions")
    boundaries = time_boundaries(time_fractions)
    generator = torch.Generator(device=device).manual_seed(seed)
    parent_coords = None
    parent_rep = None
    reports: list[DynamicSampleStage] = []
    retained_coords_by_stage: list[Tensor] = []
    retained_representations_by_stage: list[Tensor] = []
    candidate_coords_by_stage: list[Tensor] = []
    candidate_representations_by_stage: list[Tensor] = []
    nonempty_probabilities_by_stage: list[Tensor] = []
    masses_by_stage: list[Tensor] = []
    for stage_index, (resolution, num_steps) in enumerate(zip(resolutions, steps)):
        if stage_index == 0:
            coords = full_grid(resolution, device=device)
            state = torch.randn(
                (len(coords), state_channels), generator=generator, device=device
            )
        else:
            assert parent_coords is not None and parent_rep is not None
            coords = expand_eight(parent_coords, resolutions[stage_index - 1])
            base = parent_rep.repeat_interleave(8, dim=0)
            noise = torch.randn(
                base.shape, generator=generator, device=device, dtype=base.dtype
            )
            noise = project_complete_sibling_detail(coords, noise)
            state = base + noise

        low, high = stage_band(stage_index, len(resolutions), boundaries)
        stage_times = np.linspace(high, low, int(num_steps) + 1)
        logits = torch.zeros(len(coords), device=device)
        for current, previous in zip(stage_times[:-1], stage_times[1:]):
            velocity, logits = model(
                state,
                coords,
                resolution=resolution,
                global_time=float(current),
                condition=condition,
            )
            if stage_index > 0:
                velocity = project_complete_sibling_detail(coords, velocity)
            delta = float(current - previous)
            if solver == "euler":
                state = state - delta * velocity
            else:
                predictor = state - delta * velocity
                corrected, _ = model(
                    predictor,
                    coords,
                    resolution=resolution,
                    global_time=float(previous),
                    condition=condition,
                )
                if stage_index > 0:
                    corrected = project_complete_sibling_detail(coords, corrected)
                state = state - delta * 0.5 * (velocity + corrected)

        # Ask the same model for a clean-endpoint occupancy decision.  This is
        # still one generator, not a separate structure network.
        _, logits = model(
            state,
            coords,
            resolution=resolution,
            global_time=float(low),
            condition=condition,
        )
        keep = select_predicted_nonempty(
            coords,
            state,
            logits,
            probability_threshold=probability_thresholds[stage_index],
            mass_threshold=mass_thresholds[stage_index],
            min_keep=min_keep,
            mode=pruning_mode,
        )
        if resolution in force_keep_resolutions:
            keep = torch.ones_like(keep)
        keep = dilate_keep_mask(
            coords,
            keep,
            resolution=resolution,
            radius=int(halo_radii[stage_index]),
        )
        probabilities = torch.sigmoid(logits)
        candidate_coords_by_stage.append(coords.detach().clone())
        # Preserve the complete generated endpoint distribution, including
        # candidates rejected by the pruning rule.  Retained parent states are
        # insufficient for supervising the occupancy head on false negatives.
        candidate_representations_by_stage.append(state.detach().clone())
        nonempty_probabilities_by_stage.append(probabilities.detach().clone())
        masses_by_stage.append(state[:, 0].detach().clone())
        reports.append(
            DynamicSampleStage(
                resolution=resolution,
                candidate_tokens=len(coords),
                retained_tokens=int(keep.sum()),
                mean_nonempty_probability=float(probabilities.mean()),
                mass_mean=float(state[:, 0].mean()),
            )
        )
        parent_coords = coords[keep]
        parent_rep = state[keep]
        retained_coords_by_stage.append(parent_coords.detach().clone())
        retained_representations_by_stage.append(parent_rep.detach().clone())

    assert parent_coords is not None and parent_rep is not None
    return DynamicSample(
        coords=parent_coords,
        representation=parent_rep,
        trellis_features=recover_trellis_features(parent_rep),
        stages=tuple(reports),
        retained_coords_by_stage=tuple(retained_coords_by_stage),
        retained_representations_by_stage=tuple(retained_representations_by_stage),
        candidate_coords_by_stage=tuple(candidate_coords_by_stage),
        candidate_representations_by_stage=tuple(candidate_representations_by_stage),
        nonempty_probabilities_by_stage=tuple(nonempty_probabilities_by_stage),
        masses_by_stage=tuple(masses_by_stage),
    )
