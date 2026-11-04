"""Dynamic sparse RDD hierarchy for TRELLIS-style structured latents.

This module deliberately does *not* assume that a separate structure model has
already predicted the finest support.  Missing fine voxels are assigned the
canonical empty representation ``[mass=0, feature_mass=0]``.  Ordinary fixed
2x2x2 averaging then produces a hierarchy in which a parent is exactly empty
iff all eight descendants are empty.

Only candidate voxels are materialized:

* the base grid is complete (normally 8^3);
* every later candidate grid is the eight children of the preceding stage's
  non-empty ground-truth parents during teacher-forced training;
* inference uses the same operation on predicted non-empty parents.

The final TRELLIS latent is recovered as ``feature_mass / mass`` for voxels
classified as non-empty.  At the finest training level mass is exactly one for
an active voxel, so this is identical to the original 32-D SLat.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import torch
from torch import Tensor


@dataclass(frozen=True)
class DynamicSparseStage:
    resolution: int
    logical_coords: Tensor
    model_coords: Tensor
    target_rep: Tensor
    nonempty: Tensor
    parent_rep: Tensor | None
    detail: Tensor | None


@dataclass(frozen=True)
class DynamicSparsePyramid:
    stages: tuple[DynamicSparseStage, ...]
    feature_channels: int
    fine_resolution: int


def _validate_coords(coords: Tensor, resolution: int) -> None:
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape [N, 3]")
    if coords.numel() and (
        bool(torch.any(coords < 0)) or bool(torch.any(coords >= resolution))
    ):
        raise ValueError(f"coordinates must lie in [0, {resolution})")


def coordinate_keys(coords: Tensor, resolution: int) -> Tensor:
    """Return collision-free integer keys for coordinates in one lattice."""

    _validate_coords(coords, resolution)
    coords = coords.long()
    return (coords[:, 0] * resolution + coords[:, 1]) * resolution + coords[:, 2]


def full_grid(resolution: int, *, device: torch.device | str | None = None) -> Tensor:
    axis = torch.arange(resolution, device=device, dtype=torch.long)
    return torch.cartesian_prod(axis, axis, axis).reshape(-1, 3)


def _coalesce_mean(coords: Tensor, values: Tensor, resolution: int) -> tuple[Tensor, Tensor]:
    _validate_coords(coords, resolution)
    if values.ndim != 2 or values.shape[0] != coords.shape[0]:
        raise ValueError("values must have shape [N, C]")
    if coords.shape[0] == 0:
        return coords.long(), values
    unique, inverse, counts = torch.unique(
        coords.long(), dim=0, sorted=True, return_inverse=True, return_counts=True
    )
    result = torch.zeros(
        (len(unique), values.shape[1]), dtype=values.dtype, device=values.device
    )
    result.index_add_(0, inverse, values)
    result = result / counts.to(values.dtype)[:, None]
    return unique, result


def pack_fine_representation(fine_feats: Tensor) -> Tensor:
    """Convert active TRELLIS features to ``[mass, feature_mass]``."""

    if fine_feats.ndim != 2:
        raise ValueError("fine_feats must have shape [N, C]")
    mass = torch.ones(
        (fine_feats.shape[0], 1), dtype=fine_feats.dtype, device=fine_feats.device
    )
    return torch.cat((mass, fine_feats), dim=1)


def fixed_eight_pool(
    child_coords: Tensor,
    child_rep: Tensor,
    child_resolution: int,
) -> tuple[Tensor, Tensor]:
    """Pool sparse non-empty children while treating missing children as zero.

    Division is always by eight, not by the number of materialized children.
    Consequently the leading mass channel is the occupied child fraction.
    """

    if child_resolution % 2:
        raise ValueError("child_resolution must be even")
    _validate_coords(child_coords, child_resolution)
    if child_rep.ndim != 2 or child_rep.shape[0] != child_coords.shape[0]:
        raise ValueError("child_rep must have shape [N, C]")
    parent_resolution = child_resolution // 2
    if len(child_coords) == 0:
        return (
            torch.empty((0, 3), dtype=torch.long, device=child_coords.device),
            torch.empty(
                (0, child_rep.shape[1]), dtype=child_rep.dtype, device=child_rep.device
            ),
        )
    parent_query = torch.div(child_coords.long(), 2, rounding_mode="floor")
    parent_coords, inverse = torch.unique(
        parent_query, dim=0, sorted=True, return_inverse=True
    )
    parent_rep = torch.zeros(
        (len(parent_coords), child_rep.shape[1]),
        dtype=child_rep.dtype,
        device=child_rep.device,
    )
    parent_rep.index_add_(0, inverse, child_rep)
    parent_rep = parent_rep / 8.0
    _validate_coords(parent_coords, parent_resolution)
    return parent_coords, parent_rep


def gather_sparse_values(
    source_coords: Tensor,
    source_values: Tensor,
    query_coords: Tensor,
    resolution: int,
    *,
    missing_value: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Gather sparse values and return a boolean presence mask."""

    _validate_coords(source_coords, resolution)
    _validate_coords(query_coords, resolution)
    if source_values.ndim != 2 or source_values.shape[0] != source_coords.shape[0]:
        raise ValueError("source_values must have shape [N, C]")
    output = torch.full(
        (len(query_coords), source_values.shape[1]),
        missing_value,
        dtype=source_values.dtype,
        device=source_values.device,
    )
    present = torch.zeros(len(query_coords), dtype=torch.bool, device=query_coords.device)
    if len(source_coords) == 0 or len(query_coords) == 0:
        return output, present
    order = torch.argsort(coordinate_keys(source_coords, resolution))
    sorted_keys = coordinate_keys(source_coords[order], resolution)
    query_keys = coordinate_keys(query_coords, resolution)
    positions = torch.searchsorted(sorted_keys, query_keys)
    in_range = positions < len(sorted_keys)
    safe_positions = positions.clamp(max=max(len(sorted_keys) - 1, 0))
    matched = in_range & (sorted_keys[safe_positions] == query_keys)
    if bool(matched.any()):
        output[matched] = source_values[order[safe_positions[matched]]]
        present[matched] = True
    return output, present


def expand_eight(parent_coords: Tensor, parent_resolution: int) -> Tensor:
    """Expand every retained parent to its complete 2x2x2 child block."""

    _validate_coords(parent_coords, parent_resolution)
    if len(parent_coords) == 0:
        return torch.empty((0, 3), dtype=torch.long, device=parent_coords.device)
    offsets = torch.tensor(
        list(product((0, 1), repeat=3)), dtype=torch.long, device=parent_coords.device
    )
    children = parent_coords.long()[:, None, :] * 2 + offsets[None, :, :]
    return children.reshape(-1, 3)


def nearest_parent_rep(
    parent_coords: Tensor,
    parent_rep: Tensor,
    child_coords: Tensor,
    parent_resolution: int,
) -> Tensor:
    queries = torch.div(child_coords.long(), 2, rounding_mode="floor")
    gathered, present = gather_sparse_values(
        parent_coords, parent_rep, queries, parent_resolution
    )
    if not bool(torch.all(present)):
        raise ValueError("candidate child has no retained parent")
    return gathered


def project_complete_sibling_detail(coords: Tensor, values: Tensor) -> Tensor:
    """Remove the mean of every complete sibling group from ``values``."""

    if values.ndim != 2 or values.shape[0] != coords.shape[0]:
        raise ValueError("values must have shape [N, C]")
    if len(coords) == 0:
        return values
    parent = torch.div(coords.long(), 2, rounding_mode="floor")
    _, inverse, counts = torch.unique(
        parent, dim=0, sorted=True, return_inverse=True, return_counts=True
    )
    if not bool(torch.all(counts == 8)):
        raise ValueError("detail projection requires complete 2x2x2 sibling groups")
    means = torch.zeros(
        (len(counts), values.shape[1]), dtype=values.dtype, device=values.device
    )
    means.index_add_(0, inverse, values)
    means = means / 8.0
    return values - means[inverse]


def build_dynamic_sparse_pyramid(
    fine_coords: Tensor,
    fine_feats: Tensor,
    *,
    fine_resolution: int = 32,
    resolutions: tuple[int, ...] = (8, 16, 32),
) -> DynamicSparsePyramid:
    """Build teacher-forced candidate sets for dynamic sparse RDD training."""

    if tuple(sorted(resolutions)) != resolutions or resolutions[-1] != fine_resolution:
        raise ValueError("resolutions must increase and end at fine_resolution")
    if any(b != 2 * a for a, b in zip(resolutions, resolutions[1:])):
        raise ValueError("resolutions must form a dyadic hierarchy")
    _validate_coords(fine_coords, fine_resolution)
    if fine_feats.ndim != 2 or fine_feats.shape[0] != fine_coords.shape[0]:
        raise ValueError("fine_feats must have shape [N, C]")

    fine_coords, fine_feats = _coalesce_mean(
        fine_coords.long(), fine_feats, fine_resolution
    )
    active: dict[int, tuple[Tensor, Tensor]] = {
        fine_resolution: (fine_coords, pack_fine_representation(fine_feats))
    }
    current_resolution = fine_resolution
    while current_resolution > resolutions[0]:
        coords, rep = active[current_resolution]
        parent_coords, parent_rep = fixed_eight_pool(coords, rep, current_resolution)
        current_resolution //= 2
        active[current_resolution] = (parent_coords, parent_rep)

    stages: list[DynamicSparseStage] = []
    previous_active_coords: Tensor | None = None
    previous_active_rep: Tensor | None = None
    for stage_index, resolution in enumerate(resolutions):
        active_coords, active_rep = active[resolution]
        if stage_index == 0:
            candidates = full_grid(resolution, device=fine_coords.device)
            parent_rep = None
            detail = None
        else:
            assert previous_active_coords is not None and previous_active_rep is not None
            candidates = expand_eight(previous_active_coords, resolutions[stage_index - 1])
            parent_rep = nearest_parent_rep(
                previous_active_coords,
                previous_active_rep,
                candidates,
                resolutions[stage_index - 1],
            )
            detail = None
        target, nonempty = gather_sparse_values(
            active_coords, active_rep, candidates, resolution
        )
        if parent_rep is not None:
            detail = target - parent_rep
        stride = fine_resolution // resolution
        stages.append(
            DynamicSparseStage(
                resolution=resolution,
                logical_coords=candidates,
                model_coords=candidates * stride,
                target_rep=target,
                nonempty=nonempty,
                parent_rep=parent_rep,
                detail=detail,
            )
        )
        previous_active_coords = active_coords
        previous_active_rep = active_rep

    return DynamicSparsePyramid(tuple(stages), fine_feats.shape[1], fine_resolution)


def build_generated_parent_stage(
    ground_truth_stage: DynamicSparseStage,
    *,
    parent_coords: Tensor,
    parent_rep: Tensor,
    parent_resolution: int,
    fine_resolution: int,
) -> DynamicSparseStage:
    """Build a refinement example from a model-generated parent state.

    Teacher-forced refinement only sees exact ground-truth parents.  At
    inference, however, both the retained support and the parent representation
    come from the preceding generated stage.  This helper reconstructs that
    inference candidate set while gathering supervision from the ordinary-
    average ground-truth pyramid.

    A generated parent generally does not equal the mean of its ground-truth
    children.  RDD refinement is deliberately restricted to zero-mean sibling
    detail, so the velocity target uses the closest admissible zero-mean detail
    instead of pretending that a later stage can repair a coarse mean error.
    The non-empty target remains exact and can therefore train pruning under
    the real generated-state distribution.
    """

    if ground_truth_stage.resolution != 2 * parent_resolution:
        raise ValueError("ground-truth stage must be twice parent_resolution")
    if parent_coords.ndim != 2 or parent_coords.shape[1] != 3:
        raise ValueError("parent_coords must have shape [N, 3]")
    if parent_rep.ndim != 2 or len(parent_rep) != len(parent_coords):
        raise ValueError("parent_rep must have shape [N, C]")
    if parent_rep.shape[1] != ground_truth_stage.target_rep.shape[1]:
        raise ValueError("parent and target representations must have equal channels")

    candidates = expand_eight(parent_coords.long(), parent_resolution)
    repeated_parent = parent_rep.repeat_interleave(8, dim=0)
    target, nonempty = gather_sparse_values(
        ground_truth_stage.logical_coords,
        ground_truth_stage.target_rep,
        candidates,
        ground_truth_stage.resolution,
    )
    admissible_detail = project_complete_sibling_detail(
        candidates, target - repeated_parent
    )
    stride = fine_resolution // ground_truth_stage.resolution
    return DynamicSparseStage(
        resolution=ground_truth_stage.resolution,
        logical_coords=candidates,
        model_coords=candidates * stride,
        target_rep=target,
        nonempty=nonempty,
        parent_rep=repeated_parent,
        detail=admissible_detail,
    )


def add_empty_parent_groups(
    stage: DynamicSparseStage,
    *,
    parent_resolution: int,
    fine_resolution: int,
    ratio: float,
    generator: torch.Generator | None = None,
) -> DynamicSparseStage:
    """Append children of sampled GT-empty parents as hard-negative groups.

    Teacher forcing normally expands only GT-non-empty parents.  A generated
    false-positive parent, however, also gets expanded at inference.  Without
    these groups a refinement stage never learns how to delete that branch.
    The appended parent, child target, and detail are all exactly zero, so the
    ordinary-average invariant is preserved.
    """

    if ratio <= 0:
        return stage
    if stage.parent_rep is None or stage.detail is None:
        raise ValueError("empty parent augmentation only applies after the base stage")
    if stage.resolution != parent_resolution * 2:
        raise ValueError("stage resolution must be twice parent_resolution")
    if fine_resolution % stage.resolution:
        raise ValueError("fine_resolution must be divisible by stage resolution")

    existing_parents = torch.unique(
        torch.div(stage.logical_coords.long(), 2, rounding_mode="floor"),
        dim=0,
        sorted=True,
    )
    all_parents = full_grid(parent_resolution, device=stage.logical_coords.device)
    marker = torch.ones(
        (len(existing_parents), 1),
        dtype=stage.target_rep.dtype,
        device=stage.target_rep.device,
    )
    _, present = gather_sparse_values(
        existing_parents, marker, all_parents, parent_resolution
    )
    empty_parents = all_parents[~present]
    requested = int(round(len(existing_parents) * float(ratio)))
    count = min(max(requested, 0), len(empty_parents))
    if count == 0:
        return stage
    selected = empty_parents[
        torch.randperm(len(empty_parents), generator=generator, device=empty_parents.device)[
            :count
        ]
    ]
    extra_coords = expand_eight(selected, parent_resolution)
    extra_rep = torch.zeros(
        (len(extra_coords), stage.target_rep.shape[1]),
        dtype=stage.target_rep.dtype,
        device=stage.target_rep.device,
    )
    stride = fine_resolution // stage.resolution
    return DynamicSparseStage(
        resolution=stage.resolution,
        logical_coords=torch.cat((stage.logical_coords, extra_coords), dim=0),
        model_coords=torch.cat((stage.model_coords, extra_coords * stride), dim=0),
        target_rep=torch.cat((stage.target_rep, extra_rep), dim=0),
        nonempty=torch.cat(
            (
                stage.nonempty,
                torch.zeros(
                    len(extra_coords),
                    dtype=torch.bool,
                    device=stage.nonempty.device,
                ),
            ),
            dim=0,
        ),
        parent_rep=torch.cat((stage.parent_rep, extra_rep), dim=0),
        detail=torch.cat((stage.detail, extra_rep), dim=0),
    )


def select_predicted_nonempty(
    coords: Tensor,
    representation: Tensor,
    nonempty_logits: Tensor,
    *,
    probability_threshold: float,
    mass_threshold: float = 0.0,
    min_keep: int = 1,
    mode: str = "or",
) -> Tensor:
    """Recall-first pruning using both the auxiliary head and generated mass."""

    if representation.ndim != 2 or representation.shape[0] != len(coords):
        raise ValueError("representation must have shape [N, C]")
    logits = nonempty_logits.reshape(-1)
    if len(logits) != len(coords):
        raise ValueError("nonempty_logits must have one value per coordinate")
    probabilities = torch.sigmoid(logits)
    probability_keep = probabilities >= probability_threshold
    mass_keep = representation[:, 0] >= mass_threshold
    if mode == "or":
        keep = probability_keep | mass_keep
    elif mode == "and":
        keep = probability_keep & mass_keep
    elif mode == "probability":
        keep = probability_keep
    elif mode == "mass":
        keep = mass_keep
    else:
        raise ValueError(f"unknown pruning mode: {mode}")
    if len(coords) and int(keep.sum()) < min(min_keep, len(coords)):
        top = torch.topk(probabilities, k=min(min_keep, len(coords))).indices
        keep[top] = True
    return keep


def dilate_keep_mask(
    coords: Tensor,
    keep: Tensor,
    *,
    resolution: int,
    radius: int,
) -> Tensor:
    """Retain already-computed candidates near a retained token."""

    if radius <= 0 or not len(coords) or not bool(keep.any()):
        return keep
    retained = coords[keep]
    offsets = torch.tensor(
        list(product(range(-radius, radius + 1), repeat=3)),
        dtype=torch.long,
        device=coords.device,
    )
    queries = retained[:, None, :] + offsets[None, :, :]
    valid = torch.all((queries >= 0) & (queries < resolution), dim=2)
    halo_keys = torch.unique(
        coordinate_keys(queries[valid], resolution), sorted=True
    )
    candidate_keys = coordinate_keys(coords, resolution)
    positions = torch.searchsorted(halo_keys, candidate_keys)
    in_range = positions < len(halo_keys)
    safe = positions.clamp(max=max(len(halo_keys) - 1, 0))
    return keep | (in_range & (halo_keys[safe] == candidate_keys))


def recover_trellis_features(
    representation: Tensor,
    *,
    normalize_by_mass: bool = False,
    mass_floor: float = 0.25,
) -> Tensor:
    """Convert final non-empty ``[mass, feature_mass]`` rows to SLat.

    A true active finest-level target has mass exactly one, hence its feature
    mass already equals the SLat.  Returning that channel directly is much more
    stable off-manifold: dividing a false-positive token by a generated mass
    near zero can otherwise create an arbitrarily large decoder input.

    ``normalize_by_mass=True`` is retained for inspecting coarse semantic means
    and clamps the denominator to an explicit floor.
    """

    if representation.ndim != 2 or representation.shape[1] < 2:
        raise ValueError("representation must have shape [N, 1+C]")
    feature_mass = representation[:, 1:]
    if not normalize_by_mass:
        return feature_mass
    if mass_floor <= 0:
        raise ValueError("mass_floor must be positive")
    mass = representation[:, :1].clamp_min(mass_floor)
    return feature_mass / mass
