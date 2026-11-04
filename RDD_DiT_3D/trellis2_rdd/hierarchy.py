"""Topology-conditioned multiresolution decomposition of TRELLIS.2 SLat.

The sparse-structure model is assumed to have produced the finest support.  The
support is therefore data/condition dependent but fixed while the shape latent
is generated.  On that fixed support, active-child averaging and nearest-parent
replication define an idempotent linear projector.  Its complement is the
zero-mean detail subspace used for RDD noise filling and flow matching.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class LatentLevel:
    resolution: int
    logical_coords: Tensor
    model_coords: Tensor
    feats: Tensor


@dataclass(frozen=True)
class DetailTransition:
    parent: LatentLevel
    child: LatentLevel
    upsampled: Tensor
    detail: Tensor


@dataclass(frozen=True)
class MaskedLatentPyramid:
    levels: tuple[LatentLevel, ...]
    transitions: tuple[DetailTransition, ...]


def _group_mean(coords: Tensor, feats: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    unique, inverse, counts = torch.unique(
        coords, dim=0, sorted=True, return_inverse=True, return_counts=True
    )
    means = torch.zeros((unique.shape[0], feats.shape[1]), dtype=feats.dtype, device=feats.device)
    means.index_add_(0, inverse, feats)
    means = means / counts.to(feats.dtype)[:, None]
    return unique, means, inverse


def _keys(coords: Tensor, resolution: int) -> Tensor:
    return (coords[:, 0].long() * resolution + coords[:, 1]) * resolution + coords[:, 2]


def _gather_parent(parent_coords: Tensor, parent_feats: Tensor, child_coords: Tensor, parent_resolution: int) -> Tensor:
    query = torch.div(child_coords, 2, rounding_mode="floor")
    order = torch.argsort(_keys(parent_coords, parent_resolution))
    keys = _keys(parent_coords[order], parent_resolution)
    query_keys = _keys(query, parent_resolution)
    positions = torch.searchsorted(keys, query_keys)
    if bool(torch.any(positions >= keys.numel())):
        raise ValueError("child support contains a missing parent")
    positions = order[positions]
    if not bool(torch.all(parent_coords[positions] == query)):
        raise ValueError("child support contains a missing parent")
    return parent_feats[positions]


def expand_parent_features(parent: LatentLevel, child: LatentLevel, parent_feats: Tensor | None = None) -> Tensor:
    """Nearest-neighbor copy dynamic parent features onto a known child support."""

    if parent_feats is None:
        parent_feats = parent.feats
    if parent_feats.shape[0] != parent.logical_coords.shape[0]:
        raise ValueError("parent_feats must match the parent support")
    return _gather_parent(
        parent.logical_coords, parent_feats, child.logical_coords, parent.resolution
    )


def make_level(fine_coords: Tensor, fine_feats: Tensor, fine_resolution: int, resolution: int) -> LatentLevel:
    if fine_coords.ndim != 2 or fine_coords.shape[1] != 3:
        raise ValueError("fine_coords must have shape [N,3]")
    if fine_feats.ndim != 2 or fine_feats.shape[0] != fine_coords.shape[0]:
        raise ValueError("fine_feats must have shape [N,C]")
    if fine_resolution % resolution:
        raise ValueError("level resolution must divide the fine resolution")
    stride = fine_resolution // resolution
    if stride & (stride - 1):
        raise ValueError("only dyadic levels are supported")
    logical = torch.div(fine_coords, stride, rounding_mode="floor")
    logical, feats, _ = _group_mean(logical, fine_feats)
    # The official model is configured for the finest resolution.  Scaling the
    # lattice coordinates preserves relative physical distances for RoPE.
    model_coords = logical * stride
    return LatentLevel(resolution, logical, model_coords, feats)


def project_detail(logical_coords: Tensor, feats: Tensor) -> Tensor:
    """Project values into the zero-active-sibling-mean detail subspace."""

    parent_coords = torch.div(logical_coords, 2, rounding_mode="floor")
    _, parent_mean, inverse = _group_mean(parent_coords, feats)
    return feats - parent_mean[inverse]


def build_masked_latent_pyramid(
    fine_coords: Tensor,
    fine_feats: Tensor,
    *,
    fine_resolution: int = 32,
    resolutions: tuple[int, ...] = (8, 16, 32),
) -> MaskedLatentPyramid:
    if tuple(sorted(resolutions)) != resolutions or resolutions[-1] != fine_resolution:
        raise ValueError("resolutions must be increasing and end at fine_resolution")
    if any(b != 2 * a for a, b in zip(resolutions, resolutions[1:])):
        raise ValueError("adjacent resolutions must differ by 2x")
    if fine_coords.ndim != 2 or fine_coords.shape[1] != 3:
        raise ValueError("fine_coords must have shape [N,3]")
    if fine_feats.ndim != 2 or fine_feats.shape[0] != fine_coords.shape[0]:
        raise ValueError("fine_feats must have shape [N,C]")
    if fine_coords.numel() and (
        bool(torch.any(fine_coords < 0)) or bool(torch.any(fine_coords >= fine_resolution))
    ):
        raise ValueError("fine_coords must lie inside the finest lattice")

    # Build recursively from adjacent levels.  This is deliberately *not* the
    # same as averaging every fine token directly into every coarse cell: an
    # active-child projector gives each active child one vote, irrespective of
    # how many still-finer tokens that child contains.  Recursive construction
    # is what makes every transition an idempotent active-child projector and
    # therefore makes each detail have exactly zero active-sibling mean.
    logical, feats, _ = _group_mean(fine_coords.long(), fine_feats)
    levels_descending = [LatentLevel(fine_resolution, logical, logical, feats)]
    for parent_resolution in reversed(resolutions[:-1]):
        child = levels_descending[-1]
        parent_logical, parent_feats, _ = _group_mean(
            torch.div(child.logical_coords, 2, rounding_mode="floor"), child.feats
        )
        stride = fine_resolution // parent_resolution
        levels_descending.append(
            LatentLevel(
                parent_resolution,
                parent_logical,
                parent_logical * stride,
                parent_feats,
            )
        )
    levels = tuple(reversed(levels_descending))
    transitions = []
    for parent, child in zip(levels, levels[1:]):
        upsampled = expand_parent_features(parent, child)
        detail = child.feats - upsampled
        transitions.append(DetailTransition(parent, child, upsampled, detail))
    return MaskedLatentPyramid(levels, tuple(transitions))
