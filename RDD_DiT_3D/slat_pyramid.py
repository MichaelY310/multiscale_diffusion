"""Exact sparse dyadic pyramids for RDD on structured 3D latents.

The operators in this file deliberately implement *dense-zero* average pooling:
missing children contribute zero and every parent is divided by eight.  This is
different from the common sparse-convolution convention of averaging only the
active children.  Dense-zero pooling is required for ``P = up(down(.))`` to be
a fixed linear projector and for the RDD drift ``P(x) - x`` to have the stated
algebraic properties.

Coordinates are integer tensors with shape ``[N, 4]`` and layout
``(batch, x, y, z)``.  Features have shape ``[N, C]``.  Coordinates must be
unique, but they may be unsorted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor


@dataclass(frozen=True)
class SparseGrid:
    """A batched sparse voxel grid."""

    coords: Tensor
    feats: Tensor
    resolution: int

    def __post_init__(self) -> None:
        if self.coords.ndim != 2 or self.coords.shape[1] != 4:
            raise ValueError("coords must have shape [N, 4] in (batch,x,y,z) order")
        if self.feats.ndim != 2 or self.feats.shape[0] != self.coords.shape[0]:
            raise ValueError("feats must have shape [N, C] and match coords")
        if self.coords.dtype not in (torch.int32, torch.int64):
            raise TypeError("coords must be an integer tensor")
        if self.resolution <= 0:
            raise ValueError("resolution must be positive")
        if self.coords.numel():
            xyz = self.coords[:, 1:]
            if int(xyz.min()) < 0 or int(xyz.max()) >= self.resolution:
                raise ValueError("coordinate lies outside the declared resolution")
            if torch.unique(self.coords, dim=0).shape[0] != self.coords.shape[0]:
                raise ValueError("coords must be unique")

    @property
    def channels(self) -> int:
        return self.feats.shape[1]

    @property
    def num_tokens(self) -> int:
        return self.coords.shape[0]


@dataclass(frozen=True)
class PyramidTransition:
    """Exact decomposition from one fine level to its dyadic parent.

    ``candidate`` contains all eight children of every active parent.  Fine
    voxels absent from the input have an exact zero target on that candidate
    support.  Consequently ``target = upsampled + residual`` exactly, including
    the negative residuals needed to cancel inactive siblings.
    """

    coarse: SparseGrid
    candidate: SparseGrid
    target: SparseGrid
    residual: SparseGrid
    target_active: Tensor

    @property
    def active_fraction(self) -> float:
        if self.target_active.numel() == 0:
            return 0.0
        return float(self.target_active.float().mean())

    @property
    def candidate_waste(self) -> float:
        return 1.0 - self.active_fraction


@dataclass(frozen=True)
class MaskedPyramidTransition:
    """Projection on a known fine support.

    Here pooling divides by the number of active children.  Conditioned on the
    (fixed) topology mask this is a linear, idempotent projection on the active
    feature vector.  It is the operator used by the proposed TRELLIS.2 shape
    fine-tune after the separate structure model has supplied the fine support.
    """

    coarse: SparseGrid
    upsampled_on_active: SparseGrid
    residual: SparseGrid


_CHILD_OFFSETS = torch.tensor(
    [[x, y, z] for x in range(2) for y in range(2) for z in range(2)],
    dtype=torch.long,
)


def _lexicographic_keys(coords: Tensor, resolution: int) -> Tensor:
    """Return collision-free sortable keys for (batch,x,y,z) coordinates."""

    if coords.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=coords.device)
    r = int(resolution)
    return (((coords[:, 0].long() * r + coords[:, 1]) * r + coords[:, 2]) * r + coords[:, 3])


def sort_grid(grid: SparseGrid) -> SparseGrid:
    order = torch.argsort(_lexicographic_keys(grid.coords, grid.resolution))
    return SparseGrid(grid.coords[order], grid.feats[order], grid.resolution)


def zero_filled_downsample(grid: SparseGrid) -> SparseGrid:
    """Dyadically pool with missing children equal to zero.

    The sum of observed child features is divided by eight, never by the number
    of observed children.  The output support contains one parent for each
    parent block with at least one active child.
    """

    if grid.resolution % 2:
        raise ValueError("dyadic downsampling requires an even resolution")
    if grid.num_tokens == 0:
        return SparseGrid(
            grid.coords.new_empty((0, 4)),
            grid.feats.new_empty((0, grid.channels)),
            grid.resolution // 2,
        )

    parent_coords = grid.coords.clone()
    parent_coords[:, 1:] = torch.div(parent_coords[:, 1:], 2, rounding_mode="floor")
    unique, inverse = torch.unique(parent_coords, dim=0, sorted=True, return_inverse=True)
    pooled = torch.zeros(
        (unique.shape[0], grid.channels), dtype=grid.feats.dtype, device=grid.feats.device
    )
    pooled.index_add_(0, inverse, grid.feats)
    pooled = pooled / 8.0
    return sort_grid(SparseGrid(unique, pooled, grid.resolution // 2))


def masked_downsample(grid: SparseGrid) -> SparseGrid:
    """Average active children only, conditional on a fixed topology mask."""

    if grid.resolution % 2:
        raise ValueError("dyadic downsampling requires an even resolution")
    if grid.num_tokens == 0:
        return SparseGrid(
            grid.coords.new_empty((0, 4)),
            grid.feats.new_empty((0, grid.channels)),
            grid.resolution // 2,
        )
    parent_coords = grid.coords.clone()
    parent_coords[:, 1:] = torch.div(parent_coords[:, 1:], 2, rounding_mode="floor")
    unique, inverse, counts = torch.unique(
        parent_coords, dim=0, sorted=True, return_inverse=True, return_counts=True
    )
    pooled = torch.zeros(
        (unique.shape[0], grid.channels), dtype=grid.feats.dtype, device=grid.feats.device
    )
    pooled.index_add_(0, inverse, grid.feats)
    pooled = pooled / counts.to(grid.feats.dtype)[:, None]
    return sort_grid(SparseGrid(unique, pooled, grid.resolution // 2))


def nearest_on_known_support(coarse: SparseGrid, fine_coords: Tensor) -> SparseGrid:
    """Replicate parents only at the supplied active fine coordinates."""

    parent_query = fine_coords.clone()
    parent_query[:, 1:] = torch.div(parent_query[:, 1:], 2, rounding_mode="floor")
    feats, found = gather_from_grid(coarse, parent_query)
    if not bool(torch.all(found)):
        raise ValueError("fine support contains a child without an active parent")
    return SparseGrid(fine_coords, feats, coarse.resolution * 2)


def decompose_masked_level(fine: SparseGrid) -> MaskedPyramidTransition:
    """Decompose known-support features into parent means and zero-mean detail."""

    coarse = masked_downsample(fine)
    upsampled = nearest_on_known_support(coarse, fine.coords)
    residual = SparseGrid(fine.coords, fine.feats - upsampled.feats, fine.resolution)
    return MaskedPyramidTransition(coarse, upsampled, residual)


def masked_projector(grid: SparseGrid) -> SparseGrid:
    return decompose_masked_level(grid).upsampled_on_active


def expand_all_children(coarse: SparseGrid) -> SparseGrid:
    """Nearest-neighbor replicate every parent feature to its eight children."""

    if coarse.num_tokens == 0:
        return SparseGrid(
            coarse.coords.new_empty((0, 4)),
            coarse.feats.new_empty((0, coarse.channels)),
            coarse.resolution * 2,
        )
    offsets = _CHILD_OFFSETS.to(device=coarse.coords.device, dtype=coarse.coords.dtype)
    n = coarse.num_tokens
    children = coarse.coords[:, None, :].expand(n, 8, 4).clone()
    children[:, :, 1:] = 2 * children[:, :, 1:] + offsets[None]
    feats = coarse.feats[:, None, :].expand(n, 8, coarse.channels).reshape(-1, coarse.channels)
    return sort_grid(SparseGrid(children.reshape(-1, 4), feats, coarse.resolution * 2))


def gather_from_grid(source: SparseGrid, query_coords: Tensor, *, default: float = 0.0) -> tuple[Tensor, Tensor]:
    """Gather sparse features at coordinates and return ``(features, found)``."""

    if query_coords.ndim != 2 or query_coords.shape[1] != 4:
        raise ValueError("query_coords must have shape [N, 4]")
    out = torch.full(
        (query_coords.shape[0], source.channels),
        default,
        dtype=source.feats.dtype,
        device=source.feats.device,
    )
    found = torch.zeros(query_coords.shape[0], dtype=torch.bool, device=query_coords.device)
    if source.num_tokens == 0 or query_coords.shape[0] == 0:
        return out, found

    sorted_source = sort_grid(source)
    source_keys = _lexicographic_keys(sorted_source.coords, source.resolution)
    query_keys = _lexicographic_keys(query_coords, source.resolution)
    positions = torch.searchsorted(source_keys, query_keys)
    within = positions < source_keys.numel()
    safe_positions = positions.clamp(max=max(source_keys.numel() - 1, 0))
    found = within & (source_keys[safe_positions] == query_keys)
    out[found] = sorted_source.feats[safe_positions[found]]
    return out, found


def decompose_level(fine: SparseGrid) -> PyramidTransition:
    """Build one exact coarse-to-fine Haar-like RDD transition."""

    coarse = zero_filled_downsample(fine)
    candidate = expand_all_children(coarse)
    target_feats, active = gather_from_grid(fine, candidate.coords, default=0.0)
    target = SparseGrid(candidate.coords, target_feats, fine.resolution)
    residual = SparseGrid(candidate.coords, target_feats - candidate.feats, fine.resolution)
    return PyramidTransition(coarse, candidate, target, residual, active)


def build_pyramid(finest: SparseGrid, min_resolution: int) -> list[PyramidTransition]:
    """Return fine-to-coarse transitions until ``min_resolution`` is reached."""

    if min_resolution <= 0 or min_resolution > finest.resolution:
        raise ValueError("min_resolution must be in (0, finest.resolution]")
    transitions: list[PyramidTransition] = []
    current = finest
    while current.resolution > min_resolution:
        if current.resolution % 2:
            raise ValueError("all intermediate resolutions must be dyadic")
        transition = decompose_level(current)
        transitions.append(transition)
        current = transition.coarse
    if current.resolution != min_resolution:
        raise ValueError("min_resolution is not on the dyadic hierarchy")
    return transitions


def projector_on_candidate(fine: SparseGrid) -> SparseGrid:
    """Apply ``P = Up(Down(.))`` on the complete child candidate support."""

    return decompose_level(fine).candidate


def dense_from_sparse(grid: SparseGrid, batch_size: int | None = None) -> Tensor:
    """Materialize a small sparse grid for tests and diagnostics only."""

    if batch_size is None:
        batch_size = int(grid.coords[:, 0].max()) + 1 if grid.num_tokens else 1
    dense = torch.zeros(
        (batch_size, grid.resolution, grid.resolution, grid.resolution, grid.channels),
        dtype=grid.feats.dtype,
        device=grid.feats.device,
    )
    if grid.num_tokens:
        b, x, y, z = grid.coords.long().unbind(dim=1)
        dense[b, x, y, z] = grid.feats
    return dense


def sparse_from_dense(dense: Tensor, active: Tensor | None = None) -> SparseGrid:
    """Create a sparse grid from ``[B,R,R,R,C]`` data."""

    if dense.ndim != 5 or len(set(dense.shape[1:4])) != 1:
        raise ValueError("dense must have shape [B,R,R,R,C]")
    if active is None:
        active = torch.any(dense != 0, dim=-1)
    if active.shape != dense.shape[:-1]:
        raise ValueError("active mask must have shape [B,R,R,R]")
    coords = torch.nonzero(active, as_tuple=False)
    feats = dense[active]
    return SparseGrid(coords.to(torch.long), feats, dense.shape[1])


def transition_table(transitions: Iterable[PyramidTransition]) -> list[dict[str, float | int]]:
    """Create serializable token/sparsity statistics for reports."""

    rows: list[dict[str, float | int]] = []
    for transition in reversed(list(transitions)):
        rows.append(
            {
                "coarse_resolution": transition.coarse.resolution,
                "fine_resolution": transition.target.resolution,
                "coarse_tokens": transition.coarse.num_tokens,
                "candidate_children": transition.candidate.num_tokens,
                "active_fine_tokens": int(transition.target_active.sum()),
                "active_fraction": transition.active_fraction,
                "candidate_waste": transition.candidate_waste,
                "residual_rms": float(transition.residual.feats.float().square().mean().sqrt())
                if transition.residual.num_tokens
                else 0.0,
            }
        )
    return rows
