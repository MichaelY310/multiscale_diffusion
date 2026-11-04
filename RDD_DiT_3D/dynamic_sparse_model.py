"""Small dependency-free sparse voxel denoiser for the first RDD-3D gate.

This is intentionally not a replacement for a large TRELLIS backbone.  It is a
minimal model that can test whether dynamic empty-voxel pruning is learnable
before spending compute on a 1.3B parameter experiment.  Computation is only
performed for candidate coordinates.  Spatial communication uses a hashed
3x3x3 neighborhood, so its complexity is O(27N), not O(R^3) or O(N^2).
"""

from __future__ import annotations

from itertools import product
import math

import torch
from torch import Tensor, nn

from dynamic_sparse_representation import coordinate_keys


def sinusoidal_embedding(value: Tensor, dim: int, max_period: float = 10_000.0) -> Tensor:
    if dim % 2:
        raise ValueError("embedding dimension must be even")
    value = value.reshape(-1).float()
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(dim // 2, device=value.device, dtype=value.dtype)
        / (dim // 2)
    )
    angles = value[:, None] * frequencies[None]
    return torch.cat((torch.cos(angles), torch.sin(angles)), dim=1)


def build_neighbor_index(coords: Tensor, resolution: int, radius: int = 1) -> Tensor:
    """Return token indices for spatial neighbors, using -1 for missing voxels."""

    if radius < 0:
        raise ValueError("radius must be non-negative")
    coords = coords.long()
    if len(coords) == 0:
        return torch.empty((0, 0), dtype=torch.long, device=coords.device)
    offsets = torch.tensor(
        list(product(range(-radius, radius + 1), repeat=3)),
        dtype=torch.long,
        device=coords.device,
    )
    query = coords[:, None, :] + offsets[None, :, :]
    valid = torch.all((query >= 0) & (query < resolution), dim=2)
    safe_query = query.clamp(0, resolution - 1).reshape(-1, 3)
    order = torch.argsort(coordinate_keys(coords, resolution))
    keys = coordinate_keys(coords[order], resolution)
    query_keys = coordinate_keys(safe_query, resolution)
    positions = torch.searchsorted(keys, query_keys)
    in_range = positions < len(keys)
    safe_positions = positions.clamp(max=len(keys) - 1)
    matched = in_range & (keys[safe_positions] == query_keys)
    indices = torch.full_like(positions, -1)
    indices[matched] = order[safe_positions[matched]]
    indices = indices.reshape(len(coords), len(offsets))
    indices[~valid] = -1
    return indices


def neighbor_mean(values: Tensor, neighbor_index: Tensor) -> Tensor:
    if len(values) == 0:
        return values
    valid = neighbor_index >= 0
    safe = neighbor_index.clamp_min(0)
    gathered = values[safe] * valid[..., None].to(values.dtype)
    counts = valid.sum(dim=1, keepdim=True).clamp_min(1).to(values.dtype)
    return gathered.sum(dim=1) / counts


class SparseVoxelMixerBlock(nn.Module):
    def __init__(self, width: int, context_width: int, expansion: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.self_proj = nn.Linear(width, width)
        self.neighbor_proj = nn.Linear(width, width)
        self.global_proj = nn.Linear(width, width)
        self.context_proj = nn.Linear(context_width, width * 2)
        self.out_proj = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * expansion),
            nn.SiLU(),
            nn.Linear(width * expansion, width),
        )

    def forward(self, x: Tensor, neighbors: Tensor, context: Tensor) -> Tensor:
        h = self.norm1(x)
        local = neighbor_mean(h, neighbors)
        global_feature = h.mean(dim=0, keepdim=True)
        scale, shift = self.context_proj(context).chunk(2, dim=-1)
        mixed = (
            self.self_proj(h)
            + self.neighbor_proj(local)
            + self.global_proj(global_feature)
        )
        mixed = mixed * (1 + scale) + shift
        x = x + self.out_proj(torch.nn.functional.silu(mixed))
        return x + self.mlp(self.norm2(x))


class DynamicSparseRDDModel(nn.Module):
    """Shared 8/16/32(/64) denoiser with an auxiliary non-empty head."""

    def __init__(
        self,
        *,
        state_channels: int = 33,
        condition_channels: int = 384,
        width: int = 192,
        depth: int = 6,
        position_frequencies: int = 6,
    ):
        super().__init__()
        self.state_channels = int(state_channels)
        self.condition_channels = int(condition_channels)
        self.width = int(width)
        self.position_frequencies = int(position_frequencies)
        position_dim = 3 * 2 * self.position_frequencies
        self.state_proj = nn.Linear(self.state_channels, self.width)
        self.position_proj = nn.Sequential(
            nn.Linear(position_dim, self.width), nn.SiLU(), nn.Linear(self.width, self.width)
        )
        self.time_proj = nn.Sequential(
            nn.Linear(self.width, self.width), nn.SiLU(), nn.Linear(self.width, self.width)
        )
        self.condition_proj = nn.Sequential(
            nn.Linear(max(self.condition_channels, 1), self.width),
            nn.SiLU(),
            nn.Linear(self.width, self.width),
        )
        self.blocks = nn.ModuleList(
            [SparseVoxelMixerBlock(self.width, self.width) for _ in range(depth)]
        )
        self.final_norm = nn.LayerNorm(self.width)
        self.velocity_head = nn.Linear(self.width, self.state_channels)
        self.nonempty_head = nn.Linear(self.width, 1)

    def _position_embedding(self, coords: Tensor, resolution: int) -> Tensor:
        denominator = max(resolution - 1, 1)
        xyz = coords.float() / denominator
        frequencies = (2.0 ** torch.arange(
            self.position_frequencies, device=coords.device, dtype=xyz.dtype
        )) * math.pi
        angles = xyz[:, :, None] * frequencies[None, None, :]
        return torch.cat((torch.sin(angles), torch.cos(angles)), dim=2).flatten(1)

    def forward(
        self,
        state: Tensor,
        coords: Tensor,
        *,
        resolution: int,
        global_time: Tensor | float,
        condition: Tensor | None = None,
        neighbor_index: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if state.ndim != 2 or state.shape[1] != self.state_channels:
            raise ValueError(f"state must have shape [N, {self.state_channels}]")
        if len(state) != len(coords):
            raise ValueError("state and coords must contain the same number of tokens")
        if neighbor_index is None:
            neighbor_index = build_neighbor_index(coords, resolution)
        time = torch.as_tensor(global_time, device=state.device, dtype=state.dtype).reshape(1)
        time_context = self.time_proj(sinusoidal_embedding(time * 1000.0, self.width))
        if condition is None:
            condition = torch.zeros(
                (1, max(self.condition_channels, 1)), device=state.device, dtype=state.dtype
            )
        condition = condition.reshape(1, -1).to(state.dtype)
        if condition.shape[1] != max(self.condition_channels, 1):
            raise ValueError(
                f"condition must have {max(self.condition_channels, 1)} channels"
            )
        context = time_context + self.condition_proj(condition)
        x = self.state_proj(state) + self.position_proj(
            self._position_embedding(coords, resolution).to(state.dtype)
        )
        for block in self.blocks:
            x = block(x, neighbor_index, context)
        x = self.final_norm(x)
        return self.velocity_head(x), self.nonempty_head(x).squeeze(-1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
