#!/usr/bin/env python3
"""RDD Unified XL sampler using real Triton sparsity and attention-score caches.

Settings whose route density is below one never form 16/32-stage dense QK
logits.  The strict cascade uses 8-stage dense attention to select 16-stage
blocks, then uses final sparse 16-stage probabilities to select 32-stage
blocks.  Score-cache settings may instead use dense refresh calls, but compute
only V on cache hits.  The metrics time denoiser loops, not VAE decoding.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
import types
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKL
from PIL import Image

from rdd_block_sparse_triton import (
    BLOCK,
    block_sparse_attention,
    block_sparse_attention_store,
    cached_probability_av,
    micro_block_sparse_attention,
    micro_block_sparse_attention_store,
    micro_cached_probability_av,
    packed_block_sparse_attention,
    packed_block_sparse_attention_store,
    packed_cached_probability_av,
)
from sample_rdd_cross_stage_sparse import (
    DEFAULT_CLASSES,
    STAGE_LENGTHS,
    _qkv,
    cfg_model_call,
    create_sampling_diffusions,
    image_metrics,
    load_unified,
    save_comparison_grid,
    tensor_to_images,
    upsample_with_noise,
)
from sample_rdd_hierarchical_attention import (
    HierarchicalSetting,
    aggregate_attention,
    expand_cross_resolution_attention,
)


KERNEL_SETTINGS = (
    HierarchicalSetting(
        "triton_cascade_fixed_r75_50",
        density16=0.75,
        density32=0.50,
        temporal_route_update=False,
    ),
    HierarchicalSetting(
        "triton_cascade_fixed_score5_early",
        density16=0.75,
        density32=0.50,
        score_interval16=5,
        score_interval32=5,
        score_layers16=7,
        score_layers32=7,
        temporal_route_update=False,
    ),
    HierarchicalSetting(
        "triton_cascade_crossA5_score5",
        density16=0.75,
        density32=0.60,
        score_interval32=5,
        score_layers32=7,
        cross_score_calls32=5,
        cross_score_layers32=4,
        temporal_route_update=False,
    ),
    HierarchicalSetting(
        "triton_dense_score5_early",
        density16=1.0,
        density32=1.0,
        score_interval16=5,
        score_interval32=5,
        score_layers16=7,
        score_layers32=7,
        temporal_route_update=False,
    ),
    HierarchicalSetting(
        "triton_dense_score3_all",
        density16=1.0,
        density32=1.0,
        score_interval16=3,
        score_interval32=3,
        score_layers16=28,
        score_layers32=28,
        temporal_route_update=False,
    ),
    HierarchicalSetting(
        "triton_dense_score_adaptive_5_3_3",
        density16=1.0,
        density32=1.0,
        score_interval16=5,
        score_interval32=5,
        middle_score_interval16=3,
        middle_score_interval32=3,
        late_score_interval16=3,
        late_score_interval32=3,
        score_layers16=28,
        score_layers32=28,
        temporal_route_update=False,
    ),
    HierarchicalSetting(
        "triton_dense_score5_all",
        density16=1.0,
        density32=1.0,
        score_interval16=5,
        score_interval32=5,
        score_layers16=28,
        score_layers32=28,
        temporal_route_update=False,
    ),
    HierarchicalSetting(
        "triton_strict_micro_layeraware_score3_all",
        density16=0.875,
        density32=0.75,
        middle_density16=1.0,
        middle_density32=0.875,
        block_side16=2,
        score_interval16=3,
        score_interval32=3,
        score_layers16=28,
        score_layers32=28,
        temporal_route_update=False,
    ),
    HierarchicalSetting(
        "triton_no_dense_balanced_score3_all",
        density16=0.9375,
        density32=0.875,
        block_side16=2,
        score_interval16=3,
        score_interval32=3,
        score_layers16=28,
        score_layers32=28,
        temporal_route_update=False,
    ),
    HierarchicalSetting(
        "triton_no_dense_aggressive_score3_all",
        density16=0.875,
        density32=0.75,
        middle_density32=0.875,
        block_side16=2,
        score_interval16=3,
        score_interval32=3,
        score_layers16=28,
        score_layers32=28,
        temporal_route_update=False,
    ),
    HierarchicalSetting(
        "triton_no_dense_cross5_score3_all",
        density16=0.9375,
        density32=0.875,
        block_side16=2,
        score_interval16=3,
        score_interval32=3,
        score_layers16=28,
        score_layers32=28,
        cross_score_calls16=5,
        cross_score_calls32=5,
        cross_score_layers16=4,
        cross_score_layers32=4,
        temporal_route_update=False,
    ),
    HierarchicalSetting(
        "triton_vsa125_cascade_refresh1",
        density16=0.125,
        density32=0.125,
        block_side16=2,
        score_interval16=1,
        score_interval32=1,
        score_layers16=0,
        score_layers32=0,
        add_self_block=False,
        temporal_route_update=False,
    ),
    HierarchicalSetting(
        "triton_vsa125_cascade_score3",
        density16=0.125,
        density32=0.125,
        block_side16=2,
        score_interval16=3,
        score_interval32=3,
        score_layers16=28,
        score_layers32=28,
        add_self_block=False,
        temporal_route_update=False,
    ),
    HierarchicalSetting(
        "triton_stage32_layeraware_score3_all",
        density16=1.0,
        density32=0.75,
        middle_density32=0.875,
        score_interval32=3,
        score_layers32=28,
        temporal_route_update=False,
    ),
    HierarchicalSetting(
        "triton_stage32_layeraware_adaptive_5_3_3",
        density16=1.0,
        density32=0.75,
        middle_density32=0.875,
        score_interval32=5,
        middle_score_interval32=3,
        late_score_interval32=3,
        score_layers32=28,
        cross_score_calls32=5,
        cross_score_layers32=4,
        temporal_route_update=False,
    ),
    # Focused 8->16->32 cascade sweep.  All variants use exactly the previous
    # stage boundary attention as their route; only density, temporal A reuse,
    # and the packed execution backend vary.
    HierarchicalSetting(
        "packed_cascade_safe_nocache",
        density16=0.9375,
        density32=0.875,
        block_side16=2,
        temporal_route_update=False,
        kernel_backend="packed64",
    ),
    HierarchicalSetting(
        "packed_cascade_safe_k2",
        density16=0.9375,
        density32=0.875,
        block_side16=2,
        score_interval16=2,
        score_interval32=2,
        score_layers16=28,
        score_layers32=28,
        temporal_route_update=False,
        kernel_backend="packed64",
    ),
    HierarchicalSetting(
        "packed_cascade_equal875_k2",
        density16=0.875,
        density32=0.875,
        block_side16=2,
        score_interval16=2,
        score_interval32=2,
        score_layers16=28,
        score_layers32=28,
        temporal_route_update=False,
        kernel_backend="packed64",
    ),
    HierarchicalSetting(
        "packed_cascade_balanced_nocache",
        density16=0.875,
        density32=0.75,
        block_side16=2,
        temporal_route_update=False,
        kernel_backend="packed64",
    ),
    HierarchicalSetting(
        "packed_cascade_balanced_k2",
        density16=0.875,
        density32=0.75,
        block_side16=2,
        score_interval16=2,
        score_interval32=2,
        score_layers16=28,
        score_layers32=28,
        temporal_route_update=False,
        kernel_backend="packed64",
    ),
    HierarchicalSetting(
        "packed_cascade_moderate_k2",
        density16=0.75,
        density32=0.625,
        block_side16=2,
        score_interval16=2,
        score_interval32=2,
        score_layers16=28,
        score_layers32=28,
        temporal_route_update=False,
        kernel_backend="packed64",
    ),
    HierarchicalSetting(
        "packed_cascade_middle_sparse_k2",
        density16=0.9375,
        density32=0.875,
        middle_density16=0.75,
        middle_density32=0.625,
        block_side16=2,
        score_interval16=2,
        score_interval32=2,
        score_layers16=28,
        score_layers32=28,
        temporal_route_update=False,
        kernel_backend="packed64",
    ),
    HierarchicalSetting(
        "packed_stage32_75_k2_control",
        density16=1.0,
        density32=0.75,
        score_interval32=2,
        score_layers32=28,
        temporal_route_update=False,
        kernel_backend="packed64",
    ),
    HierarchicalSetting(
        "packed_strided_cascade_safe_k2",
        density16=0.9375,
        density32=0.875,
        block_side16=2,
        score_interval16=2,
        score_interval32=2,
        score_layers16=28,
        score_layers32=28,
        temporal_route_update=False,
        kernel_backend="packed64_strided",
    ),
    HierarchicalSetting(
        "packed_strided_cascade_balanced_k2",
        density16=0.875,
        density32=0.75,
        block_side16=2,
        score_interval16=2,
        score_interval32=2,
        score_layers16=28,
        score_layers32=28,
        temporal_route_update=False,
        kernel_backend="packed64_strided",
    ),
    HierarchicalSetting(
        "triton_dense_score2_all",
        density16=1.0,
        density32=1.0,
        score_interval16=2,
        score_interval32=2,
        score_layers16=28,
        score_layers32=28,
        temporal_route_update=False,
    ),
    HierarchicalSetting(
        "legacy_cascade_safe_k2",
        density16=0.9375,
        density32=0.875,
        block_side16=2,
        score_interval16=2,
        score_interval32=2,
        score_layers16=28,
        score_layers32=28,
        temporal_route_update=False,
        kernel_backend="legacy",
    ),
    HierarchicalSetting(
        "legacy_cascade_safe_nocache",
        density16=0.9375,
        density32=0.875,
        block_side16=2,
        temporal_route_update=False,
        kernel_backend="legacy",
    ),
    # Exact low-density strict cascade controls.  The current spatial layout has
    # 16 route blocks at both fine stages, so 18.75% (3/16) is the nearest
    # representable density to the requested 20%.  Self blocks are not forced,
    # otherwise the measured density would be higher than the named target.
    HierarchicalSetting(
        "legacy_cascade_keep125_nocache",
        density16=0.125,
        density32=0.125,
        block_side16=2,
        add_self_block=False,
        temporal_route_update=False,
        kernel_backend="legacy",
    ),
    HierarchicalSetting(
        "packed_cascade_keep125_nocache",
        density16=0.125,
        density32=0.125,
        block_side16=2,
        add_self_block=False,
        temporal_route_update=False,
        kernel_backend="packed64",
    ),
    HierarchicalSetting(
        "legacy_cascade_keep1875_nocache",
        density16=0.1875,
        density32=0.1875,
        block_side16=2,
        add_self_block=False,
        temporal_route_update=False,
        kernel_backend="legacy",
    ),
    HierarchicalSetting(
        "packed_cascade_keep1875_nocache",
        density16=0.1875,
        density32=0.1875,
        block_side16=2,
        add_self_block=False,
        temporal_route_update=False,
        kernel_backend="packed64",
    ),
    HierarchicalSetting(
        "legacy_cascade_keep50_nocache",
        density16=0.5,
        density32=0.5,
        block_side16=2,
        add_self_block=False,
        temporal_route_update=False,
        kernel_backend="legacy",
    ),
    HierarchicalSetting(
        "packed_cascade_keep50_nocache",
        density16=0.5,
        density32=0.5,
        block_side16=2,
        add_self_block=False,
        temporal_route_update=False,
        kernel_backend="packed64",
    ),
    HierarchicalSetting(
        "legacy_cascade_balanced_k2",
        density16=0.875,
        density32=0.75,
        block_side16=2,
        score_interval16=2,
        score_interval32=2,
        score_layers16=28,
        score_layers32=28,
        temporal_route_update=False,
        kernel_backend="legacy",
    ),
    HierarchicalSetting(
        "hybrid32_cascade_safe_k2",
        density16=0.9375,
        density32=0.875,
        block_side16=2,
        score_interval16=2,
        score_interval32=2,
        score_layers16=28,
        score_layers32=28,
        temporal_route_update=False,
        kernel_backend="hybrid_packed32",
    ),
    HierarchicalSetting(
        "hybrid32_cascade_balanced_k2",
        density16=0.875,
        density32=0.75,
        block_side16=2,
        score_interval16=2,
        score_interval32=2,
        score_layers16=28,
        score_layers32=28,
        temporal_route_update=False,
        kernel_backend="hybrid_packed32",
    ),
)


def spatial_block_ids(
    tokens: int, device: torch.device, block_side: int = 4
) -> torch.Tensor:
    grid = int(math.isqrt(tokens))
    if grid * grid != tokens or grid % block_side:
        raise ValueError(
            f"expected square token grid divisible by {block_side}, got N={tokens}"
        )
    y, x = torch.meshgrid(
        torch.arange(grid, device=device),
        torch.arange(grid, device=device),
        indexing="ij",
    )
    axis = grid // block_side
    return ((y // block_side) * axis + (x // block_side)).flatten().long()


def spatial_block_token_indices(
    block_indices: torch.Tensor, tokens: int, block_side: int = 4
) -> torch.Tensor:
    """Expand spatial block ids to row-major token ids."""
    grid = int(math.isqrt(tokens))
    axis = grid // block_side
    block_size = block_side * block_side
    local = torch.arange(block_size, device=block_indices.device)
    block_y = block_indices.clamp_min(0).long() // axis
    block_x = block_indices.clamp_min(0).long() % axis
    return (
        (block_y[..., None] * block_side + local // block_side) * grid
        + block_x[..., None] * block_side
        + local % block_side
    )


def fast_block_indices_from_route(
    route: torch.Tensor,
    density: float,
    add_self: bool,
) -> torch.Tensor:
    """Vectorized Top-K route conversion with an optional non-duplicated self block."""
    blocks = route.shape[-1]
    keep = max(1, min(blocks, math.ceil(density * blocks)))
    top = route.topk(keep, dim=-1).indices
    if not add_self or keep == blocks:
        return top.to(torch.int32).contiguous()
    self_index = torch.arange(blocks, device=route.device)
    self_index = self_index.view(1, 1, blocks, 1).expand(*top.shape[:-1], 1)
    already_present = (top == self_index).any(-1, keepdim=True)
    extra = torch.where(already_present, torch.full_like(self_index, -1), self_index)
    return torch.cat((top, extra), dim=-1).to(torch.int32).contiguous()


def v_only(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Slice the fused QKV projection and compute only V on score-cache hits."""
    width = x.shape[-1]
    weight = module.qkv.weight[2 * width :]
    bias = None if module.qkv.bias is None else module.qkv.bias[2 * width :]
    value = F.linear(x, weight, bias)
    return value.reshape(x.shape[0], x.shape[1], module.num_heads, -1).permute(0, 2, 1, 3)


def sparse_probabilities_to_full_original(
    probabilities: torch.Tensor,
    indices: torch.Tensor,
    block_side: int,
) -> torch.Tensor:
    """Materialize sparse probabilities only for boundary route extraction.

    Omitted entries remain zero; no missing QK interaction is evaluated.
    """
    batch, heads, tokens, packed = probabilities.shape
    block_size = block_side * block_side
    slots = packed // block_size
    query_blocks = spatial_block_ids(tokens, indices.device, block_side)
    row_indices = indices[:, :, query_blocks, :]
    valid = row_indices >= 0
    key_tokens = spatial_block_token_indices(row_indices, tokens, block_side)
    p = probabilities.view(batch, heads, tokens, slots, block_size)
    p = p * valid[..., None]
    full = torch.zeros(
        batch, heads, tokens, tokens,
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    full.scatter_add_(-1, key_tokens.reshape(batch, heads, tokens, -1),
                      p.reshape(batch, heads, tokens, -1))
    return full


def full_original_to_sparse_probabilities(
    attention: torch.Tensor,
    indices: torch.Tensor,
    block_side: int,
) -> torch.Tensor:
    """Gather an existing coarse-derived A into the compact sparse cache."""
    batch, heads, tokens, _ = attention.shape
    slots = indices.shape[-1]
    block_size = block_side * block_side
    query_blocks = spatial_block_ids(tokens, indices.device, block_side)
    row_indices = indices[:, :, query_blocks, :]
    valid = row_indices >= 0
    key_tokens = spatial_block_token_indices(row_indices, tokens, block_side)
    selected = torch.gather(
        attention[:, :, :, None, :].expand(-1, -1, -1, slots, -1),
        -1,
        key_tokens,
    )
    selected = selected * valid[..., None]
    selected = selected.reshape(batch, heads, tokens, slots * block_size)
    return selected / selected.sum(-1, keepdim=True).clamp_min(1e-12)


class KernelController:
    def __init__(self, setting: HierarchicalSetting, total_calls: dict[int, int]):
        self.setting = setting
        self.total_calls = total_calls
        self.resolution: int | None = None
        self.call_index = {8: -1, 16: -1, 32: -1}
        self.boundary_attention: dict[int, dict[int, torch.Tensor]] = {}
        self.indices: dict[int, dict[int, torch.Tensor]] = {16: {}, 32: {}}
        self.probabilities: dict[int, dict[int, torch.Tensor]] = {16: {}, 32: {}}
        self.dense_probabilities: dict[int, dict[int, torch.Tensor]] = {16: {}, 32: {}}
        self.stats = {
            "dense_attention_calls": 0,
            "post8_dense_refresh_calls": 0,
            "sparse_attention_calls": 0,
            "sparse_refresh_store_calls": 0,
            "temporal_score_cache_hits": 0,
            "cross_resolution_score_cache_hits": 0,
            "v_only_projection_calls": 0,
            "selected_block_density_sum": {"16": 0.0, "32": 0.0},
            "selected_block_density_count": {"16": 0, "32": 0},
        }

    def begin_forward(self, x: torch.Tensor) -> None:
        self.resolution = int(x.shape[-1])
        self.call_index[self.resolution] += 1

    def end_forward(self) -> None:
        self.resolution = None

    def _density(self, resolution: int, layer: int) -> float:
        fraction = layer / 27.0
        in_middle = 0.15 <= fraction <= 0.85
        if resolution == 16:
            if in_middle and self.setting.middle_density16 is not None:
                return self.setting.middle_density16
            return self.setting.density16
        if in_middle and self.setting.middle_density32 is not None:
            return self.setting.middle_density32
        return self.setting.density32

    def _interval(self, resolution: int, layer: int) -> int:
        fraction = layer / 27.0
        if resolution == 16:
            if fraction > 0.85 and self.setting.late_score_interval16 is not None:
                return self.setting.late_score_interval16
            if 0.15 <= fraction <= 0.85 and self.setting.middle_score_interval16 is not None:
                return self.setting.middle_score_interval16
            return self.setting.score_interval16
        if fraction > 0.85 and self.setting.late_score_interval32 is not None:
            return self.setting.late_score_interval32
        if 0.15 <= fraction <= 0.85 and self.setting.middle_score_interval32 is not None:
            return self.setting.middle_score_interval32
        return self.setting.score_interval32

    def _score_layers(self, resolution: int) -> int:
        return self.setting.score_layers16 if resolution == 16 else self.setting.score_layers32

    def _block_side(self, resolution: int) -> int:
        return self.setting.block_side16 if resolution == 16 else self.setting.block_side32

    def _cross_calls(self, resolution: int) -> int:
        return self.setting.cross_score_calls16 if resolution == 16 else self.setting.cross_score_calls32

    def _cross_layers(self, resolution: int) -> int:
        return self.setting.cross_score_layers16 if resolution == 16 else self.setting.cross_score_layers32

    def _is_last_call(self, resolution: int) -> bool:
        return self.call_index[resolution] == self.total_calls[resolution] - 1

    def _use_packed_kernel(self, resolution: int) -> bool:
        backend = self.setting.kernel_backend
        return backend.startswith("packed64") or (
            backend == "hybrid_packed32" and resolution == 32
        )

    def _packed_contiguous_inputs(self) -> bool:
        return self.setting.kernel_backend != "packed64_strided"

    def _packed_kv_tile(self) -> int:
        return 32 if self.setting.kernel_backend == "hybrid_packed32" else 64

    def _ensure_indices(self, resolution: int, layer: int, tokens: int) -> torch.Tensor:
        cached = self.indices[resolution].get(layer)
        if cached is not None:
            return cached
        source = self.boundary_attention[resolution // 2][layer]
        # A target block covers half as many tokens along each axis in the
        # immediately preceding resolution.  For example, a 2x2 token block
        # at 16 corresponds to a 1x1 token block at 8.
        source_side = self._block_side(resolution) // 2
        route = aggregate_attention(source, source_side, source_side)
        if self.setting.add_neighbor_blocks:
            raise NotImplementedError("the optimized kernel path currently supports self but not neighbor dilation")
        indices = fast_block_indices_from_route(
            route, self._density(resolution, layer), self.setting.add_self_block
        )
        self.indices[resolution][layer] = indices
        key = str(resolution)
        self.stats["selected_block_density_sum"][key] += float(
            (indices >= 0).sum(-1).float().mean().item() / route.shape[-1]
        )
        self.stats["selected_block_density_count"][key] += 1
        return indices

    def _ensure_cross_probabilities(
        self, resolution: int, layer: int, tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices = self._ensure_indices(resolution, layer, tokens)
        if layer not in self.probabilities[resolution]:
            source = self.boundary_attention[resolution // 2][layer]
            lifted = expand_cross_resolution_attention(source, tokens)
            self.probabilities[resolution][layer] = full_original_to_sparse_probabilities(
                lifted, indices, self._block_side(resolution)
            )
        return indices, self.probabilities[resolution][layer]

    def use_v_only(self, layer: int) -> str | None:
        assert self.resolution is not None
        resolution = self.resolution
        if resolution == 8:
            return None
        call = self.call_index[resolution]
        if call < self._cross_calls(resolution) and layer < self._cross_layers(resolution):
            return "cross"
        interval = self._interval(resolution, layer)
        if (
            interval > 1
            and layer < self._score_layers(resolution)
            and (
                layer in self.probabilities[resolution]
                or layer in self.dense_probabilities[resolution]
            )
            and call % interval != 0
            and not self._is_last_call(resolution)
        ):
            return "temporal"
        return None

    def attention_v_only(self, layer: int, v: torch.Tensor, reason: str) -> torch.Tensor:
        assert self.resolution is not None
        resolution = self.resolution
        tokens = v.shape[-2]
        if reason == "cross":
            indices, probabilities = self._ensure_cross_probabilities(resolution, layer, tokens)
            self.stats["cross_resolution_score_cache_hits"] += 1
        elif layer in self.dense_probabilities[resolution]:
            self.stats["temporal_score_cache_hits"] += 1
            self.stats["v_only_projection_calls"] += 1
            return self.dense_probabilities[resolution][layer].to(v.dtype) @ v
        else:
            indices = self.indices[resolution][layer]
            probabilities = self.probabilities[resolution][layer]
            self.stats["temporal_score_cache_hits"] += 1
        self.stats["v_only_projection_calls"] += 1
        if self._use_packed_kernel(resolution):
            return packed_cached_probability_av(
                v,
                indices,
                probabilities,
                block_side=self._block_side(resolution),
                contiguous_inputs=self._packed_contiguous_inputs(),
                kv_tile=self._packed_kv_tile(),
            )
        if self._block_side(resolution) == 2:
            return micro_cached_probability_av(v, indices, probabilities)
        return cached_probability_av(v, indices, probabilities)

    def attention_qkv(
        self,
        layer: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        assert self.resolution is not None
        resolution = self.resolution
        if resolution == 8:
            attention = (q @ k.transpose(-2, -1) * scale).softmax(-1)
            if self._is_last_call(8):
                self.boundary_attention.setdefault(8, {})[layer] = attention.detach()
            self.stats["dense_attention_calls"] += 1
            return attention @ v

        density = self._density(resolution, layer)
        interval = self._interval(resolution, layer)
        if density >= 1.0:
            attention = (q @ k.transpose(-2, -1) * scale).softmax(-1)
            if interval > 1 and layer < self._score_layers(resolution):
                self.dense_probabilities[resolution][layer] = attention.detach()
            if resolution == 16 and self._is_last_call(16):
                self.boundary_attention.setdefault(16, {})[layer] = attention.detach()
            self.stats["post8_dense_refresh_calls"] += 1
            return attention @ v

        tokens = q.shape[-2]
        indices = self._ensure_indices(resolution, layer, tokens)

        needs_probability_cache = (
            (interval > 1 and layer < self._score_layers(resolution))
            or (resolution == 16 and self._is_last_call(resolution))
        )
        if needs_probability_cache:
            if self._use_packed_kernel(resolution):
                output, probabilities = packed_block_sparse_attention_store(
                    q,
                    k,
                    v,
                    indices,
                    block_side=self._block_side(resolution),
                    contiguous_inputs=self._packed_contiguous_inputs(),
                    kv_tile=self._packed_kv_tile(),
                )
            elif self._block_side(resolution) == 2:
                output, probabilities = micro_block_sparse_attention_store(
                    q, k, v, indices
                )
            else:
                output, probabilities = block_sparse_attention_store(
                    q, k, v, indices
                )
            self.probabilities[resolution][layer] = probabilities.detach()
            self.stats["sparse_refresh_store_calls"] += 1
            if resolution == 16 and self._is_last_call(resolution):
                full = sparse_probabilities_to_full_original(
                    probabilities, indices, self._block_side(resolution)
                )
                self.boundary_attention.setdefault(resolution, {})[layer] = full.detach()
        else:
            if self._use_packed_kernel(resolution):
                output = packed_block_sparse_attention(
                    q,
                    k,
                    v,
                    indices,
                    block_side=self._block_side(resolution),
                    contiguous_inputs=self._packed_contiguous_inputs(),
                    kv_tile=self._packed_kv_tile(),
                )
            elif self._block_side(resolution) == 2:
                output = micro_block_sparse_attention(q, k, v, indices)
            else:
                output = block_sparse_attention(q, k, v, indices)
            self.stats["sparse_attention_calls"] += 1
        return output

    def report(self) -> dict[str, Any]:
        result = {"setting": asdict(self.setting), **self.stats}
        result["mean_selected_block_density"] = {}
        for key in ("16", "32"):
            count = max(1, self.stats["selected_block_density_count"][key])
            result["mean_selected_block_density"][key] = (
                self.stats["selected_block_density_sum"][key] / count
            )
        return result


@contextmanager
def patched_kernel_attention(model: torch.nn.Module, controller: KernelController):
    records = []
    for layer, block in enumerate(model.blocks):
        module = block.attn
        original = module.forward

        def replacement(self, *args, _layer=layer):
            reason = controller.use_v_only(_layer)
            if reason is not None:
                value = v_only(self, args[0])
                output = controller.attention_v_only(_layer, value, reason)
            else:
                q, k, v, scale = _qkv(self, args, unified=True)
                output = controller.attention_qkv(_layer, q, k, v, scale)
            x = output.transpose(1, 2).reshape(args[0].shape)
            return self.proj(x)

        records.append((module, original))
        module.forward = types.MethodType(replacement, module)
    try:
        yield
    finally:
        for module, original in records:
            module.forward = original


@torch.inference_mode()
def sample(
    model,
    diffusions,
    vae,
    labels,
    num_classes,
    cfg_scale,
    seed,
    controller: KernelController | None,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    use_cfg = cfg_scale > 1.0
    if use_cfg:
        y = torch.cat((labels, torch.full_like(labels, num_classes)))
        multiplier = 2
    else:
        y, multiplier = labels, 1
    pred_xstart = None
    final_sample = None
    elapsed = 0.0
    stage_seconds: dict[str, float] = {}
    context = patched_kernel_attention(model, controller) if controller else nullcontext()
    with context:
        for stage, resolution in enumerate((8, 16, 32)):
            diffusion = diffusions[resolution]
            if stage == 0:
                state = torch.randn(
                    multiplier * len(labels), 4, resolution, resolution, device=labels.device
                )
            else:
                state = upsample_with_noise(pred_xstart, diffusion, resolution)
                if use_cfg:
                    state = torch.cat((state, state))

            def model_forward(x, t, **kwargs):
                if controller:
                    controller.begin_forward(x)
                try:
                    if use_cfg:
                        return cfg_model_call(model, x, t, kwargs["y"], cfg_scale)
                    return model(x, t, kwargs["y"])
                finally:
                    if controller:
                        controller.end_forward()

            torch.cuda.synchronize()
            start = time.perf_counter()
            output = diffusion.p_sample_loop(
                model_forward,
                state.shape,
                noise=state,
                clip_denoised=False,
                model_kwargs={"y": y},
                progress=False,
                device=labels.device,
            )
            torch.cuda.synchronize()
            stage_elapsed = time.perf_counter() - start
            elapsed += stage_elapsed
            stage_seconds[str(resolution)] = stage_elapsed
            final_sample = output["sample"].float()
            pred_xstart = output["pred_xstart"].float()
            if use_cfg:
                final_sample = final_sample[: len(labels)]
                pred_xstart = pred_xstart[: len(labels)]
    decoded = vae.decode(final_sample / 0.18215).sample.float()
    stats = {"denoiser_seconds": elapsed, "stage_seconds": stage_seconds}
    if controller:
        stats.update(controller.report())
    return decoded, final_sample, stats


def warmup_kernels(device: torch.device, dtype: torch.dtype = torch.float32) -> None:
    # +1 covers the forced self block when it is absent from route Top-K.
    for tokens, keep in ((64, 4), (256, 2), (256, 9), (256, 11), (256, 13), (256, 15)):
        q = torch.randn(1, 16, tokens, 72, device=device, dtype=dtype)
        indices = torch.arange(keep, device=device, dtype=torch.int32).view(1, 1, 1, keep)
        indices = indices.expand(1, 16, tokens // BLOCK, keep).contiguous()
        block_sparse_attention(q, q, q, indices)
        _, p = block_sparse_attention_store(q, q, q, indices)
        cached_probability_av(q, indices, p)
        packed_block_sparse_attention(q, q, q, indices, block_side=4)
        _, packed_p = packed_block_sparse_attention_store(
            q, q, q, indices, block_side=4
        )
        packed_cached_probability_av(q, indices, packed_p, block_side=4)
        if tokens == 256:
            packed_block_sparse_attention(
                q, q, q, indices, block_side=4, kv_tile=32
            )
            _, packed32_p = packed_block_sparse_attention_store(
                q, q, q, indices, block_side=4, kv_tile=32
            )
            packed_cached_probability_av(
                q, indices, packed32_p, block_side=4, kv_tile=32
            )
    # Strict 8 -> 16 routing uses finer 2x2 spatial micro-blocks.  With 87.5%
    # route density plus a missing self block, the packed width is at most 15.
    q = torch.randn(1, 16, 64, 72, device=device, dtype=dtype)
    for keep in (2, 15, 16):
        indices = torch.arange(keep, device=device, dtype=torch.int32).view(1, 1, 1, keep)
        indices = indices.expand(1, 16, 16, keep).contiguous()
        micro_block_sparse_attention(q, q, q, indices)
        _, p = micro_block_sparse_attention_store(q, q, q, indices)
        micro_cached_probability_av(q, indices, p)
        packed_block_sparse_attention(q, q, q, indices, block_side=2)
        _, packed_p = packed_block_sparse_attention_store(
            q, q, q, indices, block_side=2
        )
        packed_cached_probability_av(q, indices, packed_p, block_side=2)
    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unified-ckpt", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--classes", default=",".join(map(str, DEFAULT_CLASSES)))
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--sample-steps", default="80,80,90")
    parser.add_argument("--settings", default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-fraction", type=float, default=0.45)
    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=1,
        help="repeat each measured sampler and report the median denoiser time",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    labels = torch.tensor([int(x) for x in args.classes.split(",")], device=device)
    steps = tuple(int(x) for x in args.sample_steps.split(","))
    total_calls = dict(zip((8, 16, 32), steps))
    requested = None if args.settings == "all" else set(args.settings.split(","))
    settings = tuple(x for x in KERNEL_SETTINGS if requested is None or x.name in requested)
    unknown = set() if requested is None else requested - {x.name for x in KERNEL_SETTINGS}
    if unknown:
        raise ValueError(f"unknown settings: {sorted(unknown)}")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    vae = AutoencoderKL.from_pretrained(args.vae_path, local_files_only=True).to(device)
    vae.eval().requires_grad_(False)
    model, num_classes = load_unified(Path(args.unified_ckpt), device)
    diffusions = create_sampling_diffusions(steps)
    warmup_kernels(device)

    all_images: dict[str, list[Image.Image]] = {}
    all_decoded: dict[str, torch.Tensor] = {}
    all_latents: dict[str, torch.Tensor] = {}
    report: dict[str, Any] = {
        "scope": (
            "true Triton sparse inference for density<1; temporal score-cache "
            "variants use dense QK only on refresh calls and V-only projection on hits"
        ),
        "classes": labels.cpu().tolist(),
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "stage_lengths_finest_to_coarsest": list(STAGE_LENGTHS),
        "sample_steps_coarse_to_fine": list(steps),
        "variants": {},
    }

    # One untimed dense warmup, then a measured dense baseline with identical noise.
    sample(model, diffusions, vae, labels[:1], num_classes, args.cfg_scale, args.seed, None)
    dense_times = []
    dense_stage_times: list[dict[str, float]] = []
    for _ in range(args.timing_repeats):
        decoded, latent, stats = sample(
            model, diffusions, vae, labels, num_classes, args.cfg_scale, args.seed, None
        )
        dense_times.append(stats["denoiser_seconds"])
        dense_stage_times.append(stats["stage_seconds"])
    stats["denoiser_seconds_runs"] = dense_times
    stats["denoiser_seconds_mean"] = statistics.fmean(dense_times)
    stats["denoiser_seconds"] = statistics.median(dense_times)
    stats["stage_seconds_runs"] = dense_stage_times
    stats["stage_seconds"] = {
        resolution: statistics.median(run[resolution] for run in dense_stage_times)
        for resolution in ("8", "16", "32")
    }
    all_decoded["dense"], all_latents["dense"] = decoded.cpu(), latent.cpu()
    all_images["dense"] = tensor_to_images(all_decoded["dense"])
    report["variants"]["dense"] = stats
    del decoded, latent
    gc.collect()
    torch.cuda.empty_cache()

    for setting in settings:
        print("sampling", setting.name, flush=True)
        variant_times = []
        variant_stage_times: list[dict[str, float]] = []
        for _ in range(args.timing_repeats):
            controller = KernelController(setting, total_calls)
            decoded, latent, stats = sample(
                model, diffusions, vae, labels, num_classes, args.cfg_scale, args.seed, controller
            )
            variant_times.append(stats["denoiser_seconds"])
            variant_stage_times.append(stats["stage_seconds"])
        stats["denoiser_seconds_runs"] = variant_times
        stats["denoiser_seconds_mean"] = statistics.fmean(variant_times)
        stats["denoiser_seconds"] = statistics.median(variant_times)
        stats["stage_seconds_runs"] = variant_stage_times
        stats["stage_seconds"] = {
            resolution: statistics.median(run[resolution] for run in variant_stage_times)
            for resolution in ("8", "16", "32")
        }
        all_decoded[setting.name], all_latents[setting.name] = decoded.cpu(), latent.cpu()
        all_images[setting.name] = tensor_to_images(all_decoded[setting.name])
        variant_dir = outdir / setting.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(all_images[setting.name]):
            image.save(variant_dir / f"class_{int(labels[index])}_seed_{args.seed}.png")
        stats["image_vs_dense"] = image_metrics(all_decoded["dense"], all_decoded[setting.name])
        stats["latent_vs_dense"] = image_metrics(all_latents["dense"], all_latents[setting.name])
        report["variants"][setting.name] = stats
        del decoded, latent
        gc.collect()
        torch.cuda.empty_cache()

    dense_dir = outdir / "dense"
    dense_dir.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(all_images["dense"]):
        image.save(dense_dir / f"class_{int(labels[index])}_seed_{args.seed}.png")
    grid = outdir / "triton_hierarchical_comparison.png"
    save_comparison_grid(
        all_images,
        labels.cpu().tolist(),
        grid,
        "RDD Unified XL: dense vs true Triton hierarchical sparse attention",
    )
    report["comparison_grid"] = str(grid)
    (outdir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("saved", grid, flush=True)


if __name__ == "__main__":
    main()
