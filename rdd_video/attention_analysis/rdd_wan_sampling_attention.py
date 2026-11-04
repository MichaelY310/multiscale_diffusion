"""Training-free attention experiments for the exploratory RDD-Wan sampler.

This module deliberately leaves checkpoint parameters untouched.  It can:

* capture a block route at an RDD stage endpoint;
* lift that route to the next spatiotemporal resolution;
* execute the selected fine blocks with FastVideo's Triton64 sparse kernel;
* optionally reuse a complete self-attention *output* for a few solver steps.

The route experiment is only a sampling sanity check.  A dense Wan checkpoint
has not learned either RDD's drifted path or sparse attention, so visual quality
from this module is not evidence for a trained RDD-Wan model.
"""

from __future__ import annotations

import importlib.util
import math
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch


@dataclass(frozen=True)
class PackedLayout:
    grid: tuple[int, int, int]
    block_grid: tuple[int, int, int]
    pack_index: torch.Tensor
    valid_positions: torch.Tensor
    actual_positions: torch.Tensor
    variable_sizes: torch.Tensor


@dataclass
class BlockRoute:
    block_grid: tuple[int, int, int]
    indices: torch.Tensor
    counts: torch.Tensor


def _load_fastvideo_forward(kernel_root: str | Path) -> Callable:
    root = Path(kernel_root).resolve()
    candidates = [
        root / "python/fastvideo_kernel/triton_kernels/block_sparse_attn_triton.py",
        root / "fastvideo-kernel/python/fastvideo_kernel/triton_kernels/block_sparse_attn_triton.py",
    ]
    source = next((item for item in candidates if item.exists()), None)
    if source is None:
        raise FileNotFoundError(
            "Cannot find FastVideo Triton sparse attention under "
            f"{root}; expected one of {candidates}"
        )
    spec = importlib.util.spec_from_file_location("rdd_wan_fastvideo_sparse", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.triton_block_sparse_attn_forward


def _flat(coord: tuple[int, int, int], grid: tuple[int, int, int]) -> int:
    return (coord[0] * grid[1] + coord[1]) * grid[2] + coord[2]


class RDDWanAttentionController:
    """Monkey-patch Wan self-attention without changing model parameters."""

    tile = (4, 4, 4)  # 64 tokens, matching FastVideo VSA.

    def __init__(
        self,
        model,
        safe_attention: Callable,
        rope_apply: Callable,
        mode: str = "route",
        route_density: float = 0.75,
        reuse_every: int = 1,
        fastvideo_kernel_root: str | Path | None = None,
    ) -> None:
        if mode not in {"dense", "reuse", "route", "route_reuse"}:
            raise ValueError(f"unknown attention mode: {mode}")
        if not 0.0 < route_density <= 1.0:
            raise ValueError("route_density must be in (0,1]")
        if reuse_every < 1:
            raise ValueError("reuse_every must be >=1")
        if mode in {"route", "route_reuse"} and fastvideo_kernel_root is None:
            raise ValueError("sparse modes require fastvideo_kernel_root")

        self.model = model
        self.safe_attention = safe_attention
        self.rope_apply = rope_apply
        self.mode = mode
        self.route_density = float(route_density)
        self.reuse_every = int(reuse_every)
        self.sparse_forward = (
            _load_fastvideo_forward(fastvideo_kernel_root)
            if mode in {"route", "route_reuse"}
            else None
        )
        self.layouts: dict[tuple[tuple[int, int, int], str], PackedLayout] = {}
        self.routes: dict[tuple[int, str, int], BlockRoute] = {}
        self.lifted_routes: dict[tuple[int, str, int, tuple[int, int, int]], BlockRoute] = {}
        self.output_cache: dict[tuple[int, str, int], torch.Tensor] = {}
        self.last_sigma: dict[tuple[int, str], float] = {}
        self.step_index: dict[tuple[int, str], int] = {}
        self.current_stage = 0
        self.current_branch = "cond"
        self.current_sigma = 1.0
        self.capture_route = False
        self.force_compute = False
        self.stats = {
            "dense_attention_calls": 0,
            "sparse_attention_calls": 0,
            "reused_attention_calls": 0,
            "captured_routes": 0,
            "sparse_fallbacks": 0,
            "effective_density_sum": 0.0,
            "effective_density_count": 0,
        }
        self._install()

    def begin_model_forward(
        self,
        stage: int,
        sigma_times_1000: torch.Tensor,
        branch: str,
    ) -> None:
        sigma = float(sigma_times_1000.flatten()[0].item()) / 1000.0
        key = (int(stage), branch)
        previous = self.last_sigma.get(key)
        if previous is None or abs(previous - sigma) > 1e-7:
            self.step_index[key] = self.step_index.get(key, -1) + 1
            self.last_sigma[key] = sigma
        self.current_stage = int(stage)
        self.current_branch = branch
        self.current_sigma = sigma

    def set_capture_route(self, enabled: bool) -> None:
        self.capture_route = bool(enabled)
        self.force_compute = bool(enabled)

    def summary(self) -> dict:
        result = dict(self.stats)
        count = int(result.pop("effective_density_count"))
        total = float(result.pop("effective_density_sum"))
        result["mean_effective_sparse_density"] = total / count if count else None
        result.update(
            {
                "mode": self.mode,
                "requested_route_density": self.route_density,
                "reuse_every": self.reuse_every,
                "route_records": len(self.routes),
            }
        )
        return result

    def _install(self) -> None:
        for layer_id, block in enumerate(self.model.blocks):
            module = block.self_attn

            def wrapped(this, x, seq_lens, grid_sizes, freqs, *, _layer=layer_id):
                return self._forward(_layer, this, x, seq_lens, grid_sizes, freqs)

            module.forward = types.MethodType(wrapped, module)

    def _layout(self, grid: tuple[int, int, int], device: torch.device) -> PackedLayout:
        key = (grid, str(device))
        cached = self.layouts.get(key)
        if cached is not None:
            return cached
        block_grid = tuple(math.ceil(g / tile) for g, tile in zip(grid, self.tile))
        packed: list[int] = []
        sizes: list[int] = []
        for bt in range(block_grid[0]):
            for bh in range(block_grid[1]):
                for bw in range(block_grid[2]):
                    valid: list[int] = []
                    for dt in range(self.tile[0]):
                        for dh in range(self.tile[1]):
                            for dw in range(self.tile[2]):
                                t = bt * self.tile[0] + dt
                                h = bh * self.tile[1] + dh
                                w = bw * self.tile[2] + dw
                                if t < grid[0] and h < grid[1] and w < grid[2]:
                                    valid.append(_flat((t, h, w), grid))
                    sizes.append(len(valid))
                    packed.extend(valid)
                    packed.extend([-1] * (64 - len(valid)))
        pack_index = torch.tensor(packed, dtype=torch.long, device=device)
        valid_positions = torch.nonzero(pack_index >= 0, as_tuple=False).flatten()
        actual_positions = pack_index.index_select(0, valid_positions)
        cached = PackedLayout(
            grid=grid,
            block_grid=block_grid,
            pack_index=pack_index,
            valid_positions=valid_positions,
            actual_positions=actual_positions,
            variable_sizes=torch.tensor(sizes, dtype=torch.int32, device=device),
        )
        self.layouts[key] = cached
        return cached

    @staticmethod
    def _pack(x: torch.Tensor, layout: PackedLayout) -> torch.Tensor:
        # Wan uses [B,S,H,D]; FastVideo's executor uses [B,H,S,D].
        source = x.transpose(1, 2)
        gathered = source.index_select(2, layout.pack_index.clamp_min(0))
        mask = (layout.pack_index >= 0).view(1, 1, -1, 1)
        return gathered.masked_fill(~mask, 0).contiguous()

    @staticmethod
    def _unpack(x: torch.Tensor, layout: PackedLayout, sequence: int) -> torch.Tensor:
        output = x.new_empty((x.shape[0], x.shape[1], sequence, x.shape[-1]))
        valid = x.index_select(2, layout.valid_positions)
        output.index_copy_(2, layout.actual_positions, valid)
        return output.transpose(1, 2).contiguous()

    def _capture(self, layer: int, q: torch.Tensor, k: torch.Tensor, layout: PackedLayout) -> None:
        q_packed = self._pack(q, layout)
        k_packed = self._pack(k, layout)
        batch, heads, _, dim = q_packed.shape
        blocks = math.prod(layout.block_grid)
        valid = layout.variable_sizes.to(q_packed.dtype).clamp_min(1).view(1, 1, blocks, 1)
        q_center = q_packed.view(batch, heads, blocks, 64, dim).sum(3) / valid
        k_center = k_packed.view(batch, heads, blocks, 64, dim).sum(3) / valid
        scores = torch.einsum("bhqd,bhkd->bhqk", q_center.float(), k_center.float())
        keep = max(1, min(blocks, round(self.route_density * blocks)))
        indices = scores.topk(keep, dim=-1, sorted=True).indices
        # A local/self block is a cheap guard against catastrophic route holes.
        self_idx = torch.arange(blocks, device=q.device).view(1, 1, blocks, 1)
        missing = ~(indices == self_idx).any(dim=-1, keepdim=True)
        indices[..., -1:] = torch.where(missing, self_idx, indices[..., -1:])
        indices = indices.to(torch.int32).contiguous()
        counts = torch.full(
            (batch, heads, blocks), keep, dtype=torch.int32, device=q.device
        )
        self.routes[(self.current_stage, self.current_branch, layer)] = BlockRoute(
            layout.block_grid, indices, counts
        )
        self.stats["captured_routes"] += 1

    @staticmethod
    def _parent_and_children(
        coarse: tuple[int, int, int], fine: tuple[int, int, int]
    ) -> tuple[list[int], list[list[int]]]:
        coarse_count = math.prod(coarse)
        children: list[list[int]] = [[] for _ in range(coarse_count)]
        parent: list[int] = []
        for t in range(fine[0]):
            for h in range(fine[1]):
                for w in range(fine[2]):
                    coord = (
                        min(coarse[0] - 1, t * coarse[0] // fine[0]),
                        min(coarse[1] - 1, h * coarse[1] // fine[1]),
                        min(coarse[2] - 1, w * coarse[2] // fine[2]),
                    )
                    p = _flat(coord, coarse)
                    idx = _flat((t, h, w), fine)
                    parent.append(p)
                    children[p].append(idx)
        return parent, children

    def _lift(self, source: BlockRoute, target_grid: tuple[int, int, int]) -> BlockRoute:
        parent, children = self._parent_and_children(source.block_grid, target_grid)
        source_cpu = source.indices.detach().cpu()
        batch, heads, _, _ = source_cpu.shape
        rows: list[list[list[list[int]]]] = []
        max_keep = 1
        for b in range(batch):
            batch_rows = []
            for head in range(heads):
                head_rows = []
                for query, parent_query in enumerate(parent):
                    values: list[int] = []
                    for parent_key in source_cpu[b, head, parent_query].tolist():
                        values.extend(children[int(parent_key)])
                    if query not in values:
                        values.append(query)
                    # Parent child sets are disjoint, but keep this defensive.
                    values = list(dict.fromkeys(values))
                    max_keep = max(max_keep, len(values))
                    head_rows.append(values)
                batch_rows.append(head_rows)
            rows.append(batch_rows)
        indices = torch.full(
            (batch, heads, len(parent), max_keep),
            -1,
            dtype=torch.int32,
            device=source.indices.device,
        )
        counts = torch.empty(
            (batch, heads, len(parent)), dtype=torch.int32, device=source.indices.device
        )
        for b in range(batch):
            for head in range(heads):
                for query, values in enumerate(rows[b][head]):
                    counts[b, head, query] = len(values)
                    indices[b, head, query, : len(values)] = torch.tensor(
                        values, dtype=torch.int32, device=source.indices.device
                    )
        return BlockRoute(target_grid, indices.contiguous(), counts.contiguous())

    def _route_for(self, layer: int, target: PackedLayout) -> BlockRoute | None:
        if self.current_stage <= 0:
            return None
        source_key = (self.current_stage - 1, self.current_branch, layer)
        source = self.routes.get(source_key)
        if source is None:
            return None
        cache_key = (self.current_stage, self.current_branch, layer, target.block_grid)
        lifted = self.lifted_routes.get(cache_key)
        if lifted is None:
            lifted = self._lift(source, target.block_grid)
            self.lifted_routes[cache_key] = lifted
        return lifted

    def _should_reuse(self, layer: int) -> bool:
        if self.mode not in {"reuse", "route_reuse"} or self.reuse_every <= 1 or self.force_compute:
            return False
        key = (self.current_stage, self.current_branch, layer)
        step = self.step_index.get((self.current_stage, self.current_branch), 0)
        return step % self.reuse_every != 0 and key in self.output_cache

    def _forward(self, layer, module, x, seq_lens, grid_sizes, freqs):
        cache_key = (self.current_stage, self.current_branch, layer)
        if self._should_reuse(layer):
            self.stats["reused_attention_calls"] += 1
            return self.output_cache[cache_key]

        batch, sequence = x.shape[:2]
        heads, dim = module.num_heads, module.head_dim
        q = module.norm_q(module.q(x)).view(batch, sequence, heads, dim)
        k = module.norm_k(module.k(x)).view(batch, sequence, heads, dim)
        v = module.v(x).view(batch, sequence, heads, dim)
        q = self.rope_apply(q, grid_sizes, freqs)
        k = self.rope_apply(k, grid_sizes, freqs)
        grid = tuple(int(value) for value in grid_sizes[0].tolist())
        layout = self._layout(grid, x.device)

        route = self._route_for(layer, layout) if self.mode in {"route", "route_reuse"} else None
        if route is None:
            output = self.safe_attention(
                q, k, v, k_lens=seq_lens, window_size=module.window_size
            )
            self.stats["dense_attention_calls"] += 1
            if self.mode in {"route", "route_reuse"} and self.current_stage > 0:
                self.stats["sparse_fallbacks"] += 1
        else:
            q_packed = self._pack(q, layout)
            k_packed = self._pack(k, layout)
            v_packed = self._pack(v, layout)
            output_packed = self.sparse_forward(
                q_packed,
                k_packed,
                v_packed,
                route.indices,
                route.counts,
                layout.variable_sizes,
            )[0]
            output = self._unpack(output_packed, layout, sequence)
            self.stats["sparse_attention_calls"] += 1
            density = float(route.counts.float().mean().item()) / math.prod(layout.block_grid)
            self.stats["effective_density_sum"] += density
            self.stats["effective_density_count"] += 1

        if self.capture_route and self.mode in {"route", "route_reuse"}:
            self._capture(layer, q, k, layout)
        output = module.o(output.flatten(2))
        if self.mode in {"reuse", "route_reuse"}:
            self.output_cache[cache_key] = output.detach()
        return output
