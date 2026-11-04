"""Cross-resolution block routing for Diffusers' WanTransformer3DModel.

This is the FastWan/Diffusers counterpart of ``rdd_wan_sampling_attention``.
It replaces only self-attention processors, leaves every checkpoint parameter
unchanged, captures a coarse 4x4x4-block route, and lifts that route to the
next RDD resolution before executing FastVideo's Triton64 sparse kernel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from rdd_wan_sampling_attention import (
    BlockRoute,
    PackedLayout,
    _flat,
    _load_fastvideo_forward,
)


def _apply_rotary(
    value: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    first, second = value.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    output = torch.empty_like(value)
    output[..., 0::2] = first * cos - second * sin
    output[..., 1::2] = first * sin + second * cos
    return output.type_as(value)


@dataclass
class _Processor:
    controller: "FastWanRDDRouteController"
    layer: int

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states=None,
        attention_mask=None,
        rotary_emb=None,
        **kwargs,
    ) -> torch.Tensor:
        if encoder_hidden_states is not None:
            raise RuntimeError("RDD route processor must only replace Wan self-attention")

        query = attn.norm_q(attn.to_q(hidden_states))
        key = attn.norm_k(attn.to_k(hidden_states))
        value = attn.to_v(hidden_states)
        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))
        if rotary_emb is not None:
            query = _apply_rotary(query, *rotary_emb)
            key = _apply_rotary(key, *rotary_emb)

        output = self.controller.attend(self.layer, query, key, value)
        output = output.flatten(2, 3).type_as(query)
        output = attn.to_out[0](output)
        return attn.to_out[1](output)


class FastWanRDDRouteController:
    """Install previous-stage routing on a Diffusers Wan transformer."""

    tile = (4, 4, 4)

    def __init__(
        self,
        model,
        route_density: float,
        fastvideo_kernel_root: str | Path,
    ) -> None:
        if not 0 < route_density <= 1:
            raise ValueError("route_density must be in (0, 1]")
        self.model = model
        self.route_density = float(route_density)
        self.sparse_forward = _load_fastvideo_forward(fastvideo_kernel_root)
        self.current_stage = 0
        self.current_grid = (1, 1, 1)
        self.capture_route = False
        self.layouts: dict[tuple[tuple[int, int, int], str], PackedLayout] = {}
        self.routes: dict[tuple[int, int], BlockRoute] = {}
        self.lifted_routes: dict[tuple[int, int, tuple[int, int, int]], BlockRoute] = {}
        self.original_processors = []
        self.stats = {
            "dense_attention_calls": 0,
            "sparse_attention_calls": 0,
            "captured_routes": 0,
            "sparse_fallbacks": 0,
            "effective_density_sum": 0.0,
            "effective_density_count": 0,
        }
        for layer, block in enumerate(model.blocks):
            self.original_processors.append(block.attn1.processor)
            block.attn1.set_processor(_Processor(self, layer))

    def begin_stage(self, stage: int, latent_shape: tuple[int, int, int], capture: bool) -> None:
        patch = tuple(int(x) for x in self.model.config.patch_size)
        self.current_stage = int(stage)
        self.current_grid = tuple(int(x // p) for x, p in zip(latent_shape, patch))
        self.capture_route = bool(capture)

    def restore(self) -> None:
        for block, processor in zip(self.model.blocks, self.original_processors):
            block.attn1.set_processor(processor)

    def summary(self) -> dict:
        result = dict(self.stats)
        count = int(result.pop("effective_density_count"))
        total = float(result.pop("effective_density_sum"))
        result["mean_effective_sparse_density"] = total / count if count else None
        result["requested_route_density"] = self.route_density
        result["route_records"] = len(self.routes)
        return result

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
                                t, h, w = (
                                    bt * self.tile[0] + dt,
                                    bh * self.tile[1] + dh,
                                    bw * self.tile[2] + dw,
                                )
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
    def _pack(value: torch.Tensor, layout: PackedLayout) -> torch.Tensor:
        source = value.transpose(1, 2)
        gathered = source.index_select(2, layout.pack_index.clamp_min(0))
        mask = (layout.pack_index >= 0).view(1, 1, -1, 1)
        return gathered.masked_fill(~mask, 0).contiguous()

    @staticmethod
    def _unpack(value: torch.Tensor, layout: PackedLayout, sequence: int) -> torch.Tensor:
        output = value.new_empty((value.shape[0], value.shape[1], sequence, value.shape[-1]))
        valid = value.index_select(2, layout.valid_positions)
        output.index_copy_(2, layout.actual_positions, valid)
        return output.transpose(1, 2).contiguous()

    def _capture(self, layer: int, query: torch.Tensor, key: torch.Tensor, layout: PackedLayout) -> None:
        q = self._pack(query, layout)
        k = self._pack(key, layout)
        batch, heads, _, dim = q.shape
        blocks = math.prod(layout.block_grid)
        valid = layout.variable_sizes.to(q.dtype).clamp_min(1).view(1, 1, blocks, 1)
        q_center = q.view(batch, heads, blocks, 64, dim).sum(3) / valid
        k_center = k.view(batch, heads, blocks, 64, dim).sum(3) / valid
        scores = torch.einsum("bhqd,bhkd->bhqk", q_center.float(), k_center.float())
        keep = max(1, min(blocks, round(self.route_density * blocks)))
        indices = scores.topk(keep, dim=-1, sorted=True).indices
        self_index = torch.arange(blocks, device=query.device).view(1, 1, blocks, 1)
        missing = ~(indices == self_index).any(dim=-1, keepdim=True)
        indices[..., -1:] = torch.where(missing, self_index, indices[..., -1:])
        counts = torch.full(
            (batch, heads, blocks), keep, dtype=torch.int32, device=query.device
        )
        self.routes[(self.current_stage, layer)] = BlockRoute(
            layout.block_grid, indices.to(torch.int32).contiguous(), counts
        )
        self.stats["captured_routes"] += 1

    @staticmethod
    def _parent_and_children(coarse, fine):
        children: list[list[int]] = [[] for _ in range(math.prod(coarse))]
        parents: list[int] = []
        for t in range(fine[0]):
            for h in range(fine[1]):
                for w in range(fine[2]):
                    parent_coord = (
                        min(coarse[0] - 1, t * coarse[0] // fine[0]),
                        min(coarse[1] - 1, h * coarse[1] // fine[1]),
                        min(coarse[2] - 1, w * coarse[2] // fine[2]),
                    )
                    parent = _flat(parent_coord, coarse)
                    child = _flat((t, h, w), fine)
                    parents.append(parent)
                    children[parent].append(child)
        return parents, children

    def _lift(self, source: BlockRoute, target_grid: tuple[int, int, int]) -> BlockRoute:
        parents, children = self._parent_and_children(source.block_grid, target_grid)
        source_cpu = source.indices.detach().cpu()
        batch, heads = source_cpu.shape[:2]
        rows = []
        max_keep = 1
        for batch_index in range(batch):
            batch_rows = []
            for head in range(heads):
                head_rows = []
                for query, parent_query in enumerate(parents):
                    values: list[int] = []
                    for parent_key in source_cpu[batch_index, head, parent_query].tolist():
                        values.extend(children[int(parent_key)])
                    if query not in values:
                        values.append(query)
                    values = list(dict.fromkeys(values))
                    max_keep = max(max_keep, len(values))
                    head_rows.append(values)
                batch_rows.append(head_rows)
            rows.append(batch_rows)
        indices = torch.full(
            (batch, heads, len(parents), max_keep),
            -1,
            dtype=torch.int32,
            device=source.indices.device,
        )
        counts = torch.empty(
            (batch, heads, len(parents)), dtype=torch.int32, device=source.indices.device
        )
        for batch_index in range(batch):
            for head in range(heads):
                for query, values in enumerate(rows[batch_index][head]):
                    counts[batch_index, head, query] = len(values)
                    indices[batch_index, head, query, : len(values)] = torch.tensor(
                        values, dtype=torch.int32, device=source.indices.device
                    )
        return BlockRoute(target_grid, indices.contiguous(), counts.contiguous())

    def _route(self, layer: int, layout: PackedLayout) -> BlockRoute | None:
        if self.current_stage == 0:
            return None
        source = self.routes.get((self.current_stage - 1, layer))
        if source is None:
            return None
        key = (self.current_stage, layer, layout.block_grid)
        route = self.lifted_routes.get(key)
        if route is None:
            route = self._lift(source, layout.block_grid)
            self.lifted_routes[key] = route
        return route

    def attend(
        self,
        layer: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        layout = self._layout(self.current_grid, query.device)
        route = self._route(layer, layout)
        if route is None:
            output = F.scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                dropout_p=0.0,
                is_causal=False,
            ).transpose(1, 2)
            self.stats["dense_attention_calls"] += 1
            if self.current_stage > 0:
                self.stats["sparse_fallbacks"] += 1
        else:
            q = self._pack(query, layout)
            k = self._pack(key, layout)
            v = self._pack(value, layout)
            packed = self.sparse_forward(
                q, k, v, route.indices, route.counts, layout.variable_sizes
            )[0]
            output = self._unpack(packed, layout, query.shape[1])
            self.stats["sparse_attention_calls"] += 1
            density = float(route.counts.float().mean().item()) / math.prod(layout.block_grid)
            self.stats["effective_density_sum"] += density
            self.stats["effective_density_count"] += 1

        if self.capture_route:
            self._capture(layer, query, key, layout)
        return output
