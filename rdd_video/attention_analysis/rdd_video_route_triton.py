"""Triton route-lifting kernel for RDD video attention.

The previous resolution stores a compact list of selected 64-token KV cubes
for every query cube.  At a resolution transition this kernel maps each new
query cube to its parent and expands every selected parent KV cube to its child
cubes.  No fine-resolution QK matrix is formed.

This is inference-only infrastructure: it does not depend on a checkpoint and
does not change old model/checkpoint behavior.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _lift_routes_kernel(
    coarse_idx,
    query_parent,
    child_flat,
    child_offsets,
    out_idx,
    out_counts,
    q_coarse: tl.constexpr,
    q_fine: tl.constexpr,
    k_keep: tl.constexpr,
    max_children: tl.constexpr,
    out_block: tl.constexpr,
):
    row = tl.program_id(0)
    q = row % q_fine
    bh = row // q_fine
    parent_q = tl.load(query_parent + q)
    coarse_row = coarse_idx + (bh * q_coarse + parent_q) * k_keep

    pos = tl.arange(0, out_block)
    value = tl.full((out_block,), -1, tl.int32)
    cursor = tl.zeros((1,), tl.int32)
    for slot in tl.static_range(0, k_keep):
        parent_k = tl.load(coarse_row + slot)
        begin = tl.load(child_offsets + parent_k)
        end = tl.load(child_offsets + parent_k + 1)
        count = end - begin
        valid = (pos >= cursor) & (pos < cursor + count)
        local = pos - cursor
        child = tl.load(child_flat + begin + local, mask=valid, other=0)
        value = tl.where(valid, child, value)
        cursor += count

    max_out = k_keep * max_children
    tl.store(out_idx + row * max_out + pos, value, mask=pos < max_out)
    tl.store(out_counts + row, tl.sum(cursor, axis=0))


def _flat(coord: tuple[int, int, int], grid: tuple[int, int, int]) -> int:
    return (coord[0] * grid[1] + coord[1]) * grid[2] + coord[2]


def make_hierarchy_lut(
    coarse_block_grid: tuple[int, int, int],
    fine_block_grid: tuple[int, int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Return fine-query parent map and inverse coarse-key child lists."""
    coarse_count = math.prod(coarse_block_grid)
    children: list[list[int]] = [[] for _ in range(coarse_count)]
    parent_q: list[int] = []
    for t in range(fine_block_grid[0]):
        for h in range(fine_block_grid[1]):
            for w in range(fine_block_grid[2]):
                parent_coord = (
                    min(coarse_block_grid[0] - 1, t * coarse_block_grid[0] // fine_block_grid[0]),
                    min(coarse_block_grid[1] - 1, h * coarse_block_grid[1] // fine_block_grid[1]),
                    min(coarse_block_grid[2] - 1, w * coarse_block_grid[2] // fine_block_grid[2]),
                )
                parent = _flat(parent_coord, coarse_block_grid)
                child = _flat((t, h, w), fine_block_grid)
                parent_q.append(parent)
                children[parent].append(child)

    offsets = [0]
    flat_children: list[int] = []
    for group in children:
        flat_children.extend(group)
        offsets.append(len(flat_children))
    max_children = max(len(group) for group in children)
    return (
        torch.tensor(parent_q, dtype=torch.int32, device=device),
        torch.tensor(flat_children, dtype=torch.int32, device=device),
        torch.tensor(offsets, dtype=torch.int32, device=device),
        max_children,
    )


def lift_routes(
    coarse_idx: torch.Tensor,
    query_parent: torch.Tensor,
    child_flat: torch.Tensor,
    child_offsets: torch.Tensor,
    max_children: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lift [B,H,Qc,Kc] parent routes to [B,H,Qf,Kout] child routes."""
    if coarse_idx.dtype != torch.int32 or not coarse_idx.is_contiguous():
        raise ValueError("coarse_idx must be contiguous int32")
    batch, heads, q_coarse, keep = coarse_idx.shape
    q_fine = query_parent.numel()
    max_out = keep * max_children
    out = torch.empty((batch, heads, q_fine, max_out), dtype=torch.int32, device=coarse_idx.device)
    counts = torch.empty((batch, heads, q_fine), dtype=torch.int32, device=coarse_idx.device)
    out_block = triton.next_power_of_2(max_out)
    _lift_routes_kernel[(batch * heads * q_fine,)](
        coarse_idx,
        query_parent,
        child_flat,
        child_offsets,
        out,
        counts,
        q_coarse=q_coarse,
        q_fine=q_fine,
        k_keep=keep,
        max_children=max_children,
        out_block=out_block,
    )
    return out, counts
