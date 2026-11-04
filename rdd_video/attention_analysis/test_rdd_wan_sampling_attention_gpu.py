#!/usr/bin/env python3
"""Numerical contract for the FastVideo-kernel RDD-Wan sampling hook."""

from __future__ import annotations

import argparse
import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from rdd_wan_sampling_attention import BlockRoute, RDDWanAttentionController


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fastvideo-kernel-root", required=True)
    args = parser.parse_args()
    device = torch.device("cuda:0")
    controller = RDDWanAttentionController(
        SimpleNamespace(blocks=[]),
        safe_attention=lambda *unused, **unused_kw: None,
        rope_apply=lambda q, *unused: q,
        mode="route",
        route_density=1.0,
        fastvideo_kernel_root=args.fastvideo_kernel_root,
    )
    grid = (3, 5, 7)
    sequence = math.prod(grid)
    torch.manual_seed(20260905)
    q = torch.randn(1, sequence, 2, 128, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    layout = controller._layout(grid, device)
    q_packed = controller._pack(q, layout)
    k_packed = controller._pack(k, layout)
    v_packed = controller._pack(v, layout)
    blocks = math.prod(layout.block_grid)
    indices = torch.arange(blocks, dtype=torch.int32, device=device)
    indices = indices.view(1, 1, 1, blocks).expand(1, 2, blocks, blocks).contiguous()
    counts = torch.full((1, 2, blocks), blocks, dtype=torch.int32, device=device)
    route = BlockRoute(layout.block_grid, indices, counts)
    sparse_packed = controller.sparse_forward(
        q_packed, k_packed, v_packed,
        route.indices, route.counts, layout.variable_sizes,
    )[0]
    sparse = controller._unpack(sparse_packed, layout, sequence)
    dense = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)
    unpacked_q = controller._unpack(q_packed, layout, sequence)
    torch.cuda.synchronize()
    max_pack_error = float((unpacked_q - q).abs().max())
    cosine = float(F.cosine_similarity(sparse.float().flatten(), dense.float().flatten(), dim=0))
    rel_l2 = float((sparse.float() - dense.float()).norm() / dense.float().norm())
    print({
        "grid": grid,
        "block_grid": layout.block_grid,
        "pack_max_abs_error": max_pack_error,
        "full_route_cosine_vs_dense": cosine,
        "full_route_relative_l2": rel_l2,
    })
    if max_pack_error != 0.0 or cosine < 0.999 or rel_l2 > 0.03:
        raise RuntimeError("RDD-Wan sparse attention numerical contract failed")


if __name__ == "__main__":
    main()

