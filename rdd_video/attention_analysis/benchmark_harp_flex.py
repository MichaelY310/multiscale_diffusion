#!/usr/bin/env python3
"""Prototype benchmark for Hierarchical Attention Routing for Pyramids (HARP).

FlexAttention is used as an executable correctness/performance prototype.  The
paper implementation should pass the same cached route indices to FastVideo's
64-token VSA kernel, avoiding VSA's per-layer fine-resolution router.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, flex_attention


def parse_ints(text: str, expected: int) -> tuple[int, ...]:
    values = tuple(int(x) for x in text.split(","))
    if len(values) != expected:
        raise ValueError(f"expected {expected} comma separated integers")
    return values


def root_coordinates(root_grid: tuple[int, int, int], device: torch.device) -> torch.Tensor:
    return torch.stack(
        torch.meshgrid(
            *(torch.arange(x, device=device) for x in root_grid), indexing="ij"
        ),
        dim=-1,
    ).reshape(-1, 3)


def make_route(
    heads: int,
    root_grid: tuple[int, int, int],
    density: float,
    local_radius: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    roots = math.prod(root_grid)
    keep = max(1, min(roots, math.ceil(density * roots)))
    coords = root_coordinates(root_grid, device)
    local = (coords[:, None] - coords[None]).abs().amax(-1) <= local_radius
    route = local[None].expand(heads, -1, -1).clone()
    generator = torch.Generator(device=device).manual_seed(seed)
    random_scores = torch.rand((heads, roots, roots), generator=generator, device=device)
    random_scores.masked_fill_(route, 2.0)
    idx = random_scores.topk(keep, dim=-1).indices
    route.scatter_(-1, idx, True)
    return route


def block_mask_from_route(route: torch.Tensor, block_size: int, seq: int) -> BlockMask:
    """Build CSR BlockMask directly; avoids create_block_mask's dense vmap path."""
    counts = route.sum(dim=-1, dtype=torch.int32)
    # PyTorch's public ``from_kv_blocks`` currently reconstructs a dense
    # transpose using ``kv_indices.shape[-1]`` as the number of KV columns.
    # Consequently this dimension must contain *all* root-block columns, not
    # only max(counts), even though entries after counts[row] are ignored by
    # the FlexAttention forward kernel.  Keeping a compact CSR array here
    # triggers an index-out-of-bounds assertion whenever a selected column id
    # is larger than max(counts)-1.
    indices = torch.argsort(route.to(torch.int8), dim=-1, descending=True, stable=True)
    indices = indices.to(torch.int32)
    counts = counts.unsqueeze(0)
    indices = indices.unsqueeze(0)
    return BlockMask.from_kv_blocks(
        counts,
        indices,
        BLOCK_SIZE=block_size,
        seq_lengths=(seq, seq),
    )


def cuda_ms(fn, warmup: int, repeats: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    values = torch.tensor(samples)
    return float(values.median()), float(torch.quantile(values, 0.90))


def validate_flex(device: torch.device, dtype: torch.dtype, block_size: int) -> dict[str, float]:
    heads, roots, head_dim = 2, 8, 64
    seq = roots * block_size
    q = torch.randn(1, heads, seq, head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    route = make_route(heads, (2, 2, 2), 0.25, 0, 7, device)

    block_mask = block_mask_from_route(route, block_size, seq)
    compiled = torch.compile(flex_attention, dynamic=False)
    kernel_options = {
        "BLOCK_M": block_size,
        "BLOCK_N": block_size,
        "ROWS_GUARANTEED_SAFE": True,
    }
    got = compiled(q, k, v, block_mask=block_mask, kernel_options=kernel_options)
    token_mask = route[:, :, None, :, None].expand(
        heads, roots, block_size, roots, block_size
    ).reshape(heads, seq, seq)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=token_mask[None])
    return {
        "max_abs": float((got - ref).abs().max()),
        "relative_l2": float((got - ref).float().norm() / ref.float().norm().clamp_min(1e-12)),
        "cosine": float(F.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fine-grid", default="24,28,52", help="padded fine DiT grid T,H,W")
    parser.add_argument("--root-grid", default="6,7,13", help="coarse/root grid T,H,W")
    parser.add_argument("--tile", default="4,4,4")
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--densities", default="0.10,0.20,0.30")
    parser.add_argument("--local-radius", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    fine_grid = parse_ints(args.fine_grid, 3)
    root_grid = parse_ints(args.root_grid, 3)
    tile = parse_ints(args.tile, 3)
    expected = tuple(r * b for r, b in zip(root_grid, tile))
    if fine_grid != expected:
        raise ValueError(f"fine grid must be root_grid*tile: {fine_grid} != {expected}")
    block_size = math.prod(tile)
    seq = math.prod(fine_grid)
    roots = math.prod(root_grid)
    dtype = torch.bfloat16
    torch.manual_seed(args.seed)
    q = torch.randn(1, args.heads, seq, args.head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    compiled_flex = torch.compile(flex_attention, dynamic=False)
    kernel_options = {
        "BLOCK_M": block_size,
        "BLOCK_N": block_size,
        "ROWS_GUARANTEED_SAFE": True,
    }

    report = {
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "dtype": str(dtype),
        "fine_grid": fine_grid,
        "root_grid": root_grid,
        "tile": tile,
        "tokens_including_padding": seq,
        "root_tiles": roots,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "correctness": validate_flex(device, dtype, block_size),
        "benchmarks": [],
    }

    dense_median, dense_p90 = cuda_ms(
        lambda: F.scaled_dot_product_attention(q, k, v), args.warmup, args.repeats
    )
    report["dense_sdpa_ms"] = {"median": dense_median, "p90": dense_p90}

    # A VSA-like self router recomputes a root score matrix from fine Q/K.
    q_root = q.reshape(1, args.heads, roots, block_size, args.head_dim).mean(3)
    k_root = k.reshape(1, args.heads, roots, block_size, args.head_dim).mean(3)

    for density in (float(x) for x in args.densities.split(",")):
        route = make_route(
            args.heads, root_grid, density, args.local_radius, args.seed, device
        )
        effective_density = float(route.float().mean())

        mask_build_start = time.perf_counter()
        block_mask = block_mask_from_route(route, block_size, seq)
        mask_build_seconds = time.perf_counter() - mask_build_start
        sparse_median, sparse_p90 = cuda_ms(
            lambda: compiled_flex(
                q, k, v, block_mask=block_mask, kernel_options=kernel_options
            ),
            args.warmup,
            args.repeats,
        )
        keep = max(1, math.ceil(density * roots))
        router_median, router_p90 = cuda_ms(
            lambda: torch.topk(
                torch.matmul(q_root, k_root.transpose(-1, -2)), keep, dim=-1
            ).indices,
            args.warmup,
            args.repeats,
        )
        cached_idx = torch.topk(
            torch.matmul(q_root, k_root.transpose(-1, -2)), keep, dim=-1
        ).indices
        transfer_median, transfer_p90 = cuda_ms(
            lambda: cached_idx.clone(), args.warmup, args.repeats
        )
        report["benchmarks"].append(
            {
                "requested_density": density,
                "effective_density_with_local_union": effective_density,
                "flex_sparse_ms": {"median": sparse_median, "p90": sparse_p90},
                "speedup_vs_dense": dense_median / sparse_median,
                "vsa_like_fine_router_ms": {"median": router_median, "p90": router_p90},
                "cached_route_transfer_ms": {"median": transfer_median, "p90": transfer_p90},
                "python_block_mask_build_seconds_one_time": mask_build_seconds,
                "attention_pair_reduction": 1.0 / effective_density,
            }
        )

    report["notes"] = [
        "FlexAttention is a functional prototype, not the final FastVideo VSA kernel.",
        "The one-time Python BlockMask construction is excluded from steady-state kernel latency.",
        "Production stores sorted CSR/CSC route indices per layer/head and reuses them under CUDA Graphs.",
        "Padded 24x28x52 corresponds to Wan 81-frame 448x832 video after VAE and patchification.",
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
