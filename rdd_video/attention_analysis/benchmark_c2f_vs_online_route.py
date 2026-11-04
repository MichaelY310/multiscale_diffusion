#!/usr/bin/env python3
"""Compare C2F route lifting with the online VSA route builder.

This intentionally measures routing only.  Both methods feed the same
FastVideo block-sparse attention executor in the end-to-end experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rdd_video.attention_analysis.rdd_c2f_route_triton import lift_c2f_route


def timed_ms(fn, warmup: int, repeats: int) -> dict[str, float]:
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
        samples.append(float(start.elapsed_time(end)))
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "mean_ms": statistics.mean(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--density", type=float, default=0.20)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=1000)
    args = parser.parse_args()

    from fastvideo_kernel.triton_kernels.fused_compress_topk import (
        fused_block_mean,
        fused_topk_mask,
    )
    from fastvideo_kernel.triton_kernels.index import map_to_index

    device = "cuda"
    dtype = torch.bfloat16
    block_tokens = 64
    coarse_grid = (1, 7, 13)
    fine_grid = (2, 7, 13)
    coarse_blocks = math.prod(coarse_grid)
    fine_blocks = math.prod(fine_grid)
    coarse_keep = math.ceil(coarse_blocks * args.density)
    fine_keep = math.ceil(fine_blocks * args.density)

    source_scores = torch.randn(
        1, args.heads, coarse_blocks, coarse_blocks, device=device, dtype=dtype
    )
    source_indices = source_scores.topk(coarse_keep, dim=-1).indices.to(torch.int32)
    source_counts = torch.full(
        (1, args.heads, coarse_blocks),
        coarse_keep,
        device=device,
        dtype=torch.int32,
    )
    q = torch.randn(
        1,
        args.heads,
        fine_blocks * block_tokens,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    k = torch.randn_like(q)
    variable_sizes = torch.full(
        (fine_blocks,), block_tokens, device=device, dtype=torch.int32
    )

    def c2f_route():
        return lift_c2f_route(
            source_indices, source_counts, coarse_grid, fine_grid
        )

    def online_vsa_route():
        q_center = fused_block_mean(q, variable_sizes, block_tokens)
        k_center = fused_block_mean(k, variable_sizes, block_tokens)
        scores = torch.matmul(q_center, k_center.transpose(-2, -1)) / math.sqrt(
            args.head_dim
        )
        mask = fused_topk_mask(scores, fine_keep)
        return map_to_index(mask)

    c2f = timed_ms(c2f_route, args.warmup, args.repeats)
    online = timed_ms(online_vsa_route, args.warmup, args.repeats)
    result = {
        "device": torch.cuda.get_device_name(0),
        "shape": {
            "coarse_grid": coarse_grid,
            "fine_grid": fine_grid,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "block_tokens": block_tokens,
            "density": args.density,
            "coarse_keep": coarse_keep,
            "fine_keep": fine_keep,
        },
        "c2f_lift": c2f,
        "online_vsa_route": online,
        "route_builder_speedup": online["median_ms"] / c2f["median_ms"],
        "scope_note": "routing only; excludes QKV projections and sparse attention",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
