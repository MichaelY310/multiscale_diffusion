#!/usr/bin/env python3
"""Microbenchmark the RDD C2F route-lift kernel on FastWan tile grids."""

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


def make_route(grid: tuple[int, int, int], density: float, heads: int, device: str):
    blocks = math.prod(grid)
    keep = max(1, math.ceil(blocks * density))
    scores = torch.randn(1, heads, blocks, blocks, device=device)
    indices = scores.topk(keep, dim=-1).indices.to(torch.int32).contiguous()
    counts = torch.full((1, heads, blocks), keep, dtype=torch.int32, device=device)
    return indices, counts


def timed_ms(fn, warmup: int, repeats: int) -> tuple[float, float, float]:
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
    return statistics.median(samples), min(samples), statistics.mean(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--density", type=float, default=0.20)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=1000)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    cases = [
        ((1, 2, 4), (1, 4, 7)),
        ((1, 4, 7), (2, 7, 13)),
        ((1, 1, 4), (1, 2, 7)),
        ((1, 2, 7), (2, 4, 13)),
    ]
    result = {
        "device": torch.cuda.get_device_name(0),
        "density": args.density,
        "heads": args.heads,
        "cases": [],
    }
    for coarse, fine in cases:
        indices, counts = make_route(coarse, args.density, args.heads, "cuda")
        # Compile before the timed region.
        out_i, out_c = lift_c2f_route(indices, counts, coarse, fine)
        torch.cuda.synchronize()
        median, minimum, mean = timed_ms(
            lambda: lift_c2f_route(indices, counts, coarse, fine),
            args.warmup,
            args.repeats,
        )
        result["cases"].append(
            {
                "coarse_grid": coarse,
                "fine_grid": fine,
                "coarse_blocks": math.prod(coarse),
                "fine_blocks": math.prod(fine),
                "output_slots": int(out_i.shape[-1]),
                "count_min": int(out_c.min().item()),
                "count_max": int(out_c.max().item()),
                "latency_ms_median": median,
                "latency_ms_min": minimum,
                "latency_ms_mean": mean,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
