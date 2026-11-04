#!/usr/bin/env python3
"""Analytic cost and cache-size model for RDD-native routing versus VSA.

The model intentionally reports conditional bounds rather than pretending that
FLOPs equal wall-clock time.  It separates:

1. multiresolution denoiser work,
2. VSA's per-fine-step coarse router overhead,
3. sparse fine attention, and
4. the memory cost of caching token-level attention probabilities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def gibibytes(value: float) -> float:
    return value / 2**30


def probability_cache_bytes(
    tokens: int, heads: int, density: float, bytes_per_value: float, layers: int
) -> float:
    return tokens * tokens * heads * density * bytes_per_value * layers


def route_cache_bytes(
    tokens: int, tile_tokens: int, heads: int, bytes_per_value: float, layers: int
) -> float:
    cubes = (tokens + tile_tokens - 1) // tile_tokens
    return cubes * cubes * heads * bytes_per_value * layers


def output_cache_bytes(tokens: int, width: int, bytes_per_value: float, layers: int) -> float:
    return tokens * width * bytes_per_value * layers


def multires_pair_flop_ratio(stage_tokens: list[int], stage_steps: list[int]) -> float:
    fine_tokens = stage_tokens[-1]
    full = sum(stage_steps) * fine_tokens * fine_tokens
    rdd = sum(step * tokens * tokens for tokens, step in zip(stage_tokens, stage_steps))
    return rdd / full


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokens", default="16384,23296,61440")
    parser.add_argument("--heads", default="12,40")
    parser.add_argument("--layers", type=int, default=30)
    parser.add_argument("--width", type=int, default=1536)
    parser.add_argument("--density", type=float, default=0.125)
    parser.add_argument("--tile-tokens", type=int, default=64)
    parser.add_argument("--bytes-per-value", type=float, default=2.0)
    parser.add_argument("--vsa-coarse-runtime-fraction", type=float, default=0.14)
    parser.add_argument("--post-vsa-attention-e2e-fraction", type=float, default=0.20)
    args = parser.parse_args()

    tokens = [int(value) for value in args.tokens.split(",")]
    heads = [int(value) for value in args.heads.split(",")]
    rows = []
    for n in tokens:
        for h in heads:
            full_probability = probability_cache_bytes(
                n, h, args.density, args.bytes_per_value, args.layers
            )
            route = route_cache_bytes(
                n, args.tile_tokens, h, args.bytes_per_value, args.layers
            )
            output = output_cache_bytes(n, args.width, args.bytes_per_value, args.layers)
            rows.append(
                {
                    "tokens": n,
                    "heads": h,
                    "density": args.density,
                    "all_layer_probability_cache_GiB": gibibytes(full_probability),
                    "all_layer_coarse_route_cache_GiB": gibibytes(route),
                    "all_layer_attention_output_cache_GiB": gibibytes(output),
                }
            )

    coarse = args.vsa_coarse_runtime_fraction
    attn_fraction = args.post_vsa_attention_e2e_fraction
    result = {
        "assumptions": vars(args),
        "conditional_speed_bounds": {
            "rdd_native_vs_vsa_attention_speedup_if_same_sparse_kernel": 1.0 / (1.0 - coarse),
            "rdd_native_vs_vsa_e2e_speedup_if_attention_is_given_fraction": 1.0
            / (1.0 - attn_fraction * coarse),
            "probability_reuse_attention_core_speedup_k2": 2.0 / (1.0 + 1.0 / 2.0),
            "probability_reuse_attention_core_speedup_k3": 2.0 / (1.0 + 1.0 / 3.0),
            "probability_reuse_attention_core_speedup_k5": 2.0 / (1.0 + 1.0 / 5.0),
        },
        "example_multiresolution_pair_flops": {
            "stage_tokens": [4096, 9216, 16384],
            "stage_steps": [20, 15, 15],
            "rdd_over_all_fine_pair_flops": multires_pair_flop_ratio(
                [4096, 9216, 16384], [20, 15, 15]
            ),
        },
        "cache_sizes": rows,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
