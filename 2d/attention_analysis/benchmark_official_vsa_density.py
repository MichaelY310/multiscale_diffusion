#!/usr/bin/env python3
"""Benchmark the unmodified FastVideo VSA Triton forward kernel.

This benchmark deliberately separates two questions:

1. At what sequence length / retained KV-block ratio does the official
   64x64 VSA kernel beat PyTorch SDPA?
2. On the small square token grids used by RDD-2D, how does our packed
   micro-block kernel compare under the same synthetic Q/K/V workload?

The official kernel is loaded directly from the FastVideo checkout so that
FastVideo's optional package-level dependencies cannot silently select a
different backend.  No line of the official kernel is changed here.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cyclic_indices(
    batch: int,
    heads: int,
    query_blocks: int,
    key_blocks: int,
    retained_blocks: int,
    device: torch.device,
) -> torch.Tensor:
    q = torch.arange(query_blocks, device=device, dtype=torch.int32)
    j = torch.arange(retained_blocks, device=device, dtype=torch.int32)
    route = (q[:, None] + j[None, :]) % key_blocks
    return route[None, None].expand(batch, heads, -1, -1).contiguous()


def event_times_ms(
    fn: Callable[[], torch.Tensor], warmup: int, repeats: int
) -> List[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times: List[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))
    return times


def summarize_times(values: Sequence[float]) -> Dict[str, float]:
    ordered = sorted(values)
    return {
        "median_ms": statistics.median(ordered),
        "mean_ms": statistics.fmean(ordered),
        "min_ms": ordered[0],
        "p90_ms": ordered[min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)],
    }


def cosine_and_rel_l2(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    cosine = F.cosine_similarity(af, bf, dim=0).item()
    rel_l2 = ((af - bf).norm() / bf.norm().clamp_min(1e-12)).item()
    return {"cosine_vs_dense": cosine, "relative_l2_vs_dense": rel_l2}


def repeats_for_n(n: int) -> tuple[int, int]:
    if n <= 1024:
        return 20, 100
    if n <= 4096:
        return 10, 50
    if n <= 16384:
        return 5, 20
    return 3, 8


def benchmark_official_vsa(
    official,
    n_values: Iterable[int],
    densities: Iterable[float],
    batch: int,
    heads: int,
    dim: int,
    device: torch.device,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    dtype = torch.bfloat16
    for n in n_values:
        if n % 64:
            raise ValueError(f"official VSA requires N multiple of 64, got {n}")
        torch.manual_seed(20260823 + n)
        q = torch.randn(batch, heads, n, dim, device=device, dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        n_blocks = n // 64
        variable_sizes = torch.full(
            (n_blocks,), 64, device=device, dtype=torch.int32
        )
        warmup, repeats = repeats_for_n(n)

        def dense_call():
            return F.scaled_dot_product_attention(q, k, v)

        dense_out = dense_call()
        dense_stats = summarize_times(event_times_ms(dense_call, warmup, repeats))
        rows.append(
            {
                "family": "torch_sdpa_dense",
                "sequence_length": n,
                "heads": heads,
                "head_dim": dim,
                "requested_density": 1.0,
                "actual_density": 1.0,
                "retained_blocks": n_blocks,
                "total_blocks": n_blocks,
                **dense_stats,
                "speedup_vs_dense": 1.0,
            }
        )

        # Sparse-first order matches how the official autotuner is normally
        # first exercised by VSA. Its cache key is (N_CTX_Q, HEAD_DIM), not K.
        for density in sorted(densities):
            keep = max(1, min(n_blocks, int(round(density * n_blocks))))
            actual = keep / n_blocks
            indices = cyclic_indices(
                batch, heads, n_blocks, n_blocks, keep, device
            )
            counts = torch.full(
                (batch, heads, n_blocks), keep, device=device, dtype=torch.int32
            )

            def sparse_call():
                return official.triton_block_sparse_attn_forward(
                    q, k, v, indices, counts, variable_sizes
                )[0]

            # The first invocation includes Triton compilation/autotuning and
            # is intentionally outside the measured region.
            sparse_out = sparse_call()
            stats = summarize_times(event_times_ms(sparse_call, warmup, repeats))
            quality = (
                cosine_and_rel_l2(sparse_out, dense_out)
                if keep == n_blocks
                else {}
            )
            rows.append(
                {
                    "family": "official_vsa_triton64",
                    "sequence_length": n,
                    "heads": heads,
                    "head_dim": dim,
                    "requested_density": density,
                    "actual_density": actual,
                    "retained_blocks": keep,
                    "total_blocks": n_blocks,
                    **stats,
                    "speedup_vs_dense": dense_stats["median_ms"] / stats["median_ms"],
                    **quality,
                }
            )
        del q, k, v, dense_out
        torch.cuda.empty_cache()
    return rows


def benchmark_rdd_packed(
    rdd,
    n_values: Iterable[int],
    densities: Iterable[float],
    batch: int,
    heads: int,
    dim: int,
    device: torch.device,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    dtype = torch.bfloat16
    for n in n_values:
        side = math.isqrt(n)
        if side * side != n:
            continue
        torch.manual_seed(20260824 + n)
        q = torch.randn(batch, heads, n, dim, device=device, dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        warmup, repeats = repeats_for_n(n)

        def dense_call():
            return F.scaled_dot_product_attention(q, k, v)

        dense_stats = summarize_times(event_times_ms(dense_call, warmup, repeats))
        for block_side in (2, 4):
            if side % block_side:
                continue
            block_tokens = block_side * block_side
            n_blocks = n // block_tokens
            for density in sorted(densities):
                keep = max(1, min(n_blocks, int(round(density * n_blocks))))
                actual = keep / n_blocks
                indices = cyclic_indices(
                    batch, heads, n_blocks, n_blocks, keep, device
                )

                def sparse_call():
                    return rdd.packed_block_sparse_attention(
                        q, k, v, indices, block_side=block_side
                    )

                sparse_call()
                stats = summarize_times(event_times_ms(sparse_call, warmup, repeats))
                rows.append(
                    {
                        "family": f"rdd_packed_{block_tokens}token_blocks",
                        "sequence_length": n,
                        "heads": heads,
                        "head_dim": dim,
                        "requested_density": density,
                        "actual_density": actual,
                        "retained_blocks": keep,
                        "total_blocks": n_blocks,
                        **stats,
                        "speedup_vs_dense": dense_stats["median_ms"] / stats["median_ms"],
                    }
                )
        del q, k, v
        torch.cuda.empty_cache()
    return rows


def write_outputs(rows: List[Dict[str, object]], output_dir: Path, metadata: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "results": rows}
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    keys = sorted({key for row in rows for key in row})
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Official VSA density benchmark",
        "",
        "Median CUDA-event time; route metadata is precomputed.",
        "",
        "| family | N | density | kept blocks | median ms | speedup vs SDPA |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['family']} | {row['sequence_length']} | "
            f"{100 * float(row['actual_density']):.2f}% | "
            f"{row['retained_blocks']}/{row['total_blocks']} | "
            f"{float(row['median_ms']):.4f} | "
            f"{float(row['speedup_vs_dense']):.3f}x |"
        )
    (output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-kernel", type=Path, required=True)
    parser.add_argument("--rdd-kernel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--sequence-lengths", type=int, nargs="+", default=[256, 1024, 4096, 16384, 49152]
    )
    parser.add_argument(
        "--densities", type=float, nargs="+", default=[0.125, 0.25, 0.5, 0.75, 0.875, 1.0]
    )
    parser.add_argument(
        "--rdd-sequence-lengths", type=int, nargs="+", default=[256, 1024, 4096]
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    official = load_module("official_vsa_sparse_kernel", args.official_kernel.resolve())
    rdd = load_module("rdd_packed_sparse_kernel", args.rdd_kernel.resolve())

    started = time.time()
    rows = benchmark_official_vsa(
        official,
        args.sequence_lengths,
        args.densities,
        batch=1,
        heads=args.heads,
        dim=args.head_dim,
        device=device,
    )
    rows.extend(
        benchmark_rdd_packed(
            rdd,
            args.rdd_sequence_lengths,
            args.densities,
            batch=1,
            heads=args.heads,
            dim=args.head_dim,
            device=device,
        )
    )
    metadata = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "python": platform.python_version(),
        "hostname": platform.node(),
        "official_kernel": str(args.official_kernel),
        "rdd_kernel": str(args.rdd_kernel),
        "heads": args.heads,
        "head_dim": args.head_dim,
        "elapsed_seconds": time.time() - started,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "notes": [
            "Official VSA source is loaded unchanged and forced to its Triton64 forward path.",
            "Sparse route construction is excluded; wrapper allocation and kernel launch are included.",
            "Official autotune cache key is (N_CTX_Q, HEAD_DIM); densities run sparse-to-dense.",
        ],
    }
    write_outputs(rows, args.output_dir, metadata)
    print(json.dumps(metadata, indent=2))
    print(f"wrote {args.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
