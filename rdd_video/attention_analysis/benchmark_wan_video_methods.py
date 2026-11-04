#!/usr/bin/env python3
"""Checkpoint-free Wan2.1-1.3B video attention/runtime proxy benchmark.

The benchmark uses random BF16 tensors and the *same unmodified FastVideo VSA
Triton sparse executor* for both:

* VSA: current-fine Q/K/V compression -> coarse QK/AV -> top-k -> sparse QK/AV.
* RDD-native routing: an already cached route from the preceding resolution ->
  the same sparse QK/AV executor.

This removes kernel quality as a confounder when comparing VSA routing overhead
with RDD's external-route idea.  It also times dense SDPA and the non-attention
parts of a Wan block with random matrices, then composes stage schedules into a
denoiser-only latency projection.  It never loads a checkpoint.
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Shape:
    name: str
    grid: tuple[int, int, int]
    padded_grid: tuple[int, int, int]

    @property
    def tokens(self) -> int:
        return math.prod(self.grid)

    @property
    def padded_tokens(self) -> int:
        return math.prod(self.padded_grid)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summary_ms(fn: Callable[[], torch.Tensor], warmup: int, repeats: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        fn()
        end.record()
        end.synchronize()
        values.append(float(begin.elapsed_time(end)))
    values.sort()
    return {
        "median_ms": statistics.median(values),
        "min_ms": values[0],
        "p90_ms": values[min(len(values) - 1, math.ceil(0.9 * len(values)) - 1)],
    }


def cyclic_indices(batch: int, heads: int, blocks: int, keep: int, device: torch.device) -> torch.Tensor:
    q = torch.arange(blocks, dtype=torch.int32, device=device)
    offset = torch.arange(keep, dtype=torch.int32, device=device)
    route = (q[:, None] + offset[None, :]) % blocks
    return route[None, None].expand(batch, heads, -1, -1).contiguous()


def padded_shape(grid: tuple[int, int, int], tile: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(((x + t - 1) // t) * t for x, t in zip(grid, tile))


def make_shapes(tile: tuple[int, int, int]) -> dict[str, Shape]:
    # Wan VAE: 81x448x832 -> 21x56x104 latent; patch (1,2,2) -> 21x28x52 tokens.
    grids = {
        "coarse": (6, 7, 13),
        "middle": (11, 14, 26),
        "fine": (21, 28, 52),
        # TPDiff-style matched proxy: reduce time only, preserve spatial grid.
        "temporal_coarse": (6, 28, 52),
        "temporal_middle": (11, 28, 52),
    }
    return {name: Shape(name, grid, padded_shape(grid, tile)) for name, grid in grids.items()}


def block_grid(shape: Shape, tile: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(x // t for x, t in zip(shape.padded_grid, tile))


def benchmark_route_lift(
    source: Shape,
    target: Shape,
    densities: list[float],
    heads: int,
    tile: tuple[int, int, int],
    device: torch.device,
    route_kernel,
    warmup: int,
    repeats: int,
) -> dict:
    source_grid = block_grid(source, tile)
    target_grid = block_grid(target, tile)
    source_blocks = math.prod(source_grid)
    target_blocks = math.prod(target_grid)
    parent, child_flat, child_offsets, max_children = route_kernel.make_hierarchy_lut(
        source_grid, target_grid, device
    )
    rows = {}
    for density in densities:
        keep = max(1, min(source_blocks, round(source_blocks * density)))
        idx = cyclic_indices(1, heads, source_blocks, keep, device)

        def lift_call():
            return route_kernel.lift_routes(idx, parent, child_flat, child_offsets, max_children)

        lifted, counts = lift_call()
        torch.cuda.synchronize()
        valid_mask = torch.arange(lifted.shape[-1], device=device)[None, None, None, :] < counts[..., None]
        valid_lifted = lifted[valid_mask]
        if (int(counts.max()) > lifted.shape[-1] or int(valid_lifted.min()) < 0
                or int(valid_lifted.max()) >= target_blocks):
            raise RuntimeError(f"invalid lifted route for {source.name}->{target.name}")
        # Exact first-head comparison against the simple CPU definition:
        # concatenate all child blocks of each selected parent KV block.
        idx_cpu = idx[0, 0].cpu()
        lifted_cpu = lifted[0, 0].cpu()
        counts_cpu = counts[0, 0].cpu()
        parent_cpu = parent.cpu()
        child_cpu = child_flat.cpu()
        offsets_cpu = child_offsets.cpu()
        for fine_q in range(target_blocks):
            expected: list[int] = []
            parent_q = int(parent_cpu[fine_q])
            for parent_k_tensor in idx_cpu[parent_q]:
                parent_k = int(parent_k_tensor)
                begin, end = int(offsets_cpu[parent_k]), int(offsets_cpu[parent_k + 1])
                expected.extend(int(x) for x in child_cpu[begin:end])
            observed = [int(x) for x in lifted_cpu[fine_q, :int(counts_cpu[fine_q])]]
            if observed != expected:
                raise RuntimeError(f"route lift mismatch at {source.name}->{target.name}, q={fine_q}")
        rows[f"{density:.6f}"] = {
            "source_keep_blocks": keep,
            "source_blocks": source_blocks,
            "target_blocks": target_blocks,
            "target_max_kept_blocks": int(counts.max()),
            "target_mean_kept_blocks": float(counts.float().mean()),
            "target_mean_density": float(counts.float().mean() / target_blocks),
            "exact_cpu_reference": True,
            "latency": summary_ms(lift_call, warmup, repeats),
        }
    return {
        "source": source.name,
        "target": target.name,
        "source_block_grid": source_grid,
        "target_block_grid": target_grid,
        "max_children_per_parent": max_children,
        "densities": rows,
    }


def benchmark_attention_shape(
    shape: Shape,
    densities: list[float],
    heads: int,
    head_dim: int,
    device: torch.device,
    official,
    fused,
    warmup: int,
    repeats: int,
) -> dict:
    batch = 1
    dtype = torch.bfloat16
    block_tokens = 64
    padded_n = shape.padded_tokens
    if padded_n % block_tokens:
        raise ValueError(f"{shape.name}: padded N={padded_n} is not divisible by 64")
    blocks = padded_n // block_tokens
    torch.manual_seed(20260904 + padded_n)
    q = torch.randn(batch, heads, padded_n, head_dim, dtype=dtype, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    variable_sizes = torch.full((blocks,), block_tokens, dtype=torch.int32, device=device)

    def dense_call():
        return F.scaled_dot_product_attention(q[:, :, :shape.tokens], k[:, :, :shape.tokens], v[:, :, :shape.tokens])

    dense = summary_ms(dense_call, warmup, repeats)
    result = {
        "shape": asdict(shape),
        "tokens": shape.tokens,
        "padded_tokens": padded_n,
        "blocks": blocks,
        "dense_sdpa": dense,
        "densities": {},
    }
    for density in densities:
        keep = max(1, min(blocks, round(blocks * density)))
        actual_density = keep / blocks
        static_idx = cyclic_indices(batch, heads, blocks, keep, device)
        static_num = torch.full((batch, heads, blocks), keep, dtype=torch.int32, device=device)

        def native_call():
            return official.triton_block_sparse_attn_forward(
                q, k, v, static_idx, static_num, variable_sizes
            )[0]

        # This is an optimistic, fully fused-friendly VSA router.  It uses the
        # official block-mean kernels; top-k returns compact indices directly
        # instead of constructing a bool mask and compacting it again.
        def vsa_router_call():
            q_c = fused.fused_block_mean(q, variable_sizes, block_tokens)
            k_c = fused.fused_block_mean(k, variable_sizes, block_tokens)
            v_c = fused.fused_block_mean(v, variable_sizes, block_tokens)
            scores = torch.matmul(q_c, k_c.transpose(-2, -1)) / math.sqrt(head_dim)
            coarse = torch.matmul(torch.softmax(scores, dim=-1), v_c)
            idx = torch.topk(scores, keep, dim=-1, sorted=False).indices.to(torch.int32)
            return idx, coarse.repeat_interleave(block_tokens, dim=2)

        def vsa_call():
            idx, coarse_full = vsa_router_call()
            counts = torch.full((batch, heads, blocks), keep, dtype=torch.int32, device=device)
            sparse = official.triton_block_sparse_attn_forward(q, k, v, idx, counts, variable_sizes)[0]
            return sparse + coarse_full

        native_call()
        vsa_call()
        result["densities"][f"{density:.6f}"] = {
            "requested_density": density,
            "actual_density": actual_density,
            "kept_blocks": keep,
            "native_external_route": summary_ms(native_call, warmup, repeats),
            "vsa_online_router_only": summary_ms(vsa_router_call, warmup, repeats),
            "vsa_online_route_and_sparse": summary_ms(vsa_call, warmup, repeats),
        }
    del q, k, v
    torch.cuda.empty_cache()
    return result


def benchmark_wan_nonattention(
    shape: Shape,
    heads: int,
    head_dim: int,
    ffn_dim: int,
    text_tokens: int,
    device: torch.device,
    reuse_kernel,
    warmup: int,
    repeats: int,
) -> dict[str, dict[str, float]]:
    dtype = torch.bfloat16
    n = shape.tokens
    dim = heads * head_dim
    torch.manual_seed(20260905 + n)
    x = torch.randn(n, dim, dtype=dtype, device=device)
    text = torch.randn(text_tokens, dim, dtype=dtype, device=device)
    w_qkv = torch.randn(3 * dim, dim, dtype=dtype, device=device) / math.sqrt(dim)
    w_out = torch.randn(dim, dim, dtype=dtype, device=device) / math.sqrt(dim)
    w_ff1 = torch.randn(ffn_dim, dim, dtype=dtype, device=device) / math.sqrt(dim)
    w_ff2 = torch.randn(dim, ffn_dim, dtype=dtype, device=device) / math.sqrt(ffn_dim)
    w_text_kv = torch.randn(2 * dim, dim, dtype=dtype, device=device) / math.sqrt(dim)
    q_cross = torch.randn(1, heads, n, head_dim, dtype=dtype, device=device)
    kv_cross = torch.randn(1, heads, text_tokens, head_dim, dtype=dtype, device=device)
    cached = torch.randn_like(x)

    def qkv_call():
        return F.linear(x, w_qkv)

    def out_call():
        return F.linear(x, w_out)

    def ffn_call():
        return F.linear(F.gelu(F.linear(x, w_ff1), approximate="tanh"), w_ff2)

    def text_kv_call():
        return F.linear(text, w_text_kv)

    def cross_attn_call():
        return F.scaled_dot_product_attention(q_cross, kv_cross, kv_cross)

    def cache_hit_call():
        return reuse_kernel.cached_output_residual(x, cached, 1.0)

    result = {
        "qkv_projection": summary_ms(qkv_call, warmup, repeats),
        "one_dim_to_dim_projection": summary_ms(out_call, warmup, repeats),
        "ffn": summary_ms(ffn_call, warmup, repeats),
        "text_kv_projection": summary_ms(text_kv_call, warmup, repeats),
        "cross_attention": summary_ms(cross_attn_call, warmup, repeats),
        "cached_output_residual_add": summary_ms(cache_hit_call, warmup, repeats),
    }
    cached_output = cache_hit_call()
    cache_reference = x + cached
    result["reuse_kernel_correctness"] = {
        "max_abs_error": float((cached_output.float() - cache_reference.float()).abs().max()),
        "allclose_atol_1e-2": bool(torch.allclose(cached_output.float(), cache_reference.float(), atol=1e-2, rtol=1e-2)),
    }
    del x, text, w_qkv, w_out, w_ff1, w_ff2, w_text_kv, q_cross, kv_cross, cached
    torch.cuda.empty_cache()
    return result


def median(component: dict[str, float]) -> float:
    return float(component["median_ms"])


def block_times(shape_result: dict, linear: dict, density: float) -> dict[str, float]:
    entry = shape_result["densities"][f"{density:.6f}"]
    out = median(linear["one_dim_to_dim_projection"])
    common = (
        median(linear["ffn"])
        + 2.0 * out  # cross-attention Q and output projections
        + median(linear["text_kv_projection"])
        + median(linear["cross_attention"])
    )
    qkv = median(linear["qkv_projection"])
    native_attention = median(entry["native_external_route"])
    vsa_router = median(entry["vsa_online_router_only"])
    return {
        "common_ms": common,
        "dense_ms": common + qkv + median(shape_result["dense_sdpa"]) + out,
        "native_ms": common + qkv + native_attention + out,
        # Use component sum for the ranking.  On a shared GPU, separately timed
        # intervals can be interrupted, but this construction cannot produce
        # the impossible result that online routing is cheaper than the exact
        # same sparse executor alone.
        "vsa_ms": common + qkv + native_attention + vsa_router + out,
        "output_cache_hit_ms": common + median(linear["cached_output_residual_add"]),
    }


def sum_schedule(times: dict[str, dict[str, float]], calls: dict[str, int], key: str) -> float:
    return 30.0 * sum(times[stage][key] * count for stage, count in calls.items())


def sum_rdd_route_schedule(times: dict[str, dict[str, float]], calls: dict[str, int]) -> float:
    """RDD has no previous stage for coarse: coarse is dense, middle/fine are routed."""
    return 30.0 * sum(
        times[stage]["dense_ms" if stage == "coarse" else "native_ms"] * count
        for stage, count in calls.items()
    )


def composed_rankings(
    attention: dict[str, dict],
    linears: dict[str, dict],
    route_lifts: dict[str, dict],
    density: float,
    reuse_every: int,
) -> list[dict]:
    bt = {name: block_times(attention[name], linears[name], density) for name in ("coarse", "middle", "fine")}
    temporal_bt = {
        "coarse": block_times(attention["temporal_coarse"], linears["temporal_coarse"], density),
        "middle": block_times(attention["temporal_middle"], linears["temporal_middle"], density),
        "fine": bt["fine"],
    }
    baseline = sum_schedule(bt, {"fine": 50}, "dense_ms")
    route_boundary_ms = 30.0 * sum(
        item["densities"][f"{density:.6f}"]["latency"]["median_ms"]
        for item in route_lifts.values()
    )
    rows = []

    def add(regime: str, method: str, calls: dict[str, int], key: str, status: str, *,
            cache_reuse: bool = False, rdd_route: bool = False):
        if cache_reuse:
            value = 0.0
            for stage, count in calls.items():
                misses = (count + reuse_every - 1) // reuse_every
                hits = count - misses
                miss_key = "dense_ms" if stage == "coarse" else "native_ms"
                value += 30.0 * (misses * bt[stage][miss_key] + hits * bt[stage]["output_cache_hit_ms"])
        elif rdd_route:
            value = sum_rdd_route_schedule(bt, calls)
        else:
            value = sum_schedule(bt, calls, key)
        if cache_reuse or rdd_route:
            # Once per layer at coarse->middle and middle->fine, not every step.
            value += route_boundary_ms
        rows.append({
            "regime": regime,
            "method": method,
            "stage_calls": "/".join(str(calls.get(x, 0)) for x in ("coarse", "middle", "fine")),
            "projected_denoiser_ms": value,
            "speedup_vs_wan_dense50": baseline / value,
            "status": status,
        })

    matched = "matched_50_model_calls"
    add(matched, "Wan2.1 dense full-resolution", {"fine": 50}, "dense_ms", "released baseline")
    add(matched, "FastVideo VSA only", {"fine": 50}, "vsa_ms", "released method")
    # Temporal proxy uses its own token shapes.
    temporal_value = sum_schedule(temporal_bt, {"coarse": 20, "middle": 20, "fine": 10}, "dense_ms")
    rows.append({"regime": matched, "method": "TPDiff-style temporal-only pyramid",
                 "stage_calls": "20/20/10", "projected_denoiser_ms": temporal_value,
                 "speedup_vs_wan_dense50": baseline / temporal_value, "status": "matched Wan proxy"})
    add(matched, "PyramidalWan / boundary-only pyramid", {"coarse": 20, "middle": 20, "fine": 10},
        "dense_ms", "released path / matched proxy")
    add(matched, "RDD dense strict 50-forward", {"coarse": 19, "middle": 20, "fine": 11},
        "dense_ms", "our planned method")
    add(matched, "RDD + previous-stage route", {"coarse": 19, "middle": 20, "fine": 11},
        "native_ms", "our proposed method", rdd_route=True)
    add(matched, f"RDD + route + output reuse k={reuse_every}",
        {"coarse": 19, "middle": 20, "fine": 11}, "native_ms", "quality unverified", cache_reuse=True)

    few = "paper_default_or_hypothetical_few_step"
    add(few, "FastVideo DMD3 + VSA", {"fine": 3}, "vsa_ms", "released method")
    add(few, "PyramidalWan DMD 2/2/1", {"coarse": 2, "middle": 2, "fine": 1},
        "dense_ms", "released method")
    add(few, "RDD distilled 2/2/1 dense", {"coarse": 2, "middle": 2, "fine": 1},
        "dense_ms", "hypothetical until trained")
    add(few, "RDD distilled 2/2/1 + previous-stage route", {"coarse": 2, "middle": 2, "fine": 1},
        "native_ms", "hypothetical until trained", rdd_route=True)
    add(few, "Pyramid Flow 10/10/10 schedule on Wan proxy", {"coarse": 10, "middle": 10, "fine": 10},
        "dense_ms", "official schedule / matched Wan proxy")

    allocation_regime = "rdd_step_allocation_sweep"
    allocations = {
        "strict 18/19/10 (+ endpoints)": {"coarse": 19, "middle": 20, "fine": 11},
        "balanced 20/20/10 (+ endpoints)": {"coarse": 21, "middle": 21, "fine": 11},
        "fine-heavy 10/15/25 (+ endpoints)": {"coarse": 11, "middle": 16, "fine": 26},
        "coarse-heavy 30/15/5 (+ endpoints)": {"coarse": 31, "middle": 16, "fine": 6},
    }
    for allocation_name, allocation_calls in allocations.items():
        add(allocation_regime, f"RDD dense — {allocation_name}", allocation_calls,
            "dense_ms", "speed-only allocation proxy")
        add(allocation_regime, f"RDD + previous-stage route — {allocation_name}", allocation_calls,
            "native_ms", "speed-only allocation proxy", rdd_route=True)
        add(allocation_regime, f"RDD + route + output reuse k={reuse_every} — {allocation_name}",
            allocation_calls, "native_ms", "quality unverified", cache_reuse=True)

    for regime in (matched, few, allocation_regime):
        subset = sorted((r for r in rows if r["regime"] == regime), key=lambda x: x["projected_denoiser_ms"])
        for rank, row in enumerate(subset, 1):
            row["rank"] = rank
    return sorted(rows, key=lambda x: (x["regime"], x["rank"]))


def write_results(output_dir: Path, payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = [row for density_rows in payload["rankings"].values() for row in density_rows]
    with (output_dir / "rankings.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Checkpoint-free Wan2.1 video speed benchmark",
        "",
        "> Random BF16 tensors. Same official FastVideo Triton64 sparse executor for VSA and RDD external-route calls.",
        "> The composed result is a 30-block denoiser proxy; it excludes VAE/T5, communication and quality.",
    ]
    for density, density_rows in payload["rankings"].items():
        lines.extend(["", f"## Retained attention {100 * float(density):.1f}%", ""])
        for regime in ("matched_50_model_calls", "paper_default_or_hypothetical_few_step",
                       "rdd_step_allocation_sweep"):
            lines.extend([f"### {regime}", "",
                          "| rank | method | C/M/F calls | projected ms | speedup | status |",
                          "|---:|---|---:|---:|---:|---|"])
            for row in (r for r in density_rows if r["regime"] == regime):
                lines.append(f"| {row['rank']} | {row['method']} | {row['stage_calls']} | "
                             f"{row['projected_denoiser_ms']:.1f} | {row['speedup_vs_wan_dense50']:.3f}× | "
                             f"{row['status']} |")
            lines.append("")
    (output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-kernel", type=Path, required=True)
    parser.add_argument("--fused-router-kernel", type=Path, required=True)
    parser.add_argument("--rdd-route-kernel", type=Path, required=True)
    parser.add_argument("--rdd-reuse-kernel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--densities", type=float, nargs="+", default=[0.125, 0.20, 0.25])
    parser.add_argument("--reuse-every", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if any(not 0 < x <= 1 for x in args.densities):
        raise ValueError("densities must be in (0,1]")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    official = load_module("fastvideo_vsa_triton64_unmodified", args.official_kernel.resolve())
    fused = load_module("fastvideo_vsa_fused_router_unmodified", args.fused_router_kernel.resolve())
    route_kernel = load_module("rdd_video_route_triton", args.rdd_route_kernel.resolve())
    reuse_kernel = load_module("rdd_video_reuse_triton", args.rdd_reuse_kernel.resolve())
    tile = (4, 4, 4)
    shapes = make_shapes(tile)
    started = time.time()
    attention, linears = {}, {}
    for name in ("coarse", "middle", "temporal_coarse", "temporal_middle", "fine"):
        print(f"benchmarking {name}: valid={shapes[name].tokens}, padded={shapes[name].padded_tokens}", flush=True)
        attention[name] = benchmark_attention_shape(
            shapes[name], args.densities, 12, 128, device, official, fused, args.warmup, args.repeats
        )
        linears[name] = benchmark_wan_nonattention(
            shapes[name], 12, 128, 8960, 512, device, reuse_kernel, args.warmup, args.repeats
        )

    route_lifts = {
        "coarse_to_middle": benchmark_route_lift(
            shapes["coarse"], shapes["middle"], args.densities, 12, tile, device,
            route_kernel, args.warmup, args.repeats
        ),
        "middle_to_fine": benchmark_route_lift(
            shapes["middle"], shapes["fine"], args.densities, 12, tile, device,
            route_kernel, args.warmup, args.repeats
        ),
    }
    rankings = {
        f"{density:.6f}": composed_rankings(attention, linears, route_lifts, density, args.reuse_every)
        for density in args.densities
    }
    payload = {
        "metadata": {
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "hostname": platform.node(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "elapsed_seconds": time.time() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "model": "Wan2.1-T2V-1.3B random-tensor proxy",
            "model_config": {"dim": 1536, "ffn_dim": 8960, "heads": 12, "layers": 30,
                             "head_dim": 128, "text_tokens": 512},
            "warnings": [
                "No checkpoint was loaded and no generated-video quality was measured.",
                "Shared-GPU measurements are preliminary; rerun exclusively for publication latency.",
                "RDD external-route and VSA use the same unmodified FastVideo Triton64 sparse executor.",
                "RDD output-cache hits use a fused Triton gate-plus-residual kernel; full probability caches are excluded as impractical.",
                "VSA timing includes current-step Q/K/V block means, coarse QK/AV, top-k, sparse QK/AV and coarse-output broadcast.",
                "Composed VSA time is sparse-executor time plus separately measured online-router time; raw combined calls are retained for diagnostics.",
                "Composed latency is a 30-layer denoiser proxy, not end-to-end generation time.",
            ],
        },
        "shapes": {name: asdict(shape) | {"tokens": shape.tokens, "padded_tokens": shape.padded_tokens}
                   for name, shape in shapes.items()},
        "attention": attention,
        "nonattention": linears,
        "route_lifts": route_lifts,
        "rankings": rankings,
    }
    write_results(args.output_dir, payload)
    print(json.dumps(payload["metadata"], indent=2), flush=True)
    print(f"wrote {args.output_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
