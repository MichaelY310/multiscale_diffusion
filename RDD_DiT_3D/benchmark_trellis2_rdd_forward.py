#!/usr/bin/env python3
"""Warm-cache forward benchmark for flat versus multiresolution TRELLIS.2 SLat.

This isolates denoiser cost from DINO, mesh decoding, and first-call kernel
compilation.  It uses real sparse supports from encoded SLat assets and reports
both first-call wall time and steady-state CUDA latency.  CFG multiplies every
stage by the same two model calls, so the reported relative sampling speed is
unchanged by omitting CFG here.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from trellis2 import models
from trellis2.modules import sparse as sp

from trellis2_rdd.hierarchy import build_masked_latent_pyramid


def make_sparse(level, channels: int, dtype: torch.dtype) -> sp.SparseTensor:
    batch = torch.zeros(
        (len(level.model_coords), 1), dtype=torch.int32, device=level.model_coords.device
    )
    coords = torch.cat((batch, level.model_coords.int()), dim=1)
    feats = torch.randn((len(coords), channels), dtype=dtype, device=coords.device)
    return sp.SparseTensor(feats=feats, coords=coords)


def measure(model, sample, timestep, cond, warmup: int, iterations: int) -> dict:
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(sample, timestep, cond)
    torch.cuda.synchronize()
    first_call = time.perf_counter() - start
    del output

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(warmup):
            output = model(sample, timestep, cond)
        torch.cuda.synchronize()
        del output

        timings = []
        for _ in range(iterations):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            output = model(sample, timestep, cond)
            end.record()
            torch.cuda.synchronize()
            timings.append(begin.elapsed_time(end))
            del output
    peak = torch.cuda.max_memory_allocated()
    return {
        "first_call_seconds": first_call,
        "latency_ms_mean": float(np.mean(timings)),
        "latency_ms_median": float(np.median(timings)),
        "latency_ms_std": float(np.std(timings)),
        "peak_increment_mib": (peak - baseline) / 2**20,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("latents", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-assets", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--cond-tokens", type=int, default=1029)
    parser.add_argument("--steps", type=int, nargs=3, default=(8, 12, 20))
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_cfg = config["models"]["denoiser"]
    model = getattr(models, model_cfg["name"])(**model_cfg["args"])
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    # Match training: FP32 master parameters with BF16 autocast in ``measure``.
    model = model.cuda().eval()
    channels = int(model_cfg["args"]["in_channels"])
    cond_channels = int(model_cfg["args"]["cond_channels"])
    cond = torch.randn(
        (1, args.cond_tokens, cond_channels), device="cuda", dtype=torch.bfloat16
    )

    all_files = sorted(args.latents.glob("*.npz"))
    if not all_files:
        raise SystemExit(f"no latent npz files under {args.latents}")
    token_counts = []
    for path in all_files:
        with np.load(path) as data:
            token_counts.append(len(data["coords"]))
    order = np.argsort(token_counts)
    positions = np.linspace(0, len(order) - 1, min(args.num_assets, len(order))).round().astype(int)
    files = [all_files[int(order[position])] for position in positions]

    fine_resolution = int(model_cfg["args"]["resolution"])
    resolutions = (fine_resolution // 4, fine_resolution // 2, fine_resolution)
    # Global clean->noise bands are fine [0,.5], middle [.5,.8], coarse [.8,1].
    stage_t = dict(zip(resolutions, (900.0, 650.0, 250.0)))
    rows = []
    for path in files:
        with np.load(path) as data:
            coords = torch.from_numpy(data["coords"]).long().cuda()
            feats = torch.from_numpy(data["feats"]).float().cuda()
        pyramid = build_masked_latent_pyramid(
            coords,
            feats,
            fine_resolution=fine_resolution,
            resolutions=resolutions,
        )
        row = {"asset": path.stem, "fine_tokens": len(coords), "stages": {}}
        for level in pyramid.levels:
            sample = make_sparse(level, channels, torch.bfloat16)
            timestep = torch.tensor([stage_t[level.resolution]], device="cuda")
            stats = measure(model, sample, timestep, cond, args.warmup, args.iterations)
            stats["tokens"] = len(level.logical_coords)
            row["stages"][str(level.resolution)] = stats
            del sample
            torch.cuda.empty_cache()
        flat = row["stages"][str(fine_resolution)]["latency_ms_median"] * sum(args.steps)
        rdd = sum(
            count * row["stages"][str(resolution)]["latency_ms_median"]
            for count, resolution in zip(args.steps, resolutions)
        )
        row["flat_40step_ms"] = flat
        row["rdd_schedule_ms"] = rdd
        row["steady_state_speedup"] = flat / rdd
        rows.append(row)

    aggregate = {}
    for resolution in resolutions:
        values = [row["stages"][str(resolution)] for row in rows]
        aggregate[str(resolution)] = {
            "tokens_mean": float(np.mean([value["tokens"] for value in values])),
            "latency_ms_median_of_assets": float(
                np.median([value["latency_ms_median"] for value in values])
            ),
            "peak_increment_mib_mean": float(
                np.mean([value["peak_increment_mib"] for value in values])
            ),
        }
    aggregate["sampling"] = {
        "steps_coarse_middle_fine": list(args.steps),
        "steady_state_speedup_mean": float(np.mean([row["steady_state_speedup"] for row in rows])),
        "steady_state_speedup_median": float(
            np.median([row["steady_state_speedup"] for row in rows])
        ),
    }
    report = {
        "note": "random condition features; steady-state denoiser-only benchmark; CFG ratio unchanged",
        "assets": rows,
        "aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
