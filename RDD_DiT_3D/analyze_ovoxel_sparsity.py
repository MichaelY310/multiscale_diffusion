#!/usr/bin/env python3
"""Analyze real O-Voxel support growth without materializing dense volumes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def unique_rows(values: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(values)
    packed = values.view(np.dtype((np.void, values.dtype.itemsize * values.shape[1])))
    return np.unique(packed).view(values.dtype).reshape(-1, values.shape[1])


def support_stats(coords: np.ndarray, finest_resolution: int, min_resolution: int) -> list[dict[str, float | int]]:
    current = coords.astype(np.int64, copy=False)
    resolution = finest_resolution
    levels: list[dict[str, float | int]] = []
    while True:
        tokens = len(current)
        levels.append(
            {
                "resolution": resolution,
                "active_tokens": tokens,
                "dense_tokens": resolution**3,
                "density": tokens / float(resolution**3),
            }
        )
        if resolution == min_resolution:
            break
        if resolution % 2 or resolution < min_resolution:
            raise ValueError("resolutions must form a dyadic hierarchy")
        parent = unique_rows(current // 2)
        levels[-1]["active_parents"] = len(parent)
        levels[-1]["candidate_children_from_parent"] = 8 * len(parent)
        levels[-1]["candidate_waste"] = 1.0 - tokens / float(8 * len(parent))
        current = parent
        resolution //= 2
    return levels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="a .vxz file or directory")
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--min-resolution", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("rdd3d_ovoxel_sparsity_report.json"))
    args = parser.parse_args()

    try:
        import o_voxel
    except ImportError as exc:
        raise SystemExit("o_voxel is required; run scripts/rdd3d/setup_trellis2_pilot.sh") from exc

    files = [args.input] if args.input.is_file() else sorted(args.input.rglob("*.vxz"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"no .vxz files found under {args.input}")

    samples = []
    failures = []
    for path in files:
        try:
            coords, _ = o_voxel.io.read_vxz(str(path), num_threads=4)
            coords = np.asarray(coords.cpu() if hasattr(coords, "cpu") else coords)
            samples.append({"file": path.name, "levels": support_stats(coords, args.resolution, args.min_resolution)})
        except Exception as exc:
            failures.append({"file": str(path), "error": f"{type(exc).__name__}: {exc}"})

    aggregate = {}
    for resolution in sorted({row["resolution"] for sample in samples for row in sample["levels"]}, reverse=True):
        rows = [row for sample in samples for row in sample["levels"] if row["resolution"] == resolution]
        aggregate[str(resolution)] = {
            "num_shapes": len(rows),
            "active_tokens_mean": float(np.mean([row["active_tokens"] for row in rows])),
            "active_tokens_median": float(np.median([row["active_tokens"] for row in rows])),
            "active_tokens_p95": float(np.percentile([row["active_tokens"] for row in rows], 95)),
            "density_mean": float(np.mean([row["density"] for row in rows])),
        }
        if "candidate_waste" in rows[0]:
            aggregate[str(resolution)]["candidate_waste_mean"] = float(
                np.mean([row["candidate_waste"] for row in rows])
            )
            aggregate[str(resolution)]["candidate_waste_p95"] = float(
                np.percentile([row["candidate_waste"] for row in rows], 95)
            )

    report = {
        "num_success": len(samples),
        "num_failures": len(failures),
        "aggregate": aggregate,
        "samples": samples,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
