#!/usr/bin/env python3
"""Summarize sampler-only runtime for a sparse policy and force-dense control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def collect(paths: list[Path], *, total_limit: int, warmup: int) -> dict:
    wall = []
    peak_mib = []
    token_nfe = []
    per_manifest = []
    for path in paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        steps = [int(value) for value in manifest.get("steps", (8, 12, 20))]
        rows = manifest["samples"][:total_limit]
        measured = rows[warmup:]
        if not measured:
            raise ValueError(f"no measured samples after warmup: {path}")
        times = [float(row["sampling_wall_seconds"]) for row in measured]
        peaks = [
            int(row["cuda_peak_allocated_bytes"]) / 2**20
            for row in measured
            if row.get("cuda_peak_allocated_bytes") is not None
        ]
        nfes = [
            sum(
                int(stage["candidate_tokens"]) * step
                for stage, step in zip(row["stages"], steps)
            )
            for row in measured
        ]
        wall.extend(times)
        peak_mib.extend(peaks)
        token_nfe.extend(nfes)
        per_manifest.append(
            {
                "manifest": str(path),
                "measured_samples": len(measured),
                "wall_seconds_mean": float(np.mean(times)),
                "candidate_token_nfe_mean": float(np.mean(nfes)),
                "cuda_peak_allocated_mib_max": float(max(peaks)) if peaks else None,
            }
        )
    return {
        "manifests": per_manifest,
        "measured_samples": len(wall),
        "wall_seconds_mean": float(np.mean(wall)),
        "wall_seconds_std": float(np.std(wall, ddof=0)),
        "candidate_token_nfe_mean": float(np.mean(token_nfe)),
        "cuda_peak_allocated_mib_max": float(max(peak_mib)) if peak_mib else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparse-manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--dense-manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--total-limit", type=int, default=16)
    parser.add_argument("--warmup-per-manifest", type=int, default=1)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    sparse = collect(
        args.sparse_manifests,
        total_limit=args.total_limit,
        warmup=args.warmup_per_manifest,
    )
    dense = collect(
        args.dense_manifests,
        total_limit=args.total_limit,
        warmup=args.warmup_per_manifest,
    )
    report = {
        "protocol": {
            "total_limit_per_manifest": args.total_limit,
            "warmup_samples_per_manifest": args.warmup_per_manifest,
            "metric_scope": "sample_dynamic_sparse_rdd only; excludes model load and NPZ write",
        },
        "sparse": sparse,
        "force_dense": dense,
        "wall_clock_speedup": dense["wall_seconds_mean"]
        / max(sparse["wall_seconds_mean"], 1e-12),
        "token_nfe_reduction": dense["candidate_token_nfe_mean"]
        / max(sparse["candidate_token_nfe_mean"], 1.0),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "| Policy | Samples | Wall/asset (s) | token-NFE | Peak allocated MiB |",
        "|---|---:|---:|---:|---:|",
        f"| selected sparse | {sparse['measured_samples']} | {sparse['wall_seconds_mean']:.4f}±{sparse['wall_seconds_std']:.4f} | {sparse['candidate_token_nfe_mean']:.0f} | {sparse['cuda_peak_allocated_mib_max']:.0f} |",
        f"| force-dense | {dense['measured_samples']} | {dense['wall_seconds_mean']:.4f}±{dense['wall_seconds_std']:.4f} | {dense['candidate_token_nfe_mean']:.0f} | {dense['cuda_peak_allocated_mib_max']:.0f} |",
        "",
        f"Sampler-only wall-clock speedup: **{report['wall_clock_speedup']:.2f}x**  ",
        f"Measured token-NFE reduction: **{report['token_nfe_reduction']:.2f}x**",
    ]
    args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
