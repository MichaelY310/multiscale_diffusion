#!/usr/bin/env python3
"""Measure oracle dynamic-RDD candidate counts on real TRELLIS SLat files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dynamic_sparse_representation import (
    build_dynamic_sparse_pyramid,
    project_complete_sibling_detail,
)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fine-resolution", type=int, default=32)
    parser.add_argument("--resolutions", type=int, nargs="+", default=(8, 16, 32))
    parser.add_argument("--steps", type=int, nargs="+", default=(8, 12, 20))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if len(args.resolutions) != len(args.steps):
        raise SystemExit("resolutions and steps must have the same length")
    files = sorted(args.latent_dir.glob("*.npz"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"no .npz files under {args.latent_dir}")

    rows = []
    for path in files:
        with np.load(path) as data:
            coords = data["coords"].astype(np.int64)
            feats = data["feats"].astype(np.float32)
        if coords.shape[1] == 4:
            coords = coords[:, 1:]
        pyramid = build_dynamic_sparse_pyramid(
            torch.from_numpy(coords),
            torch.from_numpy(feats),
            fine_resolution=args.fine_resolution,
            resolutions=tuple(args.resolutions),
        )
        stages = []
        exact_error = 0.0
        detail_mean_error = 0.0
        for stage in pyramid.stages:
            if stage.detail is not None:
                exact_error = max(
                    exact_error,
                    float((stage.parent_rep + stage.detail - stage.target_rep).abs().max()),
                )
                projected = project_complete_sibling_detail(
                    stage.logical_coords, stage.detail
                )
                detail_mean_error = max(
                    detail_mean_error, float((projected - stage.detail).abs().max())
                )
            candidates = len(stage.logical_coords)
            nonempty = int(stage.nonempty.sum())
            dense = stage.resolution**3
            stages.append(
                {
                    "resolution": stage.resolution,
                    "candidates": candidates,
                    "nonempty": nonempty,
                    "empty_candidates": candidates - nonempty,
                    "candidate_empty_fraction": (candidates - nonempty) / max(candidates, 1),
                    "dense_over_candidate": dense / max(candidates, 1),
                }
            )
        dynamic_token_nfe = sum(
            len(stage.logical_coords) * step
            for stage, step in zip(pyramid.stages, args.steps)
        )
        dense_progressive_token_nfe = sum(
            resolution**3 * step for resolution, step in zip(args.resolutions, args.steps)
        )
        dense_fine_token_nfe = args.fine_resolution**3 * sum(args.steps)
        rows.append(
            {
                "asset": path.stem,
                "stages": stages,
                "dynamic_token_nfe": dynamic_token_nfe,
                "dense_progressive_token_nfe": dense_progressive_token_nfe,
                "dense_fine_token_nfe": dense_fine_token_nfe,
                "dense_progressive_reduction": dense_progressive_token_nfe
                / dynamic_token_nfe,
                "dense_fine_reduction": dense_fine_token_nfe / dynamic_token_nfe,
                "exact_reconstruction_max_error": exact_error,
                "detail_group_mean_max_error": detail_mean_error,
            }
        )

    aggregate = {
        "assets": len(rows),
        "resolutions": list(args.resolutions),
        "steps": list(args.steps),
        "per_stage": [],
        "dynamic_token_nfe": summarize([row["dynamic_token_nfe"] for row in rows]),
        "dense_progressive_reduction": summarize(
            [row["dense_progressive_reduction"] for row in rows]
        ),
        "dense_fine_reduction": summarize([row["dense_fine_reduction"] for row in rows]),
        "exact_reconstruction_max_error": max(
            row["exact_reconstruction_max_error"] for row in rows
        ),
        "detail_group_mean_max_error": max(
            row["detail_group_mean_max_error"] for row in rows
        ),
    }
    for index, resolution in enumerate(args.resolutions):
        aggregate["per_stage"].append(
            {
                "resolution": resolution,
                "candidates": summarize([row["stages"][index]["candidates"] for row in rows]),
                "nonempty": summarize([row["stages"][index]["nonempty"] for row in rows]),
                "candidate_empty_fraction": summarize(
                    [row["stages"][index]["candidate_empty_fraction"] for row in rows]
                ),
                "dense_over_candidate": summarize(
                    [row["stages"][index]["dense_over_candidate"] for row in rows]
                ),
            }
        )
    report = {"aggregate": aggregate, "samples": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
