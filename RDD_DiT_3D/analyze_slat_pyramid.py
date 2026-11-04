#!/usr/bin/env python3
"""Measure whether a directory of sparse 3D latents is suitable for RDD.

Accepted ``.npz`` layouts include the common key pairs
``coords/features``, ``coords/feats``, ``indices/values`` and
``coords/latent``.  Coordinates may be ``[N,3]`` or batched ``[N,4]``.
PyTorch ``.pt`` dictionaries with the same keys are also supported.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from slat_pyramid import (
    SparseGrid,
    build_pyramid,
    decompose_masked_level,
    transition_table,
)


KEY_PAIRS = (
    ("coords", "features"),
    ("coords", "feats"),
    ("coords", "latent"),
    ("indices", "values"),
)


def _select_arrays(obj: Any, path: Path) -> tuple[np.ndarray, np.ndarray]:
    keys = set(obj.keys())
    for coord_key, feat_key in KEY_PAIRS:
        if coord_key in keys and feat_key in keys:
            coords = obj[coord_key]
            feats = obj[feat_key]
            if torch.is_tensor(coords):
                coords = coords.detach().cpu().numpy()
            if torch.is_tensor(feats):
                feats = feats.detach().cpu().numpy()
            return np.asarray(coords), np.asarray(feats)
    raise KeyError(f"{path}: could not find a coordinate/feature key pair; keys={sorted(keys)}")


def load_grid(path: Path, resolution: int | None) -> SparseGrid:
    if path.suffix == ".npz":
        with np.load(path) as data:
            coords, feats = _select_arrays(data, path)
    elif path.suffix in (".pt", ".pth"):
        data = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(data, dict):
            raise TypeError(f"{path}: expected a dictionary")
        coords, feats = _select_arrays(data, path)
    else:
        raise ValueError(f"unsupported file: {path}")

    coords_t = torch.as_tensor(coords, dtype=torch.long)
    feats_t = torch.as_tensor(feats, dtype=torch.float32)
    if coords_t.ndim != 2 or coords_t.shape[1] not in (3, 4):
        raise ValueError(f"{path}: coordinates must have shape [N,3] or [N,4]")
    if coords_t.shape[1] == 3:
        coords_t = torch.cat((torch.zeros((coords_t.shape[0], 1), dtype=torch.long), coords_t), dim=1)
    inferred = int(coords_t[:, 1:].max()) + 1 if coords_t.numel() else 1
    grid_resolution = resolution or 1 << max(0, (inferred - 1).bit_length())
    return SparseGrid(coords_t, feats_t, grid_resolution)


def summarize_file(
    path: Path,
    resolution: int | None,
    min_resolution: int,
    step_allocation: list[int] | None = None,
) -> dict[str, Any]:
    grid = load_grid(path, resolution)
    transitions = build_pyramid(grid, min_resolution)
    rows = transition_table(transitions)
    masked_by_pair: dict[tuple[int, int], dict[str, float]] = {}
    current = grid
    level_tokens = {current.resolution: current.num_tokens}
    while current.resolution > min_resolution:
        masked = decompose_masked_level(current)
        reconstructed = masked.upsampled_on_active.feats + masked.residual.feats
        projected_detail = decompose_masked_level(masked.residual).upsampled_on_active.feats
        pair = (masked.coarse.resolution, current.resolution)
        masked_by_pair[pair] = {
            "masked_detail_rms": float(masked.residual.feats.float().square().mean().sqrt()),
            "masked_reconstruction_max_abs": float((reconstructed - current.feats).abs().max()),
            "masked_detail_parent_mean_max_abs": float(projected_detail.abs().max()),
        }
        current = masked.coarse
        level_tokens[current.resolution] = current.num_tokens
    for row in rows:
        row["file"] = path.name
        row.update(masked_by_pair[(row["coarse_resolution"], row["fine_resolution"])])

    ordered_resolutions = sorted(level_tokens)
    tokens = [level_tokens[r] for r in ordered_resolutions]
    if step_allocation is not None:
        if len(step_allocation) != len(tokens) or any(step <= 0 for step in step_allocation):
            raise ValueError("step allocation must be positive and match pyramid levels")
        step_weights = step_allocation
    elif len(tokens) == 3:
        step_weights = [8, 12, 20]
    else:
        # Fall back to equal stages for a non-default diagnostic hierarchy.
        step_weights = [1] * len(tokens)
    flat_steps = sum(step_weights)
    rdd_token_nfe = sum(w * n for w, n in zip(step_weights, tokens))
    flat_token_nfe = flat_steps * tokens[-1]
    rdd_attention_proxy = sum(w * n * n for w, n in zip(step_weights, tokens))
    flat_attention_proxy = flat_steps * tokens[-1] * tokens[-1]
    return {
        "file": path.name,
        "resolution": grid.resolution,
        "channels": grid.channels,
        "fine_tokens": grid.num_tokens,
        "fine_density": grid.num_tokens / float(grid.resolution**3),
        "level_tokens": {str(r): level_tokens[r] for r in ordered_resolutions},
        "rdd_step_allocation": step_weights,
        "flat_token_nfe": flat_token_nfe,
        "rdd_token_nfe": rdd_token_nfe,
        "token_nfe_reduction": flat_token_nfe / rdd_token_nfe,
        "flat_attention_proxy": flat_attention_proxy,
        "rdd_attention_proxy": rdd_attention_proxy,
        "attention_proxy_reduction": flat_attention_proxy / rdd_attention_proxy,
        "transitions": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="latent file or directory")
    parser.add_argument("--resolution", type=int, default=None, help="override finest latent resolution")
    parser.add_argument("--min-resolution", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="0 means all files")
    parser.add_argument("--step-allocation", type=int, nargs="+", default=None)
    parser.add_argument("--output", type=Path, default=Path("rdd3d_slat_pyramid_report.json"))
    args = parser.parse_args()

    files = [args.input] if args.input.is_file() else sorted((*args.input.rglob("*.npz"), *args.input.rglob("*.pt")))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"no .npz/.pt latent files found under {args.input}")

    samples: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for path in files:
        try:
            samples.append(
                summarize_file(
                    path, args.resolution, args.min_resolution, args.step_allocation
                )
            )
        except Exception as exc:  # keep a large dataset scan useful despite a bad asset
            failures.append({"file": str(path), "error": f"{type(exc).__name__}: {exc}"})

    transition_rows = [row for sample in samples for row in sample["transitions"]]
    grouped: dict[str, dict[str, float]] = {}
    for key in sorted({f"{r['coarse_resolution']}->{r['fine_resolution']}" for r in transition_rows}):
        rows = [r for r in transition_rows if f"{r['coarse_resolution']}->{r['fine_resolution']}" == key]
        grouped[key] = {
            "num_shapes": len(rows),
            "mean_coarse_tokens": float(np.mean([r["coarse_tokens"] for r in rows])),
            "mean_active_fine_tokens": float(np.mean([r["active_fine_tokens"] for r in rows])),
            "mean_candidate_waste": float(np.mean([r["candidate_waste"] for r in rows])),
            "p95_candidate_waste": float(np.percentile([r["candidate_waste"] for r in rows], 95)),
            "mean_residual_rms": float(np.mean([r["residual_rms"] for r in rows])),
            "mean_masked_detail_rms": float(np.mean([r["masked_detail_rms"] for r in rows])),
            "max_masked_reconstruction_error": float(
                np.max([r["masked_reconstruction_max_abs"] for r in rows])
            ),
            "max_masked_detail_parent_mean": float(
                np.max([r["masked_detail_parent_mean_max_abs"] for r in rows])
            ),
        }

    speed = {
        "token_nfe_reduction_mean": float(np.mean([s["token_nfe_reduction"] for s in samples])),
        "token_nfe_reduction_median": float(np.median([s["token_nfe_reduction"] for s in samples])),
        "attention_proxy_reduction_mean": float(
            np.mean([s["attention_proxy_reduction"] for s in samples])
        ),
        "attention_proxy_reduction_median": float(
            np.median([s["attention_proxy_reduction"] for s in samples])
        ),
    } if samples else {}

    report = {
        "operators": {
            "candidate_analysis": "zero-filled /8 average pool + all-child nearest replication",
            "rdd_training": "fixed-support active-child average + masked nearest replication",
        },
        "num_success": len(samples),
        "num_failures": len(failures),
        "aggregate": grouped,
        "speed_proxy": speed,
        "samples": samples,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    csv_path = args.output.with_suffix(".csv")
    if transition_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(transition_rows[0]))
            writer.writeheader()
            writer.writerows(transition_rows)
    print(
        json.dumps(
            {
                "report": str(args.output),
                "csv": str(csv_path),
                "speed_proxy": speed,
                **report["aggregate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
