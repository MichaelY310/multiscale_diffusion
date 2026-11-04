#!/usr/bin/env python3
"""Measure SLat feature-distribution shift relative to a resumed checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--latent-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    reference_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float64)
    reference_std = np.asarray(checkpoint["feature_std"], dtype=np.float64)
    files = sorted(args.latent_dir.glob("*.npz"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"no .npz files under {args.latent_dir}")

    feature_sum = np.zeros_like(reference_mean)
    feature_square_sum = np.zeros_like(reference_mean)
    tokens = 0
    for path in files:
        with np.load(path) as data:
            features = data["feats"].astype(np.float64)
        if features.ndim != 2 or features.shape[1] != len(reference_mean):
            raise ValueError(f"feature shape mismatch: {path} {features.shape}")
        if not np.isfinite(features).all():
            raise ValueError(f"non-finite features: {path}")
        feature_sum += features.sum(axis=0)
        feature_square_sum += np.square(features).sum(axis=0)
        tokens += len(features)

    observed_mean = feature_sum / tokens
    observed_variance = np.maximum(
        feature_square_sum / tokens - np.square(observed_mean), 0.0
    )
    observed_std = np.sqrt(observed_variance)
    mean_z_shift = (observed_mean - reference_mean) / reference_std
    std_ratio = observed_std / reference_std
    report = {
        "checkpoint": str(args.checkpoint),
        "latent_dir": str(args.latent_dir),
        "assets": len(files),
        "tokens": tokens,
        "channels": len(reference_mean),
        "mean_z_shift_rms": float(np.sqrt(np.mean(np.square(mean_z_shift)))),
        "mean_z_shift_abs_max": float(np.max(np.abs(mean_z_shift))),
        "std_ratio_mean": float(np.mean(std_ratio)),
        "std_ratio_median": float(np.median(std_ratio)),
        "std_ratio_min": float(np.min(std_ratio)),
        "std_ratio_max": float(np.max(std_ratio)),
        "per_channel": {
            "reference_mean": reference_mean.tolist(),
            "reference_std": reference_std.tolist(),
            "observed_mean": observed_mean.tolist(),
            "observed_std": observed_std.tolist(),
            "mean_z_shift": mean_z_shift.tolist(),
            "std_ratio": std_ratio.tolist(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "per_channel"}, indent=2))


if __name__ == "__main__":
    main()
