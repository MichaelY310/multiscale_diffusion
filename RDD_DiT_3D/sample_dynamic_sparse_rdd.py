#!/usr/bin/env python3
"""Generate a sparse TRELLIS SLat with stage-wise RDD pruning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dynamic_sparse_flow import sample_dynamic_sparse_rdd
from dynamic_sparse_model import DynamicSparseRDDModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--condition", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=(8, 12, 20))
    parser.add_argument(
        "--probability-thresholds", type=float, nargs="+", default=(0.02, 0.05, 0.20)
    )
    parser.add_argument(
        "--mass-thresholds", type=float, nargs="+", default=(0.05, 0.10, 0.25)
    )
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = DynamicSparseRDDModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint[args.weights])
    model.eval().requires_grad_(False).to(args.device)
    rdd = checkpoint["rdd_config"]
    resolutions = tuple(int(value) for value in rdd["resolutions"])
    if not (
        len(args.steps)
        == len(args.probability_thresholds)
        == len(args.mass_thresholds)
        == len(resolutions)
    ):
        raise SystemExit("steps and threshold lists must match checkpoint resolutions")
    condition = None
    if checkpoint["model_config"]["condition_channels"]:
        if args.condition is None:
            raise SystemExit("this checkpoint requires --condition")
        condition = torch.from_numpy(
            np.load(args.condition).astype(np.float32).reshape(-1)
        ).to(args.device)
    sample = sample_dynamic_sparse_rdd(
        model,
        resolutions=resolutions,
        steps=tuple(args.steps),
        time_fractions=tuple(float(value) for value in rdd["time_fractions"]),
        probability_thresholds=tuple(args.probability_thresholds),
        mass_thresholds=tuple(args.mass_thresholds),
        condition=condition,
        state_channels=checkpoint["model_config"]["state_channels"],
        sigma_min=float(rdd["sigma_min"]),
        seed=args.seed,
        device=args.device,
    )
    normalized = sample.trellis_features.float().cpu().numpy()
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)[None]
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32)[None]
    feats = normalized * std + mean
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        coords=sample.coords.int().cpu().numpy(),
        feats=feats.astype(np.float32),
        normalized_feats=normalized.astype(np.float32),
        mass=sample.representation[:, 0].float().cpu().numpy(),
    )
    manifest = {
        "checkpoint": str(args.checkpoint),
        "condition": None if args.condition is None else str(args.condition),
        "seed": args.seed,
        "steps": list(args.steps),
        "probability_thresholds": list(args.probability_thresholds),
        "mass_thresholds": list(args.mass_thresholds),
        "stages": [stage.__dict__ for stage in sample.stages],
        "output": str(args.output),
    }
    manifest_path = args.output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
