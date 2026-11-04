#!/usr/bin/env python3
"""Calibrate recall-first non-empty thresholds on held-out or pilot SLat files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dynamic_sparse_flow import stage_band, time_boundaries
from dynamic_sparse_model import DynamicSparseRDDModel
from dynamic_sparse_representation import build_dynamic_sparse_pyramid


def threshold_for_recall(probabilities: np.ndarray, labels: np.ndarray, target: float):
    positive = np.sort(probabilities[labels])
    if not len(positive):
        return 0.0
    allowed_false_negatives = int(np.floor((1.0 - target) * len(positive)))
    index = min(max(allowed_false_negatives, 0), len(positive) - 1)
    threshold = float(positive[index])
    predicted = probabilities >= threshold
    tp = int(np.sum(predicted & labels))
    recall = tp / max(int(labels.sum()), 1)
    precision = tp / max(int(predicted.sum()), 1)
    return threshold, recall, precision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--latent-dir", type=Path, required=True)
    parser.add_argument("--condition-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-recall", type=float, nargs="+", default=(0.999, 0.995, 0.99))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = DynamicSparseRDDModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint[args.weights])
    model.eval().requires_grad_(False).to(args.device)
    rdd = checkpoint["rdd_config"]
    resolutions = tuple(rdd["resolutions"])
    if len(args.target_recall) != len(resolutions):
        raise SystemExit("target-recall must match checkpoint stage count")
    boundaries = time_boundaries(tuple(rdd["time_fractions"]))
    feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    files = sorted(args.latent_dir.glob("*.npz"))
    if args.limit:
        files = files[: args.limit]
    all_probabilities = [[] for _ in resolutions]
    all_labels = [[] for _ in resolutions]
    for path in files:
        with np.load(path) as data:
            coords = data["coords"].astype(np.int64)
            feats = data["feats"].astype(np.float32)
        if coords.shape[1] == 4:
            coords = coords[:, 1:]
        normalized = (feats - feature_mean) / feature_std
        pyramid = build_dynamic_sparse_pyramid(
            torch.from_numpy(coords),
            torch.from_numpy(normalized),
            fine_resolution=int(rdd["fine_resolution"]),
            resolutions=resolutions,
        )
        condition = None
        if checkpoint["model_config"]["condition_channels"]:
            if args.condition_dir is None:
                raise SystemExit("checkpoint requires --condition-dir")
            condition = torch.from_numpy(
                np.load(args.condition_dir / f"{path.stem}.npy").astype(np.float32)
            ).to(args.device)
        for stage_index, stage in enumerate(pyramid.stages):
            low, _ = stage_band(stage_index, len(resolutions), boundaries)
            state = stage.target_rep.to(args.device)
            coords_t = stage.logical_coords.to(args.device)
            with torch.inference_mode(), torch.autocast(
                device_type=torch.device(args.device).type,
                dtype=torch.bfloat16,
                enabled=torch.device(args.device).type == "cuda",
            ):
                _, logits = model(
                    state,
                    coords_t,
                    resolution=stage.resolution,
                    global_time=low,
                    condition=condition,
                )
            all_probabilities[stage_index].append(
                torch.sigmoid(logits).float().cpu().numpy()
            )
            all_labels[stage_index].append(stage.nonempty.numpy())

    stages = []
    for index, resolution in enumerate(resolutions):
        probabilities = np.concatenate(all_probabilities[index])
        labels = np.concatenate(all_labels[index]).astype(bool)
        threshold, recall, precision = threshold_for_recall(
            probabilities, labels, args.target_recall[index]
        )
        stages.append(
            {
                "resolution": resolution,
                "target_recall": args.target_recall[index],
                "probability_threshold": threshold,
                "measured_recall": recall,
                "measured_precision": precision,
                "tokens": len(labels),
                "positives": int(labels.sum()),
            }
        )
    report = {
        "checkpoint": str(args.checkpoint),
        "weights": args.weights,
        "note": "clean/teacher-forced calibration; generated-state recall must also be measured",
        "stages": stages,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
