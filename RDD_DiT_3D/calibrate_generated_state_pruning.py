#!/usr/bin/env python3
"""Calibrate sparse RDD pruning on free-running generated stage states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def coordinate_keys(coords: np.ndarray, resolution: int) -> np.ndarray:
    coords = coords.astype(np.int64)
    return (coords[:, 0] * resolution + coords[:, 1]) * resolution + coords[:, 2]


def candidate_labels(
    candidates: np.ndarray,
    fine_coords: np.ndarray,
    *,
    resolution: int,
    fine_resolution: int,
) -> np.ndarray:
    if fine_coords.shape[1] == 4:
        fine_coords = fine_coords[:, 1:]
    stride = fine_resolution // resolution
    gt = np.unique(fine_coords.astype(np.int64) // stride, axis=0)
    return np.isin(
        coordinate_keys(candidates, resolution),
        coordinate_keys(gt, resolution),
        assume_unique=False,
    )


def metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    keep = scores >= threshold
    tp = int(np.count_nonzero(keep & labels))
    fp = int(np.count_nonzero(keep & ~labels))
    fn = int(np.count_nonzero(~keep & labels))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "iou": tp / max(tp + fp + fn, 1),
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "keep_fraction": float(np.mean(keep)),
        "kept": int(np.count_nonzero(keep)),
    }


def calibrate_score(scores: np.ndarray, labels: np.ndarray) -> dict:
    finite = np.isfinite(scores)
    scores = scores[finite].astype(np.float64)
    labels = labels[finite]
    quantiles = np.linspace(0.0, 1.0, min(len(scores), 2001))
    thresholds = np.unique(np.quantile(scores, quantiles))
    candidates = [metrics(scores, labels, threshold) for threshold in thresholds]
    best_iou = max(candidates, key=lambda row: (row["iou"], row["recall"]))
    best_f1 = max(candidates, key=lambda row: (row["f1"], row["recall"]))
    recall_targets = {}
    for target in (0.99, 0.97, 0.95, 0.90, 0.85):
        eligible = [row for row in candidates if row["recall"] >= target]
        recall_targets[str(target)] = (
            max(eligible, key=lambda row: (row["precision"], row["threshold"]))
            if eligible
            else None
        )
    positive_scores = scores[labels]
    negative_scores = scores[~labels]
    return {
        "positives": int(np.count_nonzero(labels)),
        "negatives": int(np.count_nonzero(~labels)),
        "positive_quantiles": {
            str(q): float(np.quantile(positive_scores, q))
            for q in (0.01, 0.05, 0.10, 0.50)
        },
        "negative_quantiles": {
            str(q): float(np.quantile(negative_scores, q))
            for q in (0.50, 0.90, 0.95, 0.99)
        },
        "best_iou": best_iou,
        "best_f1": best_f1,
        "recall_targets": recall_targets,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolutions", type=int, nargs="+", default=(8, 16, 32))
    parser.add_argument("--fine-resolution", type=int, default=32)
    args = parser.parse_args()

    per_stage: dict[int, dict[str, list[np.ndarray]]] = {
        resolution: {"labels": [], "probability": [], "mass": []}
        for resolution in args.resolutions
    }
    used_assets = []
    for prediction_path in sorted(args.prediction_dir.glob("*.npz")):
        gt_path = args.ground_truth_dir / prediction_path.name
        if not gt_path.exists():
            continue
        with np.load(prediction_path) as pred, np.load(gt_path) as gt:
            fine_coords = gt["coords"].astype(np.int64)
            for resolution in args.resolutions:
                coord_key = f"candidate_coords_{resolution}"
                if coord_key not in pred:
                    continue
                candidates = pred[coord_key].astype(np.int64)
                per_stage[resolution]["labels"].append(
                    candidate_labels(
                        candidates,
                        fine_coords,
                        resolution=resolution,
                        fine_resolution=args.fine_resolution,
                    )
                )
                per_stage[resolution]["probability"].append(
                    pred[f"nonempty_probability_{resolution}"].astype(np.float64)
                )
                per_stage[resolution]["mass"].append(
                    pred[f"candidate_mass_{resolution}"].astype(np.float64)
                )
        used_assets.append(prediction_path.stem)
    if not used_assets:
        raise SystemExit("no prediction/ground-truth pairs with diagnostics")

    report = {"assets": used_assets, "stages": {}}
    for resolution, arrays in per_stage.items():
        if not arrays["labels"]:
            continue
        labels = np.concatenate(arrays["labels"])
        report["stages"][str(resolution)] = {
            "candidate_count": int(len(labels)),
            "occupancy_rate": float(np.mean(labels)),
            "probability": calibrate_score(
                np.concatenate(arrays["probability"]), labels
            ),
            "mass": calibrate_score(np.concatenate(arrays["mass"]), labels),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
