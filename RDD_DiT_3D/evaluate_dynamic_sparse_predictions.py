#!/usr/bin/env python3
"""Evaluate generated dynamic supports and SLat features before mesh decoding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def key(coords: np.ndarray, resolution: int) -> np.ndarray:
    coords = coords.astype(np.int64)
    return (coords[:, 0] * resolution + coords[:, 1]) * resolution + coords[:, 2]


def support_metrics(pred_coords: np.ndarray, gt_coords: np.ndarray, resolution: int):
    pred_keys = key(pred_coords, resolution)
    gt_keys = key(gt_coords, resolution)
    true_positive = len(np.intersect1d(pred_keys, gt_keys))
    union = len(pred_keys) + len(gt_keys) - true_positive
    return {
        "predicted_tokens": len(pred_keys),
        "ground_truth_tokens": len(gt_keys),
        "true_positive": true_positive,
        "support_precision": true_positive / max(len(pred_keys), 1),
        "support_recall": true_positive / max(len(gt_keys), 1),
        "support_iou": true_positive / max(union, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--stage-resolutions", type=int, nargs="*", default=(8, 16, 32))
    args = parser.parse_args()

    rows = []
    for prediction_path in sorted(args.prediction_dir.glob("*.npz")):
        gt_path = args.ground_truth_dir / prediction_path.name
        if not gt_path.exists():
            continue
        with np.load(prediction_path) as pred, np.load(gt_path) as gt:
            pred_coords = pred["coords"].astype(np.int64)
            pred_feats = pred["feats"].astype(np.float32)
            gt_coords = gt["coords"].astype(np.int64)
            gt_feats = gt["feats"].astype(np.float32)
            predicted_stages = {
                stage_resolution: pred[f"coords_{stage_resolution}"].astype(np.int64)
                for stage_resolution in args.stage_resolutions
                if f"coords_{stage_resolution}" in pred
            }
        if pred_coords.shape[1] == 4:
            pred_coords = pred_coords[:, 1:]
        if gt_coords.shape[1] == 4:
            gt_coords = gt_coords[:, 1:]
        pred_keys = key(pred_coords, args.resolution)
        gt_keys = key(gt_coords, args.resolution)
        pred_order = np.argsort(pred_keys)
        gt_order = np.argsort(gt_keys)
        pred_sorted = pred_keys[pred_order]
        gt_sorted = gt_keys[gt_order]
        common, pred_index, gt_index = np.intersect1d(
            pred_sorted, gt_sorted, return_indices=True
        )
        tp = len(common)
        precision = tp / max(len(pred_keys), 1)
        recall = tp / max(len(gt_keys), 1)
        union = len(pred_keys) + len(gt_keys) - tp
        feature_mse = None
        if tp:
            feature_mse = float(
                np.mean(
                    (
                        pred_feats[pred_order[pred_index]]
                        - gt_feats[gt_order[gt_index]]
                    )
                    ** 2
                )
            )
        row = {
                "asset": prediction_path.stem,
                "predicted_tokens": len(pred_keys),
                "ground_truth_tokens": len(gt_keys),
                "true_positive": tp,
                "support_precision": precision,
                "support_recall": recall,
                "support_iou": tp / max(union, 1),
                "feature_mse_on_intersection": feature_mse,
            }
        if predicted_stages:
            row["stages"] = {}
            for stage_resolution, stage_pred_coords in predicted_stages.items():
                stride = args.resolution // stage_resolution
                stage_gt_coords = np.unique(gt_coords // stride, axis=0)
                row["stages"][str(stage_resolution)] = support_metrics(
                    stage_pred_coords, stage_gt_coords, stage_resolution
                )
        rows.append(row)
    if not rows:
        raise SystemExit("no matching prediction/ground-truth files")
    aggregate = {
        key_name: float(np.mean([row[key_name] for row in rows]))
        for key_name in ("support_precision", "support_recall", "support_iou")
    }
    feature_values = [
        row["feature_mse_on_intersection"]
        for row in rows
        if row["feature_mse_on_intersection"] is not None
    ]
    if feature_values:
        aggregate["feature_mse_on_intersection"] = float(np.mean(feature_values))
    stage_aggregate = {}
    for stage_resolution in args.stage_resolutions:
        stage_rows = [
            row.get("stages", {}).get(str(stage_resolution)) for row in rows
        ]
        stage_rows = [row for row in stage_rows if row is not None]
        if stage_rows:
            stage_aggregate[str(stage_resolution)] = {
                metric: float(np.mean([row[metric] for row in stage_rows]))
                for metric in (
                    "predicted_tokens",
                    "ground_truth_tokens",
                    "support_precision",
                    "support_recall",
                    "support_iou",
                )
            }
    report = {"aggregate": aggregate, "stages": stage_aggregate, "samples": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
