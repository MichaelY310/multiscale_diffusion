#!/usr/bin/env python3
"""Calibrate pruning thresholds on generated (on-policy) parent states.

Unlike clean teacher-forced calibration, this consumes sampler replay files and
therefore measures the probabilities seen under the actual autoregressive
coarse-to-fine support trajectory.  Calibration assets must be training-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from calibrate_dynamic_sparse_pruning import threshold_for_recall


def coordinate_keys(coords: np.ndarray, resolution: int) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.int64)
    if coords.ndim == 2 and coords.shape[1] == 4:
        coords = coords[:, 1:]
    return (coords[:, 0] * resolution + coords[:, 1]) * resolution + coords[:, 2]


def stage_ground_truth(
    fine_coords: np.ndarray, fine_resolution: int, resolution: int
) -> np.ndarray:
    if fine_coords.ndim == 2 and fine_coords.shape[1] == 4:
        fine_coords = fine_coords[:, 1:]
    stride = fine_resolution // resolution
    return np.unique(fine_coords.astype(np.int64) // stride, axis=0)


def threshold_for_objective(
    probabilities: np.ndarray,
    labels: np.ndarray,
    objective: str,
    beta: float,
) -> tuple[float, float, float, float]:
    """Select a generated-state threshold without looking at evaluation assets."""

    labels = labels.astype(bool)
    positives = int(labels.sum())
    if not positives:
        return 1.0, 0.0, 0.0, 0.0
    order = np.argsort(-probabilities, kind="stable")
    sorted_probability = probabilities[order]
    sorted_labels = labels[order]
    true_positive = np.cumsum(sorted_labels, dtype=np.int64)
    predicted = np.arange(1, len(labels) + 1, dtype=np.int64)
    # A threshold includes every item tied at that value, so score only the
    # final index of each equal-probability group.
    valid = np.r_[sorted_probability[1:] != sorted_probability[:-1], True]
    tp = true_positive[valid].astype(np.float64)
    pred = predicted[valid].astype(np.float64)
    precision = tp / np.maximum(pred, 1.0)
    recall = tp / positives
    if objective == "iou":
        score = tp / np.maximum(positives + pred - tp, 1.0)
    elif objective == "fbeta":
        beta_squared = beta**2
        score = (1.0 + beta_squared) * precision * recall / np.maximum(
            beta_squared * precision + recall, 1e-12
        )
    else:
        raise ValueError(f"unsupported objective: {objective}")
    valid_indices = np.flatnonzero(valid)
    best_position = int(np.argmax(score))
    best_index = int(valid_indices[best_position])
    return (
        float(sorted_probability[best_index]),
        float(recall[best_position]),
        float(precision[best_position]),
        float(score[best_position]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fine-resolution", type=int, default=32)
    parser.add_argument("--resolutions", type=int, nargs="+", default=(8, 16, 32))
    parser.add_argument(
        "--target-recall", type=float, nargs="+", default=(0.999, 0.995, 0.99)
    )
    parser.add_argument(
        "--objective",
        choices=("target-recall", "iou", "fbeta"),
        default="target-recall",
    )
    parser.add_argument("--fbeta", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if len(args.resolutions) != len(args.target_recall):
        raise SystemExit("resolutions and target-recall must have identical lengths")
    if args.fine_resolution % min(args.resolutions):
        raise SystemExit("resolutions must divide fine-resolution")

    probabilities = [[] for _ in args.resolutions]
    labels = [[] for _ in args.resolutions]
    gt_counts = [0 for _ in args.resolutions]
    candidate_gt_counts = [0 for _ in args.resolutions]
    assets = set()
    trajectory_count = 0
    for replay_dir in args.replay_dirs:
        paths = sorted(replay_dir.glob("*.npz"))
        if args.limit:
            paths = paths[: args.limit]
        for replay_path in paths:
            gt_path = args.ground_truth_dir / replay_path.name
            if not gt_path.is_file():
                raise FileNotFoundError(gt_path)
            with np.load(gt_path) as gt_data:
                fine_coords = gt_data["coords"].astype(np.int64)
            with np.load(replay_path) as replay:
                for index, resolution in enumerate(args.resolutions):
                    coords_key = f"candidate_coords_{resolution}"
                    probability_key = f"nonempty_probability_{resolution}"
                    if coords_key not in replay or probability_key not in replay:
                        raise ValueError(
                            f"{replay_path} lacks {coords_key}/{probability_key}"
                        )
                    candidate_coords = replay[coords_key].astype(np.int64)
                    candidate_probability = replay[probability_key].astype(
                        np.float32
                    )
                    gt_coords = stage_ground_truth(
                        fine_coords, args.fine_resolution, resolution
                    )
                    candidate_keys = coordinate_keys(candidate_coords, resolution)
                    gt_keys = coordinate_keys(gt_coords, resolution)
                    candidate_labels = np.isin(candidate_keys, gt_keys)
                    probabilities[index].append(candidate_probability)
                    labels[index].append(candidate_labels)
                    gt_counts[index] += len(gt_keys)
                    candidate_gt_counts[index] += int(candidate_labels.sum())
            assets.add(replay_path.stem)
            trajectory_count += 1

    if not trajectory_count:
        raise SystemExit("no replay .npz files found")
    stages = []
    for index, resolution in enumerate(args.resolutions):
        stage_probabilities = np.concatenate(probabilities[index])
        stage_labels = np.concatenate(labels[index]).astype(bool)
        selection_score = None
        if stage_labels.any() and args.objective == "target-recall":
            threshold, conditional_recall, precision = threshold_for_recall(
                stage_probabilities, stage_labels, args.target_recall[index]
            )
        elif stage_labels.any():
            threshold, conditional_recall, precision, selection_score = (
                threshold_for_objective(
                    stage_probabilities, stage_labels, args.objective, args.fbeta
                )
            )
        else:
            threshold, conditional_recall, precision = 0.0, 0.0, 0.0
        predicted = stage_probabilities >= threshold
        true_positive = int(np.sum(predicted & stage_labels))
        coverage = candidate_gt_counts[index] / max(gt_counts[index], 1)
        stages.append(
            {
                "resolution": resolution,
                "target_recall": args.target_recall[index],
                "probability_threshold": threshold,
                "measured_recall": conditional_recall,
                "measured_precision": precision,
                "selection_score": selection_score,
                "candidate_coverage_of_gt": coverage,
                "global_recall_at_threshold": true_positive
                / max(gt_counts[index], 1),
                "tokens": len(stage_labels),
                "positives_in_candidates": int(stage_labels.sum()),
                "ground_truth_positives": gt_counts[index],
            }
        )
    report = {
        "calibration": "generated-parent/on-policy",
        "objective": args.objective,
        "fbeta": args.fbeta if args.objective == "fbeta" else None,
        "replay_dirs": [str(path) for path in args.replay_dirs],
        "ground_truth_dir": str(args.ground_truth_dir),
        "unique_assets": len(assets),
        "trajectories": trajectory_count,
        "note": (
            "measured_recall is conditional on a GT token entering the candidate set; "
            "candidate_coverage_of_gt exposes unrecoverable upstream pruning errors"
        ),
        "stages": stages,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
