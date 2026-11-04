#!/usr/bin/env python3
"""Fit the pruning head on actual sampler endpoint states.

Generated-parent replay exposes refinement to model-produced *parent* support,
but its within-stage state still follows the analytic FM interpolation.  This
small follow-up trains only ``nonempty_head`` on endpoint states saved by the
sampler.  The velocity path and every shared denoiser parameter are frozen, so
the experiment isolates support ranking from generation quality.

All replay assets must belong to the training split.  Threshold calibration and
held-out evaluation are deliberately separate downstream steps.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from calibrate_dynamic_sparse_pruning_replay import threshold_for_objective
from dynamic_sparse_flow import stage_band, time_boundaries
from dynamic_sparse_model import DynamicSparseRDDModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--latent-dir", type=Path, required=True)
    parser.add_argument("--condition-dir", type=Path, required=True)
    parser.add_argument("--replay-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--weights", choices=("model", "ema"), default="model")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--positive-weight", type=float, default=3.0)
    parser.add_argument(
        "--stage-probabilities", type=float, nargs="+", default=(0.40, 0.30, 0.30)
    )
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--report-examples-per-stage", type=int, default=96)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("none", "bf16"), default="bf16")
    return parser.parse_args()


def stage_coords(fine_coords: np.ndarray, fine_resolution: int, resolution: int) -> np.ndarray:
    if fine_resolution % resolution:
        raise ValueError("resolution must divide fine resolution")
    return np.unique(fine_coords // (fine_resolution // resolution), axis=0)


def coordinate_keys(coords: np.ndarray, resolution: int) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.int64)
    return (coords[:, 0] * resolution + coords[:, 1]) * resolution + coords[:, 2]


def candidate_labels(
    candidate_coords: np.ndarray,
    fine_coords: np.ndarray,
    *,
    fine_resolution: int,
    resolution: int,
) -> np.ndarray:
    truth = stage_coords(fine_coords, fine_resolution, resolution)
    return np.isin(
        coordinate_keys(candidate_coords, resolution),
        coordinate_keys(truth, resolution),
    )


def load_examples(
    *,
    replay_dirs: list[Path],
    latent_dir: Path,
    condition_dir: Path,
    resolutions: tuple[int, ...],
    fine_resolution: int,
    state_channels: int,
) -> list[dict]:
    examples = []
    for replay_dir in replay_dirs:
        for replay_path in sorted(replay_dir.glob("*.npz")):
            latent_path = latent_dir / replay_path.name
            condition_path = condition_dir / f"{replay_path.stem}.npy"
            if not latent_path.is_file():
                raise FileNotFoundError(latent_path)
            if not condition_path.is_file():
                raise FileNotFoundError(condition_path)
            with np.load(latent_path) as latent:
                fine_coords = latent["coords"].astype(np.int64)
                if fine_coords.ndim == 2 and fine_coords.shape[1] == 4:
                    fine_coords = fine_coords[:, 1:]
            condition = np.load(condition_path).astype(np.float32).reshape(-1)
            with np.load(replay_path) as replay:
                for stage_index, resolution in enumerate(resolutions):
                    coords_key = f"candidate_coords_{resolution}"
                    state_key = f"candidate_representation_{resolution}"
                    if coords_key not in replay or state_key not in replay:
                        raise ValueError(
                            f"{replay_path} lacks {coords_key}/{state_key}; regenerate "
                            "the trajectory with the endpoint-state sampler"
                        )
                    coords = replay[coords_key].astype(np.int64)
                    state = replay[state_key].astype(np.float32)
                    if coords.ndim != 2 or coords.shape[1] != 3:
                        raise ValueError(f"invalid coordinates in {replay_path}:{resolution}")
                    if state.shape != (len(coords), state_channels):
                        raise ValueError(f"invalid endpoint state in {replay_path}:{resolution}")
                    if not np.isfinite(state).all():
                        raise ValueError(f"non-finite endpoint state in {replay_path}:{resolution}")
                    labels = candidate_labels(
                        coords,
                        fine_coords,
                        fine_resolution=fine_resolution,
                        resolution=resolution,
                    )
                    examples.append(
                        {
                            "asset": replay_path.stem,
                            "trajectory": replay_dir.name,
                            "stage_index": stage_index,
                            "resolution": resolution,
                            "coords": coords,
                            "state": state,
                            "labels": labels,
                            "condition": condition,
                        }
                    )
    if not examples:
        raise SystemExit("no endpoint replay examples found")
    return examples


def evaluate_head(
    model: DynamicSparseRDDModel,
    examples: list[dict],
    *,
    time_fractions: tuple[float, ...],
    device: torch.device,
    amp: str,
    limit_per_stage: int,
) -> dict:
    selected = defaultdict(list)
    for example in examples:
        if len(selected[example["resolution"]]) < limit_per_stage:
            selected[example["resolution"]].append(example)
    result = {}
    boundaries = time_boundaries(time_fractions)
    with torch.no_grad():
        for resolution, stage_examples in selected.items():
            probabilities, labels = [], []
            stage_index = int(stage_examples[0]["stage_index"])
            low, _ = stage_band(stage_index, len(time_fractions), boundaries)
            for example in stage_examples:
                state = torch.from_numpy(example["state"]).to(device)
                coords = torch.from_numpy(example["coords"]).to(device)
                condition = torch.from_numpy(example["condition"]).to(device)
                enabled = amp == "bf16" and device.type == "cuda"
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=enabled
                ):
                    _, logits = model(
                        state,
                        coords,
                        resolution=resolution,
                        global_time=low,
                        condition=condition,
                    )
                probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
                labels.append(example["labels"])
            stage_probability = np.concatenate(probabilities)
            stage_labels = np.concatenate(labels).astype(bool)
            threshold, recall, precision, iou = threshold_for_objective(
                stage_probability, stage_labels, "iou", beta=2.0
            )
            result[str(resolution)] = {
                "examples": len(stage_examples),
                "tokens": int(len(stage_labels)),
                "positives": int(stage_labels.sum()),
                "best_iou_threshold": float(threshold),
                "best_iou_recall": float(recall),
                "best_iou_precision": float(precision),
                "best_iou": float(iou),
            }
    return result


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = DynamicSparseRDDModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint[args.weights])
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.nonempty_head.parameters():
        parameter.requires_grad_(True)

    rdd = checkpoint["rdd_config"]
    resolutions = tuple(int(value) for value in rdd["resolutions"])
    time_fractions = tuple(float(value) for value in rdd["time_fractions"])
    fine_resolution = int(resolutions[-1])
    if len(args.stage_probabilities) != len(resolutions):
        raise SystemExit("stage-probabilities must match checkpoint resolutions")
    if any(value < 0 for value in args.stage_probabilities) or not any(
        value > 0 for value in args.stage_probabilities
    ):
        raise SystemExit("stage-probabilities must be non-negative and not all zero")

    examples = load_examples(
        replay_dirs=list(args.replay_dirs),
        latent_dir=args.latent_dir,
        condition_dir=args.condition_dir,
        resolutions=resolutions,
        fine_resolution=fine_resolution,
        state_channels=int(checkpoint["model_config"]["state_channels"]),
    )
    by_stage = defaultdict(list)
    for example in examples:
        by_stage[int(example["stage_index"])].append(example)

    frozen_before = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if not name.startswith("nonempty_head.")
    }
    before = evaluate_head(
        model,
        examples,
        time_fractions=time_fractions,
        device=device,
        amp=args.amp,
        limit_per_stage=args.report_examples_per_stage,
    )

    optimizer = torch.optim.AdamW(
        model.nonempty_head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    boundaries = time_boundaries(time_fractions)
    log_rows = []
    for step in range(1, args.steps + 1):
        stage_index = random.choices(
            range(len(resolutions)), weights=args.stage_probabilities, k=1
        )[0]
        example = random.choice(by_stage[stage_index])
        state = torch.from_numpy(example["state"]).to(device)
        coords = torch.from_numpy(example["coords"]).to(device)
        target = torch.from_numpy(example["labels"].astype(np.float32)).to(device)
        condition = torch.from_numpy(example["condition"]).to(device)
        low, _ = stage_band(stage_index, len(resolutions), boundaries)
        enabled = args.amp == "bf16" and device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled):
            _, logits = model(
                state,
                coords,
                resolution=int(example["resolution"]),
                global_time=low,
                condition=condition,
            )
            loss = F.binary_cross_entropy_with_logits(
                logits.float(),
                target,
                pos_weight=torch.tensor(args.positive_weight, device=device),
            )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite occupancy loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.nonempty_head.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            with torch.no_grad():
                predicted = logits >= 0
                positive = target.bool()
                true_positive = int((predicted & positive).sum())
                precision = true_positive / max(int(predicted.sum()), 1)
                recall = true_positive / max(int(positive.sum()), 1)
            row = {
                "step": step,
                "asset": example["asset"],
                "trajectory": example["trajectory"],
                "resolution": int(example["resolution"]),
                "tokens": len(coords),
                "loss": float(loss),
                "precision_0.5": precision,
                "recall_0.5": recall,
                "grad_norm": float(grad_norm),
            }
            log_rows.append(row)
            print(json.dumps(row), flush=True)

    after = evaluate_head(
        model,
        examples,
        time_fractions=time_fractions,
        device=device,
        amp=args.amp,
        limit_per_stage=args.report_examples_per_stage,
    )
    frozen_max_error = 0.0
    for name, tensor in model.state_dict().items():
        if name.startswith("nonempty_head."):
            continue
        frozen_max_error = max(
            frozen_max_error,
            float((tensor.detach().cpu() - frozen_before[name]).abs().max()),
        )
    if frozen_max_error != 0.0:
        raise AssertionError(f"a frozen denoiser parameter changed: {frozen_max_error}")

    trained_state = {
        name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
    }
    output_checkpoint = copy.copy(checkpoint)
    output_checkpoint["model"] = trained_state
    output_checkpoint["ema"] = copy.deepcopy(trained_state)
    output_checkpoint["optimizer"] = None
    output_checkpoint["occupancy_endpoint_replay"] = {
        "source_checkpoint": str(args.checkpoint),
        "source_weights": args.weights,
        "replay_dirs": [str(path) for path in args.replay_dirs],
        "steps": args.steps,
        "lr": args.lr,
        "positive_weight": args.positive_weight,
        "stage_probabilities": list(args.stage_probabilities),
        "seed": args.seed,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.nonempty_head.parameters()
        ),
        "frozen_parameter_max_error": frozen_max_error,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, args.output)

    report = {
        "checkpoint": str(args.checkpoint),
        "output": str(args.output),
        "assets": len({example["asset"] for example in examples}),
        "trajectories": len(args.replay_dirs),
        "examples": len(examples),
        "before": before,
        "after": after,
        "frozen_parameter_max_error": frozen_max_error,
        "log": log_rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
