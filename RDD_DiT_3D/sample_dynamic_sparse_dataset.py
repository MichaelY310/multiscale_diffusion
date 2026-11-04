#!/usr/bin/env python3
"""Sample a small conditioned asset set with one loaded dynamic RDD model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from dynamic_sparse_flow import sample_dynamic_sparse_rdd
from dynamic_sparse_model import DynamicSparseRDDModel


class ResolutionDispatchModel:
    """Dispatch a shared sampler call to an optional model per resolution."""

    def __init__(self, models: dict[int, DynamicSparseRDDModel]):
        self.models = dict(models)

    def __call__(self, *args, resolution: int, **kwargs):
        try:
            model = self.models[int(resolution)]
        except KeyError as exc:
            raise ValueError(f"no model configured for resolution {resolution}") from exc
        return model(*args, resolution=resolution, **kwargs)


def load_sampling_model(
    checkpoint: dict,
    *,
    checkpoint_path: Path,
    resolutions: tuple[int, ...],
    weights: str,
    stage_checkpoints: tuple[Path, ...],
    stage_weights: tuple[str, ...],
    device: str,
):
    if not stage_checkpoints:
        model = DynamicSparseRDDModel(**checkpoint["model_config"])
        model.load_state_dict(checkpoint[weights])
        return model.eval().requires_grad_(False).to(device), [str(checkpoint_path)] * len(
            resolutions
        ), [weights] * len(resolutions)
    if len(stage_checkpoints) != len(resolutions):
        raise SystemExit("stage-checkpoints must match checkpoint resolutions")
    if stage_weights and len(stage_weights) != len(resolutions):
        raise SystemExit("stage-weights must match checkpoint resolutions")
    selected_weights = stage_weights or (weights,) * len(resolutions)
    cache: dict[tuple[str, str], DynamicSparseRDDModel] = {}
    models = {}
    base_mean = np.asarray(checkpoint["feature_mean"])
    base_std = np.asarray(checkpoint["feature_std"])
    for resolution, path, selected in zip(
        resolutions, stage_checkpoints, selected_weights
    ):
        key = (str(path.resolve()), selected)
        if key not in cache:
            stage_checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if stage_checkpoint["model_config"] != checkpoint["model_config"]:
                raise ValueError(f"model config mismatch: {path}")
            if stage_checkpoint["rdd_config"] != checkpoint["rdd_config"]:
                raise ValueError(f"RDD config mismatch: {path}")
            if not np.array_equal(np.asarray(stage_checkpoint["feature_mean"]), base_mean):
                raise ValueError(f"feature mean mismatch: {path}")
            if not np.array_equal(np.asarray(stage_checkpoint["feature_std"]), base_std):
                raise ValueError(f"feature std mismatch: {path}")
            stage_model = DynamicSparseRDDModel(**stage_checkpoint["model_config"])
            stage_model.load_state_dict(stage_checkpoint[selected])
            cache[key] = stage_model.eval().requires_grad_(False).to(device)
        models[int(resolution)] = cache[key]
    return ResolutionDispatchModel(models), [str(path) for path in stage_checkpoints], list(
        selected_weights
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--latent-dir", type=Path, required=True)
    parser.add_argument("--condition-dir", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=(8, 12, 20))
    parser.add_argument("--mass-thresholds", type=float, nargs="+", default=(0.05, 0.10, 0.25))
    parser.add_argument("--probability-thresholds", type=float, nargs="+", default=None)
    parser.add_argument("--solver", choices=("euler", "heun"), default="euler")
    parser.add_argument(
        "--pruning-mode",
        choices=("or", "and", "probability", "mass"),
        default="or",
    )
    parser.add_argument("--force-keep-resolutions", type=int, nargs="*", default=())
    parser.add_argument("--halo-radii", type=int, nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument(
        "--stage-checkpoints",
        type=Path,
        nargs="+",
        default=(),
        help="optional checkpoint per coarse-to-fine resolution",
    )
    parser.add_argument(
        "--stage-weights",
        choices=("ema", "model"),
        nargs="+",
        default=(),
        help="optional weights key per stage checkpoint",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    threshold_report = json.loads(args.thresholds.read_text(encoding="utf-8"))
    probability_thresholds = tuple(
        float(stage["probability_threshold"]) for stage in threshold_report["stages"]
    )
    if args.probability_thresholds is not None:
        probability_thresholds = tuple(args.probability_thresholds)
    rdd = checkpoint["rdd_config"]
    resolutions = tuple(int(value) for value in rdd["resolutions"])
    model, stage_checkpoint_manifest, stage_weight_manifest = load_sampling_model(
        checkpoint,
        checkpoint_path=args.checkpoint,
        resolutions=resolutions,
        weights=args.weights,
        stage_checkpoints=tuple(args.stage_checkpoints),
        stage_weights=tuple(args.stage_weights),
        device=args.device,
    )
    if not (
        len(args.steps)
        == len(args.mass_thresholds)
        == len(probability_thresholds)
        == len(resolutions)
    ):
        raise SystemExit("all stage lists must match checkpoint resolutions")
    halo_radii = tuple(args.halo_radii or (0,) * len(resolutions))
    if len(halo_radii) != len(resolutions):
        raise SystemExit("halo-radii must match checkpoint resolutions")
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)[None]
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32)[None]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    latent_paths = sorted(args.latent_dir.glob("*.npz"))
    selected_paths = latent_paths[
        args.start_index : args.start_index + args.limit
    ]
    for index, latent_path in enumerate(selected_paths, start=args.start_index):
        condition = torch.from_numpy(
            np.load(args.condition_dir / f"{latent_path.stem}.npy")
            .astype(np.float32)
            .reshape(-1)
        ).to(args.device)
        using_cuda = str(args.device).startswith("cuda") and torch.cuda.is_available()
        if using_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        sampling_started = time.perf_counter()
        sample = sample_dynamic_sparse_rdd(
            model,
            resolutions=resolutions,
            steps=tuple(args.steps),
            time_fractions=tuple(float(value) for value in rdd["time_fractions"]),
            probability_thresholds=probability_thresholds,
            mass_thresholds=tuple(args.mass_thresholds),
            condition=condition,
            state_channels=checkpoint["model_config"]["state_channels"],
            sigma_min=float(rdd["sigma_min"]),
            seed=args.seed + index,
            solver=args.solver,
            pruning_mode=args.pruning_mode,
            force_keep_resolutions=tuple(args.force_keep_resolutions),
            halo_radii=halo_radii,
            device=args.device,
        )
        if using_cuda:
            torch.cuda.synchronize()
        sampling_wall_seconds = time.perf_counter() - sampling_started
        cuda_peak_allocated_bytes = (
            int(torch.cuda.max_memory_allocated()) if using_cuda else None
        )
        normalized = sample.trellis_features.float().cpu().numpy()
        output = args.output_dir / latent_path.name
        payload = dict(
            coords=sample.coords.int().cpu().numpy(),
            feats=(normalized * std + mean).astype(np.float32),
            normalized_feats=normalized.astype(np.float32),
            mass=sample.representation[:, 0].float().cpu().numpy(),
        )
        for resolution, retained_coords, retained_representation in zip(
            resolutions,
            sample.retained_coords_by_stage,
            sample.retained_representations_by_stage,
        ):
            payload[f"coords_{resolution}"] = retained_coords.int().cpu().numpy()
            payload[f"representation_{resolution}"] = (
                retained_representation.float().cpu().numpy()
            )
        for resolution, candidate_coords, candidate_representation, probabilities, masses in zip(
            resolutions,
            sample.candidate_coords_by_stage,
            sample.candidate_representations_by_stage,
            sample.nonempty_probabilities_by_stage,
            sample.masses_by_stage,
        ):
            payload[f"candidate_coords_{resolution}"] = (
                candidate_coords.int().cpu().numpy()
            )
            payload[f"candidate_representation_{resolution}"] = (
                candidate_representation.float().cpu().numpy()
            )
            payload[f"nonempty_probability_{resolution}"] = (
                probabilities.float().cpu().numpy()
            )
            payload[f"candidate_mass_{resolution}"] = masses.float().cpu().numpy()
        np.savez_compressed(output, **payload)
        rows.append(
            {
                "asset": latent_path.stem,
                "output": str(output),
                "stages": [stage.__dict__ for stage in sample.stages],
                "sampling_wall_seconds": sampling_wall_seconds,
                "cuda_peak_allocated_bytes": cuda_peak_allocated_bytes,
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    manifest = {
        "checkpoint": str(args.checkpoint),
        "weights": args.weights,
        "stage_checkpoints": stage_checkpoint_manifest,
        "stage_weights": stage_weight_manifest,
        "probability_thresholds": list(probability_thresholds),
        "mass_thresholds": list(args.mass_thresholds),
        "solver": args.solver,
        "pruning_mode": args.pruning_mode,
        "force_keep_resolutions": list(args.force_keep_resolutions),
        "halo_radii": list(halo_radii),
        "steps": list(args.steps),
        "samples": rows,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
