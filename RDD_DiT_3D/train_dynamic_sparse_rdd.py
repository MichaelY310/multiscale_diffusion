#!/usr/bin/env python3
"""Train the minimal shared dynamic-sparse RDD denoiser.

The first gate intentionally targets 512-output / 32^3 SLat with stages
8^3->16^3->32^3.  It has no dependency on TRELLIS sparse CUDA extensions and
can therefore validate empty-block prediction before a large-backbone port.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
from dataclasses import replace
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from dynamic_sparse_flow import make_rdd_fm_training_state, make_stage_training_state
from dynamic_sparse_model import DynamicSparseRDDModel, parameter_count
from dynamic_sparse_representation import (
    DynamicSparseStage,
    add_empty_parent_groups,
    build_dynamic_sparse_pyramid,
    build_generated_parent_stage,
    project_complete_sibling_detail,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-dir", type=Path, required=True)
    parser.add_argument("--condition-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--resume-weights", choices=("model", "ema"), default="model")
    parser.add_argument("--reset-optimizer", action="store_true")
    parser.add_argument("--fine-resolution", type=int, default=32)
    parser.add_argument("--resolutions", type=int, nargs="+", default=(8, 16, 32))
    parser.add_argument(
        "--time-fractions",
        type=float,
        nargs="+",
        default=(0.5, 0.3, 0.2),
        help="ordered fine-to-coarse",
    )
    parser.add_argument("--max-assets", type=int, default=32)
    parser.add_argument(
        "--steps", type=int, default=2_000,
        help="number of optimizer updates (not gradient-accumulation microsteps)",
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=None,
        help="number of 3-D shapes per optimizer update; defaults to DDP world size",
    )
    parser.add_argument(
        "--flow-path",
        choices=("legacy-local", "rdd-fm-global"),
        default="legacy-local",
        help=(
            "legacy-local preserves old checkpoints; rdd-fm-global exactly follows "
            "the 2-D RDD-FM noise-to-data path and global-time velocity target"
        ),
    )
    parser.add_argument(
        "--all-stages-per-step",
        action="store_true",
        help="sum every resolution-stage loss for every shape, as in 2-D Unified FM",
    )
    parser.add_argument(
        "--backward-per-stage",
        action="store_true",
        help=(
            "backpropagate each resolution loss immediately instead of retaining all "
            "stage graphs; this is gradient-equivalent and lowers peak memory"
        ),
    )
    parser.add_argument(
        "--stage-probabilities",
        type=float,
        nargs="+",
        default=None,
        help="training probabilities ordered coarse-to-fine; default is uniform",
    )
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--ema-rate", type=float, default=0.999)
    parser.add_argument("--occupancy-loss-weight", type=float, default=0.25)
    parser.add_argument("--positive-weight", type=float, default=8.0)
    parser.add_argument(
        "--negative-parent-ratio",
        type=float,
        default=1.0,
        help="sampled GT-empty parents per GT-active parent at refinement stages",
    )
    parser.add_argument(
        "--endpoint-time-probability",
        type=float,
        default=0.2,
        help="probability of training exactly at the clean pruning endpoint",
    )
    parser.add_argument("--mass-velocity-weight", type=float, default=4.0)
    parser.add_argument(
        "--velocity-loss-mode",
        choices=("legacy", "group-balanced", "channel-balanced"),
        default="legacy",
        help=(
            "legacy keeps historical band/mass weighting; group-balanced gives "
            "mass and the 32-D feature group one half each; channel-balanced is "
            "ordinary elementwise MSE across all channels"
        ),
    )
    parser.add_argument(
        "--generated-parent-dir",
        type=Path,
        default=None,
        help="sampling directory containing coords_R and representation_R replay states",
    )
    parser.add_argument(
        "--generated-parent-dirs",
        type=Path,
        nargs="+",
        default=None,
        help="multiple replay directories; one trajectory is sampled per training example",
    )
    parser.add_argument(
        "--generated-parent-probability",
        type=float,
        default=0.0,
        help="probability of replacing a refinement teacher parent with replayed generated state",
    )
    parser.add_argument(
        "--generated-endpoint-time-probability",
        type=float,
        default=0.5,
        help="clean-endpoint probability for generated-parent examples",
    )
    parser.add_argument(
        "--generated-parent-velocity-weight",
        type=float,
        default=0.25,
        help="velocity-loss multiplier on generated-parent examples; occupancy is unscaled",
    )
    parser.add_argument("--sigma-min", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument(
        "--no-save-final",
        action="store_true",
        help="skip the final checkpoint (intended only for disposable memory/timing smoke tests)",
    )
    parser.add_argument(
        "--latest-every",
        type=int,
        default=0,
        help=(
            "atomically refresh checkpoints/latest.pt at this interval; 0 disables "
            "the rolling checkpoint"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("none", "bf16"), default="bf16")
    return parser.parse_args()


def read_latent(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        if "coords" not in data or "feats" not in data:
            raise ValueError(f"{path} must contain coords and feats")
        coords = data["coords"].astype(np.int64)
        feats = data["feats"].astype(np.float32)
    if coords.ndim == 2 and coords.shape[1] == 4:
        coords = coords[:, 1:]
    return coords, feats


def load_condition(path: Path | None, asset: str) -> np.ndarray | None:
    if path is None:
        return None
    condition_path = path / f"{asset}.npy"
    if not condition_path.exists():
        raise FileNotFoundError(condition_path)
    condition = np.load(condition_path).astype(np.float32).reshape(-1)
    return condition


def load_generated_parents(
    path: Path | None,
    asset: str,
    resolutions: tuple[int, ...],
) -> dict[int, tuple[torch.Tensor, torch.Tensor]] | None:
    if path is None:
        return None
    replay_path = path / f"{asset}.npz"
    if not replay_path.exists():
        raise FileNotFoundError(replay_path)
    parents: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    with np.load(replay_path) as data:
        for resolution in resolutions[:-1]:
            coords_key = f"coords_{resolution}"
            representation_key = f"representation_{resolution}"
            if coords_key not in data or representation_key not in data:
                raise ValueError(
                    f"{replay_path} is missing {coords_key}/{representation_key}; "
                    "regenerate replay samples with the current sampler"
                )
            coords = torch.from_numpy(data[coords_key].astype(np.int64))
            representation = torch.from_numpy(
                data[representation_key].astype(np.float32)
            )
            parents[int(resolution)] = (coords, representation)
    return parents


def jsonable_arg(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [jsonable_arg(item) for item in value]
    return value


def stage_to(stage: DynamicSparseStage, device: torch.device) -> DynamicSparseStage:
    return replace(
        stage,
        logical_coords=stage.logical_coords.to(device),
        model_coords=stage.model_coords.to(device),
        target_rep=stage.target_rep.to(device),
        nonempty=stage.nonempty.to(device),
        parent_rep=None if stage.parent_rep is None else stage.parent_rep.to(device),
        detail=None if stage.detail is None else stage.detail.to(device),
    )


def update_ema(ema, model, rate: float) -> None:
    with torch.no_grad():
        for ema_parameter, parameter in zip(ema.parameters(), model.parameters()):
            ema_parameter.lerp_(parameter, 1.0 - rate)


def initialize_distributed(requested_device: str) -> tuple[torch.device, int, int]:
    """Initialize torchrun DDP while preserving the single-process entrypoint."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return torch.device("cuda", local_rank), rank, world_size
    return torch.device(requested_device), rank, world_size


def reduce_mean(value: torch.Tensor, world_size: int) -> torch.Tensor:
    value = value.detach().float().clone()
    if world_size > 1:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= world_size
    return value


def reduce_sum(value: torch.Tensor, world_size: int) -> torch.Tensor:
    value = value.detach().clone()
    if world_size > 1:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def reduce_max(value: torch.Tensor, world_size: int) -> torch.Tensor:
    value = value.detach().clone()
    if world_size > 1:
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return value


def save_checkpoint(
    path: Path,
    *,
    model,
    ema,
    optimizer,
    step: int,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    condition_channels: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
            "step": step,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "model_config": {
                "state_channels": int(len(feature_mean) + 1),
                "condition_channels": condition_channels,
                "width": args.width,
                "depth": args.depth,
            },
            "rdd_config": {
                "fine_resolution": args.fine_resolution,
                "resolutions": list(args.resolutions),
                "time_fractions": list(args.time_fractions),
                "sigma_min": args.sigma_min,
            },
            "args": vars(args),
        }
    # A pre-emption or manual kill during torch.save must never destroy the
    # previous rolling checkpoint.  Write beside the target and publish with
    # an atomic rename only after the payload is complete.
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    args = parse_args()
    if len(args.resolutions) != len(args.time_fractions):
        raise SystemExit("resolutions and time-fractions must have the same length")
    if args.resolutions[-1] != args.fine_resolution:
        raise SystemExit("last resolution must equal fine-resolution")
    if args.stage_probabilities is not None and len(args.stage_probabilities) != len(
        args.resolutions
    ):
        raise SystemExit("stage-probabilities must match resolutions")
    if args.all_stages_per_step and args.stage_probabilities is not None:
        raise SystemExit("stage-probabilities are incompatible with --all-stages-per-step")
    if args.flow_path == "rdd-fm-global":
        if not args.all_stages_per_step:
            raise SystemExit("rdd-fm-global parity runs require --all-stages-per-step")
        if args.velocity_loss_mode == "legacy":
            raise SystemExit(
                "rdd-fm-global requires group-balanced or channel-balanced velocity loss"
            )
        if args.generated_parent_probability != 0:
            raise SystemExit("rdd-fm-global baseline requires generated-parent-probability=0")
        if args.endpoint_time_probability != 0:
            raise SystemExit("rdd-fm-global baseline requires endpoint-time-probability=0")
    if not 0.0 <= args.generated_parent_probability <= 1.0:
        raise SystemExit("generated-parent-probability must be in [0, 1]")
    if not 0.0 <= args.generated_endpoint_time_probability <= 1.0:
        raise SystemExit("generated-endpoint-time-probability must be in [0, 1]")
    resume_checkpoint = None
    if args.resume is not None:
        resume_checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    generated_parent_dirs = list(args.generated_parent_dirs or ())
    if args.generated_parent_dir is not None:
        generated_parent_dirs.insert(0, args.generated_parent_dir)
    device, rank, world_size = initialize_distributed(args.device)
    is_main = rank == 0
    global_batch_size = args.global_batch_size or world_size
    if global_batch_size <= 0 or global_batch_size % world_size != 0:
        raise SystemExit(
            f"global-batch-size={global_batch_size} must be positive and divisible "
            f"by world-size={world_size}"
        )
    accumulation_steps = global_batch_size // world_size
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    files = sorted(args.latent_dir.glob("*.npz"))[: args.max_assets]
    if not files:
        raise SystemExit(f"no .npz files under {args.latent_dir}")
    raw = []
    all_features = []
    condition_channels = 0
    for path in files:
        coords, feats = read_latent(path)
        condition = load_condition(args.condition_dir, path.stem)
        if condition is not None:
            if condition_channels and len(condition) != condition_channels:
                raise ValueError("condition dimensions are inconsistent")
            condition_channels = len(condition)
        raw.append((path.stem, coords, feats, condition))
        all_features.append(feats)
    if resume_checkpoint is None:
        concatenated = np.concatenate(all_features, axis=0).astype(np.float64)
        feature_mean = concatenated.mean(axis=0).astype(np.float32)
        feature_std = np.maximum(concatenated.std(axis=0), 1e-4).astype(np.float32)
    else:
        feature_mean = np.asarray(resume_checkpoint["feature_mean"], dtype=np.float32)
        feature_std = np.asarray(resume_checkpoint["feature_std"], dtype=np.float32)
        if all_features[0].shape[1] != len(feature_mean):
            raise ValueError("resume checkpoint feature dimension does not match dataset")
    examples = []
    for asset, coords, feats, condition in raw:
        normalized = (feats - feature_mean) / feature_std
        pyramid = build_dynamic_sparse_pyramid(
            torch.from_numpy(coords),
            torch.from_numpy(normalized.astype(np.float32)),
            fine_resolution=args.fine_resolution,
            resolutions=tuple(args.resolutions),
        )
        generated_parent_sets = [
            load_generated_parents(directory, asset, tuple(args.resolutions))
            for directory in generated_parent_dirs
        ]
        examples.append((asset, pyramid, condition, generated_parent_sets))

    model = DynamicSparseRDDModel(
        state_channels=len(feature_mean) + 1,
        condition_channels=condition_channels,
        width=args.width,
        depth=args.depth,
    ).to(device)
    start_step = 0
    if resume_checkpoint is not None:
        expected_config = resume_checkpoint["model_config"]
        if int(expected_config["width"]) != args.width or int(
            expected_config["depth"]
        ) != args.depth:
            raise ValueError("resume checkpoint width/depth do not match command line")
        if int(expected_config["condition_channels"]) != condition_channels:
            raise ValueError("resume checkpoint condition dimension does not match dataset")
        model.load_state_dict(resume_checkpoint[args.resume_weights])
        start_step = int(resume_checkpoint.get("step", 0))
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    if resume_checkpoint is not None and "ema" in resume_checkpoint:
        ema.load_state_dict(resume_checkpoint["ema"])
    train_model = model
    if world_size > 1:
        train_model = DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
        )
    optimizer = torch.optim.AdamW(
        train_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    if (
        resume_checkpoint is not None
        and not args.reset_optimizer
        and args.resume_weights == "model"
        and "optimizer" in resume_checkpoint
    ):
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
    if is_main:
        args.output.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    metadata = {
        "assets": len(examples),
        "parameters": parameter_count(model),
        "condition_channels": condition_channels,
        "feature_channels": len(feature_mean),
        "resolutions": list(args.resolutions),
        "world_size": world_size,
        "gradient_accumulation_steps": accumulation_steps,
        "effective_shapes_per_optimizer_step": global_batch_size,
        "start_step": start_step,
    }
    if is_main:
        (args.output / "config.json").write_text(
            json.dumps(
                {
                    **metadata,
                    "args": {k: jsonable_arg(v) for k, v in vars(args).items()},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    log_path = args.output / "train_log.jsonl"
    # DDP ranks see different assets while sharing one model.  Optional
    # accumulation makes the effective shape batch match the 2-D Unified run.
    random.seed(args.seed + rank * 100_003)
    np.random.seed(args.seed + rank * 100_003)
    torch.manual_seed(args.seed + rank * 100_003)
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    stage_probabilities = args.stage_probabilities
    if stage_probabilities is None:
        stage_probabilities = [1.0] * len(args.resolutions)
    if any(probability < 0 for probability in stage_probabilities) or not any(
        probability > 0 for probability in stage_probabilities
    ):
        raise SystemExit("stage-probabilities must be non-negative and not all zero")
    for local_step in range(1, args.steps + 1):
        step = start_step + local_step
        accumulated_loss = torch.zeros((), device=device)
        accumulated_velocity = torch.zeros((), device=device)
        accumulated_mass = torch.zeros((), device=device)
        accumulated_feature = torch.zeros((), device=device)
        accumulated_mass_contribution = torch.zeros((), device=device)
        accumulated_feature_contribution = torch.zeros((), device=device)
        accumulated_occupancy = torch.zeros((), device=device)
        true_positive = torch.zeros((), dtype=torch.long, device=device)
        positive_count = torch.zeros((), dtype=torch.long, device=device)
        predicted_count = torch.zeros((), dtype=torch.long, device=device)
        stage_calls = 0
        token_calls = 0
        last_asset = ""
        last_resolution: int | str = "all" if args.all_stages_per_step else -1
        any_generated_parent = False

        for accumulation_index in range(accumulation_steps):
            asset, pyramid, condition_np, generated_parent_sets = random.choice(examples)
            last_asset = asset
            condition = None
            if condition_np is not None:
                condition = torch.from_numpy(condition_np).to(device)
            if args.all_stages_per_step:
                stage_indices = range(len(pyramid.stages))
            else:
                stage_indices = (
                    random.choices(
                        range(len(pyramid.stages)), weights=stage_probabilities, k=1
                    )[0],
                )

            synchronize = accumulation_index == accumulation_steps - 1
            sync_context = (
                nullcontext()
                if world_size == 1 or synchronize
                else train_model.no_sync()
            )
            with sync_context:
                micro_loss = torch.zeros((), device=device)
                for stage_index in stage_indices:
                    stage = pyramid.stages[stage_index]
                    use_generated_parent = bool(
                        stage_index > 0
                        and bool(generated_parent_sets)
                        and random.random() < args.generated_parent_probability
                    )
                    any_generated_parent = any_generated_parent or use_generated_parent
                    if use_generated_parent:
                        generated_parents = random.choice(generated_parent_sets)
                        assert generated_parents is not None
                        parent_resolution = int(args.resolutions[stage_index - 1])
                        parent_coords, parent_rep = generated_parents[parent_resolution]
                        stage = build_generated_parent_stage(
                            stage,
                            parent_coords=parent_coords,
                            parent_rep=parent_rep,
                            parent_resolution=parent_resolution,
                            fine_resolution=args.fine_resolution,
                        )
                    elif stage_index > 0 and args.negative_parent_ratio > 0:
                        stage = add_empty_parent_groups(
                            stage,
                            parent_resolution=args.resolutions[stage_index - 1],
                            fine_resolution=args.fine_resolution,
                            ratio=args.negative_parent_ratio,
                        )
                    stage = stage_to(stage, device)
                    last_resolution = "all" if args.all_stages_per_step else stage.resolution
                    endpoint_probability = (
                        args.generated_endpoint_time_probability
                        if use_generated_parent
                        else args.endpoint_time_probability
                    )
                    if random.random() < endpoint_probability:
                        local_p = torch.zeros((), device=device)
                    else:
                        local_p = torch.rand((), device=device)
                    if args.flow_path == "rdd-fm-global":
                        training = make_rdd_fm_training_state(
                            stage,
                            stage_index=stage_index,
                            num_stages=len(pyramid.stages),
                            local_p=local_p,
                            time_fractions=tuple(args.time_fractions),
                        )
                    else:
                        training = make_stage_training_state(
                            stage,
                            stage_index=stage_index,
                            num_stages=len(pyramid.stages),
                            local_p=local_p,
                            time_fractions=tuple(args.time_fractions),
                            sigma_min=args.sigma_min,
                        )
                    amp_enabled = args.amp == "bf16" and device.type == "cuda"
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=amp_enabled,
                    ):
                        velocity, logits = train_model(
                            training.state,
                            stage.logical_coords,
                            resolution=stage.resolution,
                            global_time=training.global_time,
                            condition=condition,
                        )
                        if args.flow_path == "legacy-local" and stage_index > 0:
                            velocity = project_complete_sibling_detail(
                                stage.logical_coords, velocity
                            )
                        error = velocity.float() - training.target_velocity.float()
                        mass_mse = error[:, 0].square().mean()
                        feature_mse = error[:, 1:].square().mean()
                        if args.velocity_loss_mode == "legacy":
                            mass_contribution = (
                                training.band_width**2
                                * args.mass_velocity_weight
                                * mass_mse
                            )
                            feature_contribution = (
                                training.band_width**2 * feature_mse
                            )
                            velocity_loss = mass_contribution + feature_contribution
                        elif args.velocity_loss_mode == "group-balanced":
                            mass_contribution = 0.5 * mass_mse
                            feature_contribution = 0.5 * feature_mse
                            velocity_loss = mass_contribution + feature_contribution
                        else:
                            channel_count = error.shape[1]
                            mass_contribution = mass_mse / channel_count
                            feature_contribution = (
                                (channel_count - 1) * feature_mse / channel_count
                            )
                            velocity_loss = mass_contribution + feature_contribution
                        if use_generated_parent:
                            velocity_loss = (
                                velocity_loss * args.generated_parent_velocity_weight
                            )
                            mass_contribution = (
                                mass_contribution * args.generated_parent_velocity_weight
                            )
                            feature_contribution = (
                                feature_contribution * args.generated_parent_velocity_weight
                            )
                        occupancy_target = stage.nonempty.float()
                        occupancy_loss = F.binary_cross_entropy_with_logits(
                            logits.float(),
                            occupancy_target,
                            pos_weight=torch.tensor(args.positive_weight, device=device),
                        )
                        stage_loss = (
                            velocity_loss + args.occupancy_loss_weight * occupancy_loss
                        )
                    if not bool(torch.isfinite(stage_loss)):
                        raise FloatingPointError(
                            f"non-finite loss at step {step}, stage {stage.resolution}"
                        )
                    if args.backward_per_stage:
                        (stage_loss / accumulation_steps).backward()
                    else:
                        micro_loss = micro_loss + stage_loss
                    accumulated_loss += stage_loss.detach()
                    accumulated_velocity += velocity_loss.detach()
                    accumulated_mass += mass_mse.detach()
                    accumulated_feature += feature_mse.detach()
                    accumulated_mass_contribution += mass_contribution.detach()
                    accumulated_feature_contribution += feature_contribution.detach()
                    accumulated_occupancy += occupancy_loss.detach()
                    predicted = torch.sigmoid(logits.detach().float()) >= 0.5
                    positives = stage.nonempty
                    true_positive += (predicted & positives).sum()
                    positive_count += positives.sum()
                    predicted_count += predicted.sum()
                    stage_calls += 1
                    token_calls += len(stage.logical_coords)

                if not args.backward_per_stage:
                    (micro_loss / accumulation_steps).backward()

        max_grad_norm = args.grad_clip if args.grad_clip > 0 else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update_ema(ema, model, args.ema_rate)

        with torch.no_grad():
            logged_loss = reduce_mean(accumulated_loss / accumulation_steps, world_size)
            logged_velocity = reduce_mean(
                accumulated_velocity / accumulation_steps, world_size
            )
            logged_mass = reduce_mean(accumulated_mass / stage_calls, world_size)
            logged_feature = reduce_mean(accumulated_feature / stage_calls, world_size)
            logged_mass_contribution = reduce_mean(
                accumulated_mass_contribution / stage_calls, world_size
            )
            logged_feature_contribution = reduce_mean(
                accumulated_feature_contribution / stage_calls, world_size
            )
            logged_occupancy = reduce_mean(
                accumulated_occupancy / stage_calls, world_size
            )
            true_positive = reduce_sum(true_positive, world_size)
            positive_count = reduce_sum(positive_count, world_size)
            predicted_count = reduce_sum(predicted_count, world_size)
            recall = true_positive.float() / positive_count.clamp_min(1)
            precision = true_positive.float() / predicted_count.clamp_min(1)
            logged_grad = reduce_mean(torch.as_tensor(grad_norm, device=device), world_size)
            if device.type == "cuda":
                peak_allocated_gib = reduce_max(
                    torch.as_tensor(
                        torch.cuda.max_memory_allocated(device) / 2**30,
                        device=device,
                    ),
                    world_size,
                )
                peak_reserved_gib = reduce_max(
                    torch.as_tensor(
                        torch.cuda.max_memory_reserved(device) / 2**30,
                        device=device,
                    ),
                    world_size,
                )
            else:
                peak_allocated_gib = torch.zeros((), device=device)
                peak_reserved_gib = torch.zeros((), device=device)
        row = {
            "step": step,
            "asset": last_asset,
            "stage": last_resolution,
            "generated_parent": any_generated_parent,
            "mean_tokens_per_stage_call": token_calls / stage_calls,
            "global_shapes": global_batch_size,
            "gradient_accumulation_steps": accumulation_steps,
            "stage_calls_per_rank": stage_calls,
            "loss": float(logged_loss),
            "velocity_loss": float(logged_velocity),
            "mass_mse": float(logged_mass),
            "feature_mse": float(logged_feature),
            "mass_loss_contribution": float(logged_mass_contribution),
            "feature_loss_contribution": float(logged_feature_contribution),
            "mass_to_feature_contribution_ratio": float(
                logged_mass_contribution / logged_feature_contribution.clamp_min(1e-12)
            ),
            "occupancy_loss": float(logged_occupancy),
            "occupancy_recall_0.5": float(recall),
            "occupancy_precision_0.5": float(precision),
            "grad_norm": float(logged_grad),
            "peak_gpu_allocated_gib": float(peak_allocated_gib),
            "peak_gpu_reserved_gib": float(peak_reserved_gib),
        }
        if is_main:
            with log_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(row) + "\n")
        if is_main and (local_step == 1 or step % args.log_every == 0):
            print(json.dumps(row), flush=True)
        save_milestone = args.save_every > 0 and step % args.save_every == 0
        save_latest = args.latest_every > 0 and step % args.latest_every == 0
        save_final = local_step == args.steps and not args.no_save_final
        if is_main and (save_milestone or save_latest or save_final):
            checkpoint_path = (
                args.output / "checkpoints" / f"step{step:07d}.pt"
                if save_milestone or save_final
                else args.output / "checkpoints" / "latest.pt"
            )
            save_checkpoint(
                checkpoint_path,
                model=model,
                ema=ema,
                optimizer=optimizer,
                step=step,
                feature_mean=feature_mean,
                feature_std=feature_std,
                condition_channels=condition_channels,
                args=args,
            )
            if checkpoint_path.name != "latest.pt":
                latest_path = args.output / "checkpoints" / "latest.pt"
                temporary_link = latest_path.with_name(
                    f".{latest_path.name}.tmp-{os.getpid()}"
                )
                if temporary_link.exists() or temporary_link.is_symlink():
                    temporary_link.unlink()
                temporary_link.symlink_to(checkpoint_path.name)
                os.replace(temporary_link, latest_path)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
