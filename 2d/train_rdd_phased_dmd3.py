#!/usr/bin/env python3
"""Original-DMD distribution matching for a three-NFE Unified RDD generator.

One conditional DiT call is used at each resolution (8, 16, 32).  The complete
three-call composition produces one final 32x32 clean latent.  A frozen RDD
teacher estimates the target score and an online fake-score model estimates the
current final generator distribution.  Lower-resolution score queries use the
same average-pooled final clean latent used to train Unified RDD, and gradients
flow through the complete 8 -> 16 -> 32 generator composition.

The distribution-matching gradient follows DMD Eq. 7 and Eq. 8 after directly
substituting this repository's RDD ``q_sample`` expression.  No endpoint
preservation, selectable path correction, repeated fake-score updates, GAN
term, or gradient-value clipping is used.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers.models import AutoencoderKL
from torch.nn.parallel import DistributedDataParallel as DDP


ROOT = Path(__file__).resolve().parents[1]
DIT_ROOT = ROOT / "DiT"
sys.path.insert(0, str(DIT_ROOT))
from models_multires import DiTMultiRes_models  # noqa: E402
sys.path.remove(str(DIT_ROOT))
sys.path.insert(0, str(ROOT))

from diffusion import create_diffusion_with_drift  # noqa: E402
from diffusion.rdd_phased_dmd import (  # noqa: E402
    dmd_surrogate_loss,
    downsample_clean,
    fresh_transition,
    rdd_dmd_gradient,
    rdd_renoise,
    sampling_phases,
)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_csv_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not result or any(item <= 0 for item in result):
        raise ValueError(f"invalid positive integer list: {value!r}")
    return result


def set_linear_warmup_lr(
    optimizer: torch.optim.Optimizer,
    base_lr: float,
    step: int,
    warmup_steps: int,
) -> float:
    """Set a restart-safe linear warmup using the absolute outer step."""

    scale = 1.0 if warmup_steps <= 0 else min(1.0, float(step + 1) / warmup_steps)
    learning_rate = float(base_lr) * scale
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return learning_rate


def create_logger(output_dir: Path, rank: int) -> logging.Logger:
    logger = logging.getLogger(f"rdd_phased_dmd_rank{rank}")
    logger.handlers.clear()
    logger.propagate = False
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        file_handler = logging.FileHandler(output_dir / "log.txt", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(stream)
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
    else:
        logger.addHandler(logging.NullHandler())
    return logger


def set_seed(seed: int, rank: int) -> None:
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_num_classes(weights: dict[str, torch.Tensor]) -> int:
    for key, value in weights.items():
        if key.endswith("y_embedder.embedding_table.weight"):
            # Unified RDD checkpoints use classifier-free label dropout, hence
            # the final embedding row is the null class.
            return int(value.shape[0]) - 1
    raise KeyError("could not infer num_classes from label embedding")


def build_model(
    architecture: str,
    resolutions: Sequence[int],
    weights: dict[str, torch.Tensor],
) -> torch.nn.Module:
    patch_size = int(architecture.split("/")[1])
    repa_dim = None
    repa_projector_dim = 2048
    if "repa_projector.2.weight" in weights:
        repa_dim = int(weights["repa_projector.2.weight"].shape[0])
        repa_projector_dim = int(weights["repa_projector.2.weight"].shape[1])
    model = DiTMultiRes_models[architecture](
        num_classes=infer_num_classes(weights),
        base_size=int(resolutions[0]) // patch_size,
        repa_dim=repa_dim,
        repa_projector_dim=repa_projector_dim,
    )
    model.load_state_dict(weights, strict=True)
    return model


def extract_teacher_weights(checkpoint_path: str) -> tuple[dict[str, torch.Tensor], dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("teacher checkpoint must be a dictionary")
    weights = checkpoint.get("ema", checkpoint.get("model", checkpoint))
    if not isinstance(weights, dict):
        raise TypeError("teacher checkpoint has no model/ema state dictionary")
    metadata = {
        "stage_lengths": checkpoint.get("stage_lengths"),
        "resolutions": checkpoint.get("resolutions"),
        "step": checkpoint.get("step"),
    }
    return weights, metadata


def top_timestep(diffusion, batch: int, device: torch.device) -> torch.Tensor:
    return torch.full(
        (int(batch),),
        int(diffusion.num_timesteps) - 1,
        device=device,
        dtype=torch.long,
    )


def random_denoising_timestep(
    diffusion, batch: int, device: torch.device
) -> torch.Tensor:
    """DMD Algorithm 1 samples the fake-score denoising time over all times."""

    return torch.randint(
        0,
        int(diffusion.num_timesteps),
        (int(batch),),
        device=device,
        dtype=torch.long,
    )


def score_timestep_candidates(
    diffusions: Sequence,
    resolutions: Sequence[int],
    diffusion_steps: int,
) -> tuple[tuple[object, torch.Tensor], ...]:
    """Stage-local indices giving DMD's uniform global [2%, 98%) interval."""

    global_low = int(round(int(diffusion_steps) * 0.02))
    global_high = int(round(int(diffusion_steps) * 0.98))
    result = []
    for phase in sampling_phases(resolutions):
        diffusion = diffusions[phase.diffusion_index]
        mapping = torch.as_tensor(diffusion.timestep_map, dtype=torch.long)
        valid = torch.nonzero(
            (mapping >= global_low) & (mapping < global_high), as_tuple=False
        ).flatten()
        if valid.numel() == 0:
            raise ValueError(
                f"resolution {phase.resolution} has no DMD timesteps in "
                f"global [{global_low}, {global_high})"
            )
        result.append((phase, valid))
    if sum(int(indices.numel()) for _, indices in result) != global_high - global_low:
        raise ValueError("RDD stages do not partition the global DMD timestep interval")
    return tuple(result)


def sample_score_phase_and_timestep(
    candidates: Sequence[tuple[object, torch.Tensor]],
    batch: int,
    device: torch.device,
) -> tuple[object, torch.Tensor]:
    """Sample an unbiased global time while keeping one tensor shape per batch."""

    counts = [int(indices.numel()) for _, indices in candidates]
    total = sum(counts)
    draw = torch.zeros(1, device=device, dtype=torch.long)
    if dist.get_rank() == 0:
        draw.random_(0, total)
    dist.broadcast(draw, src=0)
    value = int(draw.item())
    selected_phase = None
    selected_indices = None
    offset = 0
    for (phase, indices), count in zip(candidates, counts):
        if value < offset + count:
            selected_phase = phase
            selected_indices = indices
            break
        offset += count
    if selected_phase is None or selected_indices is None:
        raise RuntimeError("failed to select an RDD score phase")
    selected_indices = selected_indices.to(device=device)
    choices = torch.randint(
        0, selected_indices.numel(), (int(batch),), device=device
    )
    return selected_phase, selected_indices[choices]


def predict_clean(
    diffusion,
    model: Callable,
    x: torch.Tensor,
    local_t: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    return diffusion.p_mean_variance(
        model,
        x,
        local_t,
        clip_denoised=False,
        model_kwargs={"y": labels},
    )["pred_xstart"]


class HistoricalCFGForward:
    """Match this repository's published sampler: guide epsilon channels 0:3."""

    def __init__(self, teacher: torch.nn.Module, num_classes: int, scale: float):
        self.teacher = teacher
        self.num_classes = int(num_classes)
        self.scale = float(scale)

    def __call__(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        conditional = self.teacher(x, t, y)
        if self.scale == 1.0:
            return conditional
        null_labels = torch.full_like(y, self.num_classes)
        unconditional = self.teacher(x, t, null_labels)
        result = conditional.clone()
        result[:, :3] = unconditional[:, :3] + self.scale * (
            conditional[:, :3] - unconditional[:, :3]
        )
        return result


def build_diffusions(
    stage_lengths: Sequence[int], resolutions: Sequence[int], diffusion_steps: int
) -> list:
    boundaries = [0]
    for length in stage_lengths:
        boundaries.append(boundaries[-1] + int(length))
    result = []
    for index, resolution in enumerate(resolutions):
        scale = int(resolution) // int(resolutions[index + 1]) if index + 1 < len(resolutions) else 2
        result.append(
            create_diffusion_with_drift(
                timestep_respacing="",
                diffusion_steps=int(diffusion_steps),
                c=index if index + 1 < len(resolutions) else None,
                T_c=boundaries[index],
                T_c_plus_1=boundaries[index + 1],
                sample_scale=scale,
            )
        )
    return result


def rollout_all_phases_from_paired_noise(
    *,
    student_ddp: torch.nn.Module,
    diffusions: Sequence,
    resolutions: Sequence[int],
    labels: torch.Tensor,
    noise_8: torch.Tensor,
    noise_16: torch.Tensor,
    noise_32: torch.Tensor,
) -> torch.Tensor:
    """Run the exact 3-NFE student on the fixed noise stored in a DMD pair."""

    phases = sampling_phases(resolutions)
    if phases[0].resolution != 8 or phases[-1].resolution != 32:
        raise ValueError("paired DMD rollout expects 8 -> 16 -> 32")
    boundary_noises = (noise_16, noise_32)
    state = noise_8
    clean = None
    for phase in phases:
        diffusion = diffusions[phase.diffusion_index]
        local_t = top_timestep(diffusion, labels.shape[0], labels.device)
        clean = predict_clean(diffusion, student_ddp, state, local_t, labels)
        if phase.phase_index + 1 < len(phases):
            next_phase = phases[phase.phase_index + 1]
            state = fresh_transition(
                clean,
                next_phase.resolution,
                diffusions[next_phase.diffusion_index],
                noise=boundary_noises[phase.phase_index],
            )
    if clean is None:
        raise RuntimeError("paired student rollout produced no output")
    return clean


class RegressionPairStream:
    """Random batches from the fixed teacher-pair dataset required by DMD Eq.9."""

    def __init__(
        self,
        root: str,
        rank: int,
        world_size: int,
        seed: int,
        batch_size: int,
    ) -> None:
        self.root = Path(root).resolve()
        metadata_path = self.root / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"missing DMD pair metadata: {metadata_path}")
        with open(metadata_path, "r", encoding="utf-8") as handle:
            self.metadata = json.load(handle)
        paths = []
        for manifest_path in sorted(self.root.glob("manifest_rank*.jsonl")):
            with open(manifest_path, "r", encoding="utf-8") as manifest:
                for line in manifest:
                    entry = json.loads(line)
                    if int(entry["count"]) >= int(batch_size):
                        paths.append(self.root / entry["file"])
        if not paths:
            raise FileNotFoundError(
                f"no complete DMD regression shards found under {self.root}"
            )
        self.paths = paths[rank::world_size]
        if not self.paths:
            raise FileNotFoundError(f"no DMD pair shards assigned to rank {rank}")
        self.batch_size = int(batch_size)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed) + int(rank) * 1_000_003)

    def next(self, device: torch.device) -> dict[str, torch.Tensor]:
        shard_index = int(
            torch.randint(0, len(self.paths), (1,), generator=self.generator).item()
        )
        payload = torch.load(
            self.paths[shard_index], map_location="cpu", weights_only=False
        )
        count = int(payload["labels"].shape[0])
        if count < self.batch_size:
            raise ValueError(
                f"pair shard {self.paths[shard_index]} has {count} entries, "
                f"smaller than training microbatch {self.batch_size}"
            )
        selection = torch.randperm(count, generator=self.generator)[: self.batch_size]
        result = {}
        for key in ("labels", "noise_8", "noise_16", "noise_32", "target_latent"):
            result[key] = payload[key][selection].to(device=device, non_blocking=True)
        result["labels"] = result["labels"].long()
        for key in ("noise_8", "noise_16", "noise_32", "target_latent"):
            result[key] = result[key].float()
        return result


def decode_for_lpips(vae: AutoencoderKL, latent: torch.Tensor) -> torch.Tensor:
    image = vae.decode(latent / 0.18215).sample
    image = (image / 2.0 + 0.5).clamp(0.0, 1.0)
    return F.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)


def save_checkpoint(
    *,
    output_dir: Path,
    step: int,
    student: torch.nn.Module,
    critic: torch.nn.Module,
    student_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    rank: int,
    export_student: bool,
) -> None:
    if rank != 0:
        return
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "student": student.state_dict(),
        # Preserve the established sampler key. Original DMD itself does not
        # introduce a separate EMA model, so this is the current generator.
        "ema": student.state_dict(),
        "critic": critic.state_dict(),
        "student_opt": student_optimizer.state_dict(),
        "critic_opt": critic_optimizer.state_dict(),
        "stage_lengths": parse_csv_ints(args.stage_lengths),
        "resolutions": parse_csv_ints(args.resolutions),
        "distillation": "RDD-Original-DMD-3NFE-FullOutput-v2",
        "args": vars(args),
    }
    temporary = checkpoint_dir / f"latest.pt.tmp-{os.getpid()}"
    torch.save(payload, temporary)
    os.replace(temporary, checkpoint_dir / "latest.pt")
    if export_student:
        export = {
            "ema": student.state_dict(),
            "step": int(step),
            "stage_lengths": parse_csv_ints(args.stage_lengths),
            "resolutions": parse_csv_ints(args.resolutions),
            "distillation": "RDD-Original-DMD-3NFE-FullOutput-v2",
            "teacher_checkpoint": args.teacher_checkpoint,
            "teacher_cfg": args.teacher_cfg,
            "regression_data": args.regression_data,
        }
        export_temp = checkpoint_dir / f"{step:07d}.pt.tmp-{os.getpid()}"
        torch.save(export, export_temp)
        os.replace(export_temp, checkpoint_dir / f"{step:07d}.pt")


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("RDD phased DMD requires CUDA")
    # Bind the CUDA device before constructing the NCCL process group.  This is
    # important on multi-root PCIe systems: otherwise NCCL has to infer a device
    # at the first collective and may initialize its watchdog on cuda:0.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    set_seed(args.seed, rank)

    output_dir = Path(args.output_dir).resolve()
    logger = create_logger(output_dir, rank)
    resolutions = parse_csv_ints(args.resolutions)
    stage_lengths = parse_csv_ints(args.stage_lengths)
    if len(resolutions) != 3 or len(stage_lengths) != 3:
        raise ValueError("the three-step experiment requires exactly three stages")
    if sum(stage_lengths) != args.diffusion_steps:
        raise ValueError("stage lengths must sum to diffusion_steps")
    if args.batch_per_gpu <= 0:
        raise ValueError("batch_per_gpu must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    effective_global_batch = (
        world_size * args.batch_per_gpu * args.gradient_accumulation_steps
    )

    try:
        import piq
    except ImportError as error:
        raise RuntimeError(
            "Original DMD regression requires PIQ LPIPS; install requirements-dmd.txt"
        ) from error

    weights, teacher_metadata = extract_teacher_weights(args.teacher_checkpoint)
    if teacher_metadata["stage_lengths"] is not None and tuple(teacher_metadata["stage_lengths"]) != stage_lengths:
        raise ValueError(
            f"teacher stage_lengths={teacher_metadata['stage_lengths']} do not match {stage_lengths}"
        )
    num_classes = infer_num_classes(weights)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
            json.dump(vars(args), handle, indent=2, sort_keys=True)
        logger.info("RDD-Phased-DMD-3NFE: one call each at 8x8, 16x16, 32x32")
        logger.info(
            "Teacher=%s; architecture=%s; stage_lengths=%s; world_size=%d; "
            "micro_batch_per_gpu=%d; accumulation=%d; effective_global_batch=%d",
            args.teacher_checkpoint,
            args.model,
            stage_lengths,
            world_size,
            args.batch_per_gpu,
            args.gradient_accumulation_steps,
            effective_global_batch,
        )
        logger.info(
            "Original DMD Eq.7/8: teacher_cfg=%.3f, one generator update and "
            "one fresh fake-score update per outer step, lr_warmup=%d",
            args.teacher_cfg,
            args.lr_warmup_steps,
        )

    pair_stream = RegressionPairStream(
        args.regression_data,
        rank,
        world_size,
        args.seed,
        args.batch_per_gpu,
    )
    if tuple(pair_stream.metadata["stage_lengths"]) != stage_lengths:
        raise ValueError("DMD regression-pair stage lengths do not match training")
    if tuple(pair_stream.metadata["resolutions"]) != resolutions:
        raise ValueError("DMD regression-pair resolutions do not match training")
    if int(pair_stream.metadata.get("sampler_version", 0)) != 2:
        raise ValueError(
            "DMD pairs must use validated RDD marginal-DDIM sampler_version=2"
        )

    student = build_model(args.model, resolutions, weights).to(device)
    critic = build_model(args.model, resolutions, weights).to(device)
    teacher = build_model(args.model, resolutions, weights).to(
        device=device, dtype=torch.bfloat16 if args.bf16 else torch.float32
    )
    del weights
    torch.cuda.synchronize(device)
    if rank == 0:
        logger.info("Three DiT replicas loaded on every rank; constructing DDP reducers")

    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    # Eval mode disables class-label dropout but does not disable gradients.
    student.eval()
    critic.eval()
    teacher.eval()

    vae = AutoencoderKL.from_pretrained(
        args.vae_path, local_files_only=True
    ).to(device).eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    lpips = piq.LPIPS(reduction="mean").to(device).eval()
    for parameter in lpips.parameters():
        parameter.requires_grad_(False)

    # The generator reuses the same DiT three times inside a fixed regression
    # graph. static_graph lets DDP reduce that shared-weight graph directly.
    student_ddp = DDP(
        student,
        device_ids=[local_rank],
        broadcast_buffers=False,
        init_sync=False,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        static_graph=True,
    )
    critic = DDP(
        critic,
        device_ids=[local_rank],
        broadcast_buffers=False,
        init_sync=False,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
    )
    if rank == 0:
        logger.info("Student and fake-score DDP reducers ready")
    student_optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.student_lr,
        betas=(args.student_beta1, args.student_beta2),
        weight_decay=args.weight_decay,
    )
    critic_optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=args.critic_lr,
        betas=(args.critic_beta1, args.critic_beta2),
        weight_decay=args.weight_decay,
    )
    diffusions = build_diffusions(stage_lengths, resolutions, args.diffusion_steps)
    score_candidates = score_timestep_candidates(
        diffusions, resolutions, args.diffusion_steps
    )
    teacher_forward = HistoricalCFGForward(teacher, num_classes, args.teacher_cfg)

    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        student.load_state_dict(checkpoint["student"])
        critic.module.load_state_dict(checkpoint["critic"])
        student_optimizer.load_state_dict(checkpoint["student_opt"])
        critic_optimizer.load_state_dict(checkpoint["critic_opt"])
        start_step = int(checkpoint["step"])
        logger.info("Resumed exact training state at outer step %d", start_step)
    dist.barrier()

    autocast = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.bf16
        else nullcontext()
    )
    running = {"critic": 0.0, "dmd": 0.0, "regression": 0.0, "student_grad": 0.0}
    running_phase_counts = torch.zeros(3, dtype=torch.long)
    running_count = 0
    wall_start = time.time()
    stop_requested = False
    inference_phases = sampling_phases(resolutions)

    for step in range(start_step, args.max_steps):
        set_linear_warmup_lr(
            student_optimizer, args.student_lr, step, args.lr_warmup_steps
        )
        set_linear_warmup_lr(
            critic_optimizer, args.critic_lr, step, args.lr_warmup_steps
        )
        dmd_value = 0.0
        regression_value = 0.0
        student_optimizer.zero_grad(set_to_none=True)
        fake_score_batches = []
        for micro_index in range(args.gradient_accumulation_steps):
            labels_student = torch.randint(
                0,
                num_classes,
                (args.batch_per_gpu,),
                device=device,
                dtype=torch.long,
            )
            with nullcontext():
                with autocast():
                    generated_final = rollout_all_phases_from_paired_noise(
                        student_ddp=student_ddp,
                        diffusions=diffusions,
                        resolutions=resolutions,
                        labels=labels_student,
                        noise_8=torch.randn(
                            args.batch_per_gpu,
                            4,
                            inference_phases[0].resolution,
                            inference_phases[0].resolution,
                            device=device,
                        ),
                        noise_16=torch.randn(
                            args.batch_per_gpu,
                            4,
                            inference_phases[1].resolution,
                            inference_phases[1].resolution,
                            device=device,
                        ),
                        noise_32=torch.randn(
                            args.batch_per_gpu,
                            4,
                            inference_phases[2].resolution,
                            inference_phases[2].resolution,
                            device=device,
                        ),
                    )
                    selected_phase, score_t = sample_score_phase_and_timestep(
                        score_candidates,
                        args.batch_per_gpu,
                        device,
                    )
                    phase_diffusion = diffusions[selected_phase.diffusion_index]
                    generated = downsample_clean(
                        generated_final, selected_phase.resolution
                    )
                    running_phase_counts[selected_phase.phase_index] += 1
                with torch.no_grad(), autocast():
                    noisy = rdd_renoise(
                        phase_diffusion,
                        generated.detach(),
                        score_t,
                        noise=torch.randn_like(generated),
                    )
                    real_clean = predict_clean(
                        phase_diffusion,
                        teacher_forward,
                        noisy,
                        score_t,
                        labels_student,
                    ).float()
                    fake_clean = predict_clean(
                        phase_diffusion,
                        critic.module,
                        noisy,
                        score_t,
                        labels_student,
                    ).float()
                    gradient = rdd_dmd_gradient(
                        phase_diffusion,
                        generated.detach().float(),
                        real_clean,
                        fake_clean,
                        score_t,
                    )

                dmd_loss = dmd_surrogate_loss(generated, gradient)

                # Original DMD Eq.9 uses a separate fixed noise/teacher-output
                # stream and LPIPS. It is not an online one-step teacher target.
                pair = pair_stream.next(device)
                with autocast():
                    paired_generated = rollout_all_phases_from_paired_noise(
                        student_ddp=student_ddp,
                        diffusions=diffusions,
                        resolutions=resolutions,
                        labels=pair["labels"],
                        noise_8=pair["noise_8"],
                        noise_16=pair["noise_16"],
                        noise_32=pair["noise_32"],
                    )
                    paired_image = decode_for_lpips(vae, paired_generated)
                    with torch.no_grad():
                        target_image = decode_for_lpips(
                            vae, pair["target_latent"]
                        )
                regression_loss = lpips(
                    paired_image.float(), target_image.float()
                ).mean()
                student_loss = dmd_loss + 0.25 * regression_loss
                (student_loss / args.gradient_accumulation_steps).backward()
            dmd_value += float(dmd_loss.detach()) / args.gradient_accumulation_steps
            regression_value += (
                float(regression_loss.detach()) / args.gradient_accumulation_steps
            )
            fake_score_batches.append(
                (generated.detach().float(), labels_student, phase_diffusion)
            )

        student_grad = torch.nn.utils.clip_grad_norm_(
            student.parameters(), args.max_grad_norm
        )
        student_optimizer.step()
        student_grad_value = float(student_grad.detach())

        # Original DMD Algorithm 1 performs one fake-score update using the same
        # freshly generated data stream after each generator update.
        critic_optimizer.zero_grad(set_to_none=True)
        critic_loss_value = 0.0
        for micro_index, (critic_clean, labels, phase_diffusion) in enumerate(
            fake_score_batches
        ):
            critic_t = random_denoising_timestep(
                phase_diffusion, args.batch_per_gpu, device
            )
            sync_context = (
                nullcontext()
                if micro_index + 1 == args.gradient_accumulation_steps
                else critic.no_sync()
            )
            with sync_context:
                with autocast():
                    critic_terms = phase_diffusion.training_losses(
                        critic,
                        critic_clean,
                        critic_t,
                        model_kwargs={"y": labels},
                    )
                    critic_loss = critic_terms["loss"].mean()
                (critic_loss / args.gradient_accumulation_steps).backward()
            critic_loss_value += (
                float(critic_loss.detach()) / args.gradient_accumulation_steps
            )
        torch.nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
        critic_optimizer.step()
        del fake_score_batches

        running["critic"] += critic_loss_value
        running["dmd"] += dmd_value
        running["regression"] += regression_value
        running["student_grad"] += student_grad_value
        running_count += 1
        completed = step + 1

        if completed % args.log_every == 0:
            values = torch.tensor(
                [
                    running["critic"],
                    running["dmd"],
                    running["regression"],
                    running["student_grad"],
                ],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
            denom = float(running_count * world_size)
            elapsed = time.time() - wall_start
            logger.info(
                "step=%07d critic=%.6g dmd=%.6g regression=%.6g "
                "student_grad=%.6g score_batches_8_16_32=%d/%d/%d step_time=%.3fs",
                completed,
                values[0].item() / denom,
                values[1].item() / denom,
                values[2].item() / denom,
                values[3].item() / denom,
                int(running_phase_counts[0]),
                int(running_phase_counts[1]),
                int(running_phase_counts[2]),
                elapsed / running_count,
            )
            running = {key: 0.0 for key in running}
            running_phase_counts.zero_()
            running_count = 0
            wall_start = time.time()

        checkpoint_request_tensor = torch.tensor(
            [
                1
                if args.checkpoint_request_file
                and os.path.exists(args.checkpoint_request_file)
                else 0
            ],
            device=device,
            dtype=torch.int32,
        )
        dist.all_reduce(checkpoint_request_tensor, op=dist.ReduceOp.MAX)
        checkpoint_requested = bool(checkpoint_request_tensor.item())

        stop_tensor = torch.tensor(
            [1 if args.stop_file and os.path.exists(args.stop_file) else 0],
            device=device,
            dtype=torch.int32,
        )
        dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
        stop_requested = bool(stop_tensor.item())
        periodic = completed % args.checkpoint_every == 0
        final = completed == args.max_steps or stop_requested
        if (periodic or final or checkpoint_requested) and not args.no_save:
            save_checkpoint(
                output_dir=output_dir,
                step=completed,
                student=student,
                critic=critic.module,
                student_optimizer=student_optimizer,
                critic_optimizer=critic_optimizer,
                args=args,
                rank=rank,
                export_student=(periodic or final),
            )
            dist.barrier()
            logger.info("Saved exact checkpoint at step %07d", completed)
        if checkpoint_requested:
            if rank == 0 and args.checkpoint_request_file:
                try:
                    os.unlink(args.checkpoint_request_file)
                except FileNotFoundError:
                    pass
            dist.barrier()
            logger.info(
                "Checkpoint-only request completed; training continues: %s",
                args.checkpoint_request_file,
            )
        if stop_requested:
            logger.info("Job-scoped stop token observed: %s", args.stop_file)
            break

    logger.info("Done")
    dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--regression-data", required=True)
    parser.add_argument(
        "--vae-path",
        default="checkpoints/pretrained/stabilityai/sd-vae-ft-ema",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="DiT-L/2")
    parser.add_argument("--resolutions", default="32,16,8")
    parser.add_argument("--stage-lengths", default="300,300,400")
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument(
        "--batch-per-gpu",
        type=int,
        default=8,
        help="Per-rank microbatch; effective global batch also multiplies accumulation.",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--student-lr", type=float, default=1e-5)
    parser.add_argument("--critic-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--student-beta1", type=float, default=0.9)
    parser.add_argument("--student-beta2", type=float, default=0.999)
    parser.add_argument("--critic-beta1", type=float, default=0.0)
    parser.add_argument("--critic-beta2", type=float, default=0.999)
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--teacher-cfg", type=float, default=1.5)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--stop-file", default=None)
    parser.add_argument("--checkpoint-request-file", default=None)
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run the complete training path without writing a checkpoint (smoke tests only).",
    )
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
