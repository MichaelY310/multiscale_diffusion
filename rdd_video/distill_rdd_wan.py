"""Two-phase, stage-aware few-step distillation for RDD-Wan.

Phase A compresses a many-step RDD teacher trajectory into fixed student calls
that never cross a resolution boundary.  Phase B performs RDD-aware DMD2:
the fake-score model is trained on samples produced by the student and the
student follows the teacher-vs-fake score difference on the *RDD* re-noising
path.  The default inference budget is 2/2/1 calls.

This is deliberately separate from ``train_rdd_wan.py --data-mode distill``.
That legacy mode only generates terminal synthetic data and is not few-step
distillation.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import random
from copy import deepcopy
from pathlib import Path
from time import time
from typing import Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from flow_matching_video import RDDVideoSchedule
from lora_utils import inject_lora, load_lora_state_dict, lora_state_dict, trainable_parameters
from rdd_distillation import (
    DistillationCall,
    RDDDistillationSchedule,
    normalized_dmd_surrogate,
    parse_positive_ints,
    rdd_renoise,
    recover_clean_from_velocity,
    trajectory_distillation_loss,
)
from train_rdd_wan import (
    DummyLatentDataset,
    MultiResLatentDataset,
    WanModel,
    compute_seq_len,
    multires_collate_fn,
    parse_stage_configs,
)


def set_seed(seed: int, rank: int) -> None:
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_logger(directory: str, rank: int) -> logging.Logger:
    logger = logging.getLogger(f"rdd_distill_rank{rank}")
    logger.propagate = False
    logger.handlers.clear()
    if rank == 0:
        os.makedirs(directory, exist_ok=True)
        formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        file_handler = logging.FileHandler(os.path.join(directory, "log.txt"))
        file_handler.setFormatter(formatter)
        logger.addHandler(stream)
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
    else:
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.WARNING)
    return logger


def _strip_state_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state and all(key.startswith("module.") for key in state):
        return {key[len("module.") :]: value for key, value in state.items()}
    return state


def load_rdd_teacher(
    base_model_dir: str,
    teacher_checkpoint: Optional[str],
    *,
    device: torch.device,
    dtype: torch.dtype,
    allow_base_teacher: bool,
) -> WanModel:
    """Load official Wan weights, then overlay a converged RDD checkpoint."""

    if not os.path.isdir(base_model_dir):
        raise FileNotFoundError(f"Wan base model directory not found: {base_model_dir}")
    try:
        model = WanModel.from_pretrained(
            base_model_dir,
            torch_dtype=dtype,
            local_files_only=True,
        )
    except TypeError:
        model = WanModel.from_pretrained(base_model_dir, local_files_only=True)
    model = model.to(device=device, dtype=dtype)

    if teacher_checkpoint:
        checkpoint = torch.load(teacher_checkpoint, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            for key in ("ema", "model", "state_dict"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break
        if not isinstance(checkpoint, dict):
            raise TypeError("teacher checkpoint must contain a state dictionary")
        checkpoint = _strip_state_prefix(checkpoint)
        result = model.load_state_dict(checkpoint, strict=False)
        missing_non_lora = [key for key in result.missing_keys if "lora_" not in key]
        if result.unexpected_keys or missing_non_lora:
            raise RuntimeError(
                "RDD teacher checkpoint is not architecture-compatible: "
                f"missing={missing_non_lora[:20]}, unexpected={result.unexpected_keys[:20]}"
            )
    elif not allow_base_teacher:
        raise ValueError(
            "--teacher-checkpoint is required for a real RDD distillation run. "
            "Use --allow-base-teacher only for a plumbing smoke test."
        )
    return model


def wan_velocity(
    model: torch.nn.Module,
    x: torch.Tensor,
    data_t: torch.Tensor,
    context: torch.Tensor,
) -> torch.Tensor:
    """Call official Wan using its noise-time convention."""

    if data_t.ndim == 0:
        data_t = data_t.expand(x.shape[0])
    sigma_input = (1.0 - data_t.float()) * 1000.0
    _, _, frames, height, width = x.shape
    seq_len = compute_seq_len(frames, height, width, (1, 2, 2))
    result = model(
        [sample for sample in x],
        t=sigma_input,
        context=[item for item in context],
        seq_len=seq_len,
    )
    return torch.stack(result) if isinstance(result, list) else result


def cfg_teacher_velocity(
    teacher: torch.nn.Module,
    x: torch.Tensor,
    data_t: torch.Tensor,
    context: torch.Tensor,
    null_context: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    conditional = wan_velocity(teacher, x, data_t, context)
    if float(scale) == 1.0:
        return conditional
    unconditional = wan_velocity(teacher, x, data_t, null_context)
    return unconditional + float(scale) * (conditional - unconditional)


def prepare_batch(
    batch,
    data_mode: str,
    schedule: RDDVideoSchedule,
    device: torch.device,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    if data_mode == "multires":
        stage_latents, text = batch
        stage_latents = [item.to(device=device, dtype=torch.bfloat16, non_blocking=True) for item in stage_latents]
    else:
        full_latent, text = batch
        full_latent = full_latent.to(device=device, dtype=torch.bfloat16, non_blocking=True)
        stage_latents = [schedule.downsample_latent(full_latent, i) for i in range(schedule.num_stages)]
    text = text.to(device=device, dtype=torch.bfloat16, non_blocking=True)
    return stage_latents, text


def split_micro_batches(batch, data_mode: str, micro_batch_size: int):
    """Split one effective local batch without duplicating any sample."""

    if data_mode == "multires":
        stage_latents, text = batch
        batch_size = text.shape[0]
        return [
            ([latent[start : start + micro_batch_size] for latent in stage_latents],
             text[start : start + micro_batch_size])
            for start in range(0, batch_size, micro_batch_size)
        ]
    latent, text = batch
    batch_size = text.shape[0]
    return [
        (latent[start : start + micro_batch_size], text[start : start + micro_batch_size])
        for start in range(0, batch_size, micro_batch_size)
    ]


def _batch_time(value: float, x: torch.Tensor) -> torch.Tensor:
    return torch.full((x.shape[0],), float(value), device=x.device, dtype=x.dtype)


def paired_trajectory_loss(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    call: DistillationCall,
    schedule: RDDVideoSchedule,
    stage_latents: Sequence[torch.Tensor],
    context: torch.Tensor,
    null_context: torch.Tensor,
    *,
    cfg_scale: float,
    teacher_substeps: int,
    teacher_solver: str,
) -> torch.Tensor:
    fm = schedule.flow_matchers[call.stage_index]
    clean = stage_latents[call.stage_index]
    start = rdd_renoise(fm, clean, call.t_start)

    def student_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return wan_velocity(student, x, t, context)

    def teacher_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return cfg_teacher_velocity(teacher, x, t, context, null_context, cfg_scale)

    loss, _, _ = trajectory_distillation_loss(
        student_fn,
        teacher_fn,
        start,
        call,
        teacher_substeps=teacher_substeps,
        teacher_solver=teacher_solver,
    )
    return loss


def simulate_student_to_call(
    student: torch.nn.Module,
    target_call: DistillationCall,
    distilled_schedule: RDDDistillationSchedule,
    stage_latents: Sequence[torch.Tensor],
    context: torch.Tensor,
    *,
    with_grad: bool,
) -> torch.Tensor:
    """Replay earlier 2/2/1 calls detached, then predict clean at target call."""

    schedule = distilled_schedule.rdd_schedule
    x = torch.randn_like(stage_latents[0])
    for call in distilled_schedule.calls:
        if call.global_index > target_call.global_index:
            break
        fm = schedule.flow_matchers[call.stage_index]
        is_target = call.global_index == target_call.global_index
        grad_context = contextlib.nullcontext() if is_target and with_grad else torch.no_grad()
        with grad_context:
            t_batch = _batch_time(call.t_start, x)
            velocity = wan_velocity(student, x, t_batch, context)
            clean_prediction = recover_clean_from_velocity(fm, x, velocity, t_batch)
            if is_target:
                return clean_prediction
            x = x - call.dt * velocity

        stage_calls = distilled_schedule.calls_for_stage(call.stage_index)
        if call.index_in_stage == len(stage_calls) - 1:
            next_stage = call.stage_index + 1
            if next_stage < schedule.num_stages:
                target_shape = tuple(stage_latents[next_stage].shape[2:])
                x = schedule.upsample_with_noise(
                    clean_prediction,
                    from_stage=call.stage_index,
                    to_stage=next_stage,
                    target_shape=target_shape,
                )
    raise RuntimeError("target call was not reached")


def critic_flow_matching_loss(
    critic: torch.nn.Module,
    generated_x1: torch.Tensor,
    call: DistillationCall,
    schedule: RDDVideoSchedule,
    context: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    fm = schedule.flow_matchers[call.stage_index]
    batch = generated_x1.shape[0]
    margin = min(1e-4, 0.01 * call.dt)
    t = torch.rand(batch, device=generated_x1.device, dtype=generated_x1.dtype)
    t = t * (call.t_end - call.t_start - 2 * margin) + call.t_start + margin
    noise = torch.randn_like(generated_x1)
    drift = None if fm.c is None else fm.drift_scale * fm.compute_drift(generated_x1)
    noisy = fm.q_sample(generated_x1, t, noise=noise, d=drift)
    target = fm.compute_velocity_target(generated_x1, noise, t, d=drift)
    prediction = wan_velocity(critic, noisy, t, context)
    return F.mse_loss(prediction.float(), target.float()), t


def dmd_generator_loss(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    critic: torch.nn.Module,
    call: DistillationCall,
    distilled_schedule: RDDDistillationSchedule,
    stage_latents: Sequence[torch.Tensor],
    context: torch.Tensor,
    null_context: torch.Tensor,
    *,
    cfg_scale: float,
    path_correction: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    schedule = distilled_schedule.rdd_schedule
    fm = schedule.flow_matchers[call.stage_index]
    generated = simulate_student_to_call(
        student,
        call,
        distilled_schedule,
        stage_latents,
        context,
        with_grad=True,
    )
    batch = generated.shape[0]
    margin = min(1e-4, 0.01 * call.dt)
    score_t = torch.rand(batch, device=generated.device, dtype=generated.dtype)
    score_t = score_t * (call.t_end - call.t_start - 2 * margin) + call.t_start + margin
    noise = torch.randn_like(generated)
    noisy = rdd_renoise(fm, generated, score_t, noise)
    with torch.no_grad():
        teacher_velocity = cfg_teacher_velocity(
            teacher, noisy, score_t, context, null_context, cfg_scale
        )
        fake_velocity = wan_velocity(critic, noisy, score_t, context)
        teacher_x1 = recover_clean_from_velocity(fm, noisy, teacher_velocity, score_t)
        fake_x1 = recover_clean_from_velocity(fm, noisy, fake_velocity, score_t)
    loss, grad = normalized_dmd_surrogate(
        generated,
        teacher_x1,
        fake_x1,
        fm=fm,
        score_t=score_t,
        path_correction=path_correction,
    )
    return loss, grad


def teacher_preservation_loss(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    call: DistillationCall,
    schedule: RDDVideoSchedule,
    stage_latents: Sequence[torch.Tensor],
    context: torch.Tensor,
    null_context: torch.Tensor,
    *,
    cfg_scale: float,
) -> torch.Tensor:
    """Cheap supervised anchor used during DMD refinement (PyramidalWan Lteach role)."""

    fm = schedule.flow_matchers[call.stage_index]
    clean = stage_latents[call.stage_index]
    start = rdd_renoise(fm, clean, call.t_start)
    t = _batch_time(call.t_start, start)
    student_velocity = wan_velocity(student, start, t, context)
    with torch.no_grad():
        teacher_velocity = cfg_teacher_velocity(
            teacher, start, t, context, null_context, cfg_scale
        )
    return F.mse_loss(student_velocity.float(), teacher_velocity.float())


def save_checkpoint(
    path: str,
    student: DDP,
    critic: DDP,
    student_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
) -> None:
    payload = {
        "format": "rdd_wan_lora_dmd2_v1",
        "step": int(step),
        "student_lora": lora_state_dict(student.module),
        "critic_lora": lora_state_dict(critic.module),
        "student_optimizer": student_optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
        "config": vars(args),
    }
    torch.save(payload, path)


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Wan distillation requires CUDA; run CPU contract tests separately")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    set_seed(args.seed, rank)

    stage_configs = parse_stage_configs(args.stage_configs)
    stage_lengths = [int(part) for part in args.stage_lengths.split(",")]
    schedule = RDDVideoSchedule(
        stage_configs,
        stage_lengths,
        diffusion_steps=1000,
        downsample_mode="latent",
        drift_scale=args.drift_scale,
    )
    step_allocation = parse_positive_ints(args.student_steps, expected=schedule.num_stages)
    distilled_schedule = RDDDistillationSchedule(schedule, step_allocation)

    experiment_dir = os.path.join(args.results_dir, args.exp_name)
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
    dist.barrier()
    logger = create_logger(experiment_dir, rank)
    logger.info(
        "RDD-Wan distillation | boundaries=%s | student calls=%s | total NFE=%d",
        schedule.s_boundaries,
        step_allocation,
        len(distilled_schedule.calls),
    )

    teacher = load_rdd_teacher(
        args.base_model_dir,
        args.teacher_checkpoint,
        device=device,
        dtype=torch.bfloat16,
        allow_base_teacher=args.allow_base_teacher,
    )
    teacher.eval()
    teacher.requires_grad_(False)
    student = deepcopy(teacher)
    critic = deepcopy(teacher)
    student_report = inject_lora(
        student, rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout
    )
    critic_report = inject_lora(
        critic, rank=args.critic_lora_rank, alpha=args.critic_lora_alpha, dropout=0.0
    )
    student.train()
    critic.train()
    logger.info(
        "LoRA | student layers=%d trainable=%.2fM | critic layers=%d trainable=%.2fM",
        len(student_report.names),
        student_report.trainable_parameters / 1e6,
        len(critic_report.names),
        critic_report.trainable_parameters / 1e6,
    )

    student = DDP(student, device_ids=[local_rank], broadcast_buffers=False)
    critic = DDP(critic, device_ids=[local_rank], broadcast_buffers=False)
    student_optimizer = torch.optim.AdamW(
        trainable_parameters(student),
        lr=args.student_lr,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    critic_optimizer = torch.optim.AdamW(
        trainable_parameters(critic),
        lr=args.critic_lr,
        betas=(0.0, 0.999),
        weight_decay=0.0,
    )

    if args.data_mode == "multires":
        dataset = MultiResLatentDataset(
            args.data_dir,
            num_stages=schedule.num_stages,
            text_len=512,
            text_dim=4096,
            allow_missing_text=False,
        )
        collate_fn = multires_collate_fn
    else:
        dataset = DummyLatentDataset(
            num_samples=max(
                128,
                args.max_steps * args.micro_batch_size * args.grad_accum_steps * world_size,
            ),
            channels=16,
            frames=args.smoke_latent_frames,
            height=args.smoke_latent_height,
            width=args.smoke_latent_width,
        )
        collate_fn = None
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size * args.grad_accum_steps,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        load_lora_state_dict(student.module, checkpoint["student_lora"])
        load_lora_state_dict(critic.module, checkpoint["critic_lora"])
        student_optimizer.load_state_dict(checkpoint["student_optimizer"])
        critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        start_step = int(checkpoint["step"])
        logger.info("Resumed LoRA distillation at optimizer step %d", start_step)

    null_cache: dict[tuple[int, int, int], torch.Tensor] = {}
    optimizer_step = start_step
    epoch = 0
    wall_start = time()
    while optimizer_step < args.max_steps:
        sampler.set_epoch(epoch)
        for batch in loader:
            if optimizer_step >= args.max_steps:
                break
            phase_a = optimizer_step < args.trajectory_steps
            update_student = phase_a or optimizer_step % args.generator_update_interval == 0
            student_optimizer.zero_grad(set_to_none=True)
            critic_optimizer.zero_grad(set_to_none=True)
            aggregate = {"trajectory": 0.0, "dmd": 0.0, "critic": 0.0, "teach": 0.0}

            micro_batches = split_micro_batches(
                batch, args.data_mode, args.micro_batch_size
            )
            if len(micro_batches) != args.grad_accum_steps:
                raise RuntimeError(
                    f"expected {args.grad_accum_steps} microbatches, got "
                    f"{len(micro_batches)}"
                )

            for micro_index, micro_batch in enumerate(micro_batches):
                stage_latents, text_context = prepare_batch(
                    micro_batch, args.data_mode, schedule, device
                )
                cache_key = tuple(text_context.shape)
                null_context = null_cache.get(cache_key)
                if null_context is None:
                    null_context = torch.zeros_like(text_context)
                    null_cache[cache_key] = null_context
                call = distilled_schedule.calls[
                    (optimizer_step * world_size + rank + micro_index) % len(distilled_schedule.calls)
                ]

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if phase_a:
                        trajectory = paired_trajectory_loss(
                            student,
                            teacher,
                            call,
                            schedule,
                            stage_latents,
                            text_context,
                            null_context,
                            cfg_scale=args.teacher_cfg_scale,
                            teacher_substeps=args.teacher_substeps,
                            teacher_solver=args.teacher_solver,
                        )
                        (trajectory / args.grad_accum_steps).backward()
                        aggregate["trajectory"] += float(trajectory.detach())
                    else:
                        with torch.no_grad():
                            generated_for_critic = simulate_student_to_call(
                                student,
                                call,
                                distilled_schedule,
                                stage_latents,
                                text_context,
                                with_grad=False,
                            )
                        critic_loss, _ = critic_flow_matching_loss(
                            critic,
                            generated_for_critic.detach(),
                            call,
                            schedule,
                            text_context,
                        )
                        (critic_loss / args.grad_accum_steps).backward()
                        aggregate["critic"] += float(critic_loss.detach())

                        if update_student:
                            dmd_loss, _ = dmd_generator_loss(
                                student,
                                teacher,
                                critic,
                                call,
                                distilled_schedule,
                                stage_latents,
                                text_context,
                                null_context,
                                cfg_scale=args.teacher_cfg_scale,
                                path_correction=args.path_correction,
                            )
                            teach_loss = teacher_preservation_loss(
                                student,
                                teacher,
                                call,
                                schedule,
                                stage_latents,
                                text_context,
                                null_context,
                                cfg_scale=args.teacher_cfg_scale,
                            )
                            student_loss = dmd_loss + args.teacher_loss_weight * teach_loss
                            (student_loss / args.grad_accum_steps).backward()
                            aggregate["dmd"] += float(dmd_loss.detach())
                            aggregate["teach"] += float(teach_loss.detach())

            torch.nn.utils.clip_grad_norm_(list(trainable_parameters(critic)), args.grad_clip)
            if not phase_a:
                critic_optimizer.step()
            if update_student:
                torch.nn.utils.clip_grad_norm_(list(trainable_parameters(student)), args.grad_clip)
                student_optimizer.step()
            optimizer_step += 1

            if optimizer_step % args.log_every == 0 and rank == 0:
                elapsed = time() - wall_start
                logger.info(
                    "step=%d phase=%s call=%d(stage=%d,%d) trajectory=%.5f "
                    "dmd=%.5f critic=%.5f teach=%.5f sec/step=%.2f",
                    optimizer_step,
                    "trajectory" if phase_a else "dmd2",
                    call.global_index,
                    call.stage_index,
                    call.index_in_stage,
                    aggregate["trajectory"],
                    aggregate["dmd"],
                    aggregate["critic"],
                    aggregate["teach"],
                    elapsed / max(optimizer_step - start_step, 1),
                )
            if optimizer_step % args.ckpt_every == 0 or optimizer_step == args.max_steps:
                dist.barrier()
                if rank == 0:
                    save_checkpoint(
                        os.path.join(checkpoint_dir, f"{optimizer_step:07d}.pt"),
                        student,
                        critic,
                        student_optimizer,
                        critic_optimizer,
                        optimizer_step,
                        args,
                    )
                    logger.info("Saved LoRA checkpoint at step %d", optimizer_step)
                dist.barrier()
        epoch += 1
    dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RDD-aware few-step Wan distillation")
    parser.add_argument("--base-model-dir", required=True)
    parser.add_argument("--teacher-checkpoint", default=None)
    parser.add_argument("--allow-base-teacher", action="store_true")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--data-mode", choices=["multires", "dummy"], default="multires")
    parser.add_argument("--stage-configs", default="4,4,2;2,2,2;1,1,1")
    parser.add_argument("--stage-lengths", default="60,350,590")
    parser.add_argument("--student-steps", default="2,2,1")
    parser.add_argument("--drift-scale", type=float, default=1.0)
    parser.add_argument("--teacher-cfg-scale", type=float, default=5.0)
    parser.add_argument("--teacher-substeps", type=int, default=8)
    parser.add_argument("--teacher-solver", choices=["euler", "heun"], default="heun")
    parser.add_argument("--trajectory-steps", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--generator-update-interval", type=int, default=5)
    parser.add_argument("--teacher-loss-weight", type=float, default=0.01)
    parser.add_argument("--path-correction", choices=["exact", "simple"], default="exact")
    parser.add_argument("--lora-rank", type=int, default=128)
    parser.add_argument("--lora-alpha", type=float, default=128.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--critic-lora-rank", type=int, default=128)
    parser.add_argument("--critic-lora-alpha", type=float, default=128.0)
    parser.add_argument("--student-lr", type=float, default=1e-5)
    parser.add_argument("--critic-lr", type=float, default=1e-5)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", default="./results_rdd_video_distill")
    parser.add_argument("--exp-name", default="rdd_wan_dmd2_221")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--smoke-latent-frames", type=int, default=9)
    parser.add_argument("--smoke-latent-height", type=int, default=16)
    parser.add_argument("--smoke-latent-width", type=int, default=16)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
