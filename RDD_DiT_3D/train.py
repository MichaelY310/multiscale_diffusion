"""
RDD-DiT-3D Training Script (全稀疏版本).

Multi-stage training with DDP, EMA, and summed per-stage losses.
Supports: unified (single model), multi-model (per-stage independent models), decouple mode.
"""

import os
import time
import math
import argparse
import functools
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from copy import deepcopy

# TF32 for performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from model import DiT3D_RDD_models
from diffusion_3d import create_diffusion_3d
from dataset import SparseVolumeDataset, sparse_collate_fn


def get_lr(step, warmup_steps, max_lr, min_lr, total_steps):
    """Linear warmup + cosine decay learning rate schedule."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def create_argparser():
    parser = argparse.ArgumentParser(description="Train RDD-DiT-3D (Sparse)")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to precomputed sparse volume .pt files")
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--model", type=str, default="DiT3D-RDD-L",
                        choices=list(DiT3D_RDD_models.keys()))
    parser.add_argument("--in-channels", type=int, default=128)
    parser.add_argument("--num-classes", type=int, default=55)
    parser.add_argument("--base-size", type=int, default=8,
                        help="Base size for position embedding normalization")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Per-GPU batch size")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=500,
                        help="Linear warmup steps for LR scheduler")
    parser.add_argument("--min-lr", type=float, default=1e-6,
                        help="Minimum learning rate for cosine decay")
    parser.add_argument("--total-steps", type=int, default=50000,
                        help="Total training steps for cosine schedule planning")
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=False,
                        help="Enable gradient checkpointing to save memory")
    parser.add_argument("--amp", action="store_true", default=False,
                        help="Enable bf16 automatic mixed precision")
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--schedule", type=str, default="linear")
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--steps-per-stage", type=str, default="400,300,300",
                        help="Comma-separated steps per stage")
    parser.add_argument("--stage-resolutions", type=str, default="4,8,16",
                        help="Comma-separated resolutions for each stage")
    parser.add_argument("--class-dropout-prob", type=float, default=0.1,
                        help="Classifier-free guidance dropout prob (0 for unconditional)")
    # --- Mode selection ---
    parser.add_argument("--mode", type=str, default="unified",
                        choices=["unified", "multi-model", "decouple"],
                        help="Training mode")
    parser.add_argument("--model-per-stage", type=str, nargs="+", default=None,
                        help="Model names per stage for multi-model/decouple mode")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Filter samples whose max-resolution token count exceeds this (e.g. 27476)")
    return parser


def setup_distributed():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    return rank, world_size, device


def cleanup():
    dist.destroy_process_group()


@torch.no_grad()
def update_ema(ema_model, model, decay):
    ema_params = dict(ema_model.named_parameters())
    for name, param in model.named_parameters():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def _make_model_fn(model, coords, padding_mask, resolution):
    """Create a model_fn closure that binds coords, padding_mask, and resolution for diffusion calls."""
    def model_fn(x_t, t_global, y):
        return model(x_t, coords, t_global, y, padding_mask=padding_mask, resolution=resolution)
    return model_fn


def train_unified(args, rank, world_size, device, is_main):
    """Unified mode: single model handles all resolution stages, losses summed."""
    steps_per_stage = [int(x) for x in args.steps_per_stage.split(",")]
    stage_resolutions = [int(x) for x in args.stage_resolutions.split(",")]
    num_stages = len(steps_per_stage)

    # Model
    model = DiT3D_RDD_models[args.model](
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        base_size=args.base_size,
        class_dropout_prob=args.class_dropout_prob,
    ).to(device)

    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing = True
        if is_main:
            print("[INFO] Gradient checkpointing enabled")

    model = DDP(model, device_ids=[device])
    ema_model = deepcopy(model.module).to(device)
    ema_model.eval()

    # Diffusion (full-step for training)
    diffusions = create_diffusion_3d(
        schedule_name=args.schedule,
        steps_per_stage=steps_per_stage,
        diffusion_steps=args.diffusion_steps,
        stage_resolutions=stage_resolutions,
    )

    # Dataset & Loader
    max_tokens = getattr(args, 'max_tokens', None)
    dataset = SparseVolumeDataset(args.data_dir, max_tokens=max_tokens,
                                  stage_resolutions=stage_resolutions)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    collate_fn = functools.partial(sparse_collate_fn, stage_resolutions=stage_resolutions)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        num_workers=4, pin_memory=True, drop_last=True,
                        collate_fn=collate_fn)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # AMP scaler
    use_amp = args.amp
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    # Resume
    global_step = 0
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=f"cuda:{device}")
        model.module.load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        global_step = ckpt["global_step"]
        start_epoch = ckpt["epoch"]
        if is_main:
            print(f"Resumed from step {global_step}, epoch {start_epoch}")

    if is_main:
        os.makedirs(args.results_dir, exist_ok=True)
        print(f"[Unified] Training {args.model} | {num_stages} stages: "
              f"{list(zip(stage_resolutions, steps_per_stage))}")
        print(f"Dataset size: {len(dataset)}, Batch/GPU: {args.batch_size}")

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        model.train()

        for batch_idx, batch in enumerate(loader):
            t0 = time.time()
            labels = batch['label'].to(device)

            # Sum losses from all stages
            total_loss = 0.0
            stage_losses = []
            for stage_idx in range(num_stages):
                diff = diffusions[stage_idx]
                R_s = stage_resolutions[stage_idx]

                # Get this stage's data
                x_0 = batch[f'embeddings_{R_s}'].to(device)       # [B, N, C]
                coords = batch[f'coords_{R_s}'].to(device)         # [B, N, 3]
                pmask = batch[f'padding_mask_{R_s}'].to(device)    # [B, N]

                B = x_0.shape[0]
                t = torch.randint(0, diff.num_timesteps, (B,), device=device)

                # Create model_fn that binds coords/padding_mask/resolution
                model_fn = _make_model_fn(model, coords, pmask, R_s)
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                    losses = diff.training_losses(model_fn, x_0, t, labels, coords, padding_mask=pmask)
                    stage_loss = losses["loss"].mean()
                stage_losses.append(stage_loss.item())
                total_loss = total_loss + stage_loss

            total_loss = total_loss / num_stages

            # Gradient accumulation
            scaled_loss = total_loss / args.grad_accum_steps
            scaler.scale(scaled_loss).backward()

            if (batch_idx + 1) % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                update_ema(ema_model, model.module, args.ema_decay)

                global_step += 1

                # Update LR
                current_lr = get_lr(global_step, args.warmup_steps, args.lr, args.min_lr, args.total_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

                dt = time.time() - t0

                if is_main and global_step % args.log_every == 0:
                    stage_str = " | ".join([f"R{stage_resolutions[i]}={stage_losses[i]:.4f}" for i in range(num_stages)])
                    print(f"Step {global_step} | Epoch {epoch} | Loss {total_loss.item():.4f} | {stage_str} | LR {current_lr:.2e} | Time {dt*1000:.0f}ms")

                if is_main and global_step % args.ckpt_every == 0:
                    ckpt_path = os.path.join(args.results_dir, f"ckpt_{global_step:07d}.pt")
                    torch.save({
                        "model": model.module.state_dict(),
                        "ema": ema_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "global_step": global_step,
                        "epoch": epoch,
                        "args": vars(args),
                    }, ckpt_path)
                    print(f"Saved checkpoint: {ckpt_path}")


def train_multi_model(args, rank, world_size, device, is_main, decouple=False):
    """
    Multi-model / Decouple mode: independent model per stage, each with own optimizer/EMA.
    decouple=True: uses training_losses_decouple + d_prediction head (except coarsest stage).
    """
    steps_per_stage = [int(x) for x in args.steps_per_stage.split(",")]
    stage_resolutions = [int(x) for x in args.stage_resolutions.split(",")]
    num_stages = len(steps_per_stage)
    model_names = args.model_per_stage or [args.model] * num_stages

    # Create models, optimizers, EMAs per stage
    models = []
    optimizers = []
    ema_models = []
    for i in range(num_stages):
        use_d_pred = decouple and (i > 0)
        m = DiT3D_RDD_models[model_names[i]](
            in_channels=args.in_channels,
            num_classes=args.num_classes,
            base_size=args.base_size,
            d_prediction=use_d_pred,
            class_dropout_prob=args.class_dropout_prob,
        ).to(device)
        m = DDP(m, device_ids=[device])
        models.append(m)
        optimizers.append(torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=args.wd))
        ema = deepcopy(m.module).to(device)
        ema.eval()
        ema_models.append(ema)

    # Diffusion
    diffusions = create_diffusion_3d(
        schedule_name=args.schedule,
        steps_per_stage=steps_per_stage,
        diffusion_steps=args.diffusion_steps,
        stage_resolutions=stage_resolutions,
    )

    # Dataset
    dataset = SparseVolumeDataset(args.data_dir)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    collate_fn = functools.partial(sparse_collate_fn, stage_resolutions=stage_resolutions)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        num_workers=4, pin_memory=True, drop_last=True,
                        collate_fn=collate_fn)

    # Resume
    global_step = 0
    start_epoch = 0
    if args.resume and os.path.isdir(args.resume):
        for i in range(num_stages):
            R_s = stage_resolutions[i]
            ckpt_dir = os.path.join(args.resume, f"stage_{R_s}")
            latest = _find_latest_ckpt(ckpt_dir)
            if latest:
                ckpt = torch.load(latest, map_location=f"cuda:{device}")
                models[i].module.load_state_dict(ckpt["model"], strict=False)
                ema_models[i].load_state_dict(ckpt["ema"], strict=False)
                optimizers[i].load_state_dict(ckpt["optimizer"])
                global_step = max(global_step, ckpt.get("global_step", 0))
                start_epoch = max(start_epoch, ckpt.get("epoch", 0))
                if is_main:
                    print(f"Stage {i} (R={R_s}): resumed from {latest}")

    mode_name = "Decouple" if decouple else "Multi-Model"
    if is_main:
        os.makedirs(args.results_dir, exist_ok=True)
        print(f"[{mode_name}] Training {num_stages} models: {model_names}")
        print(f"Resolutions: {stage_resolutions}, Steps: {steps_per_stage}")
        print(f"Dataset size: {len(dataset)}, Batch/GPU: {args.batch_size}")

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for m in models:
            m.train()

        for batch_idx, batch in enumerate(loader):
            t0 = time.time()
            labels = batch['label'].to(device)

            total_loss_val = 0.0
            stage_losses = []
            for stage_idx in range(num_stages):
                diff = diffusions[stage_idx]
                R_s = stage_resolutions[stage_idx]
                model_s = models[stage_idx]
                opt_s = optimizers[stage_idx]

                x_0 = batch[f'embeddings_{R_s}'].to(device)
                coords = batch[f'coords_{R_s}'].to(device)
                pmask = batch[f'padding_mask_{R_s}'].to(device)

                B = x_0.shape[0]
                t = torch.randint(0, diff.num_timesteps, (B,), device=device)

                model_fn = _make_model_fn(model_s, coords, pmask, R_s)

                use_decouple_loss = decouple and (stage_idx > 0)
                if use_decouple_loss:
                    losses = diff.training_losses_decouple(model_fn, x_0, t, labels, coords, padding_mask=pmask)
                else:
                    losses = diff.training_losses(model_fn, x_0, t, labels, coords, padding_mask=pmask)

                loss = losses["loss"].mean()
                opt_s.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model_s.parameters(), 1.0)
                opt_s.step()
                update_ema(ema_models[stage_idx], model_s.module, args.ema_decay)
                stage_losses.append(loss.item())
                total_loss_val += loss.item()

            global_step += 1
            dt = time.time() - t0

            # Update LR for all optimizers
            current_lr = get_lr(global_step, args.warmup_steps, args.lr, args.min_lr, args.total_steps)
            for opt_s in optimizers:
                for param_group in opt_s.param_groups:
                    param_group['lr'] = current_lr

            if is_main and global_step % args.log_every == 0:
                stage_str = " | ".join([f"R{stage_resolutions[i]}={stage_losses[i]:.4f}" for i in range(num_stages)])
                print(f"Step {global_step} | Epoch {epoch} | Loss {total_loss_val / num_stages:.4f} | {stage_str} | LR {current_lr:.2e} | Time {dt*1000:.0f}ms")

            if is_main and global_step % args.ckpt_every == 0:
                for i in range(num_stages):
                    R_s = stage_resolutions[i]
                    stage_dir = os.path.join(args.results_dir, f"stage_{R_s}")
                    os.makedirs(stage_dir, exist_ok=True)
                    ckpt_path = os.path.join(stage_dir, f"ckpt_{global_step:07d}.pt")
                    torch.save({
                        "model": models[i].module.state_dict(),
                        "ema": ema_models[i].state_dict(),
                        "optimizer": optimizers[i].state_dict(),
                        "global_step": global_step,
                        "epoch": epoch,
                        "args": vars(args),
                    }, ckpt_path)
                print(f"Saved checkpoints at step {global_step}")


def _find_latest_ckpt(ckpt_dir):
    """Find the latest checkpoint .pt file in a directory."""
    if not os.path.isdir(ckpt_dir):
        return None
    pts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pt')])
    return os.path.join(ckpt_dir, pts[-1]) if pts else None


def main():
    args = create_argparser().parse_args()
    rank, world_size, device = setup_distributed()
    is_main = (rank == 0)

    if args.mode == "unified":
        train_unified(args, rank, world_size, device, is_main)
    elif args.mode == "multi-model":
        train_multi_model(args, rank, world_size, device, is_main, decouple=False)
    elif args.mode == "decouple":
        train_multi_model(args, rank, world_size, device, is_main, decouple=True)

    cleanup()


if __name__ == "__main__":
    main()
