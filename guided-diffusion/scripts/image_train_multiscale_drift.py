"""
Multi-scale UNet training with drift on pixel-space images.

Three UNet models, each responsible for a non-overlapping timestep sub-interval
of the global 1000-step diffusion schedule:

  Scale 0 (finest)  image_size=64  c=0     t ∈ [T0, T1) = [0,   300)  drift ON
  Scale 1 (medium)  image_size=32  c=1     t ∈ [T1, T2) = [300, 600)  drift ON
  Scale 2 (coarse)  image_size=16  c=None  t ∈ [T2, T3) = [600, 1000) drift OFF

The drift term d = upsample(downsample(x_0, 2), 2) - x_0 is the high-frequency
detail that is lost when the image is coarsened by 2×.

Usage (MPI, 4 GPUs 4-7):
    CUDA_VISIBLE_DEVICES=4,5,6,7 \\
    /usr/mpi/gcc/openmpi-4.1.5rc2/bin/mpiexec -n 4 -x CUDA_VISIBLE_DEVICES \\
        python scripts/image_train_multiscale_drift.py \\
            --data_dir /path/to/images \\
            --image_size 64 \\
            --batch_size 4
"""

import argparse
import copy
import os
import re

import torch as th
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.script_util import (
    create_model_for_drift,
    create_gaussian_diffusion_with_drift,
)


@th.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """EMA update: ema_model ← decay * ema_model + (1-decay) * model."""
    ema_params  = dict(ema_model.named_parameters())
    src_params  = dict(model.named_parameters())
    for name, param in src_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


# ---------------------------------------------------------------------------
# Timestep boundaries for the three sub-intervals
# ---------------------------------------------------------------------------
T0, T1, T2, T3 = 0, 300, 600, 1000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def downsample_image(x, target_size):
    """Average-pool [B, C, H, W] to [B, C, target_size, target_size]."""
    B, C, H, W = x.shape
    assert H == W and H % target_size == 0
    factor = H // target_size
    x = x.view(B, C, target_size, factor, target_size, factor)
    return x.mean(dim=(3, 5))


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    rank  = dist.get_rank()
    world = dist.get_world_size()
    device = dist_util.dev()

    logger.log("Creating models …")

    # ---- Scale 0: finest, image_size=64, drift ON ----
    model_0 = create_model_for_drift(
        image_size=args.image_size,
        num_channels=args.num_channels_0,
        num_res_blocks=args.num_res_blocks,
        learn_sigma=args.learn_sigma,
        class_cond=args.class_cond,
        use_checkpoint=args.use_checkpoint,
        attention_resolutions=args.attention_resolutions_0,
        num_heads=args.num_heads,
        use_scale_shift_norm=args.use_scale_shift_norm,
        dropout=args.dropout,
        use_fp16=args.use_fp16,
    ).to(device)
    ema_0 = copy.deepcopy(model_0)
    requires_grad(ema_0, False)
    ema_0.eval()
    ddp_0 = DDP(model_0, device_ids=[device], output_device=device,
                broadcast_buffers=False, find_unused_parameters=False)
    diffusion_0 = create_gaussian_diffusion_with_drift(
        steps=T3,
        learn_sigma=args.learn_sigma,
        noise_schedule=args.noise_schedule,
        c=0,
        T_c=T0,
        T_c_plus_1=T1,
        sample_scale=args.sample_scale,
    )

    # ---- Scale 1: medium, image_size = image_size/2, drift ON ----
    image_size_1 = args.image_size // 2
    model_1 = create_model_for_drift(
        image_size=image_size_1,
        num_channels=args.num_channels_1,
        num_res_blocks=args.num_res_blocks,
        learn_sigma=args.learn_sigma,
        class_cond=args.class_cond,
        use_checkpoint=args.use_checkpoint,
        attention_resolutions=args.attention_resolutions_1,
        num_heads=args.num_heads,
        use_scale_shift_norm=args.use_scale_shift_norm,
        dropout=args.dropout,
        use_fp16=args.use_fp16,
    ).to(device)
    ema_1 = copy.deepcopy(model_1)
    requires_grad(ema_1, False)
    ema_1.eval()
    ddp_1 = DDP(model_1, device_ids=[device], output_device=device,
                broadcast_buffers=False, find_unused_parameters=False)
    diffusion_1 = create_gaussian_diffusion_with_drift(
        steps=T3,
        learn_sigma=args.learn_sigma,
        noise_schedule=args.noise_schedule,
        c=1,
        T_c=T1,
        T_c_plus_1=T2,
        sample_scale=args.sample_scale,
    )

    # ---- Scale 2: coarsest, image_size = image_size/4, drift OFF ----
    image_size_2 = args.image_size // 4
    model_2 = create_model_for_drift(
        image_size=image_size_2,
        num_channels=args.num_channels_2,
        num_res_blocks=args.num_res_blocks,
        learn_sigma=args.learn_sigma,
        class_cond=args.class_cond,
        use_checkpoint=args.use_checkpoint,
        attention_resolutions=args.attention_resolutions_2,
        num_heads=args.num_heads,
        use_scale_shift_norm=args.use_scale_shift_norm,
        dropout=args.dropout,
        use_fp16=args.use_fp16,
    ).to(device)
    ema_2 = copy.deepcopy(model_2)
    requires_grad(ema_2, False)
    ema_2.eval()
    ddp_2 = DDP(model_2, device_ids=[device], output_device=device,
                broadcast_buffers=False, find_unused_parameters=False)
    diffusion_2 = create_gaussian_diffusion_with_drift(
        steps=T3,
        learn_sigma=args.learn_sigma,
        noise_schedule=args.noise_schedule,
        c=None,
        T_c=T2,
        T_c_plus_1=T3,
        sample_scale=args.sample_scale,
    )

    logger.log(f"Scale-0 params: {sum(p.numel() for p in model_0.parameters()):,}")
    logger.log(f"Scale-1 params: {sum(p.numel() for p in model_1.parameters()):,}")
    logger.log(f"Scale-2 params: {sum(p.numel() for p in model_2.parameters()):,}")

    # ---- Optimisers ----
    opt_0 = th.optim.AdamW(model_0.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_1 = th.optim.AdamW(model_1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_2 = th.optim.AdamW(model_2.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---- Initialise EMA weights ----
    update_ema(ema_0, model_0, decay=0)
    update_ema(ema_1, model_1, decay=0)
    update_ema(ema_2, model_2, decay=0)

    model_0.train(); model_1.train(); model_2.train()

    # ---- Optional checkpoint resume ----
    train_steps = 0
    if args.resume_0 and args.resume_1 and args.resume_2:
        logger.log("Resuming from checkpoints …")
        for ckpt_path, model, ema, opt, label in [
            (args.resume_0, model_0, ema_0, opt_0, "scale-0"),
            (args.resume_1, model_1, ema_1, opt_1, "scale-1"),
            (args.resume_2, model_2, ema_2, opt_2, "scale-2"),
        ]:
            logger.log(f"  {label}: {ckpt_path}")
            ckpt = th.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            ema.load_state_dict(ckpt["ema"])
            opt.load_state_dict(ckpt["opt"])
        match = re.search(r"(\d+)\.pt$", args.resume_0)
        if match:
            train_steps = int(match.group(1))
            logger.log(f"Resuming from step {train_steps}")
        dist.barrier()

    # ---- Checkpoint directories (rank-0 only) ----
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        from glob import glob
        exp_idx = len(glob(f"{args.results_dir}/*"))
        exp_dir = f"{args.results_dir}/{exp_idx:03d}-multiscale-drift"
        ckpt_dir_0 = f"{exp_dir}/checkpoints_scale0_{args.image_size}x{args.image_size}"
        ckpt_dir_1 = f"{exp_dir}/checkpoints_scale1_{image_size_1}x{image_size_1}"
        ckpt_dir_2 = f"{exp_dir}/checkpoints_scale2_{image_size_2}x{image_size_2}"
        os.makedirs(ckpt_dir_0, exist_ok=True)
        os.makedirs(ckpt_dir_1, exist_ok=True)
        os.makedirs(ckpt_dir_2, exist_ok=True)
    else:
        ckpt_dir_0 = ckpt_dir_1 = ckpt_dir_2 = None
    dist.barrier()

    # ---- Data loader ----
    logger.log("Creating data loader …")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    # ---- Training loop ----
    logger.log("Training …")
    running_loss_0 = running_loss_1 = running_loss_2 = 0.0
    log_steps = 0

    import time
    start_time = time.time()

    while True:
        x, model_kwargs = next(data)
        x = x.to(device)
        # model_kwargs contains {"y": labels} when class_cond=True, else {}

        # ---- Scale 0: full resolution, t ∈ [0, T1) ----
        t_0 = th.randint(0, T1 - T0, (x.shape[0],), device=device)
        loss_dict_0 = diffusion_0.training_losses(ddp_0, x, t_0, model_kwargs)
        loss_0 = loss_dict_0["loss"].mean()
        opt_0.zero_grad()
        loss_0.backward()
        opt_0.step()
        update_ema(ema_0, model_0)

        # ---- Scale 1: half resolution, t ∈ [0, T2-T1) ----
        x_1 = downsample_image(x.detach(), image_size_1)
        t_1 = th.randint(0, T2 - T1, (x_1.shape[0],), device=device)
        loss_dict_1 = diffusion_1.training_losses(ddp_1, x_1, t_1, model_kwargs)
        loss_1 = loss_dict_1["loss"].mean()
        opt_1.zero_grad()
        loss_1.backward()
        opt_1.step()
        update_ema(ema_1, model_1)

        # ---- Scale 2: quarter resolution, t ∈ [0, T3-T2) ----
        x_2 = downsample_image(x.detach(), image_size_2)
        t_2 = th.randint(0, T3 - T2, (x_2.shape[0],), device=device)
        loss_dict_2 = diffusion_2.training_losses(ddp_2, x_2, t_2, model_kwargs)
        loss_2 = loss_dict_2["loss"].mean()
        opt_2.zero_grad()
        loss_2.backward()
        opt_2.step()
        update_ema(ema_2, model_2)

        running_loss_0 += loss_0.item()
        running_loss_1 += loss_1.item()
        running_loss_2 += loss_2.item()
        log_steps   += 1
        train_steps += 1

        # ---- Logging ----
        if train_steps % args.log_interval == 0:
            th.cuda.synchronize()
            elapsed = time.time() - start_time
            steps_per_sec = log_steps / elapsed

            avg_0 = th.tensor(running_loss_0 / log_steps, device=device)
            avg_1 = th.tensor(running_loss_1 / log_steps, device=device)
            avg_2 = th.tensor(running_loss_2 / log_steps, device=device)
            dist.all_reduce(avg_0, op=dist.ReduceOp.SUM)
            dist.all_reduce(avg_1, op=dist.ReduceOp.SUM)
            dist.all_reduce(avg_2, op=dist.ReduceOp.SUM)
            avg_0 = avg_0.item() / world
            avg_1 = avg_1.item() / world
            avg_2 = avg_2.item() / world

            logger.log(
                f"step={train_steps:07d}  "
                f"loss[scale0]={avg_0:.4f}  "
                f"loss[scale1]={avg_1:.4f}  "
                f"loss[scale2]={avg_2:.4f}  "
                f"steps/s={steps_per_sec:.2f}"
            )

            running_loss_0 = running_loss_1 = running_loss_2 = 0.0
            log_steps = 0
            start_time = time.time()

        # ---- Checkpointing ----
        if train_steps % args.save_interval == 0 and train_steps > 0:
            if rank == 0:
                for ckpt_dir, model, ema, opt, scale_meta in [
                    (ckpt_dir_0, model_0, ema_0, opt_0,
                     {"scale": 0, "image_size": args.image_size,
                      "c": 0, "T_c": T0, "T_c_plus_1": T1}),
                    (ckpt_dir_1, model_1, ema_1, opt_1,
                     {"scale": 1, "image_size": image_size_1,
                      "c": 1, "T_c": T1, "T_c_plus_1": T2}),
                    (ckpt_dir_2, model_2, ema_2, opt_2,
                     {"scale": 2, "image_size": image_size_2,
                      "c": None, "T_c": T2, "T_c_plus_1": T3}),
                ]:
                    th.save(
                        {
                            "model": model.state_dict(),
                            "ema":   ema.state_dict(),
                            "opt":   opt.state_dict(),
                            "args":  args,
                            **scale_meta,
                        },
                        f"{ckpt_dir}/{train_steps:07d}.pt",
                    )
                logger.log(f"Saved checkpoints at step {train_steps:07d}")
            dist.barrier()

        if args.lr_anneal_steps > 0 and train_steps >= args.lr_anneal_steps:
            logger.log("Reached lr_anneal_steps, stopping.")
            break

    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def create_argparser():
    parser = argparse.ArgumentParser(
        description=(
            "Train three UNet models on pixel-space images using the "
            "multi-scale diffusion-with-drift method."
        )
    )
    # Data
    parser.add_argument("--data_dir",        type=str,   required=True)
    parser.add_argument("--image_size",      type=int,   default=64,
                        help="Spatial size for scale-0 (finest) model. "
                             "Scales 1 and 2 use image_size/2 and image_size/4.")
    parser.add_argument("--class_cond",      type=_str2bool, default=False)

    # Training
    parser.add_argument("--batch_size",      type=int,   default=4)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--weight_decay",    type=float, default=0.0)
    parser.add_argument("--lr_anneal_steps", type=int,   default=0,
                        help="Stop after this many steps (0 = run forever).")
    parser.add_argument("--ema_rate",        type=float, default=0.9999)
    parser.add_argument("--log_interval",    type=int,   default=100)
    parser.add_argument("--save_interval",   type=int,   default=10000)
    parser.add_argument("--results_dir",     type=str,   default="results_multiscale_drift")

    # Architecture (shared across scales)
    parser.add_argument("--num_res_blocks",  type=int,   default=2)
    parser.add_argument("--num_heads",       type=int,   default=4)
    parser.add_argument("--dropout",         type=float, default=0.0)
    parser.add_argument("--use_scale_shift_norm", type=_str2bool, default=True)
    parser.add_argument("--use_checkpoint",  type=_str2bool, default=False)
    parser.add_argument("--use_fp16",        type=_str2bool, default=False)

    # Per-scale channel widths and attention (sensible defaults)
    parser.add_argument("--num_channels_0",  type=int,   default=128,
                        help="Model channels for scale-0 (finest).")
    parser.add_argument("--num_channels_1",  type=int,   default=128,
                        help="Model channels for scale-1 (medium).")
    parser.add_argument("--num_channels_2",  type=int,   default=128,
                        help="Model channels for scale-2 (coarsest).")
    parser.add_argument("--attention_resolutions_0", type=str, default="16,8")
    parser.add_argument("--attention_resolutions_1", type=str, default="16,8")
    parser.add_argument("--attention_resolutions_2", type=str, default="8,4")

    # Diffusion
    parser.add_argument("--noise_schedule",  type=str,   default="linear")
    parser.add_argument("--learn_sigma",     type=_str2bool, default=False)
    parser.add_argument("--sample_scale",    type=int,   default=2,
                        help="Spatial downsampling factor used to compute drift d.")

    # Checkpoint resume (all three must be given together)
    parser.add_argument("--resume_0",        type=str,   default=None)
    parser.add_argument("--resume_1",        type=str,   default=None)
    parser.add_argument("--resume_2",        type=str,   default=None)

    return parser


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("boolean value expected")


if __name__ == "__main__":
    main()
