"""
Multi-scale UNet training with drift on ImageNet-1k-256×256.

Default mode uses three UNet models, each responsible for a non-overlapping
sub-interval of the global 1000-step diffusion schedule:

  256×256  c=0    t ∈ [T_0, T_1) = [0,   300)  drift ON
  128×128  c=1    t ∈ [T_1, T_2) = [300, 600)  drift ON
   64×64   c=None t ∈ [T_2, T_3) = [600, 1000) drift OFF

The drift term  d = upsample(downsample(x_0, 2), 2) – x_0  captures the
high-frequency detail removed when the image is spatially coarsened by 2×.
The coarsest model (64×64, c=None) uses a standard DDPM objective.

Dataset: ImageNet-1k-256×256 streamed from HuggingFace (same cache as the
DiT training script):  benjamin-paine/imagenet-1k-256x256
An optional single-model mode shares the 256×256 UNet and adds adapter paths
for the 128×128 and 64×64 stages.
All modes are class-conditional (num_classes=1000).

Usage (MPI, 4 GPUs 4-7):
    CUDA_VISIBLE_DEVICES=4,5,6,7 \\
    /usr/mpi/gcc/openmpi-4.1.5rc2/bin/mpiexec -n 4 -x CUDA_VISIBLE_DEVICES \\
        python scripts/image_train_with_drift_imagenet256.py
"""

import copy
import os
import re
import time
from glob import glob

import torch as th
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    create_model_for_drift,
    create_gaussian_diffusion_with_drift,
)

th.backends.cuda.matmul.allow_tf32 = True
th.backends.cudnn.allow_tf32 = True

# ---------------------------------------------------------------------------
# Timestep boundaries (same as train_with_drift_imagenet256.py)
# ---------------------------------------------------------------------------
T_0, T_1, T_2, T_3 = 0, 300, 600, 1000


# ---------------------------------------------------------------------------
# ImageNet HuggingFace Dataset  (mirrors the DiT training script)
# ---------------------------------------------------------------------------

class ImageNetHFDataset(Dataset):
    """
    Wraps the HuggingFace `benjamin-paine/imagenet-1k-256x256` dataset.
    Images are returned as (C, H, W) float tensors normalised to [-1, 1].
    Labels are int class indices in [0, 999].
    """

    def __init__(self, image_size: int = 256):
        from datasets import load_dataset
        self.hf_dataset = load_dataset(
            "benjamin-paine/imagenet-1k-256x256",
            split="train",
        )
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item["image"]
        label = item["label"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.transform(image), int(label)


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


@th.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = dict(ema_model.named_parameters())
    src_params = dict(model.named_parameters())
    for name, param in src_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    dist_util.setup_dist()
    logger.configure()

    rank  = dist.get_rank()
    world = dist.get_world_size()
    device = dist_util.dev()

    logger.log("Creating models …")
    image_size_128 = args.image_size // 2
    image_size_64 = args.image_size // 4

    if args.use_single_model:
        shared_model = create_model_for_drift(
            image_size=args.image_size,
            num_channels=args.num_channels_256,
            num_res_blocks=args.num_res_blocks,
            learn_sigma=args.learn_sigma,
            class_cond=True,
            use_checkpoint=args.use_checkpoint,
            attention_resolutions=args.attention_resolutions_256,
            num_heads=args.num_heads,
            use_scale_shift_norm=args.use_scale_shift_norm,
            dropout=args.dropout,
            use_fp16=args.use_fp16,
            use_single_model=True,
            sample_scale=args.sample_scale,
        ).to(device)
        ema_shared = copy.deepcopy(shared_model)
        requires_grad(ema_shared, False)
        ema_shared.eval()
        ddp_shared = DDP(
            shared_model,
            device_ids=[device],
            output_device=device,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        logger.log(f"Shared RDD UNet params : {sum(p.numel() for p in shared_model.parameters()):,}")
        opt_shared = th.optim.AdamW(shared_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        update_ema(ema_shared, shared_model, decay=0)
        shared_model.train()
    else:
        # ---- Scale 0: finest, 256×256, drift ON (c=0, t ∈ [T_0, T_1)) ----
        model_256 = create_model_for_drift(
            image_size=args.image_size,
            num_channels=args.num_channels_256,
            num_res_blocks=args.num_res_blocks,
            learn_sigma=args.learn_sigma,
            class_cond=True,
            use_checkpoint=args.use_checkpoint,
            attention_resolutions=args.attention_resolutions_256,
            num_heads=args.num_heads,
            use_scale_shift_norm=args.use_scale_shift_norm,
            dropout=args.dropout,
            use_fp16=args.use_fp16,
        ).to(device)
        ema_256 = copy.deepcopy(model_256)
        requires_grad(ema_256, False)
        ema_256.eval()
        ddp_256 = DDP(model_256, device_ids=[device], output_device=device,
                      broadcast_buffers=False, find_unused_parameters=False)

        # ---- Scale 1: medium, 128×128, drift ON (c=1, t ∈ [T_1, T_2)) ----
        model_128 = create_model_for_drift(
            image_size=image_size_128,
            num_channels=args.num_channels_128,
            num_res_blocks=args.num_res_blocks,
            learn_sigma=args.learn_sigma,
            class_cond=True,
            use_checkpoint=args.use_checkpoint,
            attention_resolutions=args.attention_resolutions_128,
            num_heads=args.num_heads,
            use_scale_shift_norm=args.use_scale_shift_norm,
            dropout=args.dropout,
            use_fp16=args.use_fp16,
        ).to(device)
        ema_128 = copy.deepcopy(model_128)
        requires_grad(ema_128, False)
        ema_128.eval()
        ddp_128 = DDP(model_128, device_ids=[device], output_device=device,
                      broadcast_buffers=False, find_unused_parameters=False)

        # ---- Scale 2: coarsest, 64×64, drift OFF (c=None, t ∈ [T_2, T_3)) ----
        model_64 = create_model_for_drift(
            image_size=image_size_64,
            num_channels=args.num_channels_64,
            num_res_blocks=args.num_res_blocks,
            learn_sigma=args.learn_sigma,
            class_cond=True,
            use_checkpoint=args.use_checkpoint,
            attention_resolutions=args.attention_resolutions_64,
            num_heads=args.num_heads,
            use_scale_shift_norm=args.use_scale_shift_norm,
            dropout=args.dropout,
            use_fp16=args.use_fp16,
        ).to(device)
        ema_64 = copy.deepcopy(model_64)
        requires_grad(ema_64, False)
        ema_64.eval()
        ddp_64 = DDP(model_64, device_ids=[device], output_device=device,
                     broadcast_buffers=False, find_unused_parameters=False)

        logger.log(f"UNet 256×256 params : {sum(p.numel() for p in model_256.parameters()):,}")
        logger.log(f"UNet 128×128 params : {sum(p.numel() for p in model_128.parameters()):,}")
        logger.log(f"UNet  64×64  params : {sum(p.numel() for p in model_64.parameters()):,}")

        opt_256 = th.optim.AdamW(model_256.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        opt_128 = th.optim.AdamW(model_128.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        opt_64  = th.optim.AdamW(model_64.parameters(),  lr=args.lr, weight_decay=args.weight_decay)

        update_ema(ema_256, model_256, decay=0)
        update_ema(ema_128, model_128, decay=0)
        update_ema(ema_64,  model_64,  decay=0)

        model_256.train(); model_128.train(); model_64.train()

    diffusion_256 = create_gaussian_diffusion_with_drift(
        steps=T_3,
        learn_sigma=args.learn_sigma,
        noise_schedule=args.noise_schedule,
        c=0,
        T_c=T_0,
        T_c_plus_1=T_1,
        sample_scale=args.sample_scale,
    )
    diffusion_128 = create_gaussian_diffusion_with_drift(
        steps=T_3,
        learn_sigma=args.learn_sigma,
        noise_schedule=args.noise_schedule,
        c=1,
        T_c=T_1,
        T_c_plus_1=T_2,
        sample_scale=args.sample_scale,
    )
    diffusion_64 = create_gaussian_diffusion_with_drift(
        steps=T_3,
        learn_sigma=args.learn_sigma,
        noise_schedule=args.noise_schedule,
        c=None,
        T_c=T_2,
        T_c_plus_1=T_3,
        sample_scale=args.sample_scale,
    )

    # ---- Checkpoint directories (rank-0 only) ----
    train_steps = 0

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        exp_idx  = len(glob(f"{args.results_dir}/*"))
        exp_dir  = f"{args.results_dir}/{exp_idx:03d}-imagenet256-multiscale-drift"
        if args.use_single_model:
            ckpt_dir_shared = f"{exp_dir}/checkpoints_shared"
            os.makedirs(ckpt_dir_shared, exist_ok=True)
        else:
            ckpt_dir_256 = f"{exp_dir}/checkpoints_256x256"
            ckpt_dir_128 = f"{exp_dir}/checkpoints_128x128"
            ckpt_dir_64  = f"{exp_dir}/checkpoints_64x64"
            os.makedirs(ckpt_dir_256, exist_ok=True)
            os.makedirs(ckpt_dir_128, exist_ok=True)
            os.makedirs(ckpt_dir_64,  exist_ok=True)
        logger.log(f"Experiment directory: {exp_dir}")
    else:
        if args.use_single_model:
            ckpt_dir_shared = None
        else:
            ckpt_dir_256 = ckpt_dir_128 = ckpt_dir_64 = None
    dist.barrier()

    # ---- Optional checkpoint resume ----
    if args.use_single_model:
        if args.resume_shared:
            logger.log("Resuming shared model checkpoint:")
            logger.log(f"  shared : {args.resume_shared}")
            ckpt = th.load(args.resume_shared, map_location=device, weights_only=False)
            shared_model.load_state_dict(ckpt["model"])
            ema_shared.load_state_dict(ckpt["ema"])
            opt_shared.load_state_dict(ckpt["opt"])
            match = re.search(r"(\d+)\.pt$", args.resume_shared)
            if match:
                train_steps = int(match.group(1))
                logger.log(f"Resuming from step {train_steps}")
            dist.barrier()
    else:
        if args.resume_256 and args.resume_128 and args.resume_64:
            logger.log("Resuming from checkpoints:")
            logger.log(f"  256×256 : {args.resume_256}")
            logger.log(f"  128×128 : {args.resume_128}")
            logger.log(f"   64×64  : {args.resume_64}")
            for ckpt_path, model, ema, opt in [
                (args.resume_256, model_256, ema_256, opt_256),
                (args.resume_128, model_128, ema_128, opt_128),
                (args.resume_64,  model_64,  ema_64,  opt_64),
            ]:
                ckpt = th.load(ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model"])
                ema.load_state_dict(ckpt["ema"])
                opt.load_state_dict(ckpt["opt"])
            match = re.search(r"(\d+)\.pt$", args.resume_256)
            if match:
                train_steps = int(match.group(1))
                logger.log(f"Resuming from step {train_steps}")
            dist.barrier()

    # ---- Data loader ----
    logger.log("Creating data loader …")
    dataset = ImageNetHFDataset(image_size=args.image_size)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world,
        rank=rank,
        shuffle=True,
        seed=args.global_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    loader_iter = iter(loader)

    # ---- Training loop ----
    logger.log(f"Training for {args.total_steps} steps …")
    running_loss_256 = running_loss_128 = running_loss_64 = 0.0
    log_steps = 0
    start_time = time.time()
    epoch = 0

    while train_steps < args.total_steps:
        # Restart loader at epoch boundary
        try:
            x, y = next(loader_iter)
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            loader_iter = iter(loader)
            x, y = next(loader_iter)

        x = x.to(device)
        y = y.to(device)

        t_256 = th.randint(0, T_1 - T_0, (x.shape[0],), device=device)
        x_128 = downsample_image(x.detach(), image_size_128)
        t_128 = th.randint(0, T_2 - T_1, (x_128.shape[0],), device=device)
        x_64 = downsample_image(x.detach(), image_size_64)
        t_64 = th.randint(0, T_3 - T_2, (x_64.shape[0],), device=device)

        if args.use_single_model:
            loss_256 = diffusion_256.training_losses(
                ddp_shared, x, t_256, {"y": y, "rdd_stage": "fine"}
            )["loss"].mean()
            loss_128 = diffusion_128.training_losses(
                ddp_shared, x_128, t_128, {"y": y, "rdd_stage": "mid"}
            )["loss"].mean()
            loss_64 = diffusion_64.training_losses(
                ddp_shared, x_64, t_64, {"y": y, "rdd_stage": "coarse"}
            )["loss"].mean()
            total_loss = loss_256 + loss_128 + loss_64
            opt_shared.zero_grad()
            total_loss.backward()
            opt_shared.step()
            update_ema(ema_shared, shared_model)
        else:
            model_kwargs = dict(y=y)

            loss_256 = diffusion_256.training_losses(ddp_256, x, t_256, model_kwargs)["loss"].mean()
            opt_256.zero_grad()
            loss_256.backward()
            opt_256.step()
            update_ema(ema_256, model_256)

            loss_128 = diffusion_128.training_losses(ddp_128, x_128, t_128, model_kwargs)["loss"].mean()
            opt_128.zero_grad()
            loss_128.backward()
            opt_128.step()
            update_ema(ema_128, model_128)

            loss_64 = diffusion_64.training_losses(ddp_64, x_64, t_64, model_kwargs)["loss"].mean()
            opt_64.zero_grad()
            loss_64.backward()
            opt_64.step()
            update_ema(ema_64, model_64)

        running_loss_256 += loss_256.item()
        running_loss_128 += loss_128.item()
        running_loss_64  += loss_64.item()
        log_steps   += 1
        train_steps += 1

        # ---- Logging ----
        if train_steps % args.log_every == 0:
            th.cuda.synchronize()
            end_time = time.time()
            steps_per_sec = log_steps / (end_time - start_time)

            avg_256 = th.tensor(running_loss_256 / log_steps, device=device)
            avg_128 = th.tensor(running_loss_128 / log_steps, device=device)
            avg_64  = th.tensor(running_loss_64  / log_steps, device=device)
            dist.all_reduce(avg_256, op=dist.ReduceOp.SUM)
            dist.all_reduce(avg_128, op=dist.ReduceOp.SUM)
            dist.all_reduce(avg_64,  op=dist.ReduceOp.SUM)
            avg_256 = avg_256.item() / world
            avg_128 = avg_128.item() / world
            avg_64  = avg_64.item()  / world

            logger.log(
                f"(step={train_steps:07d}) "
                f"Loss[256x256]={avg_256:.4f}  "
                f"Loss[128x128]={avg_128:.4f}  "
                f"Loss[64x64]={avg_64:.4f}  "
                f"Steps/s={steps_per_sec:.2f}"
            )

            running_loss_256 = running_loss_128 = running_loss_64 = 0.0
            log_steps  = 0
            start_time = time.time()

        # ---- Checkpointing ----
        if train_steps % args.ckpt_every == 0 and train_steps > 0:
            if rank == 0:
                if args.use_single_model:
                    th.save(
                        {
                            "model": shared_model.state_dict(),
                            "ema":   ema_shared.state_dict(),
                            "opt":   opt_shared.state_dict(),
                            "args":  args,
                            "resolution": "shared",
                            "mode": "single_model_adapter",
                            "T": [T_0, T_1, T_2, T_3],
                        },
                        f"{ckpt_dir_shared}/{train_steps:07d}.pt",
                    )
                else:
                    th.save(
                        {
                            "model": model_256.state_dict(),
                            "ema":   ema_256.state_dict(),
                            "opt":   opt_256.state_dict(),
                            "args":  args,
                            "resolution": "256x256",
                            "c": 0, "T_c": T_0, "T_c_plus_1": T_1,
                        },
                        f"{ckpt_dir_256}/{train_steps:07d}.pt",
                    )
                    th.save(
                        {
                            "model": model_128.state_dict(),
                            "ema":   ema_128.state_dict(),
                            "opt":   opt_128.state_dict(),
                            "args":  args,
                            "resolution": "128x128",
                            "c": 1, "T_c": T_1, "T_c_plus_1": T_2,
                        },
                        f"{ckpt_dir_128}/{train_steps:07d}.pt",
                    )
                    th.save(
                        {
                            "model": model_64.state_dict(),
                            "ema":   ema_64.state_dict(),
                            "opt":   opt_64.state_dict(),
                            "args":  args,
                            "resolution": "64x64",
                            "c": None, "T_c": T_2, "T_c_plus_1": T_3,
                        },
                        f"{ckpt_dir_64}/{train_steps:07d}.pt",
                    )
                logger.log(f"Saved checkpoints at step {train_steps:07d}")
            dist.barrier()

    logger.log("Done!")
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def create_argparser():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Multi-scale UNet training on ImageNet-1k-256×256 with drift.\n"
            "  default: three separate UNets for 256/128/64\n"
            "  optional: one shared 256×256 UNet + adapters for 128/64"
        )
    )

    # Data
    parser.add_argument("--image_size",  type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--global_seed", type=int, default=0)

    # Output
    parser.add_argument("--results_dir", type=str,
                        default="guided_diffusion/results_imagenet256",
                        help="Root directory for experiment folders and checkpoints.")

    # Training schedule
    parser.add_argument("--total_steps",  type=int,   default=1_000_000)
    parser.add_argument("--batch_size",   type=int,   default=16,
                        help="Per-GPU batch size. Global batch = batch_size × world_size.")
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--ema_rate",     type=float, default=0.9999)
    parser.add_argument("--log_every",    type=int,   default=100)
    parser.add_argument("--ckpt_every",   type=int,   default=5000)

    # Diffusion
    parser.add_argument("--noise_schedule", type=str,  default="linear")
    parser.add_argument("--learn_sigma",    action="store_true", default=False)
    parser.add_argument("--sample_scale",   type=int,  default=2)

    # Architecture – shared
    parser.add_argument("--num_res_blocks",       type=int,   default=2)
    parser.add_argument("--num_heads",            type=int,   default=4)
    parser.add_argument("--dropout",              type=float, default=0.0)
    parser.add_argument("--use_scale_shift_norm", action="store_true", default=True)
    parser.add_argument("--use_checkpoint",       action="store_true", default=True)
    parser.add_argument("--use_fp16",             action="store_true", default=False)
    parser.add_argument("--use_single_model",     action="store_true", default=False)

    # Architecture – per-scale channel widths and attention resolutions
    parser.add_argument("--num_channels_256",          type=int, default=192)
    parser.add_argument("--num_channels_128",          type=int, default=128)
    parser.add_argument("--num_channels_64",           type=int, default=64)
    parser.add_argument("--attention_resolutions_256", type=str, default="32,16,8")
    parser.add_argument("--attention_resolutions_128", type=str, default="32,16,8")
    parser.add_argument("--attention_resolutions_64",  type=str, default="16,8")

    # Checkpoint resume (all three must be provided together)
    parser.add_argument("--resume_256", type=str, default=None)
    parser.add_argument("--resume_128", type=str, default=None)
    parser.add_argument("--resume_64",  type=str, default=None)
    parser.add_argument("--resume_shared", type=str, default=None)

    return parser


if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)
