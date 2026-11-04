"""
Standard (no-drift) class-conditional UNet training on ImageNet-1k-256×256.

Single 256×256 UNet, full 1000-step DDPM schedule.
Dataset: benjamin-paine/imagenet-1k-256x256 (HuggingFace).

Usage (MPI, 4 GPUs):
    CUDA_VISIBLE_DEVICES=4,5,6,7 \\
    /usr/mpi/gcc/openmpi-4.1.5rc2/bin/mpiexec -n 4 -x CUDA_VISIBLE_DEVICES \\
        python scripts/image_train_imagenet256.py
"""

import copy
import os
import re
import time
from glob import glob

import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)

th.backends.cuda.matmul.allow_tf32 = True
th.backends.cudnn.allow_tf32 = True


# ---------------------------------------------------------------------------
# ImageNet HuggingFace Dataset
# ---------------------------------------------------------------------------

class ImageNetHFDataset(Dataset):
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
        img = item["image"].convert("RGB")
        return self.transform(img), item["label"]


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

def update_ema(ema_params, model_params, decay=0.9999):
    for e, m in zip(ema_params, model_params):
        e.detach().mul_(decay).add_(m.detach(), alpha=1 - decay)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(step, model, ema_params, opt, ckpt_dir, keep_last=5):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"{step:07d}.pt")
    th.save({
        "step": step,
        "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "ema":   ema_params,
        "opt":   opt.state_dict(),
    }, path)
    # prune old checkpoints
    ckpts = sorted(glob(os.path.join(ckpt_dir, "*.pt")))
    for old in ckpts[:-keep_last]:
        os.remove(old)


def load_checkpoint(path, model, ema_params, opt, device):
    ckpt = th.load(path, map_location=device)
    raw = model.module if hasattr(model, "module") else model
    raw.load_state_dict(ckpt["model"])
    for ep, sp in zip(ema_params, ckpt["ema"]):
        ep.copy_(sp)
    opt.load_state_dict(ckpt["opt"])
    return ckpt["step"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir",    default="results_imagenet256")
    parser.add_argument("--image_size",     type=int,   default=256)
    parser.add_argument("--num_channels",   type=int,   default=192)
    parser.add_argument("--num_res_blocks", type=int,   default=2)
    parser.add_argument("--num_heads",      type=int,   default=4)
    parser.add_argument("--attention_resolutions", default="32,16,8")
    parser.add_argument("--noise_schedule", default="linear")
    parser.add_argument("--diffusion_steps",type=int,   default=1000)
    parser.add_argument("--batch_size",     type=int,   default=48)
    parser.add_argument("--num_workers",    type=int,   default=4)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--weight_decay",   type=float, default=0.0)
    parser.add_argument("--ema_rate",       type=float, default=0.9999)
    parser.add_argument("--total_steps",    type=int,   default=1000000)
    parser.add_argument("--log_every",      type=int,   default=100)
    parser.add_argument("--ckpt_every",     type=int,   default=5000)
    parser.add_argument("--use_checkpoint", action="store_true")
    parser.add_argument("--resume",         default="",
                        help="path to a .pt checkpoint to resume from")
    args = parser.parse_args()

    # distributed setup
    dist_util.setup_dist()
    rank  = dist.get_rank()
    world = dist.get_world_size()
    device = dist_util.dev()

    # logging
    log_suffix = f"imagenet256-standard"
    exp_dirs = sorted(glob(os.path.join(args.results_dir, f"*-{log_suffix}")))
    if args.resume and exp_dirs:
        exp_dir = exp_dirs[-1]
    else:
        exp_id  = time.strftime("%Y%m%d-%H%M%S")
        exp_dir = os.path.join(args.results_dir, f"{exp_id}-{log_suffix}")
    if rank == 0:
        os.makedirs(exp_dir, exist_ok=True)
    dist.barrier()
    logger.configure(dir=exp_dir)

    ckpt_dir = os.path.join(exp_dir, "checkpoints")

    # model
    logger.log("creating model...")
    model, diffusion = create_model_and_diffusion(
        image_size=args.image_size,
        num_channels=args.num_channels,
        num_res_blocks=args.num_res_blocks,
        num_heads=args.num_heads,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions=args.attention_resolutions,
        channel_mult="",
        dropout=0.0,
        class_cond=True,
        use_checkpoint=args.use_checkpoint,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
        learn_sigma=True,
        diffusion_steps=args.diffusion_steps,
        noise_schedule=args.noise_schedule,
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )
    model.to(device)
    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        logger.log(f"model params: {n_params/1e6:.1f}M")

    # EMA
    ema_params = [copy.deepcopy(p.data) for p in model.parameters()]

    # optimizer
    opt = th.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # resume
    start_step = 0
    if args.resume and os.path.isfile(args.resume):
        logger.log(f"resuming from {args.resume}")
        start_step = load_checkpoint(args.resume, model, ema_params, opt, device)
        logger.log(f"resumed at step {start_step}")

    model = DDP(model, device_ids=[device], find_unused_parameters=False)

    # data
    logger.log("loading dataset...")
    dataset = ImageNetHFDataset(image_size=args.image_size)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True)
    loader  = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    data_iter = iter(loader)

    logger.log("training...")
    model.train()

    for step in range(start_step, args.total_steps):
        try:
            imgs, labels = next(data_iter)
        except StopIteration:
            sampler.set_epoch(step)
            data_iter = iter(loader)
            imgs, labels = next(data_iter)

        imgs   = imgs.to(device)
        labels = labels.to(device)

        t = th.randint(0, diffusion.num_timesteps, (imgs.shape[0],), device=device)
        model_kwargs = {"y": labels}

        losses = diffusion.training_losses(model, imgs, t, model_kwargs=model_kwargs)
        loss = losses["loss"].mean()

        opt.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        update_ema(ema_params, [p.data for p in model.parameters()], decay=args.ema_rate)

        if rank == 0 and (step + 1) % args.log_every == 0:
            vb   = losses.get("vb",   th.tensor(0.0)).mean().item()
            mse  = losses.get("mse",  th.tensor(0.0)).mean().item()
            logger.log(f"step={step+1:7d}  loss={loss.item():.4f}  mse={mse:.4f}  vb={vb:.4f}")

        if rank == 0 and (step + 1) % args.ckpt_every == 0:
            save_checkpoint(step + 1, model, ema_params, opt, ckpt_dir)
            logger.log(f"saved checkpoint at step {step+1}")

    if rank == 0:
        save_checkpoint(args.total_steps, model, ema_params, opt, ckpt_dir)

    dist.barrier()


if __name__ == "__main__":
    main()
