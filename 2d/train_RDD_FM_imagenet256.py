"""
RDD Flow Matching: Multi-scale DiT training with drift (Flow Matching version).

Three models, each fixed to a specific architecture and resolution:
  32x32 latent → DiT-XL/2  (c=0, drift to 16x16)
  16x16 latent → DiT-L/2   (c=1, drift to 8x8)
   8x8  latent → DiT-B/2   (c=None, coarsest, standard FM)

Time-axis partitioning (matching DDPM 300-300-400 split):
  --stage-lengths 300,300,400  →  s_boundaries [0, 0.3, 0.6, 1.0] in FM

  Stage 0 (finest,   32x32): t ∈ [0.6, 1.0) — near data, with drift
  Stage 1 (middle,   16x16): t ∈ [0.3, 0.6) — with drift
  Stage 2 (coarsest,  8x8):  t ∈ [0.0, 0.3) — no drift, standard FM

Mathematical formulation (per stage k):
  x_t = t·x_1 + (1-t)·ε + f(t)·d
  f(t) = t·(s_{k+1} - t) / L_k
  u_t = (x_1 - ε) + (s_{k+1} - 2t)/L_k · d
  Loss = E[||v_θ(x_t, t) - u_t||²]

Where d = UP(DOWN(x_1)) - x_1 is the low-pass residual.

Dataset: ImageNet-1k-256x256 (original with ADM center_crop_arr)
"""

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

import numpy as np
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
from PIL import Image
import argparse
import logging
import os
import sys
import re

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dit_path = os.path.join(parent_dir, "DiT")

sys.path.insert(0, dit_path)
from models import DiT_models
sys.path.remove(dit_path)
sys.path.insert(0, parent_dir)

from diffusion import FlowMatchingWithDrift
from diffusers.models import AutoencoderKL


#################################################################################
#                         ImageNet Original Dataset (ADM)                       #
#################################################################################

def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class ImageNetOriginalDataset(Dataset):
    def __init__(self, data_path: str, image_size: int = 256):
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        self.dataset = ImageFolder(data_path, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class ImageNetHFDataset(Dataset):
    DEFAULT_PATHS = {
        256: "/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/data/imagenet256",
    }

    def __init__(self, image_size: int = 256, dataset_path: str = None):
        from datasets import load_dataset, load_from_disk
        path = dataset_path or self.DEFAULT_PATHS.get(image_size)
        if path is None:
            raise ValueError(f"No default dataset path for image_size={image_size}.")
        try:
            self.hf_dataset = load_dataset(path, split="train")
        except ValueError:
            self.hf_dataset = load_from_disk(path)
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


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def downsample_latent(latent, target_size):
    """Average-pool [B, C, H, W] latent to [B, C, target_size, target_size]."""
    B, C, H, W = latent.shape
    assert H == W and H % target_size == 0
    factor = H // target_size
    latent = latent.view(B, C, target_size, factor, target_size, factor)
    return latent.mean(dim=(3, 5))


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    dist.destroy_process_group()


def create_logger(logging_dir):
    if dist.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt", mode="a"),
            ],
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    assert torch.cuda.is_available(), "Training requires at least one GPU."

    # DDP setup
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Experiment directories
    if rank == 0:
        experiment_dir = f"{args.results_dir}/imagenet256-fm-multiscale"
        checkpoint_dir_32 = f"{experiment_dir}/checkpoints_32x32"
        checkpoint_dir_16 = f"{experiment_dir}/checkpoints_16x16"
        checkpoint_dir_8 = f"{experiment_dir}/checkpoints_8x8"
        os.makedirs(checkpoint_dir_32, exist_ok=True)
        os.makedirs(checkpoint_dir_16, exist_ok=True)
        os.makedirs(checkpoint_dir_8, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory: {experiment_dir}")
        logger.info(f"Flow Matching RDD (multistage): 32x32→{args.model_32}, 16x16→{args.model_16}, 8x8→{args.model_8}")
    else:
        logger = create_logger(None)

    # ---------------------------------------------------------------------------
    # Models
    # ---------------------------------------------------------------------------
    NUM_CLASSES = 1000

    model_32 = DiT_models[args.model_32](input_size=32, num_classes=NUM_CLASSES)
    ema_32 = deepcopy(model_32).to(device)
    requires_grad(ema_32, False)
    model_32 = DDP(model_32.to(device), device_ids=[rank])

    model_16 = DiT_models[args.model_16](input_size=16, num_classes=NUM_CLASSES)
    ema_16 = deepcopy(model_16).to(device)
    requires_grad(ema_16, False)
    model_16 = DDP(model_16.to(device), device_ids=[rank])

    model_8 = DiT_models[args.model_8](input_size=8, num_classes=NUM_CLASSES)
    ema_8 = deepcopy(model_8).to(device)
    requires_grad(ema_8, False)
    model_8 = DDP(model_8.to(device), device_ids=[rank])

    # Flow Matching objects — with time-axis partitioning
    # Parse stage lengths and compute FM boundaries
    stage_lengths = [int(x) for x in args.stage_lengths.split(",")]
    assert len(stage_lengths) == 3, "Expect 3 stage lengths (finest, middle, coarsest)"
    diffusion_steps = sum(stage_lengths)
    T_boundaries = [0]
    for sl in stage_lengths:
        T_boundaries.append(T_boundaries[-1] + sl)
    s_boundaries = [T / diffusion_steps for T in T_boundaries]
    # s_boundaries = [0.0, 0.3, 0.6, 1.0] for 300,300,400

    # Stage mapping (same as unified):
    #   resolutions[0]=32 (finest) → stage_time_idx = num_stages-1-0 = 2 → t∈[s2, s3)=[0.6, 1.0)
    #   resolutions[1]=16 (middle) → stage_time_idx = num_stages-1-1 = 1 → t∈[s1, s2)=[0.3, 0.6)
    #   resolutions[2]=8 (coarsest) → stage_time_idx = num_stages-1-2 = 0 → t∈[s0, s1)=[0.0, 0.3)
    num_stages = 3
    # 32x32: finest, time interval [s2, s3]
    s_k_32 = s_boundaries[num_stages - 1 - 0]   # s_boundaries[2] = 0.6
    s_k1_32 = s_boundaries[num_stages - 1 - 0 + 1]  # s_boundaries[3] = 1.0
    fm_32 = FlowMatchingWithDrift(c=0, sample_scale=2, s_k=s_k_32, s_k_plus_1=s_k1_32)

    # 16x16: middle, time interval [s1, s2]
    s_k_16 = s_boundaries[num_stages - 1 - 1]   # s_boundaries[1] = 0.3
    s_k1_16 = s_boundaries[num_stages - 1 - 1 + 1]  # s_boundaries[2] = 0.6
    fm_16 = FlowMatchingWithDrift(c=1, sample_scale=2, s_k=s_k_16, s_k_plus_1=s_k1_16)

    # 8x8: coarsest, time interval [s0, s1], no drift
    s_k_8 = s_boundaries[num_stages - 1 - 2]    # s_boundaries[0] = 0.0
    s_k1_8 = s_boundaries[num_stages - 1 - 2 + 1]  # s_boundaries[1] = 0.3
    fm_8 = FlowMatchingWithDrift(c=None, s_k=s_k_8, s_k_plus_1=s_k1_8)

    logger.info(f"FM time boundaries: {s_boundaries}")
    logger.info(f"  32x32: t∈[{s_k_32:.3f}, {s_k1_32:.3f}), drift to 16x16")
    logger.info(f"  16x16: t∈[{s_k_16:.3f}, {s_k1_16:.3f}), drift to 8x8")
    logger.info(f"   8x8:  t∈[{s_k_8:.3f}, {s_k1_8:.3f}), no drift")

    # VAE
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    logger.info(f"DiT-XL/2 (32x32) params: {sum(p.numel() for p in model_32.module.parameters()):,}")
    logger.info(f"DiT-L/2  (16x16) params: {sum(p.numel() for p in model_16.module.parameters()):,}")
    logger.info(f"DiT-B/2  ( 8x8 ) params: {sum(p.numel() for p in model_8.module.parameters()):,}")

    # Optimizers
    opt_32 = torch.optim.AdamW(model_32.parameters(), lr=1e-4, weight_decay=0)
    opt_16 = torch.optim.AdamW(model_16.parameters(), lr=1e-4, weight_decay=0)
    opt_8 = torch.optim.AdamW(model_8.parameters(), lr=1e-4, weight_decay=0)

    # ---------------------------------------------------------------------------
    # Dataset & DataLoader
    # ---------------------------------------------------------------------------
    if args.data_source == "imagenet":
        if not args.data_path:
            raise ValueError("--data-path is required when --data-source=imagenet")
        dataset = ImageNetOriginalDataset(data_path=args.data_path, image_size=args.image_size)
        logger.info(f"Data source: original ImageNet (ADM center_crop_arr) from {args.data_path}")
    else:
        dataset = ImageNetHFDataset(image_size=args.image_size, dataset_path=args.data_path)
        logger.info(f"Data source: HuggingFace imagenet-1k-256x256")
    sampler = DistributedSampler(
        dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=True, seed=args.global_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Dataset size: {len(dataset):,} images")

    # ---------------------------------------------------------------------------
    # Initialize EMA
    # ---------------------------------------------------------------------------
    update_ema(ema_32, model_32.module, decay=0)
    update_ema(ema_16, model_16.module, decay=0)
    update_ema(ema_8, model_8.module, decay=0)

    model_32.train(); model_16.train(); model_8.train()
    ema_32.eval(); ema_16.eval(); ema_8.eval()

    # ---------------------------------------------------------------------------
    # Monitoring
    # ---------------------------------------------------------------------------
    train_steps = 0
    start_epoch = 0
    log_steps = 0
    running_loss_32 = running_loss_16 = running_loss_8 = 0.0
    start_time = time()

    # ---------------------------------------------------------------------------
    # Optional checkpoint resume
    # ---------------------------------------------------------------------------
    if args.resume_32 and args.resume_16 and args.resume_8:
        logger.info("Resuming from checkpoints:")
        logger.info(f"  32x32 : {args.resume_32}")
        logger.info(f"  16x16 : {args.resume_16}")
        logger.info(f"   8x8  : {args.resume_8}")

        device_str = f"cuda:{device}"

        ckpt_32 = torch.load(args.resume_32, map_location=device_str, weights_only=False)
        model_32.module.load_state_dict(ckpt_32["model"])
        ema_32.load_state_dict(ckpt_32["ema"])
        opt_32.load_state_dict(ckpt_32["opt"])

        ckpt_16 = torch.load(args.resume_16, map_location=device_str, weights_only=False)
        model_16.module.load_state_dict(ckpt_16["model"])
        ema_16.load_state_dict(ckpt_16["ema"])
        opt_16.load_state_dict(ckpt_16["opt"])

        ckpt_8 = torch.load(args.resume_8, map_location=device_str, weights_only=False)
        model_8.module.load_state_dict(ckpt_8["model"])
        ema_8.load_state_dict(ckpt_8["ema"])
        opt_8.load_state_dict(ckpt_8["opt"])

        match = re.search(r"(\d+)\.pt$", args.resume_32)
        if match:
            train_steps = int(match.group(1))
            start_epoch = train_steps // len(loader)
            logger.info(f"Resuming from step {train_steps}, epoch ~{start_epoch}")

        dist.barrier()

    # ---------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------
    timing_log_path = None
    if rank == 0:
        timing_log_path = f"{experiment_dir}/timing_log.jsonl"
        import json as _json
        with open(timing_log_path, "a") as _f:
            _f.write(_json.dumps({"event": "train_start", "step": train_steps, "timestamp": time()}) + "\n")

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        epoch_start_time = time()

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)

            model_kwargs = dict(y=y)

            # --- 32x32: Flow Matching with drift ---
            loss_32 = fm_32.training_losses(model_32, x, model_kwargs=model_kwargs)["loss"].mean()
            opt_32.zero_grad()
            loss_32.backward()
            opt_32.step()
            update_ema(ema_32, model_32.module)

            # --- 16x16: Flow Matching with drift ---
            x_16 = downsample_latent(x.detach(), target_size=16)
            loss_16 = fm_16.training_losses(model_16, x_16, model_kwargs=model_kwargs)["loss"].mean()
            opt_16.zero_grad()
            loss_16.backward()
            opt_16.step()
            update_ema(ema_16, model_16.module)

            # --- 8x8: Standard Flow Matching (no drift) ---
            x_8 = downsample_latent(x.detach(), target_size=8)
            loss_8 = fm_8.training_losses(model_8, x_8, model_kwargs=model_kwargs)["loss"].mean()
            opt_8.zero_grad()
            loss_8.backward()
            opt_8.step()
            update_ema(ema_8, model_8.module)

            # Logging
            running_loss_32 += loss_32.item()
            running_loss_16 += loss_16.item()
            running_loss_8 += loss_8.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_step_time = (end_time - start_time) / log_steps

                avg_32 = torch.tensor(running_loss_32 / log_steps, device=device)
                avg_16 = torch.tensor(running_loss_16 / log_steps, device=device)
                avg_8 = torch.tensor(running_loss_8 / log_steps, device=device)
                dist.all_reduce(avg_32, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_16, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_8, op=dist.ReduceOp.SUM)
                avg_32 = avg_32.item() / dist.get_world_size()
                avg_16 = avg_16.item() / dist.get_world_size()
                avg_8 = avg_8.item() / dist.get_world_size()

                logger.info(
                    f"(step={train_steps:07d}) "
                    f"Loss[32x32]={avg_32:.4f}  "
                    f"Loss[16x16]={avg_16:.4f}  "
                    f"Loss[8x8]={avg_8:.4f}  "
                    f"Steps/s={steps_per_sec:.2f}  "
                    f"Step_time={avg_step_time:.3f}s"
                )

                if rank == 0 and timing_log_path:
                    import json as _json
                    with open(timing_log_path, "a") as _f:
                        _f.write(_json.dumps({
                            "event": "steps", "step": train_steps, "epoch": epoch,
                            "avg_step_time_s": round(avg_step_time, 4),
                            "steps_per_sec": round(steps_per_sec, 4),
                            "timestamp": end_time
                        }) + "\n")

                running_loss_32 = running_loss_16 = running_loss_8 = 0.0
                log_steps = 0
                start_time = time()

            # Checkpointing
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    torch.save(
                        {"model": model_32.module.state_dict(), "ema": ema_32.state_dict(),
                         "opt": opt_32.state_dict(), "args": args, "resolution": "32x32", "c": 0},
                        f"{checkpoint_dir_32}/{train_steps:07d}.pt",
                    )
                    torch.save(
                        {"model": model_16.module.state_dict(), "ema": ema_16.state_dict(),
                         "opt": opt_16.state_dict(), "args": args, "resolution": "16x16", "c": 1},
                        f"{checkpoint_dir_16}/{train_steps:07d}.pt",
                    )
                    torch.save(
                        {"model": model_8.module.state_dict(), "ema": ema_8.state_dict(),
                         "opt": opt_8.state_dict(), "args": args, "resolution": "8x8", "c": None},
                        f"{checkpoint_dir_8}/{train_steps:07d}.pt",
                    )
                    logger.info(f"Saved checkpoints at step {train_steps:07d}")
                dist.barrier()

        # End of epoch
        epoch_end_time = time()
        epoch_duration = epoch_end_time - epoch_start_time
        logger.info(f"Epoch {epoch} completed in {epoch_duration:.1f}s ({epoch_duration/60:.1f}min)")
        if rank == 0 and timing_log_path:
            import json as _json
            with open(timing_log_path, "a") as _f:
                _f.write(_json.dumps({
                    "event": "epoch_end", "epoch": epoch, "step": train_steps,
                    "epoch_time_s": round(epoch_duration, 2), "timestamp": epoch_end_time
                }) + "\n")

    model_32.eval(); model_16.eval(); model_8.eval()
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RDD Flow Matching multistage training on ImageNet-256.")
    parser.add_argument("--model-32", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--model-16", type=str, choices=list(DiT_models.keys()), default="DiT-L/2")
    parser.add_argument("--model-8", type=str, choices=list(DiT_models.keys()), default="DiT-B/2")
    parser.add_argument("--results-dir", type=str, default="results_imagenet256")
    parser.add_argument("--stage-lengths", type=str, default="300,300,400",
                        help="Stage lengths for time-axis partitioning (finest,middle,coarsest)")
    parser.add_argument("--diffusion-steps", type=int, default=1000,
                        help="Total diffusion steps (for normalization, default 1000)")
    parser.add_argument("--image-size", type=int, default=256, choices=[256])
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, default="ema", choices=["ema", "mse"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--resume-32", type=str, default=None)
    parser.add_argument("--resume-16", type=str, default=None)
    parser.add_argument("--resume-8", type=str, default=None)
    parser.add_argument("--data-source", type=str, default="hf", choices=["hf", "imagenet"])
    parser.add_argument("--data-path", type=str, default=None)
    args = parser.parse_args()
    main(args)
