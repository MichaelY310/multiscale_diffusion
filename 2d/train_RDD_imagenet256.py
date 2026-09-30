"""
Multi-scale DiT training script with drift for ImageNet-1k-256x256.

Three models, each fixed to a specific architecture and resolution:
  32x32 latent → DiT-XL/2  (c=0, t ∈ [T_0, T_1) = [0,   300))
  16x16 latent → DiT-L/2   (c=1, t ∈ [0, T_2-T_1) = [0,  300))
   8x8  latent → DiT-B/2   (c=None, t ∈ [0, T_3-T_2) = [0, 400))

All models are class-conditional (num_classes=1000) and trained with
classifier-free guidance dropout (class_dropout_prob=0.1 by default in DiT).

Dataset: benjamin-paine/imagenet-1k-256x256 (HuggingFace)
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
from contextlib import ExitStack
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
# Path setup: import DiT models then switch to parent diffusion module
# ---------------------------------------------------------------------------
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dit_path = os.path.join(parent_dir, "DiT")

sys.path.insert(0, dit_path)
from models import DiT_models
sys.path.remove(dit_path)
sys.path.insert(0, parent_dir)

from diffusion import create_diffusion_with_drift
from diffusion.rdd_improvements import weighted_diffusion_loss
from diffusion.timestep_sampler import LossAwareSampler, create_named_schedule_sampler
from diffusers.models import AutoencoderKL


#################################################################################
#                              ImageNet HF Dataset                              #
#################################################################################

class ImageNetHFDataset(Dataset):
    """
    Wraps the HuggingFace `benjamin-paine/imagenet-1k-256x256` dataset as a
    standard PyTorch map-style Dataset so DistributedSampler can be used.

    Images are returned as (C,H,W) float tensors normalised to [-1, 1].
    Labels are int class indices in [0, 999].
    """

    def __init__(self, image_size: int = 256):
        from datasets import load_dataset
        self.hf_dataset = load_dataset(
            "/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/data/imagenet256",
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


class ImageNetOriginalParquetDataset(Dataset):
    """Original-resolution ILSVRC images stored in the official HF Parquet shards.

    This is deliberately different from the old
    ``benjamin-paine/imagenet-1k-256x256`` source: the official
    ``ILSVRC/imagenet-1k`` shards retain the original image dimensions.  We
    apply the same ADM ``center_crop_arr`` transform used by
    :class:`ImageNetOriginalDataset` at training time.
    """

    def __init__(self, data_path: str, image_size: int = 256):
        from datasets import load_dataset

        parquet_glob = os.path.join(data_path, "data", "train-*.parquet")
        parquet_files = sorted(glob(parquet_glob))
        if not parquet_files:
            raise FileNotFoundError(
                f"No official ImageNet train shards found at {parquet_glob}"
            )
        self.dataset = load_dataset(
            "parquet",
            data_files={"train": parquet_files},
            split="train",
        )
        self.transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.transform(image), int(item["label"])


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


def parse_stage_resolutions(spec):
    """Parse and validate the supported finest-to-coarsest topologies."""
    try:
        resolutions = tuple(int(token.strip()) for token in spec.split(",") if token.strip())
    except ValueError as exc:
        raise ValueError("--stage-resolutions must be comma-separated integers") from exc
    if resolutions not in ((32, 16, 8), (32, 8)):
        raise ValueError(
            "--stage-resolutions currently supports exactly '32,16,8' or '32,8'; "
            f"got {spec!r}"
        )
    return resolutions


def parse_train_stages(spec, available_stages=("32", "16", "8")):
    """Parse --train-stages into canonical finest-to-coarsest stage names."""
    aliases = {
        "32": "32", "32x32": "32",
        "16": "16", "16x16": "16",
        "8": "8", "8x8": "8",
    }
    tokens = [token.strip().lower() for token in spec.split(",") if token.strip()]
    if tokens == ["all"]:
        return tuple(available_stages)
    if not tokens:
        raise ValueError("--train-stages must contain at least one of: 32,16,8")

    unknown = [token for token in tokens if token not in aliases]
    if unknown:
        raise ValueError(
            f"Unknown --train-stages value(s): {unknown}; expected all or a comma-separated "
            "subset of 32,16,8 (32x32/16x16/8x8 aliases are also accepted)."
        )
    selected = {aliases[token] for token in tokens}
    unavailable = sorted(selected.difference(available_stages))
    if unavailable:
        raise ValueError(
            f"--train-stages requested {unavailable}, but --stage-resolutions only enables "
            f"{list(available_stages)}"
        )
    return tuple(stage for stage in available_stages if stage in selected)


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
    # ---------------------------------------------------------------------------
    # Timestep boundaries (global timeline)
    # ---------------------------------------------------------------------------
    # We model a single diffusion process of length `diffusion_steps` and train three
    # separate models on contiguous sub-intervals of that timeline. Each sub-model
    # sees a local timestep index in [0, T_{c+1}-T_c), and `SpacedDiffusionWithDrift`
    # maps it back to the global timeline via `T_c + t_local`.
    #
    # To keep the math consistent:
    # - stage lengths must be positive integers
    # - sum(stage_lengths) must equal diffusion_steps
    stage_resolutions = parse_stage_resolutions(args.stage_resolutions)
    stage_names = tuple(str(resolution) for resolution in stage_resolutions)
    stage_lengths = [int(x) for x in args.stage_lengths.split(",")]
    if len(stage_lengths) != len(stage_resolutions):
        raise ValueError(
            f"--stage-lengths must have {len(stage_resolutions)} values for topology "
            f"{stage_resolutions}, got: {args.stage_lengths!r}"
        )
    if any(x <= 0 for x in stage_lengths):
        raise ValueError(f"--stage-lengths must be positive, got: {stage_lengths}")
    if sum(stage_lengths) != args.diffusion_steps:
        raise ValueError(
            f"sum(--stage-lengths) must equal --diffusion-steps "
            f"({sum(stage_lengths)} != {args.diffusion_steps})"
        )

    boundaries = [0]
    for length in stage_lengths:
        boundaries.append(boundaries[-1] + length)
    stage_intervals = {
        stage: (boundaries[index], boundaries[index + 1])
        for index, stage in enumerate(stage_names)
    }
    stage_scales = {
        stage_names[index]: stage_resolutions[index] // stage_resolutions[index + 1]
        for index in range(len(stage_resolutions) - 1)
    }
    active_stages = parse_train_stages(args.train_stages, stage_names)

    assert torch.cuda.is_available(), "Training requires at least one GPU."

    # DDP setup
    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    if args.micro_batch_size is None:
        if args.global_batch_size % world_size != 0:
            raise ValueError(
                f"--global-batch-size ({args.global_batch_size}) must be divisible by "
                f"world size ({world_size}) when --micro-batch-size is omitted."
            )
        micro_batch_size = args.global_batch_size // world_size
        gradient_accumulation_steps = 1
    else:
        micro_batch_size = args.micro_batch_size
        if micro_batch_size <= 0:
            raise ValueError("--micro-batch-size must be positive")
        micro_global_batch = micro_batch_size * world_size
        if args.gradient_accumulation_steps is None:
            if args.global_batch_size % micro_global_batch != 0:
                raise ValueError(
                    f"--global-batch-size ({args.global_batch_size}) must be divisible by "
                    f"micro_batch_size * world_size ({micro_global_batch})."
                )
            gradient_accumulation_steps = args.global_batch_size // micro_global_batch
        else:
            gradient_accumulation_steps = args.gradient_accumulation_steps
        if gradient_accumulation_steps <= 0:
            raise ValueError("--gradient-accumulation-steps must be positive")
        effective_global_batch = micro_global_batch * gradient_accumulation_steps
        if effective_global_batch != args.global_batch_size:
            raise ValueError(
                "Batch-size mismatch: micro_batch_size * world_size * "
                f"gradient_accumulation_steps = {effective_global_batch}, but "
                f"--global-batch-size={args.global_batch_size}."
            )
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Experiment directories (rank-0 only)
    if rank == 0:
        experiment_dir = f"{args.results_dir}/imagenet256-multiscale"
        checkpoint_dir_32 = f"{experiment_dir}/checkpoints_32x32"
        checkpoint_dir_16 = f"{experiment_dir}/checkpoints_16x16"
        checkpoint_dir_8  = f"{experiment_dir}/checkpoints_8x8"
        checkpoint_dirs = {
            "32": checkpoint_dir_32,
            "16": checkpoint_dir_16,
            "8": checkpoint_dir_8,
        }
        for stage in active_stages:
            os.makedirs(checkpoint_dirs[stage], exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory: {experiment_dir}")
        logger.info(
            f"Diffusion steps: {args.diffusion_steps}, topology: {stage_resolutions}, "
            f"stage lengths: {stage_lengths}, boundaries: {boundaries}"
        )
        logger.info(f"Training stages: {', '.join(stage + 'x' + stage for stage in active_stages)}")
        logger.info(
            f"Batching: global={args.global_batch_size}, per-rank micro={micro_batch_size}, "
            f"world_size={world_size}, gradient_accumulation={gradient_accumulation_steps}"
        )
    else:
        logger = create_logger(None)

    # ---------------------------------------------------------------------------
    # Models  (each resolution uses a different DiT variant)
    # ---------------------------------------------------------------------------
    NUM_CLASSES = 1000
    model_32 = ema_32 = diffusion_32 = None
    model_16 = ema_16 = diffusion_16 = None
    model_8 = ema_8 = diffusion_8 = None

    # 32x32 latent → (finest detail, lowest noise, t ∈ [T_0, T_1))
    if "32" in active_stages:
        T_32_start, T_32_end = stage_intervals["32"]
        model_32 = DiT_models[args.model_32](input_size=32, num_classes=NUM_CLASSES)
        ema_32 = deepcopy(model_32).to(device)
        requires_grad(ema_32, False)
        model_32 = DDP(model_32.to(device), device_ids=[rank])
        diffusion_32 = create_diffusion_with_drift(
            timestep_respacing="",
            diffusion_steps=args.diffusion_steps,
            c=0,
            T_c=T_32_start,
            T_c_plus_1=T_32_end,
            sample_scale=stage_scales["32"],
        )

    # 16x16 latent → (mid-scale, t ∈ [T_1, T_2) → local [0, T_2-T_1))
    if "16" in active_stages:
        T_16_start, T_16_end = stage_intervals["16"]
        model_16 = DiT_models[args.model_16](input_size=16, num_classes=NUM_CLASSES)
        ema_16 = deepcopy(model_16).to(device)
        requires_grad(ema_16, False)
        model_16 = DDP(model_16.to(device), device_ids=[rank])
        diffusion_16 = create_diffusion_with_drift(
            timestep_respacing="",
            diffusion_steps=args.diffusion_steps,
            c=1,
            T_c=T_16_start,
            T_c_plus_1=T_16_end,
            sample_scale=stage_scales["16"],
        )

    # 8x8 latent → (coarsest, no drift, t ∈ [T_2, T_3) → local [0, T_3-T_2))
    if "8" in active_stages:
        T_8_start, T_8_end = stage_intervals["8"]
        model_8 = DiT_models[args.model_8](input_size=8, num_classes=NUM_CLASSES)
        ema_8 = deepcopy(model_8).to(device)
        requires_grad(ema_8, False)
        model_8 = DDP(model_8.to(device), device_ids=[rank])
        diffusion_8 = create_diffusion_with_drift(
            timestep_respacing="",
            diffusion_steps=args.diffusion_steps,
            c=None,
            T_c=T_8_start,
            T_c_plus_1=T_8_end,
        )

    timestep_samplers = None
    if args.timestep_sampler != "uniform":
        timestep_samplers = {
            stage: create_named_schedule_sampler(args.timestep_sampler, diffusion)
            for stage, diffusion in {
                "32": diffusion_32,
                "16": diffusion_16,
                "8": diffusion_8,
            }.items()
            if diffusion is not None
        }
        logger.info(f"Timestep sampler: {args.timestep_sampler} (unbiased importance correction)")
    logger.info(
        f"Loss weighting: {args.loss_weighting} "
        f"(min_snr_gamma={args.min_snr_gamma}, p2_gamma={args.p2_gamma}, p2_k={args.p2_k})"
    )

    # VAE (encode images → 4-channel latents)
    if args.vae_path:
        vae = AutoencoderKL.from_pretrained(args.vae_path, local_files_only=True).to(device)
    else:
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()

    if model_32 is not None:
        logger.info(f"{args.model_32} (32x32) params: {sum(p.numel() for p in model_32.module.parameters()):,}")
    if model_16 is not None:
        logger.info(f"{args.model_16} (16x16) params: {sum(p.numel() for p in model_16.module.parameters()):,}")
    if model_8 is not None:
        logger.info(f"{args.model_8} (8x8) params: {sum(p.numel() for p in model_8.module.parameters()):,}")

    # Optimisers
    opt_32 = torch.optim.AdamW(model_32.parameters(), lr=1e-4, weight_decay=0) if model_32 is not None else None
    opt_16 = torch.optim.AdamW(model_16.parameters(), lr=1e-4, weight_decay=0) if model_16 is not None else None
    opt_8 = torch.optim.AdamW(model_8.parameters(), lr=1e-4, weight_decay=0) if model_8 is not None else None

    # ---------------------------------------------------------------------------
    # Dataset & DataLoader
    # ---------------------------------------------------------------------------
    if args.data_source == "imagenet":
        if not args.data_path:
            raise ValueError("--data-path is required when --data-source=imagenet")
        dataset = ImageNetOriginalDataset(data_path=args.data_path, image_size=args.image_size)
        logger.info(f"Data source: original ImageNet (ADM center_crop_arr) from {args.data_path}")
    elif args.data_source == "imagenet_parquet":
        if not args.data_path:
            raise ValueError("--data-path is required when --data-source=imagenet_parquet")
        dataset = ImageNetOriginalParquetDataset(
            data_path=args.data_path,
            image_size=args.image_size,
        )
        logger.info(
            "Data source: official ILSVRC/imagenet-1k original-resolution Parquet "
            f"shards with ADM center_crop_arr from {args.data_path}"
        )
    else:
        dataset = ImageNetHFDataset(image_size=args.image_size)
        logger.info(f"Data source: HuggingFace imagenet-1k-256x256 (Lanczos preprocessed)")
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    optimizer_steps_per_epoch = len(loader) // gradient_accumulation_steps
    if optimizer_steps_per_epoch <= 0:
        raise ValueError(
            f"DataLoader has only {len(loader)} micro-batches, fewer than the requested "
            f"{gradient_accumulation_steps} accumulation steps."
        )
    usable_micro_batches_per_epoch = optimizer_steps_per_epoch * gradient_accumulation_steps
    logger.info(f"Dataset size: {len(dataset):,} images")
    logger.info(
        f"Optimizer steps/epoch: {optimizer_steps_per_epoch:,} "
        f"({usable_micro_batches_per_epoch:,}/{len(loader):,} micro-batches used)"
    )

    # ---------------------------------------------------------------------------
    # Initialise EMA weights
    # ---------------------------------------------------------------------------
    if model_32 is not None:
        update_ema(ema_32, model_32.module, decay=0)
        model_32.train()
        ema_32.eval()
    if model_16 is not None:
        update_ema(ema_16, model_16.module, decay=0)
        model_16.train()
        ema_16.eval()
    if model_8 is not None:
        update_ema(ema_8, model_8.module, decay=0)
        model_8.train()
        ema_8.eval()

    # ---------------------------------------------------------------------------
    # Monitoring variables
    # ---------------------------------------------------------------------------
    train_steps = 0
    start_epoch = 0
    log_steps = 0
    running_losses = {stage: 0.0 for stage in active_stages}
    start_time = time()

    # ---------------------------------------------------------------------------
    # Optional checkpoint resume
    # ---------------------------------------------------------------------------
    resume_paths = {"32": args.resume_32, "16": args.resume_16, "8": args.resume_8}
    inactive_resumes = [stage for stage, path in resume_paths.items()
                        if stage not in active_stages and path]
    if inactive_resumes:
        raise ValueError(
            f"Resume checkpoint(s) supplied for inactive stage(s): {inactive_resumes}. "
            f"Add them to --train-stages or remove the corresponding --resume-* argument."
        )

    active_resume_paths = {stage: resume_paths[stage] for stage in active_stages}
    if any(active_resume_paths.values()):
        missing_resumes = [stage for stage, path in active_resume_paths.items() if not path]
        if missing_resumes:
            raise ValueError(
                "When resuming, every active stage needs its checkpoint; missing: "
                + ", ".join(missing_resumes)
            )

        logger.info("Resuming from checkpoints:")
        for stage in active_stages:
            logger.info(f"  {stage}x{stage}: {resume_paths[stage]}")

        device_str = f"cuda:{device}"

        if "32" in active_stages:
            ckpt_32 = torch.load(args.resume_32, map_location=device_str, weights_only=False)
            model_32.module.load_state_dict(ckpt_32["model"])
            ema_32.load_state_dict(ckpt_32["ema"])
            opt_32.load_state_dict(ckpt_32["opt"])
        if "16" in active_stages:
            ckpt_16 = torch.load(args.resume_16, map_location=device_str, weights_only=False)
            model_16.module.load_state_dict(ckpt_16["model"])
            ema_16.load_state_dict(ckpt_16["ema"])
            opt_16.load_state_dict(ckpt_16["opt"])
        if "8" in active_stages:
            ckpt_8 = torch.load(args.resume_8, map_location=device_str, weights_only=False)
            model_8.module.load_state_dict(ckpt_8["model"])
            ema_8.load_state_dict(ckpt_8["ema"])
            opt_8.load_state_dict(ckpt_8["opt"])

        resume_steps = set()
        for path in active_resume_paths.values():
            match = re.search(r"(\d+)\.pt$", path)
            if not match:
                raise ValueError(f"Could not infer training step from checkpoint filename: {path}")
            resume_steps.add(int(match.group(1)))
        if len(resume_steps) != 1:
            raise ValueError(f"Active-stage checkpoints must have the same step, got: {sorted(resume_steps)}")
        train_steps = resume_steps.pop()
        if train_steps:
            start_epoch = train_steps // optimizer_steps_per_epoch
            logger.info(f"Resuming from step {train_steps}, epoch ~{start_epoch}")

        dist.barrier()

    # ---------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------
    # --- Timing log (auto-saved, never overwritten) ---
    timing_log_path = None
    if rank == 0:
        timing_log_path = f"{experiment_dir}/timing_log.jsonl"
        import json as _json
        with open(timing_log_path, "a") as _f:
            _f.write(_json.dumps({"event": "train_start", "step": train_steps, "timestamp": time()}) + "\n")
        logger.info(f"Timing log: {timing_log_path}")

    logger.info(f"Training for {args.epochs} epochs...")
    if args.max_train_steps is not None:
        if train_steps >= args.max_train_steps:
            raise ValueError(
                f"Checkpoint is already at step {train_steps}, which is not below "
                f"--max-train-steps={args.max_train_steps}."
            )
        logger.info(f"Stopping after optimizer step {args.max_train_steps}")

    stop_training = False
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        epoch_start_time = time()

        accumulated_step_losses = {stage: 0.0 for stage in active_stages}
        step_start_time = None
        for micro_step, (x, y) in enumerate(loader):
            if micro_step >= usable_micro_batches_per_epoch:
                break
            accumulation_index = micro_step % gradient_accumulation_steps
            sync_gradients = accumulation_index == gradient_accumulation_steps - 1
            if accumulation_index == 0:
                step_start_time = time()
                for optimizer in (opt_32, opt_16, opt_8):
                    if optimizer is not None:
                        optimizer.zero_grad(set_to_none=True)

            x = x.to(device)
            y = y.to(device)

            # DDP synchronizes only on the last micro-batch. Dividing each
            # micro loss by the accumulation count makes the resulting update
            # exactly the mean over --global-batch-size examples.
            with ExitStack() as stack:
                if not sync_gradients:
                    for model in (model_32, model_16, model_8):
                        if model is not None:
                            stack.enter_context(model.no_sync())

                # Encode to latent space: [B, 4, 32, 32] for 256x256 images
                with torch.no_grad():
                    x = vae.encode(x).latent_dist.sample().mul_(0.18215)

                model_kwargs = dict(y=y)

                def compute_stage_loss(stage, diffusion, stage_model, x_stage, local_steps):
                    importance_weights = None
                    if timestep_samplers is None:
                        # Preserve the exact legacy RNG path unless explicitly enabled.
                        timesteps = torch.randint(
                            0, local_steps, (x_stage.shape[0],), device=device
                        )
                    else:
                        timesteps, importance_weights = timestep_samplers[stage].sample(
                            x_stage.shape[0], device
                        )
                    terms = diffusion.training_losses(
                        stage_model, x_stage, timesteps, model_kwargs
                    )
                    if timestep_samplers is not None and isinstance(
                        timestep_samplers[stage], LossAwareSampler
                    ):
                        timestep_samplers[stage].update_with_local_losses(
                            timesteps, terms["loss"].detach()
                        )
                    loss_vector, _ = weighted_diffusion_loss(
                        terms,
                        diffusion,
                        timesteps,
                        args.loss_weighting,
                        min_snr_gamma=args.min_snr_gamma,
                        p2_gamma=args.p2_gamma,
                        p2_k=args.p2_k,
                        importance_weights=importance_weights,
                    )
                    return loss_vector.mean()

                # --- 32x32: t ∈ [0, T_1) (local index = original index since T_0=0) ---
                if model_32 is not None:
                    T_32_start, T_32_end = stage_intervals["32"]
                    loss_32 = compute_stage_loss(
                        "32", diffusion_32, model_32, x, T_32_end - T_32_start
                    )
                    (loss_32 / gradient_accumulation_steps).backward()
                    accumulated_step_losses["32"] += loss_32.item()

                # --- 16x16: t ∈ [0, T_2-T_1) (local index into sub-interval) ---
                if model_16 is not None:
                    T_16_start, T_16_end = stage_intervals["16"]
                    x_16 = downsample_latent(x.detach(), target_size=16)
                    loss_16 = compute_stage_loss(
                        "16", diffusion_16, model_16, x_16, T_16_end - T_16_start
                    )
                    (loss_16 / gradient_accumulation_steps).backward()
                    accumulated_step_losses["16"] += loss_16.item()

                # --- 8x8: t ∈ [0, T_3-T_2) (local index into sub-interval) ---
                if model_8 is not None:
                    T_8_start, T_8_end = stage_intervals["8"]
                    x_8 = downsample_latent(x.detach(), target_size=8)
                    loss_8 = compute_stage_loss(
                        "8", diffusion_8, model_8, x_8, T_8_end - T_8_start
                    )
                    (loss_8 / gradient_accumulation_steps).backward()
                    accumulated_step_losses["8"] += loss_8.item()

            if not sync_gradients:
                continue

            for optimizer in (opt_32, opt_16, opt_8):
                if optimizer is not None:
                    optimizer.step()
            if model_32 is not None:
                update_ema(ema_32, model_32.module)
            if model_16 is not None:
                update_ema(ema_16, model_16.module)
            if model_8 is not None:
                update_ema(ema_8, model_8.module)

            step_losses = {
                stage: accumulated_step_losses[stage] / gradient_accumulation_steps
                for stage in active_stages
            }
            accumulated_step_losses = {stage: 0.0 for stage in active_stages}

            # Logging
            for stage, loss_value in step_losses.items():
                running_losses[stage] += loss_value
            log_steps   += 1
            train_steps += 1

            # Per-step timing (write every log_every steps to reduce I/O)
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_step_time = (end_time - start_time) / log_steps

                avg_losses = {}
                for stage in active_stages:
                    avg_loss = torch.tensor(running_losses[stage] / log_steps, device=device)
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    avg_losses[stage] = avg_loss.item() / dist.get_world_size()
                loss_text = "  ".join(
                    f"Loss[{stage}x{stage}]={avg_losses[stage]:.4f}"
                    for stage in active_stages
                )

                logger.info(
                    f"(step={train_steps:07d}) "
                    f"{loss_text}  "
                    f"Steps/s={steps_per_sec:.2f}  "
                    f"Step_time={avg_step_time:.3f}s"
                )

                # Write to timing log
                if rank == 0 and timing_log_path:
                    import json as _json
                    with open(timing_log_path, "a") as _f:
                        _f.write(_json.dumps({
                            "event": "steps", "step": train_steps, "epoch": epoch,
                            "avg_step_time_s": round(avg_step_time, 4),
                            "steps_per_sec": round(steps_per_sec, 4),
                            "timestamp": end_time
                        }) + "\n")

                running_losses = {stage: 0.0 for stage in active_stages}
                log_steps = 0
                start_time = time()

            # Checkpointing
            reached_max_steps = (
                args.max_train_steps is not None and train_steps >= args.max_train_steps
            )
            scheduled_stop = False
            if args.stop_file:
                stop_requested = torch.zeros(1, dtype=torch.int32, device=device)
                if rank == 0 and os.path.isfile(args.stop_file):
                    stop_requested.fill_(1)
                dist.broadcast(stop_requested, src=0)
                scheduled_stop = bool(stop_requested.item())

            if (
                train_steps % args.ckpt_every == 0
                or reached_max_steps
                or scheduled_stop
            ) and train_steps > 0:
                if rank == 0:
                    if model_32 is not None:
                        T_32_start, T_32_end = stage_intervals["32"]
                        torch.save(
                            {
                                "model": model_32.module.state_dict(),
                                "ema": ema_32.state_dict(),
                                "opt": opt_32.state_dict(),
                                "args": args,
                                "resolution": "32x32",
                                "c": 0, "T_c": T_32_start, "T_c_plus_1": T_32_end,
                                "sample_scale": stage_scales["32"],
                            },
                            f"{checkpoint_dir_32}/{train_steps:07d}.pt",
                        )
                    if model_16 is not None:
                        T_16_start, T_16_end = stage_intervals["16"]
                        torch.save(
                            {
                                "model": model_16.module.state_dict(),
                                "ema": ema_16.state_dict(),
                                "opt": opt_16.state_dict(),
                                "args": args,
                                "resolution": "16x16",
                                "c": 1, "T_c": T_16_start, "T_c_plus_1": T_16_end,
                                "sample_scale": stage_scales["16"],
                            },
                            f"{checkpoint_dir_16}/{train_steps:07d}.pt",
                        )
                    if model_8 is not None:
                        T_8_start, T_8_end = stage_intervals["8"]
                        torch.save(
                            {
                                "model": model_8.module.state_dict(),
                                "ema": ema_8.state_dict(),
                                "opt": opt_8.state_dict(),
                                "args": args,
                                "resolution": "8x8",
                                "c": None, "T_c": T_8_start, "T_c_plus_1": T_8_end,
                            },
                            f"{checkpoint_dir_8}/{train_steps:07d}.pt",
                        )
                    saved_stages = ", ".join(stage + "x" + stage for stage in active_stages)
                    logger.info(f"Saved {saved_stages} checkpoint(s) at step {train_steps:07d}")
                dist.barrier()

            if reached_max_steps or scheduled_stop:
                if scheduled_stop:
                    logger.info(
                        f"Observed --stop-file={args.stop_file}; "
                        f"saved exact-step checkpoints at step {train_steps} and will exit."
                    )
                if reached_max_steps:
                    logger.info(f"Reached --max-train-steps={args.max_train_steps}")
                stop_training = True
                break

        # End of epoch timing
        epoch_end_time = time()
        epoch_duration = epoch_end_time - epoch_start_time
        logger.info(f"Epoch {epoch} completed in {epoch_duration:.1f}s ({epoch_duration/60:.1f}min)")
        if rank == 0 and timing_log_path:
            import json as _json
            with open(timing_log_path, "a") as _f:
                _f.write(_json.dumps({
                    "event": "epoch_end", "epoch": epoch, "step": train_steps,
                    "epoch_time_s": round(epoch_duration, 2),
                    "timestamp": epoch_end_time
                }) + "\n")
        if stop_training:
            break

    for model in (model_32, model_16, model_8):
        if model is not None:
            model.eval()
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Multi-scale DiT training on ImageNet-256 with drift. "
            "32x32→DiT-XL/2, 16x16→DiT-L/2, 8x8→DiT-B/2."
        )
    )
    parser.add_argument("--model-32", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2",
                        help="Architecture for 32x32 stage")
    parser.add_argument("--model-16", type=str, choices=list(DiT_models.keys()), default="DiT-L/2",
                        help="Architecture for 16x16 stage")
    parser.add_argument("--model-8",  type=str, choices=list(DiT_models.keys()), default="DiT-B/2",
                        help="Architecture for 8x8 stage")
    parser.add_argument("--results-dir",       type=str,   default="results_imagenet256")
    parser.add_argument("--image-size",        type=int,   default=256, choices=[256])
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=1000,
        help="Total diffusion steps for the global timeline (must equal sum of --stage-lengths).",
    )
    parser.add_argument(
        "--stage-resolutions",
        type=str,
        default="32,16,8",
        help=(
            "Finest-to-coarsest topology. Supported: '32,16,8' (three stages) "
            "or '32,8' (direct two-stage RDD)."
        ),
    )
    parser.add_argument(
        "--stage-lengths",
        type=str,
        default="300,300,400",
        help=(
            "Positive ints matching --stage-resolutions and summing to --diffusion-steps. "
            "Examples: 300,300,400 for 32/16/8 or 400,600 for 32/8."
        ),
    )
    parser.add_argument(
        "--train-stages",
        type=str,
        default="all",
        help=(
            "Stages to instantiate and train: 'all' (default), or a comma-separated subset "
            "of 32,16,8. Examples: --train-stages 8, --train-stages 16, "
            "--train-stages 16,8. Aliases 32x32/16x16/8x8 are accepted."
        ),
    )
    parser.add_argument("--epochs",            type=int,   default=1400)
    parser.add_argument("--global-batch-size", type=int,   default=256)
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=None,
        help=(
            "Per-rank micro-batch size. When set, gradient accumulation is used "
            "to preserve --global-batch-size."
        ),
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help=(
            "Number of micro-batches per optimizer update. If omitted together "
            "with --micro-batch-size, it is inferred from the global batch size."
        ),
    )
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=None,
        help="Stop after this many optimizer updates (resume step is included).",
    )
    parser.add_argument(
        "--stop-file",
        type=str,
        default=None,
        help=(
            "Optional job-scoped stop request. After an optimizer step, rank 0 checks "
            "for this file; all ranks then save exact-step checkpoints and exit cleanly."
        ),
    )
    parser.add_argument("--global-seed",       type=int,   default=0)
    parser.add_argument("--vae",               type=str,   default="ema", choices=["ema", "mse"])
    parser.add_argument(
        "--vae-path",
        type=str,
        default=None,
        help="Optional local diffusers-format VAE directory; used with local_files_only=True.",
    )
    parser.add_argument("--num-workers",       type=int,   default=4)
    parser.add_argument("--log-every",         type=int,   default=100)
    parser.add_argument("--ckpt-every",        type=int,   default=1000)
    parser.add_argument(
        "--loss-weighting", type=str, default="none",
        choices=["none", "min-snr", "p2"],
        help="Optional denoising-MSE weighting; default preserves legacy training.",
    )
    parser.add_argument("--min-snr-gamma", type=float, default=5.0)
    parser.add_argument("--p2-gamma", type=float, default=1.0)
    parser.add_argument("--p2-k", type=float, default=1.0)
    parser.add_argument(
        "--timestep-sampler", type=str, default="uniform",
        choices=["uniform", "loss-second-moment"],
    )
    parser.add_argument("--resume-32",         type=str,   default=None)
    parser.add_argument("--resume-16",         type=str,   default=None)
    parser.add_argument("--resume-8",          type=str,   default=None)
    parser.add_argument(
        "--data-source",
        type=str,
        default="hf",
        choices=["hf", "imagenet", "imagenet_parquet"],
        help=(
            "Data source: 'hf' is the legacy incorrect 256x256-preprocessed source; "
            "'imagenet' is an original ImageFolder tree; 'imagenet_parquet' is the "
            "official ILSVRC/imagenet-1k original-resolution Parquet snapshot. Both "
            "original modes use ADM center_crop_arr."
        ),
    )
    parser.add_argument("--data-path",         type=str,   default=None,
                        help="Path to original ImageNet train directory (required when --data-source=imagenet).")
    args = parser.parse_args()
    main(args)
