"""
RDD Flow Matching Unified: single DiTMultiRes model handles ALL resolution stages.

This is the Flow Matching version of RDD, mirroring the DDPM version's time-axis
partitioning. The time axis t∈[0,1] is split into stages (analogous to splitting
t∈[0,1000] in DDPM), with each stage operating at a different resolution.

``--stage-lengths`` is expressed in FM data-time order (coarsest first).
To mirror a DDPM finest/middle/coarsest allocation of 300/300/400, use:
  --stage-lengths 400,300,300  →  boundaries [0, 0.4, 0.7, 1.0] in FM

  Stage 0 (coarsest, 8x8):   t ∈ [0.0, 0.4)  — no drift, standard FM
  Stage 1 (middle, 16x16):   t ∈ [0.4, 0.7)  — with drift
  Stage 2 (finest, 32x32):   t ∈ [0.7, 1.0)  — with drift

Usage:
  torchrun --nproc_per_node=8 train_RDD_FM_imagenet256_unified.py \
    --model DiT-XL/2 \
    --resolutions 32,16,8 \
    --stage-lengths 400,300,300 \
    --data-source imagenet \
    --data-path /path/to/imagenet/train
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
from PIL import Image
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
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
from models_multires import DiTMultiRes_models
sys.path.remove(dit_path)
sys.path.insert(0, parent_dir)

from diffusion import FlowMatchingWithDrift
from diffusion.rdd_improvements import (
    pool_teacher_tokens,
    representation_alignment_loss,
    sample_fm_time,
)
from diffusion.repa import (
    dinov2_dimension,
    dinov2_patch_tokens,
    load_dinov2_teacher,
    load_state_dict_with_optional_repa,
)
from diffusers.models import AutoencoderKL


#################################################################################
#                          ADM center_crop_arr                                  #
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
    """Original ImageNet with ADM center_crop_arr preprocessing (same as DiT)."""
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


#################################################################################
#                              ImageNet HF Dataset                              #
#################################################################################

class ImageNetHFDataset(Dataset):
    """
    Wraps a HuggingFace ImageNet dataset (parquet or arrow) as a standard
    PyTorch map-style Dataset.
    """
    DEFAULT_PATHS = {
        256: "/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/data/imagenet256",
        512: "/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/data/imagenet512",
    }

    def __init__(self, image_size: int = 256, dataset_path: str = None):
        from datasets import load_dataset, load_from_disk
        path = dataset_path or self.DEFAULT_PATHS.get(image_size)
        if path is None:
            raise ValueError(f"No default dataset path for image_size={image_size}. "
                             f"Use --data-path to specify.")
        try:
            self.hf_dataset = load_dataset(path, split="train")
        except ValueError:
            self.hf_dataset = load_from_disk(path)
            if hasattr(self.hf_dataset, "column_names") and "train" in str(type(self.hf_dataset)):
                self.hf_dataset = self.hf_dataset["train"]
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
    assert H == W and H % target_size == 0, \
        f"Cannot downsample {H}x{W} to {target_size}x{target_size} (not divisible)"
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
    # ---------------------------------------------------------------------------
    # Parse resolutions and stage lengths
    # ---------------------------------------------------------------------------
    resolutions = [int(x) for x in args.resolutions.split(",")]
    num_stages = len(resolutions)

    for i in range(1, num_stages):
        if resolutions[i] >= resolutions[i-1]:
            raise ValueError(f"--resolutions must be strictly decreasing, got {resolutions}")

    vae_latent_size = args.image_size // 8
    for res in resolutions:
        if res > vae_latent_size:
            raise ValueError(f"Resolution {res} exceeds VAE latent size {vae_latent_size}.")
        if vae_latent_size % res != 0:
            raise ValueError(f"VAE latent size {vae_latent_size} not divisible by {res}")

    stage_lengths = [int(x) for x in args.stage_lengths.split(",")]
    if len(stage_lengths) != num_stages:
        raise ValueError(f"--stage-lengths must have {num_stages} entries, got {len(stage_lengths)}")
    if sum(stage_lengths) != args.diffusion_steps:
        raise ValueError(
            f"sum(--stage-lengths)={sum(stage_lengths)} != --diffusion-steps={args.diffusion_steps}"
        )

    # Convert integer boundaries to [0,1] for Flow Matching
    # e.g. stage_lengths=[400,300,300], diffusion_steps=1000
    # → T_boundaries = [0, 400, 700, 1000]
    # → s_boundaries = [0.0, 0.4, 0.7, 1.0]
    T_boundaries = [0]
    for length in stage_lengths:
        T_boundaries.append(T_boundaries[-1] + length)
    s_boundaries = [T / args.diffusion_steps for T in T_boundaries]

    # ---------------------------------------------------------------------------
    # DDP setup
    # ---------------------------------------------------------------------------
    assert torch.cuda.is_available(), "Training requires at least one GPU."
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Experiment directory
    if rank == 0:
        experiment_dir = f"{args.results_dir}/imagenet{args.image_size}-fm-unified"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory: {experiment_dir}")
        logger.info(f"Model: {args.model}, Resolutions: {resolutions}")
        logger.info(f"Stage lengths: {stage_lengths}, Boundaries (int): {T_boundaries}")
        logger.info(f"FM time boundaries: {s_boundaries}")
    else:
        logger = create_logger(None)
        experiment_dir = None
        checkpoint_dir = None

    # ---------------------------------------------------------------------------
    # Create single unified model
    # ---------------------------------------------------------------------------
    NUM_CLASSES = 1000

    if args.model not in DiTMultiRes_models:
        raise ValueError(f"Unknown model: {args.model!r}. Available: {list(DiTMultiRes_models.keys())}")

    patch_size = int(args.model.split("/")[1])
    base_size = resolutions[0] // patch_size

    repa_enabled = args.repa_weight > 0
    # Match the total REPA coefficient across the finest-only and all-stage
    # ablations; otherwise the latter silently gets num_stages times more
    # auxiliary supervision.
    repa_stage_count = num_stages if repa_enabled and args.repa_stages == "all" else 1
    repa_dim = dinov2_dimension(args.repa_model) if repa_enabled else None
    model = DiTMultiRes_models[args.model](
        num_classes=NUM_CLASSES,
        base_size=base_size,
        repa_dim=repa_dim,
        repa_depth=args.repa_depth,
        repa_projector_dim=args.repa_projector_dim,
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"DiTMultiRes model: {args.model}, base_size={base_size}, params={total_params:,}")

    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[device])

    # Single optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # ---------------------------------------------------------------------------
    # Create Flow Matching objects for each stage with proper time boundaries
    # ---------------------------------------------------------------------------
    # resolutions = [32, 16, 8] (finest first)
    # Sampling order is reversed (coarsest first): stage 2 (8x8) → stage 1 (16x16) → stage 0 (32x32)
    # But training iterates over all stages, each with its own time interval.
    #
    # This reverses the DDPM time direction:
    #   - t=0 is clean data, t=1000 is noise (DDPM)
    #   - In FM: t=0 is noise, t=1 is clean data
    # Therefore the coarsest resolution gets the first FM interval and the
    # finest resolution gets the last. The CLI lengths are already listed in
    # that FM data-time order (coarsest -> finest).

    flow_matchers = []
    for i in range(num_stages):
        # Stage i in resolution order (finest first): resolutions[i]
        # Its time interval: map to reversed order
        # Stage with resolution resolutions[i] gets time interval:
        #   stage_time_idx = num_stages - 1 - i
        #   s_k = s_boundaries[stage_time_idx]
        #   s_{k+1} = s_boundaries[stage_time_idx + 1]
        stage_time_idx = num_stages - 1 - i
        s_k = s_boundaries[stage_time_idx]
        s_k_plus_1 = s_boundaries[stage_time_idx + 1]

        # Drift: coarsest stage (last in resolution list) has no drift
        # All others have drift relative to the next coarser resolution
        if i < num_stages - 1:
            c = i  # drift stage
            # sample_scale: ratio between this resolution and the next coarser one
            scale = resolutions[i] // resolutions[i + 1]
        else:
            c = None  # coarsest, no drift
            scale = 2  # not used but needed as default

        fm = FlowMatchingWithDrift(
            c=c,
            sample_scale=scale,
            s_k=s_k,
            s_k_plus_1=s_k_plus_1,
        )
        flow_matchers.append(fm)
        logger.info(
            f"Stage {i}: res={resolutions[i]}x{resolutions[i]}, c={c}, "
            f"scale={scale}, t∈[{s_k:.3f}, {s_k_plus_1:.3f})"
        )

    # VAE
    if args.vae_path:
        vae = AutoencoderKL.from_pretrained(
            args.vae_path, local_files_only=True
        ).to(device)
    else:
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()

    repa_teacher = None
    if repa_enabled:
        repa_teacher = load_dinov2_teacher(
            args.repa_model,
            device,
            repo_or_dir=args.repa_repo,
            weights_path=args.repa_weights,
        )
        logger.info(
            f"REPA enabled: teacher={args.repa_model}, depth={args.repa_depth}, "
            f"weight={args.repa_weight}, stages={args.repa_stages}, dim={repa_dim}"
        )
    logger.info(
        f"FM time sampler: {args.time_sampler} "
        f"(mean={args.time_sampler_mean}, std={args.time_sampler_std}); "
        "sampling is stage-local and does not alter stage boundaries"
    )

    # ---------------------------------------------------------------------------
    # Dataset & DataLoader (with gradient accumulation)
    # ---------------------------------------------------------------------------
    world_size = dist.get_world_size()
    per_gpu_batch = int(args.global_batch_size // world_size)
    micro_batch_global = args.micro_batch_size if args.micro_batch_size else args.global_batch_size
    assert args.global_batch_size % micro_batch_global == 0, \
        f"global_batch_size ({args.global_batch_size}) must be divisible by micro_batch_size ({micro_batch_global})"
    assert micro_batch_global % world_size == 0, \
        f"micro_batch_size ({micro_batch_global}) must be divisible by world_size ({world_size})"
    micro_batch_per_gpu = micro_batch_global // world_size
    accum_steps = args.global_batch_size // micro_batch_global
    if accum_steps > 1:
        logger.info(f"Gradient accumulation: {accum_steps} steps × micro_batch={micro_batch_global} "
                    f"(per GPU: {micro_batch_per_gpu}) = global_batch {args.global_batch_size}")

    if args.data_source == "imagenet":
        if not args.data_path:
            raise ValueError("--data-path is required when --data-source=imagenet")
        dataset = ImageNetOriginalDataset(args.data_path, image_size=args.image_size)
        logger.info(f"Data source: original ImageNet (ADM center_crop_arr) from {args.data_path}")
    else:
        dataset = ImageNetHFDataset(image_size=args.image_size, dataset_path=args.data_path)
        logger.info(f"Data source: HuggingFace imagenet-1k-256x256")
    logger.info(f"Dataset size: {len(dataset):,} images")
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.global_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=micro_batch_per_gpu,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        drop_last=True,
    )

    # ---------------------------------------------------------------------------
    # Initialize EMA
    # ---------------------------------------------------------------------------
    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    # ---------------------------------------------------------------------------
    # Optional checkpoint resume
    # ---------------------------------------------------------------------------
    train_steps = 0
    start_epoch = 0
    resume_batch_in_epoch = 0

    if args.resume:
        logger.info(f"Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=f"cuda:{device}", weights_only=False)

        # A checkpoint trained with different FM time boundaries is not a
        # valid resume point: the same timestep would correspond to a
        # different resolution/path.  Refuse this silently-dangerous case.
        ckpt_resolutions = ckpt.get("resolutions")
        if ckpt_resolutions is not None and list(ckpt_resolutions) != resolutions:
            raise ValueError(
                f"Resume resolution mismatch: checkpoint={list(ckpt_resolutions)}, "
                f"current={resolutions}"
            )
        ckpt_stage_lengths = ckpt.get("stage_lengths")
        if ckpt_stage_lengths is not None and list(ckpt_stage_lengths) != stage_lengths:
            raise ValueError(
                "Resume FM stage-boundary mismatch: "
                f"checkpoint stage_lengths={list(ckpt_stage_lengths)}, "
                f"current={stage_lengths}. Do not resume the legacy FM-v2 "
                "300,300,400 checkpoint into the corrected 400,300,300 run."
            )
        ckpt_boundaries = ckpt.get("s_boundaries")
        if ckpt_boundaries is not None and not np.allclose(
            np.asarray(ckpt_boundaries, dtype=np.float64),
            np.asarray(s_boundaries, dtype=np.float64),
            rtol=0.0,
            atol=1e-8,
        ):
            raise ValueError(
                f"Resume FM boundary mismatch: checkpoint={ckpt_boundaries}, "
                f"current={s_boundaries}"
            )

        load_state_dict_with_optional_repa(
            model.module, ckpt["model"], allow_missing_repa=repa_enabled
        )
        load_state_dict_with_optional_repa(
            ema, ckpt["ema"], allow_missing_repa=repa_enabled
        )
        if not args.reset_optimizer_on_resume:
            opt.load_state_dict(ckpt["opt"])
        else:
            logger.info("Optimizer state reset on resume by request")
        if "step" in ckpt:
            train_steps = int(ckpt["step"])
        else:
            match = re.search(r"(\d+)\.pt$", args.resume)
            if match:
                train_steps = int(match.group(1))

        if "epoch" in ckpt and "batch_in_epoch" in ckpt:
            start_epoch = int(ckpt["epoch"])
            resume_batch_in_epoch = int(ckpt["batch_in_epoch"])
            logger.info(
                f"Resumed at step {train_steps}, epoch {start_epoch}, "
                f"next dataloader batch {resume_batch_in_epoch}"
            )
        else:
            # Backward-compatible fallback for checkpoints written before exact
            # dataloader progress was recorded.
            optimizer_steps_per_epoch = max(1, len(loader) // accum_steps)
            start_epoch = train_steps // optimizer_steps_per_epoch
            logger.info(f"Resumed at step {train_steps}, epoch ~{start_epoch}")
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

    log_steps = 0
    running_loss = 0.0
    micro_step = 0
    start_time = time()
    opt.zero_grad()

    if args.max_train_steps is not None and train_steps >= args.max_train_steps:
        logger.info(
            f"Checkpoint is already at step {train_steps}, which reaches "
            f"--max-train-steps={args.max_train_steps}. Nothing to do."
        )
        cleanup()
        return

    def save_checkpoint(step, epoch, batch_in_epoch):
        if rank == 0:
            checkpoint = {
                "model": model.module.state_dict(),
                "ema": ema.state_dict(),
                "opt": opt.state_dict(),
                "args": args,
                "step": step,
                "epoch": epoch,
                # Number of dataloader batches already consumed in this epoch.
                # Checkpoints are only written at optimizer boundaries, so a
                # resumed run can safely skip exactly this many micro-batches.
                "batch_in_epoch": batch_in_epoch,
                "resolutions": resolutions,
                "stage_lengths": stage_lengths,
                "s_boundaries": s_boundaries,
            }
            checkpoint_path = f"{checkpoint_dir}/{step:07d}.pt"
            temporary_path = f"{checkpoint_path}.tmp-{os.getpid()}"
            torch.save(checkpoint, temporary_path)
            # Atomic publication prevents a watchdog hard-stop from leaving a
            # partially written file at the canonical checkpoint path.
            os.replace(temporary_path, checkpoint_path)
            logger.info(f"Saved checkpoint at step {step:07d}")
        dist.barrier()

    def stop_file_requested():
        """Return one synchronized stop decision for every DDP rank."""
        requested = torch.zeros(1, dtype=torch.int32, device=device)
        if rank == 0 and args.stop_file and os.path.isfile(args.stop_file):
            requested.fill_(1)
        dist.broadcast(requested, src=0)
        return bool(requested.item())

    if args.max_train_steps is None:
        logger.info(f"Training for up to {args.epochs} epochs...")
    else:
        logger.info(
            f"Training until optimizer step {args.max_train_steps} "
            f"(epochs is only a safety ceiling: {args.epochs})..."
        )

    stop_training = False
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        epoch_start_time = time()

        batches_to_skip = resume_batch_in_epoch if epoch == start_epoch else 0
        if batches_to_skip:
            logger.info(
                f"Skipping {batches_to_skip} already-consumed dataloader "
                f"batches in resumed epoch {epoch}"
            )

        for batch_index, (x, y) in enumerate(loader):
            if batch_index < batches_to_skip:
                continue

            x = x.to(device)
            y = y.to(device)

            teacher_tokens = None
            if repa_enabled:
                teacher_tokens = dinov2_patch_tokens(repa_teacher, x)

            # Encode to latent space
            with torch.no_grad():
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)

            model_kwargs = dict(y=y)

            # Accumulate loss across all stages (same as DDPM version)
            # Each stage trains on its own time interval with its own resolution
            total_loss = 0.0
            for i in range(num_stages):
                res = resolutions[i]

                # Downsample latent to this stage's resolution
                if res == vae_latent_size:
                    x_stage = x
                else:
                    x_stage = downsample_latent(x.detach(), target_size=res)

                explicit_t = None
                if args.time_sampler != "uniform":
                    tau = sample_fm_time(
                        x_stage.shape[0],
                        device,
                        sampler=args.time_sampler,
                        mean=args.time_sampler_mean,
                        std=args.time_sampler_std,
                        eps=flow_matchers[i].train_eps / flow_matchers[i].stage_length,
                    )
                    explicit_t = flow_matchers[i].s_k + tau * flow_matchers[i].stage_length

                align_this_stage = repa_enabled and (
                    args.repa_stages == "all" or i == 0
                )
                captured = {}
                if align_this_stage:
                    def model_forward(x_t, t_model, **kwargs):
                        output, features = model(
                            x_t, t_model, return_features=True, **kwargs
                        )
                        captured["features"] = features
                        return output
                else:
                    model_forward = model

                # With uniform sampling, t=None preserves the historical RNG
                # path exactly.  Alternative densities are local to this stage.
                loss = flow_matchers[i].training_losses(
                    model_forward,
                    x_stage,
                    model_kwargs=model_kwargs,
                    t=explicit_t,
                )["loss"].mean()
                if align_this_stage:
                    student_features = captured["features"]
                    pooled_teacher = pool_teacher_tokens(
                        teacher_tokens, student_features.shape[1]
                    )
                    repa_loss = representation_alignment_loss(
                        student_features, pooled_teacher
                    ).mean()
                    loss = loss + (args.repa_weight / repa_stage_count) * repa_loss
                total_loss = total_loss + loss

            # Scale loss for gradient accumulation
            (total_loss / accum_steps).backward()

            running_loss += total_loss.item() / accum_steps
            micro_step += 1

            # Step optimizer after accumulation
            if micro_step % accum_steps == 0:
                opt.step()
                opt.zero_grad()
                update_ema(ema, model.module)

                log_steps += 1
                train_steps += 1

                # Logging
                if train_steps % args.log_every == 0:
                    torch.cuda.synchronize()
                    end_time = time()
                    steps_per_sec = log_steps / (end_time - start_time)
                    avg_step_time = (end_time - start_time) / log_steps
                    avg_loss = torch.tensor(running_loss / log_steps, device=device)
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / dist.get_world_size()

                    logger.info(
                        f"(step={train_steps:07d}) Loss={avg_loss:.4f}  "
                        f"Steps/s={steps_per_sec:.2f}  Step_time={avg_step_time:.3f}s"
                    )

                    if rank == 0 and timing_log_path:
                        import json as _json
                        try:
                            with open(timing_log_path, "a") as _f:
                                _f.write(_json.dumps({
                                    "event": "steps", "step": train_steps, "epoch": epoch,
                                    "avg_step_time_s": round(avg_step_time, 4),
                                    "loss": round(avg_loss, 6),
                                    "timestamp": end_time
                                }) + "\n")
                        except PermissionError:
                            pass

                    running_loss = 0.0
                    log_steps = 0
                    start_time = time()

                # Checkpointing
                if train_steps % args.ckpt_every == 0 and train_steps > 0:
                    save_checkpoint(train_steps, epoch, batch_index + 1)

                if stop_file_requested():
                    # The external SF-07:00 watchdog first requests this clean
                    # path, then has its own TERM/KILL deadline as a guarantee.
                    if train_steps % args.ckpt_every != 0:
                        save_checkpoint(train_steps, epoch, batch_index + 1)
                    logger.info(
                        f"Stop file detected at optimizer step {train_steps}: "
                        f"{args.stop_file}. Checkpoint saved; stopping cleanly."
                    )
                    stop_training = True
                    break

                if (
                    args.max_train_steps is not None
                    and train_steps >= args.max_train_steps
                ):
                    # Save a non-periodic final step as well.  For the paper
                    # run (400K with ckpt-every=5K), the periodic save above
                    # already wrote the exact target checkpoint.
                    if train_steps % args.ckpt_every != 0:
                        save_checkpoint(train_steps, epoch, batch_index + 1)
                    logger.info(
                        f"Reached --max-train-steps={args.max_train_steps}; "
                        "stopping cleanly."
                    )
                    stop_training = True
                    break

        # End of epoch
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

        resume_batch_in_epoch = 0

    model.eval()
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "RDD Flow Matching Unified: single DiTMultiRes model for all resolution stages. "
            "Time axis partitioned into stages (mirrors DDPM version)."
        )
    )
    parser.add_argument(
        "--model", type=str, default="DiT-XL/2",
        help="Model architecture (e.g. DiT-XL/2, DiT-L/2).",
    )
    parser.add_argument(
        "--resolutions", type=str, default="32,16,8",
        help="Comma-separated resolutions (strictly decreasing, finest first).",
    )
    parser.add_argument(
        "--stage-lengths", type=str, default="400,300,300",
        help="Comma-separated FM data-time interval lengths (coarsest first). "
             "For resolutions=32,16,8, 400,300,300 gives effective "
             "coarse/middle/fine allocation 400/300/300, which mirrors a "
             "DDPM finest/middle/coarse allocation of 300/300/400. "
             "Sum must equal --diffusion-steps.",
    )
    parser.add_argument("--diffusion-steps",   type=int,   default=1000,
                        help="Total diffusion steps (used to convert stage-lengths to FM [0,1] boundaries)")
    parser.add_argument("--results-dir",       type=str,   default="results_imagenet256")
    parser.add_argument("--data-path",         type=str,   default=None)
    parser.add_argument("--data-source",       type=str,   default="hf", choices=["hf", "imagenet"],
                        help="'hf' for HuggingFace parquet/arrow dataset; "
                             "'imagenet' for original ImageNet with ADM center_crop_arr.")
    parser.add_argument("--image-size",        type=int,   default=256, choices=[256, 512])
    parser.add_argument("--epochs",            type=int,   default=1400)
    parser.add_argument(
        "--max-train-steps", type=int, default=None,
        help="Stop after this many optimizer updates and save the exact final checkpoint. "
             "Useful for controlled 400K comparisons.",
    )
    parser.add_argument("--global-batch-size", type=int,   default=256)
    parser.add_argument("--micro-batch-size",  type=int,   default=None,
                        help="Global micro batch size per forward pass. "
                             "Gradient accumulation steps = global_batch_size / micro_batch_size.")
    parser.add_argument("--global-seed",       type=int,   default=0)
    parser.add_argument("--vae",               type=str,   default="ema", choices=["ema", "mse"])
    parser.add_argument(
        "--vae-path", type=str, default=None,
        help="Optional local diffusers-format VAE directory (offline-safe).",
    )
    parser.add_argument("--num-workers",       type=int,   default=4)
    parser.add_argument("--no-pin-memory",     action="store_true", help="Disable pin_memory in DataLoader")
    parser.add_argument("--log-every",         type=int,   default=100)
    parser.add_argument("--ckpt-every",        type=int,   default=5000)
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a checkpoint .pt to resume training from."
    )
    parser.add_argument(
        "--stop-file", type=str, default=None,
        help=(
            "When this file exists, all DDP ranks stop at the next optimizer "
            "boundary after writing an exact-step checkpoint. Intended for a "
            "job-scoped external watchdog."
        ),
    )
    parser.add_argument(
        "--time-sampler", type=str, default="uniform",
        choices=["uniform", "logit-normal"],
        help="Distribution inside each fixed RDD-FM stage interval.",
    )
    parser.add_argument("--time-sampler-mean", type=float, default=0.0)
    parser.add_argument("--time-sampler-std", type=float, default=1.0)
    parser.add_argument(
        "--repa-weight", type=float, default=0.0,
        help="REPA cosine-loss coefficient; zero leaves the legacy model unchanged.",
    )
    parser.add_argument("--repa-model", type=str, default="dinov2_vitb14")
    parser.add_argument("--repa-depth", type=int, default=8)
    parser.add_argument("--repa-projector-dim", type=int, default=2048)
    parser.add_argument(
        "--repa-stages",
        choices=["all", "finest"],
        default="all",
        help=(
            "Apply representation alignment at every RDD stage or only the "
            "finest stage. REPA remains disabled when --repa-weight=0."
        ),
    )
    parser.add_argument("--repa-repo", type=str, default=None)
    parser.add_argument("--repa-weights", type=str, default=None)
    parser.add_argument(
        "--reset-optimizer-on-resume", action="store_true",
        help="Reset optimizer when adding REPA parameters to a legacy checkpoint.",
    )
    args = parser.parse_args()
    main(args)
