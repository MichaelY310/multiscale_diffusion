"""
RDD Unified Training: single DiTMultiRes model handles ALL resolution stages.

Instead of using N separate DiT models for N resolutions, this script uses ONE
DiTMultiRes model with 2D RoPE that natively accepts variable-resolution inputs.

At each training step, for each stage (resolution), the same model is fed the
downsampled latent at that resolution. The 2D RoPE (Centralized Interpolative
Positional Encoding) ensures consistent positional information across resolutions.

Example:
  torchrun --nproc_per_node=8 train_RDD_imagenet256_unified.py \
    --model DiT-XL/2 \
    --resolutions 32,16,8 \
    --stage-lengths 200,300,500

Dataset: ImageNet-1k (local parquet, auto-detected by --image-size)
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

from diffusion import create_diffusion_with_drift
from diffusion.rdd_improvements import (
    pool_teacher_tokens,
    representation_alignment_loss,
    weighted_diffusion_loss,
)
from diffusion.timestep_sampler import LossAwareSampler, create_named_schedule_sampler
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
        if resolutions[i-1] % resolutions[i] != 0:
            raise ValueError(
                "Each finer resolution must be an integer multiple of the next "
                f"coarser resolution, got {resolutions[i-1]} -> {resolutions[i]}"
            )

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

    T_boundaries = [0]
    for length in stage_lengths:
        T_boundaries.append(T_boundaries[-1] + length)

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
        experiment_dir = f"{args.results_dir}/imagenet{args.image_size}-unified"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory: {experiment_dir}")
        logger.info(f"Model: {args.model}, Resolutions: {resolutions}")
        logger.info(f"Stage lengths: {stage_lengths}, Boundaries: {T_boundaries}")
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

    # base_size: the RoPE base grid size — set to the largest resolution / patch_size
    # so that the finest resolution uses positions [0, base_size] and coarser resolutions
    # use a subset (interpolated to the same range)
    patch_size = int(args.model.split("/")[1])
    base_size = resolutions[0] // patch_size  # e.g. 32/2 = 16

    repa_enabled = args.repa_weight > 0
    # Keep the total auxiliary-loss coefficient comparable between
    # ``finest`` and ``all``.  Without this normalization, multiscale REPA
    # would receive num_stages times more REPA weight and the ablation would
    # confound scale alignment with a larger regularizer.
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

    # Single optimizer for the unified model
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Create diffusion objects for each stage (only timestep boundaries differ)
    diffusions = []
    for i in range(num_stages):
        T_c = T_boundaries[i]
        T_c_plus_1 = T_boundaries[i + 1]
        # Project the clean target to the *actual next configured stage*.
        # Adjacent 32->16->8 runs use scale=2, while a valid two-stage
        # 32->8 topology must use scale=4.  Leaving the factory default at 2
        # would silently train the requested 8->32 sampler against a 16x16
        # projection target.
        projection_scale = (
            resolutions[i] // resolutions[i + 1]
            if i < num_stages - 1 else 2
        )
        # When --no-drift is set, all stages use c=None (no drift term)
        if args.no_drift:
            c = None
        else:
            c = i if i < num_stages - 1 else None
        diffusion = create_diffusion_with_drift(
            timestep_respacing="",
            diffusion_steps=args.diffusion_steps,
            c=c,
            T_c=T_c,
            T_c_plus_1=T_c_plus_1,
            sample_scale=projection_scale,
        )
        diffusions.append(diffusion)
        logger.info(
            f"Stage {i}: {resolutions[i]}x{resolutions[i]}, c={c}, "
            f"projection_scale={projection_scale}, T=[{T_c},{T_c_plus_1})"
        )
    timestep_samplers = None
    if args.timestep_sampler != "uniform":
        timestep_samplers = [
            create_named_schedule_sampler(args.timestep_sampler, diffusion)
            for diffusion in diffusions
        ]
        logger.info(f"Timestep sampler: {args.timestep_sampler} (unbiased importance correction)")
    logger.info(
        f"Loss weighting: {args.loss_weighting} "
        f"(min_snr_gamma={args.min_snr_gamma}, p2_gamma={args.p2_gamma}, p2_k={args.p2_k})"
    )
    if args.no_drift:
        logger.info("*** NO-DRIFT MODE: d^c=0 for all stages (curriculum-only ablation) ***")
    if args.drift_scale != 1.0:
        for diff in diffusions:
            diff.drift_scale_multiplier = args.drift_scale
        logger.info(f"*** DRIFT SCALE: {args.drift_scale} (drift magnitude multiplier) ***")

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

    # ---------------------------------------------------------------------------
    # Dataset & DataLoader (with gradient accumulation support)
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
        logger.info(f"Data source: HuggingFace imagenet-1k-256x256 (Lanczos preprocessed)")
    logger.info(f"Dataset size: {len(dataset):,} images")
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed,
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
    logger.info(f"Dataset size: {len(dataset):,} images")

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

    if args.resume:
        logger.info(f"Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=f"cuda:{device}", weights_only=False)
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
        match = re.search(r"(\d+)\.pt$", args.resume)
        if match:
            train_steps = int(match.group(1))
            start_epoch = train_steps // len(loader)
            logger.info(f"Resumed at step {train_steps}, epoch ~{start_epoch}")
        dist.barrier()

    if args.max_train_steps is not None and train_steps >= args.max_train_steps:
        logger.info(
            f"Checkpoint is already at step {train_steps}, reaching "
            f"--max-train-steps={args.max_train_steps}; nothing to do."
        )
        cleanup()
        return

    def save_checkpoint(reason):
        """Save one rank-0 checkpoint atomically at the current optimizer step."""
        if rank != 0:
            return
        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
        temporary_path = f"{checkpoint_path}.tmp-{os.getpid()}"
        torch.save(
            {
                "model": model.module.state_dict(),
                "ema": ema.state_dict(),
                "opt": opt.state_dict(),
                "args": args,
                "step": train_steps,
                "resolutions": resolutions,
                "stage_lengths": stage_lengths,
            },
            temporary_path,
        )
        os.replace(temporary_path, checkpoint_path)
        logger.info(f"Saved {reason} checkpoint at step {train_steps:07d}")

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
    opt.zero_grad()  # Initialize gradients for accumulation

    if args.max_train_steps is None:
        logger.info(f"Training for up to {args.epochs} epochs...")
    else:
        logger.info(
            f"Training until optimizer step {args.max_train_steps} "
            f"(epochs safety ceiling: {args.epochs})..."
        )
    stop_training = False
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        epoch_start_time = time()

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            teacher_tokens = None
            if repa_enabled:
                teacher_tokens = dinov2_patch_tokens(repa_teacher, x)

            # Encode to latent space
            with torch.no_grad():
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)

            model_kwargs = dict(y=y)

            # Accumulate loss across all stages, single backward pass
            total_loss = 0.0
            for i in range(num_stages):
                res = resolutions[i]
                T_c = T_boundaries[i]
                T_c_plus_1 = T_boundaries[i + 1]
                local_steps = T_c_plus_1 - T_c

                # Downsample latent to this stage's resolution
                if res == vae_latent_size:
                    x_stage = x
                else:
                    x_stage = downsample_latent(x.detach(), target_size=res)

                importance_weights = None
                if timestep_samplers is None:
                    # Preserve the exact legacy RNG path when optional sampling is disabled.
                    t = torch.randint(0, local_steps, (x_stage.shape[0],), device=device)
                else:
                    t, importance_weights = timestep_samplers[i].sample(x_stage.shape[0], device)

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

                terms = diffusions[i].training_losses(
                    model_forward, x_stage, t, model_kwargs
                )
                if timestep_samplers is not None and isinstance(
                    timestep_samplers[i], LossAwareSampler
                ):
                    timestep_samplers[i].update_with_local_losses(t, terms["loss"].detach())
                loss_vector, _ = weighted_diffusion_loss(
                    terms,
                    diffusions[i],
                    t,
                    args.loss_weighting,
                    min_snr_gamma=args.min_snr_gamma,
                    p2_gamma=args.p2_gamma,
                    p2_k=args.p2_k,
                    importance_weights=importance_weights,
                )
                loss = loss_vector.mean()
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

            # Only step optimizer after accumulating enough micro-batches
            if micro_step % accum_steps == 0:
                opt.step()
                opt.zero_grad()
                update_ema(ema, model.module)

                log_steps += 1
                train_steps += 1

                # A launcher-scoped watchdog creates a unique stop file at the
                # scheduled wall-clock deadline.  Synchronize the request so
                # every DDP rank takes the same branch.  This check happens at
                # optimizer-step boundaries, so an exact resumable checkpoint
                # can be written without leaving partial accumulated gradients.
                scheduled_stop = False
                if args.stop_file:
                    stop_requested = torch.tensor(
                        [1 if os.path.exists(args.stop_file) else 0],
                        device=device,
                        dtype=torch.int32,
                    )
                    dist.all_reduce(stop_requested, op=dist.ReduceOp.MAX)
                    scheduled_stop = bool(stop_requested.item())

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
                            pass  # skip timing log if no write permission

                    running_loss = 0.0
                    log_steps = 0
                    start_time = time()

                # Save once when a regular interval, final step, or scheduled
                # stop is reached.  os.replace() above prevents auto-resume
                # from observing a half-written checkpoint after a hard stop.
                reached_max_steps = (
                    args.max_train_steps is not None
                    and train_steps >= args.max_train_steps
                )
                regular_checkpoint = train_steps > 0 and train_steps % args.ckpt_every == 0
                if regular_checkpoint or reached_max_steps or scheduled_stop:
                    if regular_checkpoint:
                        reason = "regular"
                    elif scheduled_stop:
                        reason = "scheduled-stop"
                    else:
                        reason = "final"
                    save_checkpoint(reason)
                    dist.barrier()

                if scheduled_stop:
                    logger.info(
                        f"Scheduled stop token observed at optimizer step {train_steps:07d}: "
                        f"{args.stop_file}"
                    )
                    stop_training = True
                    break

                if reached_max_steps:
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

    model.eval()
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "RDD Unified: single DiTMultiRes model for all resolution stages. "
            "Uses 2D RoPE with Centralized Interpolative Positional Encoding."
        )
    )
    parser.add_argument(
        "--model", type=str, default="DiT-XL/2",
        help="Model architecture (e.g. DiT-XL/2, DiT-L/2). A SINGLE model for all resolutions.",
    )
    parser.add_argument(
        "--resolutions", type=str, default="32,16,8",
        help="Comma-separated resolutions (strictly decreasing).",
    )
    parser.add_argument(
        "--stage-lengths", type=str, default="200,300,500",
        help="Comma-separated timestep lengths per stage. Sum must equal --diffusion-steps.",
    )
    parser.add_argument("--results-dir",       type=str,   default="results_imagenet256")
    parser.add_argument("--data-path",         type=str,   default=None)
    parser.add_argument("--data-source",       type=str,   default="hf", choices=["hf", "imagenet"],
                        help="'hf' for HuggingFace parquet/arrow dataset; "
                             "'imagenet' for original ImageNet with ADM center_crop_arr.")
    parser.add_argument("--image-size",        type=int,   default=256, choices=[256, 512])
    parser.add_argument("--diffusion-steps",   type=int,   default=1000)
    parser.add_argument("--epochs",            type=int,   default=1400)
    parser.add_argument(
        "--max-train-steps", type=int, default=None,
        help="Stop after this many optimizer updates and save the exact final checkpoint.",
    )
    parser.add_argument(
        "--stop-file", type=str, default=None,
        help=(
            "Optional launcher-scoped file whose appearance requests a clean stop "
            "after the next optimizer update and an exact-step checkpoint."
        ),
    )
    parser.add_argument("--global-batch-size", type=int,   default=256)
    parser.add_argument("--micro-batch-size",  type=int,   default=None,
                        help="Global micro batch size per forward pass (split across GPUs). "
                             "Gradient accumulation steps = global_batch_size / micro_batch_size. "
                             "E.g., --global-batch-size 256 --micro-batch-size 64 means each GPU "
                             "processes 64/8=8 images per forward, accumulates 256/64=4 times. "
                             "Default: same as global_batch_size (no accumulation).")
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
        help="Path to a single checkpoint file to resume from.",
    )
    parser.add_argument(
        "--no-drift", action="store_true",
        help="Disable drift term (d^c=0). Model still trains on multi-resolution inputs "
             "with the same stage-length schedule, but the forward process has no drift. "
             "Used for the 'curriculum without drift' ablation (exp01).",
    )
    parser.add_argument(
        "--drift-scale", type=float, default=1.0,
        help="Multiplier on the drift magnitude (default 1.0). "
             "Used for the drift-lambda sweep ablation (exp06). "
             "Values <1 weaken drift, >1 strengthen it, 0 is equivalent to --no-drift.",
    )
    parser.add_argument(
        "--loss-weighting", type=str, default="none",
        choices=["none", "min-snr", "p2"],
        help="Optional denoising-MSE weighting; default preserves the legacy RDD objective.",
    )
    parser.add_argument("--min-snr-gamma", type=float, default=5.0)
    parser.add_argument("--p2-gamma", type=float, default=1.0)
    parser.add_argument("--p2-k", type=float, default=1.0)
    parser.add_argument(
        "--timestep-sampler", type=str, default="uniform",
        choices=["uniform", "loss-second-moment"],
        help="Optional unbiased importance sampler from Improved Diffusion.",
    )
    parser.add_argument(
        "--repa-weight", type=float, default=0.0,
        help="REPA cosine-loss coefficient. Zero disables the teacher and adds no model parameters.",
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
    parser.add_argument(
        "--repa-repo", type=str, default=None,
        help="Optional local facebookresearch/dinov2 repository for offline torch.hub loading.",
    )
    parser.add_argument("--repa-weights", type=str, default=None)
    parser.add_argument(
        "--reset-optimizer-on-resume", action="store_true",
        help="Required when adding REPA parameters while fine-tuning a legacy checkpoint.",
    )
    args = parser.parse_args()
    main(args)
