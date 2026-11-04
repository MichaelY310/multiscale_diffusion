"""
RDD-Wan Training Script: Multi-stage RDD-Video training with Wan 2.1 backbone.

Supports:
  - Loading pretrained Wan 2.1 checkpoint (from_pretrained or manual state_dict)
  - Two downsampling modes: "latent" (avg-pool) and "pixel" (pixel-space downsample + VAE)
  - Per-step single-stage training (sample one stage per batch, PyramidalWan style)
  - Full-parameter fine-tuning with AdamW, EMA, bfloat16 mixed precision
  - DDP distributed training with gradient clipping
  - WandB logging (optional)
  - Checkpoint saving/resuming
  - CPU pipeline validation mode

Usage (single node 8-GPU):
    torchrun --nproc_per_node=8 train_rdd_wan.py \
        --pretrained-dir /path/to/wan2.1_checkpoint \
        --data-dir /path/to/latents \
        --stage-configs "4,2,20;2,1,20;1,1,10" \
        --global-batch-size 24 \
        --max-steps 5000

Usage (CPU test):
    python train_rdd_wan.py --cpu-test
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F_torch
import math
import argparse
import logging
import os
import sys
import glob as glob_module
import json
import random
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from time import time
from typing import Optional, List, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "Wan2.1"))

# Detect CPU-only environment and patch flash_attention before importing WanModel
_CPU_ONLY = not torch.cuda.is_available()
if _CPU_ONLY:
    import importlib.util
    import types

    _wan_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "Wan2.1")

    # Pre-create wan package namespace to avoid wan/__init__.py (which imports T5 needing CUDA)
    wan_pkg = types.ModuleType("wan")
    wan_pkg.__path__ = [os.path.join(_wan_root, "wan")]
    wan_pkg.__package__ = "wan"
    sys.modules["wan"] = wan_pkg

    wan_modules_pkg = types.ModuleType("wan.modules")
    wan_modules_pkg.__path__ = [os.path.join(_wan_root, "wan", "modules")]
    wan_modules_pkg.__package__ = "wan.modules"
    sys.modules["wan.modules"] = wan_modules_pkg

    # Load attention module with CPU fallback
    _attn_path = os.path.join(_wan_root, "wan", "modules", "attention.py")
    _spec = importlib.util.spec_from_file_location("wan.modules.attention", _attn_path)
    _attn_mod = importlib.util.module_from_spec(_spec)
    sys.modules["wan.modules.attention"] = _attn_mod
    _spec.loader.exec_module(_attn_mod)

    # Patch flash_attention with CPU-compatible scaled dot product attention
    def _cpu_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.,
                       softmax_scale=None, q_scale=None, causal=False,
                       window_size=(-1, -1), deterministic=False,
                       dtype=torch.bfloat16, version=None):
        if q_scale is not None:
            q = q * q_scale
        q = q.transpose(1, 2).float()
        k = k.transpose(1, 2).float()
        v = v.transpose(1, 2).float()
        out = F_torch.scaled_dot_product_attention(q, k, v, is_causal=causal, dropout_p=dropout_p)
        return out.transpose(1, 2).contiguous()

    _attn_mod.flash_attention = _cpu_attention

    # Now load model module
    _model_path = os.path.join(_wan_root, "wan", "modules", "model.py")
    _spec_m = importlib.util.spec_from_file_location("wan.modules.model", _model_path)
    _model_mod = importlib.util.module_from_spec(_spec_m)
    sys.modules["wan.modules.model"] = _model_mod
    _spec_m.loader.exec_module(_model_mod)
    WanModel = _model_mod.WanModel
else:
    from wan.modules.model import WanModel

from flow_matching_video import RDDVideoSchedule, SpatioTemporalDownsample


# ─── Utilities ────────────────────────────────────────────────────────────────

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """Exponential moving average update."""
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir, rank):
    if rank == 0:
        os.makedirs(logging_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(),
                      logging.FileHandler(f"{logging_dir}/log.txt")])
    logger = logging.getLogger(__name__)
    if rank != 0:
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.WARNING)
    return logger


def parse_stage_configs(config_str: str) -> List[dict]:
    """
    Parse "4,2,20;2,1,20;1,1,10" -> list of dicts.
    Format: spatial_factor,temporal_factor,num_steps[,drift_spatial,drift_temporal]
    """
    stages = []
    for part in config_str.split(";"):
        parts = [p.strip() for p in part.strip().split(",")]
        cfg = {
            "spatial_factor": int(parts[0]),
            "temporal_factor": int(parts[1]),
            "num_steps": int(parts[2]),
        }
        if len(parts) >= 5:
            cfg["drift_spatial"] = int(parts[3])
            cfg["drift_temporal"] = int(parts[4])
        stages.append(cfg)
    return stages


def compute_seq_len(F, H, W, patch_size=(1, 2, 2)):
    """Compute sequence length after patchification."""
    return (F // patch_size[0]) * (H // patch_size[1]) * (W // patch_size[2])


def get_lr_schedule(step, warmup_steps, max_lr):
    """Linear warmup then constant LR."""
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    return max_lr


def set_seed(seed, rank=0):
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_stage_index(stage_lengths: List[int], mode: str = "proportional") -> int:
    """Sample one resolution stage for a batch.

    ``proportional`` is equivalent to sampling a global FM time uniformly and
    then selecting its stage. ``uniform`` intentionally over-samples short
    stages and is kept only as an ablation. ``balanced`` is handled exactly by
    the distributed accumulation loop; its standalone fallback is uniform.
    """
    if mode == "proportional":
        return random.choices(range(len(stage_lengths)), weights=stage_lengths, k=1)[0]
    if mode in {"uniform", "balanced"}:
        return random.randrange(len(stage_lengths))
    raise ValueError(f"Unknown stage sampling mode: {mode}")


# ─── Datasets ────────────────────────────────────────────────────────────────

class MultiResLatentDataset(Dataset):
    """
    Loads pre-computed multi-resolution latent files (from precompute_latents.py).
    Each file is a dict:
      - "latents": list of [C, F_i, H_i, W_i] tensors (one per stage)
      - "text_emb": [L, C_text] tensor (T5 embedding)
    This is the FASTEST training mode — no VAE or downsampling at training time.
    """

    def __init__(self, data_dir: str, num_stages: int = 3,
                 text_len: int = 512, text_dim: int = 4096,
                 allow_missing_text: bool = False):
        self.data_dir = data_dir
        self.num_stages = num_stages
        self.text_len = text_len
        self.text_dim = text_dim
        self.allow_missing_text = allow_missing_text
        self.files = sorted(glob_module.glob(os.path.join(data_dir, "**/*_multires.pt"), recursive=True))
        if not self.files:
            # Fallback: try any .pt files
            self.files = sorted(glob_module.glob(os.path.join(data_dir, "**/*.pt"), recursive=True))
        assert len(self.files) > 0, f"No multi-res .pt files found in {data_dir}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location="cpu", weights_only=True)

        if isinstance(data, dict) and "latents" in data:
            # Multi-res format from precompute_latents.py
            latents = [l.float() for l in data["latents"]]
            if len(latents) != self.num_stages:
                raise ValueError(
                    f"{self.files[idx]} has {len(latents)} stages; expected {self.num_stages}"
                )
        elif isinstance(data, dict) and "latent" in data:
            raise ValueError(
                f"{self.files[idx]} is a single-resolution record. Run "
                "precompute_latents.py or train with --data-mode single."
            )
        else:
            raise ValueError(f"Unexpected data format in {self.files[idx]}")

        text_emb = data.get("text_emb")
        if text_emb is None:
            if not self.allow_missing_text:
                raise ValueError(
                    f"{self.files[idx]} has no text_emb; refusing to silently train an "
                    "unconditional model. Regenerate/cache T5 contexts, or pass "
                    "--allow-missing-text only for a smoke test."
                )
            text_emb = torch.zeros(self.text_len, self.text_dim)
        text_emb = text_emb.float()

        # Pad/truncate text_emb
        if text_emb.shape[0] < self.text_len:
            pad = torch.zeros(self.text_len - text_emb.shape[0], text_emb.shape[1])
            text_emb = torch.cat([text_emb, pad], dim=0)
        elif text_emb.shape[0] > self.text_len:
            text_emb = text_emb[:self.text_len]

        return latents, text_emb


def multires_collate_fn(batch):
    """
    Custom collate for MultiResLatentDataset.
    Each sample has variable-sized latents per stage, so we batch per stage.
    
    Returns:
        latents_per_stage: list of [B, C, F_i, H_i, W_i] tensors
        text_embs: [B, L, D] tensor
    """
    latents_list = [item[0] for item in batch]
    text_embs = torch.stack([item[1] for item in batch])

    num_stages = len(latents_list[0])
    latents_per_stage = []
    for s in range(num_stages):
        stage_latents = [sample[s] for sample in latents_list]
        # All samples should have same shape at each stage
        latents_per_stage.append(torch.stack(stage_latents))

    return latents_per_stage, text_embs


class LatentDataset(Dataset):
    """
    Loads pre-encoded .pt latent files (single full-resolution).
    Each file should contain a dict with:
      - "latent": [C, F, H, W] tensor (float32 or bfloat16)
      - "text_emb": [L, C_text] tensor (T5 embedding)
    Or just a tensor of shape [C, F, H, W] (text_emb will be zeros).
    """

    def __init__(self, data_dir: str, text_len: int = 512, text_dim: int = 4096,
                 allow_missing_text: bool = False):
        self.data_dir = data_dir
        self.text_len = text_len
        self.text_dim = text_dim
        self.allow_missing_text = allow_missing_text
        self.files = sorted(glob_module.glob(os.path.join(data_dir, "**/*.pt"), recursive=True))
        if not self.files:
            self.files = sorted(glob_module.glob(os.path.join(data_dir, "**/*.pth"), recursive=True))
        assert len(self.files) > 0, f"No .pt/.pth files found in {data_dir}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location="cpu", weights_only=True)
        if isinstance(data, dict):
            latent = data["latent"].float()
            text_emb = data.get("text_emb")
        else:
            latent = data.float()
            text_emb = None
        if text_emb is None:
            if not self.allow_missing_text:
                raise ValueError(
                    f"{self.files[idx]} has no text_emb; refusing conditional training."
                )
            text_emb = torch.zeros(self.text_len, self.text_dim)
        text_emb = text_emb.float()
        # Ensure text_emb has correct length
        if text_emb.shape[0] < self.text_len:
            pad = torch.zeros(self.text_len - text_emb.shape[0], text_emb.shape[1])
            text_emb = torch.cat([text_emb, pad], dim=0)
        elif text_emb.shape[0] > self.text_len:
            text_emb = text_emb[:self.text_len]
        return latent, text_emb


class PixelVideoDataset(Dataset):
    """
    Loads pixel-space videos for pixel-mode training.
    Each file should be a .pt dict with:
      - "video": [C, T, H, W] tensor in [-1, 1] (pixel space)
      - "text_emb": [L, C_text] tensor
    Or a video tensor directly.
    """

    def __init__(self, data_dir: str, text_len: int = 512, text_dim: int = 4096,
                 allow_missing_text: bool = False):
        self.data_dir = data_dir
        self.text_len = text_len
        self.text_dim = text_dim
        self.allow_missing_text = allow_missing_text
        self.files = sorted(glob_module.glob(os.path.join(data_dir, "**/*.pt"), recursive=True))
        if not self.files:
            self.files = sorted(glob_module.glob(os.path.join(data_dir, "**/*.pth"), recursive=True))
        assert len(self.files) > 0, f"No .pt/.pth files found in {data_dir}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location="cpu", weights_only=True)
        if isinstance(data, dict):
            video = data["video"].float()
            text_emb = data.get("text_emb")
        else:
            video = data.float()
            text_emb = None
        if text_emb is None:
            if not self.allow_missing_text:
                raise ValueError(
                    f"{self.files[idx]} has no text_emb; refusing conditional training."
                )
            text_emb = torch.zeros(self.text_len, self.text_dim)
        text_emb = text_emb.float()
        if text_emb.shape[0] < self.text_len:
            pad = torch.zeros(self.text_len - text_emb.shape[0], text_emb.shape[1])
            text_emb = torch.cat([text_emb, pad], dim=0)
        elif text_emb.shape[0] > self.text_len:
            text_emb = text_emb[:self.text_len]
        return video, text_emb


class DummyLatentDataset(Dataset):
    """Random video latent dataset for pipeline testing."""

    def __init__(self, num_samples=1000, channels=16, frames=21, height=60, width=104,
                 text_len=512, text_dim=4096):
        self.num_samples = num_samples
        self.shape = (channels, frames, height, width)
        self.text_len = text_len
        self.text_dim = text_dim

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        video = torch.randn(*self.shape)
        text_emb = torch.randn(self.text_len, self.text_dim)
        return video, text_emb


class PromptDataset(Dataset):
    """
    Dataset that loads text prompts for self-distillation training.
    No videos needed — the teacher model generates x_1 online.
    
    Expects a text file with one prompt per line, or a JSONL file with {"prompt": "..."}.
    """

    def __init__(self, prompt_file: str, text_len: int = 512, text_dim: int = 4096):
        self.text_len = text_len
        self.text_dim = text_dim
        self.prompts = []

        if prompt_file.endswith(".jsonl"):
            with open(prompt_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        self.prompts.append(obj.get("prompt", obj.get("text", "")))
        elif prompt_file.endswith(".json"):
            with open(prompt_file, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        self.prompts.append(item)
                    elif isinstance(item, dict):
                        self.prompts.append(item.get("prompt", item.get("text", "")))
        else:
            # Plain text file, one prompt per line
            with open(prompt_file, "r") as f:
                self.prompts = [line.strip() for line in f if line.strip()]

        assert len(self.prompts) > 0, f"No prompts found in {prompt_file}"

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


def prompt_collate_fn(batch):
    """Collate function for PromptDataset — just returns list of strings."""
    return batch


# ─── Pixel-Mode VAE Processing ───────────────────────────────────────────────

def downsample_pixel_video(video: torch.Tensor, spatial_factor: int, temporal_factor: int) -> torch.Tensor:
    """
    Downsample a pixel-space video [C, T, H, W] using bilinear spatial and
    linear temporal interpolation.

    Args:
        video: [C, T, H, W] in [-1, 1]
        spatial_factor: spatial downsampling factor
        temporal_factor: temporal downsampling factor
    Returns:
        [C, T//temporal_factor, H//spatial_factor, W//spatial_factor]
    """
    C, T, H, W = video.shape

    if temporal_factor > 1 and T > 1:
        # Temporal downsample via avg pooling along time
        T_new = T // temporal_factor
        # Reshape to [C, T_new, temporal_factor, H, W] and average
        pad_t = (temporal_factor - T % temporal_factor) % temporal_factor
        if pad_t > 0:
            video = torch.cat([video, video[:, -1:].expand(-1, pad_t, -1, -1)], dim=1)
            T = T + pad_t
        T_new = T // temporal_factor
        video = video.view(C, T_new, temporal_factor, H, W).mean(dim=2)
        T = T_new

    if spatial_factor > 1:
        # Spatial downsample per frame using bilinear interpolation
        H_new = H // spatial_factor
        W_new = W // spatial_factor
        # video: [C, T, H, W] -> [T, C, H, W] for F.interpolate
        video_frames = video.permute(1, 0, 2, 3)  # [T, C, H, W]
        video_frames = F_torch.interpolate(video_frames, size=(H_new, W_new), mode='bilinear', align_corners=False)
        video = video_frames.permute(1, 0, 2, 3)  # [C, T, H_new, W_new]

    return video


def encode_pixel_videos_for_stage(vae, pixel_videos: torch.Tensor, spatial_factor: int,
                                  temporal_factor: int, device: torch.device) -> torch.Tensor:
    """
    Downsample pixel videos and encode with VAE for a specific stage.

    Args:
        vae: WanVAE instance
        pixel_videos: [B, C, T, H, W] pixel videos in [-1, 1]
        spatial_factor: spatial factor for this stage
        temporal_factor: temporal factor for this stage
        device: target device
    Returns:
        [B, C_latent, F_latent, H_latent, W_latent] encoded latents
    """
    B = pixel_videos.shape[0]
    latents = []
    for i in range(B):
        video_i = pixel_videos[i]  # [C, T, H, W]
        # Downsample in pixel space
        video_low = downsample_pixel_video(video_i, spatial_factor, temporal_factor)
        # Encode with VAE: expects list of [C, T, H, W]
        video_low = video_low.to(device)
        with torch.no_grad():
            encoded = vae.encode([video_low])  # returns list of [C_latent, F_latent, H_l, W_l]
        latents.append(encoded[0])
    return torch.stack(latents, dim=0)


# ─── Online synthetic data: teacher generates terminal x_1 ─────────────────

@torch.no_grad()
def teacher_generate_x1(
    teacher_model,
    text_embeddings: torch.Tensor,
    null_text_embeddings: torch.Tensor,
    latent_shape: tuple,
    num_steps: int = 50,
    cfg_scale: float = 5.0,
    device: torch.device = None,
):
    """
    Use the frozen original Wan model to generate a clean latent with Euler.

    Wan uses noise time sigma: x_sigma=(1-sigma)x_1+sigma*epsilon and predicts
    dx/dsigma=epsilon-x_1. Generation therefore integrates sigma from 1 to 0.
    
    This is the full-resolution single-stage generation that serves as target data
    for the student (RDD pyramidal model).
    
    Args:
        teacher_model: frozen WanModel (original pretrained)
        text_embeddings: [B, L, D] conditional T5 embeddings
        null_text_embeddings: [B, L, D] negative-prompt T5 embeddings
        latent_shape: (B, C, F, H, W) target latent shape
        num_steps: number of Euler steps for teacher ODE
        cfg_scale: classifier-free guidance scale
        device: torch device
    Returns:
        x_1: [B, C, F, H, W] generated clean latent (detached)
    """
    B, C, F, H, W = latent_shape
    seq_len = compute_seq_len(F, H, W, (1, 2, 2))
    
    # Start from pure noise at sigma=1.
    x = torch.randn(latent_shape, device=device, dtype=torch.bfloat16)
    
    # Euler integration from sigma=1 to sigma=0.
    dsigma = -1.0 / num_steps
    context = [text_embeddings[i] for i in range(B)]
    null_context = [null_text_embeddings[i] for i in range(B)]
    
    for step_i in range(num_steps):
        sigma = 1.0 + step_i * dsigma
        t_tensor = torch.full(
            (B,), sigma * 1000.0, device=device, dtype=torch.bfloat16
        )
        
        # Predict velocity with classifier-free guidance
        if cfg_scale > 1.0:
            # Conditional prediction
            v_cond = torch.stack(teacher_model(
                [x[i] for i in range(B)], t=t_tensor, context=context, seq_len=seq_len
            ))
            # Unconditional prediction
            v_uncond = torch.stack(teacher_model(
                [x[i] for i in range(B)], t=t_tensor, context=null_context, seq_len=seq_len
            ))
            # CFG combination
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = torch.stack(teacher_model(
                [x[i] for i in range(B)], t=t_tensor, context=context, seq_len=seq_len
            ))
        
        # Wan velocity is dx/dsigma and dsigma is negative during generation.
        x = x + dsigma * v
    
    return x.detach()


# ─── Training Core ────────────────────────────────────────────────────────────

def train_step_single_stage(
    model,
    schedule: RDDVideoSchedule,
    video_latents: torch.Tensor,
    text_embeddings: torch.Tensor,
    stage_idx: int,
    device: torch.device,
    patch_size=(1, 2, 2),
):
    """
    One training step: compute FM loss for a single stage.
    This follows PyramidalWan: one stage per batch per step.

    Args:
        model: WanModel (unwrapped)
        schedule: RDDVideoSchedule
        video_latents: [B, C, F, H, W] full-resolution latents (latent mode)
                       or already-downsampled latents for this stage (pixel mode)
        text_embeddings: [B, L, C_text]
        stage_idx: which stage to train
        device: torch device
        patch_size: WanModel patch size
    Returns:
        loss: scalar tensor with grad
    """
    B = video_latents.shape[0]

    # Downsample to this stage's resolution (latent mode does actual downsampling;
    # pixel mode may pass pre-downsampled latents directly)
    x_stage = schedule.downsample_latent(video_latents, stage_idx)
    _, C_s, F_s, H_s, W_s = x_stage.shape

    # Sequence length for Wan's positional encoding
    seq_len = compute_seq_len(F_s, H_s, W_s, patch_size)

    # Prepare context (list of [L, C_text] tensors)
    context = [text_embeddings[i] for i in range(B)]

    # Create model_fn closure
    # Note: training_losses already splits x_t into a list before calling model_fn
    def model_fn(x_list, t, **kwargs):
        out_list = model(x_list, t=t, context=context, seq_len=seq_len)
        return out_list

    # Compute FM loss for this stage
    loss_dict = schedule.flow_matchers[stage_idx].training_losses(
        model_fn, x_stage, model_kwargs={}
    )

    return loss_dict["loss"].mean()


# ─── Main Distributed Training ───────────────────────────────────────────────

def main_distributed(args):
    """Main training loop with DDP."""
    if args.downsample_mode != "latent":
        raise NotImplementedError(
            "The current analytic RDD-FM clean inverse is valid for the latent-space "
            "average-pool/repeat idempotent projection. Pixel downsample + VAE "
            "re-encode is nonlinear and is not a drop-in replacement; it needs an "
            "x-prediction head or a transform-specific inverse. Use --downsample-mode "
            "latent for RDD-Wan."
        )
    # Parse configs
    stage_configs = parse_stage_configs(args.stage_configs)
    stage_lengths = [int(x) for x in args.stage_lengths.split(",")]
    schedule = RDDVideoSchedule(
        stage_configs,
        stage_lengths=stage_lengths,
        downsample_mode=args.downsample_mode,
        drift_scale=args.drift_scale,
    )
    num_stages = len(stage_configs)

    # DDP setup
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    set_seed(args.seed, rank)

    if args.global_batch_size % world_size != 0:
        raise ValueError(
            f"global batch size {args.global_batch_size} must be divisible by world size {world_size}"
        )
    local_batch_size = args.global_batch_size // world_size
    micro_batch_size = args.micro_batch_size or local_batch_size
    if micro_batch_size <= 0 or local_batch_size % micro_batch_size != 0:
        raise ValueError(
            f"local batch {local_batch_size} must be divisible by micro batch "
            f"{micro_batch_size}"
        )
    grad_accum_steps = local_batch_size // micro_batch_size
    if args.stage_sampling == "balanced" and grad_accum_steps % num_stages != 0:
        raise ValueError(
            "balanced stage sampling requires gradient accumulation steps to be "
            f"divisible by the number of stages ({grad_accum_steps} vs {num_stages})"
        )
    experiment_dir = os.path.join(args.results_dir, args.exp_name)
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
    dist.barrier()
    logger = create_logger(experiment_dir, rank)

    logger.info(f"RDD-Wan Training | Stages: {stage_configs}")
    logger.info(f"  Stage lengths: {stage_lengths} → boundaries: {schedule.s_boundaries}")
    logger.info(f"  Downsample mode: {args.downsample_mode}")
    logger.info(f"  Drift scale: {args.drift_scale}")
    logger.info(
        f"  World size: {world_size}, Local batch: {local_batch_size}, "
        f"Micro batch: {micro_batch_size}, Accumulation: {grad_accum_steps}"
    )
    logger.info(f"  Max steps: {args.max_steps}, LR: {args.lr}")

    # ─── Model ────────────────────────────────────────────────────────────
    if args.pretrained_dir and os.path.isdir(args.pretrained_dir):
        logger.info(f"Loading pretrained WanModel from {args.pretrained_dir}")
        # Try diffusers from_pretrained first
        try:
            model = WanModel.from_pretrained(args.pretrained_dir).to(device)
        except Exception:
            # Fall back to manual state_dict loading
            logger.info("from_pretrained failed, trying manual state_dict loading...")
            # Load config
            config_path = os.path.join(args.pretrained_dir, "config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    config = json.load(f)
                model = WanModel(**config).to(device)
            else:
                model = WanModel(
                    model_type='t2v',
                    patch_size=(1, 2, 2),
                    text_len=512,
                    in_dim=16,
                    out_dim=16,
                    dim=args.dim,
                    ffn_dim=args.ffn_dim,
                    freq_dim=256,
                    text_dim=4096,
                    num_heads=args.num_heads,
                    num_layers=args.num_layers,
                    window_size=(-1, -1),
                    qk_norm=True,
                    cross_attn_norm=True,
                    eps=1e-6,
                ).to(device)
            # Load state dict from .pt/.safetensors
            ckpt_files = glob_module.glob(os.path.join(args.pretrained_dir, "*.pt"))
            ckpt_files += glob_module.glob(os.path.join(args.pretrained_dir, "*.pth"))
            ckpt_files += glob_module.glob(os.path.join(args.pretrained_dir, "*.bin"))
            if ckpt_files:
                state_dict = torch.load(ckpt_files[0], map_location="cpu", weights_only=True)
                if "model" in state_dict:
                    state_dict = state_dict["model"]
                model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded weights from {ckpt_files[0]}")
    elif args.pretrained_dir and os.path.isfile(args.pretrained_dir):
        logger.info(f"Loading pretrained from single file: {args.pretrained_dir}")
        model = WanModel(
            model_type='t2v',
            patch_size=(1, 2, 2),
            text_len=512,
            in_dim=16,
            out_dim=16,
            dim=args.dim,
            ffn_dim=args.ffn_dim,
            freq_dim=256,
            text_dim=4096,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            window_size=(-1, -1),
            qk_norm=True,
            cross_attn_norm=True,
            eps=1e-6,
        ).to(device)
        state_dict = torch.load(args.pretrained_dir, map_location="cpu", weights_only=True)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "ema" in state_dict:
            state_dict = state_dict["ema"]
        model.load_state_dict(state_dict, strict=False)
    else:
        logger.info("No pretrained dir specified, training from scratch.")
        model = WanModel(
            model_type='t2v',
            patch_size=(1, 2, 2),
            text_len=512,
            in_dim=16,
            out_dim=16,
            dim=args.dim,
            ffn_dim=args.ffn_dim,
            freq_dim=256,
            text_dim=4096,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            window_size=(-1, -1),
            qk_norm=True,
            cross_attn_norm=True,
            eps=1e-6,
        ).to(device)

    model.train()
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    ema.eval()

    model = DDP(model, device_ids=[local_rank])
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"WanModel: {param_count:.1f}M params")

    # ─── Optimizer ────────────────────────────────────────────────────────
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.0,  # No weight decay as specified
    )

    # ─── VAE (for pixel mode) ────────────────────────────────────────────
    vae = None
    if args.downsample_mode == "pixel":
        from wan.modules.vae import WanVAE
        assert args.vae_path is not None, "--vae-path required for pixel mode"
        logger.info(f"Loading WanVAE from {args.vae_path}")
        vae = WanVAE(z_dim=16, vae_pth=args.vae_path, dtype=torch.bfloat16, device=str(device))

    # ─── Teacher model & T5 (for distill mode) ────────────────────────────
    teacher_model = None
    t5_encoder = None
    online_synthetic = args.data_mode in {"distill", "online_synthetic"}
    if online_synthetic:
        assert args.pretrained_dir is not None, \
            "--pretrained-dir required for online_synthetic mode (used as teacher)"
        logger.warning(
            "online_synthetic generates terminal samples and applies ordinary RDD-FM; "
            "it is NOT few-step/trajectory distillation and holds student, EMA, teacher, "
            "and T5 copies per rank. Use offline cached samples for real experiments."
        )
        # Teacher is a copy of the pretrained model (frozen)
        teacher_model = deepcopy(model.module).to(device)
        teacher_model.eval()
        requires_grad(teacher_model, False)

        # Load T5 text encoder for prompt encoding
        from wan.configs.wan_t2v_1_3B import t2v_1_3B as wan_config
        t5_checkpoint = args.t5_model_dir or os.path.join(
            args.pretrained_dir, wan_config.t5_checkpoint
        )
        t5_tokenizer = args.t5_tokenizer_dir or os.path.join(
            args.pretrained_dir, wan_config.t5_tokenizer
        )
        logger.info(f"Loading T5 encoder from {t5_checkpoint}")
        from wan.modules.t5 import T5EncoderModel
        t5_encoder = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device=device,
            checkpoint_path=t5_checkpoint,
            tokenizer_path=t5_tokenizer,
        )
        t5_encoder.model.eval()

    # ─── Dataset ──────────────────────────────────────────────────────────
    if args.data_mode == "multires":
        dataset = MultiResLatentDataset(
            args.data_dir, num_stages=num_stages, text_len=512, text_dim=4096,
            allow_missing_text=args.allow_missing_text,
        )
        collate_fn = multires_collate_fn
    elif online_synthetic:
        assert args.prompt_file is not None, "--prompt-file required for online_synthetic mode"
        dataset = PromptDataset(args.prompt_file, text_len=512, text_dim=4096)
        collate_fn = prompt_collate_fn
    elif args.data_mode == "pixel":
        dataset = PixelVideoDataset(
            args.data_dir, text_len=512, text_dim=4096,
            allow_missing_text=args.allow_missing_text,
        )
        collate_fn = None
    elif args.data_mode == "dummy":
        dataset = DummyLatentDataset(
            num_samples=10000, channels=16,
            frames=args.latent_frames, height=args.latent_height, width=args.latent_width,
        )
        collate_fn = None
    else:  # "single"
        if args.data_dir and os.path.isdir(args.data_dir):
            dataset = LatentDataset(
                args.data_dir, text_len=512, text_dim=4096,
                allow_missing_text=args.allow_missing_text,
            )
        else:
            logger.info("No data_dir, using DummyLatentDataset for testing.")
            dataset = DummyLatentDataset(
                num_samples=10000, channels=16,
                frames=args.latent_frames, height=args.latent_height, width=args.latent_width,
            )
        collate_fn = None

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
    loader = DataLoader(
        dataset, batch_size=micro_batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        collate_fn=collate_fn,
    )

    # ─── Resume ───────────────────────────────────────────────────────────
    train_steps = 0
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.module.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        train_steps = ckpt.get("train_steps", 0)
        start_epoch = ckpt.get("epoch", 0)
        logger.info(f"  Resumed at step {train_steps}, epoch {start_epoch}")
        dist.barrier()

    # ─── WandB ────────────────────────────────────────────────────────────
    wandb_run = None
    if args.wandb and rank == 0:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.exp_name,
                config=vars(args),
                resume="allow" if args.resume else None,
            )
        except ImportError:
            logger.warning("wandb not installed, skipping logging.")

    # ─── Training Loop ────────────────────────────────────────────────────
    log_steps = 0
    running_loss = 0.0
    start_time = time()
    epoch = start_epoch
    micro_steps = 0
    accumulated_loss = 0.0
    accumulated_stage_counts = [0] * num_stages
    opt.zero_grad(set_to_none=True)
    logger.info(f"Starting training from step {train_steps}...")

    while train_steps < args.max_steps:
        sampler.set_epoch(epoch)
        for batch in loader:
            if train_steps >= args.max_steps:
                break

            # LR warmup schedule
            lr = get_lr_schedule(train_steps, args.warmup_steps, args.lr)
            for param_group in opt.param_groups:
                param_group['lr'] = lr

            # Match DDP.no_sync semantics without indenting the mode-specific
            # forward branches: communicate gradients only on the final micro
            # batch of each optimizer update.
            model.require_backward_grad_sync = (
                micro_steps + 1 == grad_accum_steps
            )

            # balanced exactly reproduces the paper batch composition when
            # accumulation=6 and there are three stages: two examples/stage/GPU.
            if args.stage_sampling == "balanced":
                stage_idx = (micro_steps + train_steps + rank) % num_stages
            else:
                stage_idx = sample_stage_index(stage_lengths, args.stage_sampling)

            # Prepare data based on data_mode
            if online_synthetic:
                # batch is a list of prompt strings from PromptDataset
                prompts = batch
                B = len(prompts)

                # Encode prompts with T5
                with torch.no_grad():
                    contexts = t5_encoder(prompts, device)
                    negative_prompt = getattr(wan_config, "sample_neg_prompt", "")
                    null_contexts = t5_encoder([negative_prompt] * B, device)

                    def pad_context(items):
                        padded = torch.zeros(
                            B, 512, 4096, device=device, dtype=torch.bfloat16
                        )
                        for item_idx, item in enumerate(items):
                            length = min(item.shape[0], 512)
                            padded[item_idx, :length] = item[:length]
                        return padded

                    text_embeddings = pad_context(contexts)
                    null_text_embeddings = pad_context(null_contexts)

                # Teacher generates full-resolution x_1
                # Latent shape: 81 frames → 21 latent frames (÷4), 448→56 (÷8), 832→104 (÷8)
                latent_shape = (B, 16, args.latent_frames, args.latent_height, args.latent_width)
                with torch.no_grad():
                    x_1_fullres = teacher_generate_x1(
                        teacher_model, text_embeddings, null_text_embeddings, latent_shape,
                        num_steps=args.teacher_steps, cfg_scale=5.0, device=device,
                    )

                # Downsample x_1 to the target stage resolution
                x_stage = schedule.downsample_latent(x_1_fullres, stage_idx)

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    _, C_s, F_s, H_s, W_s = x_stage.shape
                    seq_len = compute_seq_len(F_s, H_s, W_s, (1, 2, 2))
                    context = [text_embeddings[i] for i in range(B)]

                    def model_fn(x_list, t, _ctx=context, _sl=seq_len):
                        # Call the DDP wrapper so reducer hooks are registered.
                        out_list = model(x_list, t=t, context=_ctx, seq_len=_sl)
                        return out_list

                    loss_dict = schedule.flow_matchers[stage_idx].training_losses(
                        model_fn, x_stage, model_kwargs={}
                    )
                    loss = loss_dict["loss"].mean()

            elif args.data_mode == "multires":
                # batch is (latents_per_stage, text_embs) from multires_collate_fn
                batch_data, batch_text = batch
                batch_text = batch_text.to(device, dtype=torch.bfloat16)
                # Directly use the pre-computed latent for this stage
                x_stage = batch_data[stage_idx].to(device, dtype=torch.bfloat16)

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    B = x_stage.shape[0]
                    _, C_s, F_s, H_s, W_s = x_stage.shape
                    seq_len = compute_seq_len(F_s, H_s, W_s, (1, 2, 2))
                    context = [batch_text[i] for i in range(B)]

                    def model_fn(x_list, t, _ctx=context, _sl=seq_len):
                        out_list = model(x_list, t=t, context=_ctx, seq_len=_sl)
                        return out_list

                    loss_dict = schedule.flow_matchers[stage_idx].training_losses(
                        model_fn, x_stage, model_kwargs={}
                    )
                    loss = loss_dict["loss"].mean()

            elif args.data_mode == "pixel":
                # pixel mode: downsample in pixel space, encode with VAE
                batch_data, batch_text = batch
                batch_text = batch_text.to(device, dtype=torch.bfloat16)
                pixel_videos = batch_data.to(device, dtype=torch.bfloat16)
                s_factor = stage_configs[stage_idx]["spatial_factor"]
                t_factor = stage_configs[stage_idx]["temporal_factor"]
                with torch.no_grad():
                    video_latents = encode_pixel_videos_for_stage(
                        vae, pixel_videos, s_factor, t_factor, device
                    ).to(dtype=torch.bfloat16)

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    B = video_latents.shape[0]
                    _, C_s, F_s, H_s, W_s = video_latents.shape
                    seq_len = compute_seq_len(F_s, H_s, W_s, (1, 2, 2))
                    context = [batch_text[i] for i in range(B)]

                    def model_fn(x_list, t, _ctx=context, _sl=seq_len):
                        out_list = model(x_list, t=t, context=_ctx, seq_len=_sl)
                        return out_list

                    loss_dict = schedule.flow_matchers[stage_idx].training_losses(
                        model_fn, video_latents, model_kwargs={}
                    )
                    loss = loss_dict["loss"].mean()
            else:
                # single/dummy mode: full-res latent, downsample online
                batch_data, batch_text = batch
                batch_text = batch_text.to(device, dtype=torch.bfloat16)
                video_latents = batch_data.to(device, dtype=torch.bfloat16)
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    loss = train_step_single_stage(
                        model, schedule, video_latents, batch_text,
                        stage_idx, device, patch_size=(1, 2, 2),
                    )

            # Backward
            (loss / grad_accum_steps).backward()
            micro_steps += 1
            accumulated_loss += loss.item() / grad_accum_steps
            accumulated_stage_counts[stage_idx] += 1

            if micro_steps < grad_accum_steps:
                continue

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step
            opt.step()
            opt.zero_grad(set_to_none=True)

            # EMA update
            update_ema(ema, model.module, decay=0.9999)

            # Bookkeeping
            train_steps += 1
            log_steps += 1
            running_loss += accumulated_loss
            step_stage_counts = tuple(accumulated_stage_counts)
            micro_steps = 0
            accumulated_loss = 0.0
            accumulated_stage_counts = [0] * num_stages

            # Logging
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                elapsed = time() - start_time
                avg_loss = running_loss / log_steps
                # All-reduce for global average
                avg_t = torch.tensor(avg_loss, device=device)
                dist.all_reduce(avg_t, op=dist.ReduceOp.SUM)
                global_avg = avg_t.item() / world_size
                steps_per_sec = log_steps / elapsed
                logger.info(
                    f"(step={train_steps:07d}) loss={global_avg:.4f} "
                    f"lr={lr:.2e} stage_microbatches={step_stage_counts} "
                    f"steps/s={steps_per_sec:.2f}"
                )
                if wandb_run is not None:
                    wandb_run.log({
                        "loss": global_avg,
                        "lr": lr,
                        "stage_microbatches": step_stage_counts,
                        "steps_per_sec": steps_per_sec,
                    }, step=train_steps)
                running_loss, log_steps, start_time = 0.0, 0, time()

            # Checkpoint saving
            if train_steps % args.ckpt_every == 0 and rank == 0:
                ckpt_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save({
                    "model": model.module.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "stage_configs": stage_configs,
                    "stage_lengths": stage_lengths,
                    "train_steps": train_steps,
                    "epoch": epoch,
                    "args": vars(args),
                }, ckpt_path)
                logger.info(f"Saved checkpoint: {ckpt_path}")

        # Drop an incomplete accumulation at the dataset boundary. The paper
        # dataset is large enough that this is at most a handful of examples.
        if micro_steps:
            opt.zero_grad(set_to_none=True)
            micro_steps = 0
            accumulated_loss = 0.0
            accumulated_stage_counts = [0] * num_stages
        epoch += 1

    # Final save
    if rank == 0:
        final_path = f"{checkpoint_dir}/final.pt"
        torch.save({
            "model": model.module.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "stage_configs": stage_configs,
            "stage_lengths": stage_lengths,
            "train_steps": train_steps,
            "epoch": epoch,
            "args": vars(args),
        }, final_path)
        logger.info(f"Training complete. Final checkpoint: {final_path}")

    if wandb_run is not None:
        wandb_run.finish()
    dist.destroy_process_group()


# ─── CPU Test ─────────────────────────────────────────────────────────────────

def main_cpu_test(args):
    """Quick CPU test to verify the full pipeline works end-to-end."""
    print("=" * 60)
    print("RDD-Wan CPU Pipeline Test")
    print("=" * 60)

    stage_configs = parse_stage_configs(args.stage_configs)
    stage_lengths = [int(x) for x in args.stage_lengths.split(",")]
    schedule = RDDVideoSchedule(
        stage_configs,
        stage_lengths=stage_lengths,
        downsample_mode=args.downsample_mode,
        drift_scale=args.drift_scale,
    )
    num_stages = len(stage_configs)

    # Create tiny WanModel for CPU test (dim=128, layers=2)
    dim, ffn_dim, num_heads, num_layers = 128, 512, 4, 2
    model = WanModel(
        model_type='t2v',
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        out_dim=16,
        dim=dim,
        ffn_dim=ffn_dim,
        freq_dim=256,
        text_dim=4096,
        num_heads=num_heads,
        num_layers=num_layers,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
    )
    model.train()
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\nModel params: {param_count:.2f}M")

    # Create EMA
    ema = deepcopy(model)
    requires_grad(ema, False)
    ema.eval()

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.95), weight_decay=0.0)

    # Small test data: ensure divisible by all stage factors
    # For 3-stage with factors 4,2,1: need F%4=0, H%(4*2)=0, W%(4*2)=0
    # (patch_size=(1,2,2) requires H,W divisible by 2 at each resolution level)
    max_spatial = max(cfg["spatial_factor"] for cfg in stage_configs)
    max_temporal = max(cfg["temporal_factor"] for cfg in stage_configs)
    F_test = max(4, max_temporal * 2)
    H_test = max(8, max_spatial * 4)
    W_test = max(8, max_spatial * 4)
    B, C = 2, 16

    video_latents = torch.randn(B, C, F_test, H_test, W_test)
    text_embeddings = torch.randn(B, 512, 4096)

    print(f"Input: video_latents={video_latents.shape}, text={text_embeddings.shape}")
    print(f"Stage configs: {stage_configs}")
    print(f"Downsample mode: {args.downsample_mode}")

    device = torch.device("cpu")

    # Test each stage individually
    for stage_idx in range(num_stages):
        print(f"\n--- Stage {stage_idx} ---")
        x_stage = schedule.downsample_latent(video_latents, stage_idx)
        print(f"  Resolution: {x_stage.shape}")

        loss = train_step_single_stage(
            model, schedule, video_latents, text_embeddings,
            stage_idx, device, patch_size=(1, 2, 2),
        )
        print(f"  Loss: {loss.item():.4f}")

    # Full training step simulation (sample random stage, backward, optimizer step)
    print("\n--- Full Training Step Simulation ---")
    stage_idx = sample_stage_index(stage_lengths, args.stage_sampling)
    loss = train_step_single_stage(
        model, schedule, video_latents, text_embeddings,
        stage_idx, device, patch_size=(1, 2, 2),
    )
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
    opt.zero_grad()
    update_ema(ema, model, decay=0.9999)
    print(f"  Stage: {stage_idx}, Loss: {loss.item():.4f}, Grad norm: {grad_norm.item():.4f}")

    # Verify EMA diverges from model slightly
    model_p = list(model.parameters())[0].data.flatten()[:5]
    ema_p = list(ema.parameters())[0].data.flatten()[:5]
    diff = (model_p - ema_p).abs().mean().item()
    print(f"  Model-EMA diff: {diff:.6f} (should be > 0)")

    # Test checkpoint save/load roundtrip
    print("\n--- Checkpoint Save/Load Test ---")
    ckpt = {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "opt": opt.state_dict(),
        "stage_configs": stage_configs,
        "stage_lengths": stage_lengths,
        "train_steps": 1,
        "epoch": 0,
        "args": vars(args),
    }
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name
    torch.save(ckpt, ckpt_path)

    # Load back
    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(loaded["model"])
    ema.load_state_dict(loaded["ema"])
    opt.load_state_dict(loaded["opt"])
    os.unlink(ckpt_path)
    print(f"  Checkpoint roundtrip OK (step={loaded['train_steps']})")

    print("\n" + "=" * 60)
    print("CPU PIPELINE TEST PASSED ✓")
    print("=" * 60)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RDD-Wan Training")

    # Mode
    parser.add_argument("--cpu-test", action="store_true", help="Run CPU pipeline test")

    # Model
    parser.add_argument("--pretrained-dir", type=str, default=None,
                        help="Path to Wan 2.1 pretrained checkpoint directory or file")
    parser.add_argument("--dim", type=int, default=1536,
                        help="Model hidden dim (1536 for 1.3B, 2048 for 14B)")
    parser.add_argument("--ffn-dim", type=int, default=8960,
                        help="FFN intermediate dim")
    parser.add_argument("--num-heads", type=int, default=12,
                        help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=30,
                        help="Number of transformer layers")

    # RDD stage config
    parser.add_argument("--stage-configs", type=str, default="4,4,20;2,2,20;1,1,10",
                        help="spatial_factor,temporal_factor,num_steps per stage, semicolon-separated")
    parser.add_argument("--stage-lengths", type=str, default="60,350,590",
                        help="Diffusion time budget per stage (sum=1000). Coarsest first.")
    parser.add_argument("--stage-sampling", type=str, default="proportional",
                        choices=["proportional", "uniform", "balanced"],
                        help="Sample stages proportional to interval length (global-uniform t) "
                             "or uniformly as an explicit ablation. balanced assigns each "
                             "stage equally within every accumulated optimizer batch.")
    parser.add_argument("--drift-scale", type=float, default=1.0,
                        help="Residual-path strength. Use 0 for the matched boundary-only "
                             "pyramid control and 1 for RDD.")
    parser.add_argument("--downsample-mode", type=str, default="latent",
                        choices=["latent", "pixel"],
                        help="RDD transform. Only latent projection is currently enabled; "
                             "pixel re-encode needs a different clean inverse")

    # VAE (for pixel mode)
    parser.add_argument("--vae-path", type=str, default=None,
                        help="Path to Wan VAE weights (required for pixel mode)")

    # Data
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to training data (latents .pt or pixel videos .pt)")
    parser.add_argument("--prompt-file", type=str, default=None,
                        help="Path to text prompts file (for online_synthetic mode). "
                             "Supports .txt (one prompt per line), .json, .jsonl")
    parser.add_argument("--data-mode", type=str, default="multires",
                        choices=["multires", "single", "pixel", "dummy",
                                 "online_synthetic", "distill"],
                        help="Dataset mode: 'multires' (pre-computed all stages), "
                             "'single' (full-res latent, downsample online), "
                             "'pixel' (pixel video, VAE online), "
                             "'online_synthetic' (teacher terminal samples; not step distillation), "
                             "'distill' (deprecated alias for online_synthetic), "
                             "'dummy' (random data for testing)")
    parser.add_argument("--teacher-steps", type=int, default=50,
                        help="Euler steps for experimental online_synthetic teacher generation")
    parser.add_argument("--t5-model-dir", type=str, default=None,
                        help="Optional T5 checkpoint override for online_synthetic mode")
    parser.add_argument("--t5-tokenizer-dir", type=str, default=None,
                        help="Optional T5 tokenizer override for online_synthetic mode")
    parser.add_argument("--allow-missing-text", action="store_true",
                        help="Allow zero text contexts (unconditional smoke tests only)")
    parser.add_argument("--latent-frames", type=int, default=21,
                        help="Latent temporal dim (for dummy dataset)")
    parser.add_argument("--latent-height", type=int, default=60,
                        help="Latent spatial height (for dummy dataset)")
    parser.add_argument("--latent-width", type=int, default=104,
                        help="Latent spatial width (for dummy dataset)")

    # Training
    parser.add_argument("--global-batch-size", type=int, default=24)
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=0,
        help="Per-GPU micro batch. 0 disables accumulation; optimizer global batch "
             "always remains --global-batch-size.",
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=1000)
    parser.add_argument("--results-dir", type=str, default="./results_rdd_video")
    parser.add_argument("--exp-name", type=str, default="rdd_wan_t2v")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (model + ema + optimizer + step)")

    # WandB
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb-project", type=str, default="rdd-wan",
                        help="WandB project name")

    args = parser.parse_args()

    if args.cpu_test:
        main_cpu_test(args)
    else:
        main_distributed(args)
