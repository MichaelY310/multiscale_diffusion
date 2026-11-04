#!/usr/bin/env python3
"""Train a cross-scale residual adapter (CSRA) on a frozen RDD DiT.

The base RDD checkpoint is never modified.  For selected DiT blocks, the
adapter predicts the teacher's fine-stage block residual from two signals:

  1. the corresponding middle-stage residual, spatially upsampled; and
  2. the high-frequency part of the current fine-stage hidden state.

This is the inexpensive feature-distillation phase described in the CSBR
report.  A later rollout/distillation phase can plug the learned adapters into
sampling, but this script deliberately keeps that experiment separate.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "DiT"))
from models_multires import DiTMultiRes_models  # noqa: E402
from models import DiT_models  # noqa: E402
sys.path.insert(0, str(REPO_ROOT))
from diffusion import create_diffusion, create_diffusion_with_drift  # noqa: E402
from diffusers.models import AutoencoderKL  # noqa: E402


def center_crop_arr(image: Image.Image, size: int) -> Image.Image:
    while min(image.size) >= 2 * size:
        image = image.resize(tuple(x // 2 for x in image.size), Image.Resampling.BOX)
    scale = size / min(image.size)
    image = image.resize(
        tuple(round(x * scale) for x in image.size), Image.Resampling.BICUBIC
    )
    array = np.asarray(image)
    top = (array.shape[0] - size) // 2
    left = (array.shape[1] - size) // 2
    return Image.fromarray(array[top : top + size, left : left + size])


class ImageNetOriginal(Dataset):
    def __init__(self, root: str, image_size: int = 256) -> None:
        transform = transforms.Compose(
            [
                transforms.Lambda(lambda image: center_crop_arr(image, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3, inplace=True),
            ]
        )
        self.dataset = ImageFolder(root, transform=transform)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index]


def checkpoint_weights(path: Path, safe_only: bool = False) -> dict[str, torch.Tensor]:
    # Prefer PyTorch's restricted tensor-only unpickler.  Trusted checkpoints
    # from our own training runs contain argparse metadata and may require the
    # legacy fallback; externally downloaded controls are safe_only=True.
    try:
        checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except Exception:
        if safe_only:
            raise
        try:
            checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint.get("ema", checkpoint.get("model", checkpoint))


def load_frozen_teacher(
    path: Path, device: torch.device, model_name: str, teacher_family: str
) -> nn.Module:
    weights = checkpoint_weights(path, safe_only=teacher_family == "dit")
    num_classes = int(weights["y_embedder.embedding_table.weight"].shape[0] - 1)
    if teacher_family == "rdd":
        model = DiTMultiRes_models[model_name](num_classes=num_classes, base_size=16)
    elif teacher_family == "dit":
        model = DiT_models[model_name](input_size=32, num_classes=num_classes)
    else:
        raise ValueError(teacher_family)
    model.load_state_dict(weights)
    del weights
    return model.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)


def stage_drift(diffusion, clean: torch.Tensor) -> torch.Tensor | None:
    if diffusion.c is None:
        return None
    return (
        diffusion.upsample(
            diffusion.downsample(clean, diffusion.sample_scale), diffusion.sample_scale
        )
        - clean
    )


def aligned_noise(latent32: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    noise32 = torch.randn_like(latent32)
    noise16 = F.avg_pool2d(noise32, 2, 2) * 2.0
    return noise32, noise16


def upsample_tokens(tokens: torch.Tensor, target_grid: int) -> torch.Tensor:
    batch, count, channels = tokens.shape
    grid = int(round(math.sqrt(count)))
    if grid * grid != count:
        raise ValueError(f"token count {count} is not square")
    spatial = tokens.transpose(1, 2).reshape(batch, channels, grid, grid)
    spatial = F.interpolate(spatial, (target_grid, target_grid), mode="nearest")
    return spatial.flatten(2).transpose(1, 2)


def low_high_tokens(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, count, channels = tokens.shape
    grid = int(round(math.sqrt(count)))
    if grid * grid != count or grid < 2:
        raise ValueError(f"expected an even square token grid, got {count} tokens")
    spatial = tokens.transpose(1, 2).reshape(batch, channels, grid, grid)
    low = F.interpolate(F.avg_pool2d(spatial, 2, 2), (grid, grid), mode="nearest")
    low = low.flatten(2).transpose(1, 2)
    return low, tokens - low


@torch.no_grad()
def extract_block_traces(
    model: nn.Module,
    latent: torch.Tensor,
    timestep: torch.Tensor,
    labels: torch.Tensor,
    layers: set[int],
    multires: bool,
) -> dict[int, dict[str, torch.Tensor]]:
    """Extract only the needed frozen-teacher states and stop at the last layer."""
    _, _, height, width = latent.shape
    grid_h, grid_w = height // model.patch_size, width // model.patch_size
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if multires:
            hidden = model.x_embedder(latent)
            rope_cos, rope_sin = model.get_rope(grid_h, grid_w, latent.device)
        else:
            hidden = model.x_embedder(latent) + model.pos_embed
            rope_cos = rope_sin = None
        condition = model.t_embedder(timestep) + model.y_embedder(labels, False)
        traces: dict[int, dict[str, torch.Tensor]] = {}
        for layer, block in enumerate(model.blocks):
            before = hidden
            if multires:
                hidden = block(hidden, condition, rope_cos, rope_sin)
            else:
                hidden = block(hidden, condition)
            if layer in layers:
                traces[layer] = {
                    "input": before.detach().clone(),
                    "residual": (hidden - before).detach().clone(),
                }
            if layer >= max(layers):
                break
    return traces


@torch.no_grad()
def build_on_policy_examples(
    model: nn.Module,
    adapter: "CSRAAdapterBank",
    latent: torch.Tensor,
    model_timestep: torch.Tensor,
    local_timestep: torch.Tensor,
    labels: torch.Tensor,
    layers: list[int],
    coarse_inputs: dict[int, torch.Tensor],
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Collect targets on hidden states actually visited by the student.

    The adapter predictions are detached before advancing to the next block.
    This keeps memory close to teacher forcing while exposing every selected
    layer to upstream adapter errors.  A second DDP adapter forward later builds
    the gradient graph on these collected inputs.
    """
    _, _, height, width = latent.shape
    grid_h, grid_w = height // model.patch_size, width // model.patch_size
    selected = set(layers)
    fine_inputs: dict[int, torch.Tensor] = {}
    targets: dict[int, torch.Tensor] = {}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model.x_embedder(latent)
        rope_cos, rope_sin = model.get_rope(grid_h, grid_w, latent.device)
        condition = model.t_embedder(model_timestep) + model.y_embedder(labels, False)
        for layer, block in enumerate(model.blocks):
            before = hidden
            if layer in selected:
                exact = block(before, condition, rope_cos, rope_sin)
                fine_inputs[layer] = before.detach().clone()
                targets[layer] = (exact - before).detach().clone()
                with torch.autocast("cuda", enabled=False):
                    predicted = adapter.adapters[str(layer)](
                        coarse_inputs[layer], before, local_timestep
                    )
                hidden = before + predicted.to(dtype=before.dtype)
            else:
                hidden = block(before, condition, rope_cos, rope_sin)
            hidden = hidden.detach()
            if layer >= max(layers):
                break
    return fine_inputs, targets


def sinusoidal_embedding(timestep: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=timestep.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = timestep.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=1)
    if dimension % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class CrossScaleResidualAdapter(nn.Module):
    """Low-rank coarse transport plus a local fine high-frequency correction."""

    def __init__(
        self,
        hidden_size: int,
        rank: int,
        time_dim: int = 64,
        architecture: str = "legacy_v1",
    ) -> None:
        super().__init__()
        if architecture not in {"legacy_v1", "fine_context_v2", "teacher_condition_v3", "full_channel_v4", "coarse_memory_v5", "hidden_memory_v6"}:
            raise ValueError(f"unknown CSRA architecture: {architecture}")
        self.hidden_size = hidden_size
        self.rank = rank
        self.time_dim = time_dim
        self.architecture = architecture
        self.requires_condition = architecture == "teacher_condition_v3"
        self.requires_native_coarse = architecture in {"coarse_memory_v5", "hidden_memory_v6"}
        self.coarse_feature = "output" if architecture == "hidden_memory_v6" else "residual"

        self.low_down = nn.Linear(hidden_size, rank, bias=False)
        self.low_up = nn.Linear(rank, hidden_size)

        self.high_spatial = nn.Conv2d(
            hidden_size, hidden_size, kernel_size=3, padding=1,
            groups=hidden_size, bias=True,
        )
        self.high_down = nn.Linear(hidden_size, rank, bias=False)
        self.high_up = nn.Linear(rank, hidden_size)

        # V1 only exposes the high-frequency part of the current fine hidden
        # state.  A complete Transformer block also contains conditioned,
        # token-wise work that depends on its full current input.  V2 adds a
        # cheap low-rank pointwise path but still performs no token-token
        # attention, so it remains far cheaper than the skipped block.
        if architecture in {"fine_context_v2", "teacher_condition_v3", "full_channel_v4", "coarse_memory_v5", "hidden_memory_v6"}:
            self.fine_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
            self.fine_down = nn.Linear(hidden_size, rank, bias=False)
            self.fine_up = nn.Linear(rank, hidden_size)
            self.fine_time_mlp = nn.Sequential(
                nn.Linear(time_dim, 2 * rank),
                nn.SiLU(),
                nn.Linear(2 * rank, 2 * rank),
            )
        if self.requires_native_coarse:
            # Learned one-head rank-r readout, NOT old DiT Q/K/V or an exact
            # attention rewrite. Keys/values stay on the native coarse grid.
            self.memory_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
            self.memory_q = nn.Linear(hidden_size, rank, bias=False)
            self.memory_k = nn.Linear(hidden_size, rank, bias=False)
            self.memory_v = nn.Linear(hidden_size, rank, bias=False)
            self.memory_out = nn.Linear(rank, hidden_size)
            self.memory_time = nn.Sequential(
                nn.Linear(time_dim, 2 * rank), nn.SiLU(),
                nn.Linear(2 * rank, 6 * rank),
            )
            nn.init.zeros_(self.memory_out.weight)
            nn.init.zeros_(self.memory_out.bias)
            nn.init.zeros_(self.memory_time[-1].weight)
            nn.init.zeros_(self.memory_time[-1].bias)
        if architecture == "full_channel_v4":
            # Per-channel affine correction, without an H->rank->H output
            # bottleneck. Time MLP runs once per example, not once per token.
            # Zero final projection makes a V2 warm start exactly equivalent.
            self.diag_time_mlp = nn.Sequential(
                nn.Linear(time_dim, time_dim), nn.SiLU(),
                nn.Linear(time_dim, 2 * hidden_size),
            )
            nn.init.zeros_(self.diag_time_mlp[-1].weight)
            nn.init.zeros_(self.diag_time_mlp[-1].bias)
        if self.requires_condition:
            # Reuse the frozen DiT's already-computed CURRENT time+class vector.
            # Per-example FiLM only: no token-token attention or new backbone.
            self.condition_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
            self.condition_mlp = nn.Sequential(
                nn.Linear(hidden_size, 2 * rank), nn.SiLU(),
                nn.Linear(2 * rank, 6 * rank),
            )
            nn.init.zeros_(self.condition_mlp[-1].weight)
            nn.init.zeros_(self.condition_mlp[-1].bias)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, 4 * rank),
            nn.SiLU(),
            nn.Linear(4 * rank, 4 * rank),
        )
        nn.init.zeros_(self.low_up.weight)
        nn.init.zeros_(self.low_up.bias)
        nn.init.zeros_(self.high_up.weight)
        nn.init.zeros_(self.high_up.bias)
        nn.init.zeros_(self.time_mlp[-1].weight)
        nn.init.zeros_(self.time_mlp[-1].bias)
        if architecture in {"fine_context_v2", "teacher_condition_v3", "full_channel_v4", "coarse_memory_v5", "hidden_memory_v6"}:
            nn.init.zeros_(self.fine_up.weight)
            nn.init.zeros_(self.fine_up.bias)
            nn.init.zeros_(self.fine_time_mlp[-1].weight)
            nn.init.zeros_(self.fine_time_mlp[-1].bias)

    def forward(
        self,
        coarse_residual: torch.Tensor,
        fine_hidden: torch.Tensor,
        fine_timestep: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Keep the small trainable adapter in FP32.  On the Blackwell server,
        # grouped/depthwise BF16 backward can hit a cuDNN illegal-access path
        # while the frozen DiT/VAE BF16 inference path is stable.
        coarse_residual = coarse_residual.float()
        fine_hidden = fine_hidden.float()
        native_coarse = coarse_residual
        if self.requires_native_coarse:
            fine_grid = math.isqrt(fine_hidden.shape[1])
            if fine_grid**2 != fine_hidden.shape[1] or fine_grid % 2 or native_coarse.shape[1] != (fine_grid // 2)**2:
                raise ValueError("V5 requires native square coarse tokens, each spatial side half the fine grid")
            # Preserve the existing V2 local-parent path. The NEW global read
            # below uses native_coarse, never the duplicated keys/values.
            # V6 starts from a trained ZERO-source V2 student. Its old parent
            # path must keep receiving zeros. Full hidden h is read ONLY by
            # the new, zero-initialized memory path; never treat h as delta.
            coarse_residual = (torch.zeros_like(fine_hidden) if self.architecture == "hidden_memory_v6"
                               else upsample_tokens(native_coarse, fine_grid))
        _, fine_high = low_high_tokens(fine_hidden)
        batch, count, channels = fine_high.shape
        grid = int(round(math.sqrt(count)))
        high_spatial = fine_high.transpose(1, 2).reshape(batch, channels, grid, grid)
        high_spatial = self.high_spatial(high_spatial)
        fine_high = high_spatial.flatten(2).transpose(1, 2)

        time = sinusoidal_embedding(fine_timestep / 299.0, self.time_dim)
        low_scale, low_shift, high_scale, high_shift = self.time_mlp(time).chunk(4, 1)
        condition_parts = None
        if self.requires_condition:
            if condition is None or condition.shape != (batch, self.hidden_size):
                raise ValueError("V3 requires the current teacher time+class condition [B,H]")
            condition_parts = self.condition_mlp(self.condition_norm(condition.float())).chunk(6, 1)
            low_scale = low_scale + condition_parts[0]
            low_shift = low_shift + condition_parts[1]
            high_scale = high_scale + condition_parts[2]
            high_shift = high_shift + condition_parts[3]

        low = self.low_down(coarse_residual)
        low = low * (1.0 + low_scale.unsqueeze(1)) + low_shift.unsqueeze(1)
        low = self.low_up(F.silu(low))

        high = self.high_down(fine_high)
        high = high * (1.0 + high_scale.unsqueeze(1)) + high_shift.unsqueeze(1)
        high = self.high_up(F.silu(high))
        output = low + high
        if self.architecture in {"fine_context_v2", "teacher_condition_v3", "full_channel_v4", "coarse_memory_v5", "hidden_memory_v6"}:
            fine_scale, fine_shift = self.fine_time_mlp(time).chunk(2, 1)
            if condition_parts is not None:
                fine_scale = fine_scale + condition_parts[4]
                fine_shift = fine_shift + condition_parts[5]
            fine_normalized = self.fine_norm(fine_hidden)
            fine = self.fine_down(fine_normalized)
            fine = fine * (1.0 + fine_scale.unsqueeze(1)) + fine_shift.unsqueeze(1)
            output = output + self.fine_up(F.silu(fine))
            if self.architecture == "full_channel_v4":
                channel_scale, channel_shift = self.diag_time_mlp(time).chunk(2, 1)
                output = output + fine_normalized * channel_scale.unsqueeze(1) + channel_shift.unsqueeze(1)
            if self.requires_native_coarse:
                qs, qb, ks, kb, vs, vb = self.memory_time(time).chunk(6, 1)
                memory = self.memory_norm(native_coarse)
                q = self.memory_q(fine_normalized) * (1 + qs[:, None]) + qb[:, None]
                k = self.memory_k(memory) * (1 + ks[:, None]) + kb[:, None]
                v = self.memory_v(memory) * (1 + vs[:, None]) + vb[:, None]
                read = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None], dropout_p=0.)[:, 0]
                output = output + self.memory_out(read)
        return output


class CSRAAdapterBank(nn.Module):
    def __init__(
        self,
        layers: list[int],
        hidden_size: int,
        rank: int,
        architecture: str = "legacy_v1",
    ) -> None:
        super().__init__()
        self.layers = layers
        self.architecture = architecture
        self.adapters = nn.ModuleDict(
            {
                str(layer): CrossScaleResidualAdapter(
                    hidden_size, rank, architecture=architecture
                )
                for layer in layers
            }
        )

    def forward(
        self,
        coarse: dict[int, torch.Tensor],
        fine_hidden: dict[int, torch.Tensor],
        fine_timestep: torch.Tensor,
    ) -> dict[int, torch.Tensor]:
        return {
            layer: self.adapters[str(layer)](
                coarse[layer], fine_hidden[layer], fine_timestep
            )
            for layer in self.layers
        }


def normalized_band_loss(prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    prediction = prediction.float()
    target = target.float()
    pred_low, pred_high = low_high_tokens(prediction)
    target_low, target_high = low_high_tokens(target)

    def relative_mse(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        numerator = (left - right).square().mean(dim=(1, 2))
        denominator = right.square().mean(dim=(1, 2)).clamp_min(1e-6)
        return (numerator / denominator).mean()

    full = relative_mse(prediction, target)
    low = relative_mse(pred_low, target_low)
    high = relative_mse(pred_high, target_high)
    loss = full + 0.5 * low + 0.5 * high
    metrics = {
        "full": float(full.detach()),
        "low": float(low.detach()),
        "high": float(high.detach()),
    }
    return loss, metrics


def unwrap(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, DDP) else module


def initialize_adapter_weights(adapter: CSRAAdapterBank, checkpoint: dict) -> None:
    """Warm-start weights; permit V1->zero-output V2 or V2->zero-FiLM V3.

    This is intentionally not an optimizer resume: the new experiment has a
    fresh optimizer and its own update counter. Do not silently drop old keys.
    """
    if list(checkpoint["layers"]) != adapter.layers:
        raise ValueError("Warm-start layer mask must match exactly")
    first = adapter.adapters[str(adapter.layers[0])]
    if int(checkpoint["hidden_size"]) != first.hidden_size or int(checkpoint["rank"]) != first.rank:
        raise ValueError("Warm-start hidden size/rank must match exactly")
    source_arch = checkpoint.get("adapter_architecture", "legacy_v1")
    to_v2 = source_arch == "legacy_v1" and adapter.architecture == "fine_context_v2"
    to_v3 = source_arch == "fine_context_v2" and adapter.architecture == "teacher_condition_v3"
    to_v4 = source_arch == "fine_context_v2" and adapter.architecture == "full_channel_v4"
    to_v5 = source_arch == "fine_context_v2" and adapter.architecture in {"coarse_memory_v5", "hidden_memory_v6"}
    conversion = to_v2 or to_v3 or to_v4 or to_v5
    if source_arch != adapter.architecture and not conversion:
        raise ValueError("Unsupported adapter architecture conversion")
    expected = set(adapter.state_dict())
    provided = set(checkpoint["adapter"])
    missing = expected - provided
    allowed = {
        name for name in expected
        if (to_v2 and name.split(".")[2].startswith("fine_"))
        or (to_v3 and name.split(".")[2].startswith("condition_"))
        or (to_v4 and name.split(".")[2].startswith("diag_"))
        or (to_v5 and name.split(".")[2].startswith("memory_"))
    }
    if provided - expected or missing != allowed:
        raise ValueError(f"Incompatible warm-start keys: missing={missing}, extra={provided - expected}")
    adapter.load_state_dict(checkpoint["adapter"], strict=not conversion)
    if to_v2:
        for layer in adapter.layers:
            branch = adapter.adapters[str(layer)]
            nn.init.zeros_(branch.fine_up.weight)
            nn.init.zeros_(branch.fine_up.bias)
    if to_v3:
        for layer in adapter.layers:
            branch = adapter.adapters[str(layer)]
            nn.init.zeros_(branch.condition_mlp[-1].weight)
            nn.init.zeros_(branch.condition_mlp[-1].bias)
    if to_v4:
        for layer in adapter.layers:
            branch = adapter.adapters[str(layer)]
            nn.init.zeros_(branch.diag_time_mlp[-1].weight)
            nn.init.zeros_(branch.diag_time_mlp[-1].bias)
    if to_v5:
        for branch in adapter.adapters.values():
            nn.init.zeros_(branch.memory_out.weight)
            nn.init.zeros_(branch.memory_out.bias)
            nn.init.zeros_(branch.memory_time[-1].weight)
            nn.init.zeros_(branch.memory_time[-1].bias)


def atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_checkpoint(
    adapter: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    step: int,
    output_dir: Path,
) -> Path:
    path = output_dir / f"{step:07d}.pt"
    payload = {
        "version": 2 if args.adapter_architecture != "legacy_v1" else 1,
        "method": (
            "RDD-CSRA feature distillation"
            if args.teacher_family == "rdd"
            else "DiT full-resolution temporal residual adapter control"
        ),
        "step": step,
        "base_checkpoint": str(args.rdd_ckpt),
        "layers": args.layers,
        "hidden_size": args.hidden_size,
        "rank": args.rank,
        "adapter_architecture": args.adapter_architecture,
        "adapter": unwrap(adapter).state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }
    atomic_save(payload, path)
    atomic_save(payload, output_dir / "latest.pt")
    return path


def main(args: argparse.Namespace) -> None:
    if args.resume and args.init_adapter:
        raise ValueError("--resume and --init-adapter are mutually exclusive")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # Bind the CUDA device before NCCL is initialized.  This matters when a
    # node is already busy: otherwise NCCL may create its communicator on an
    # implicitly selected device and later report asynchronous illegal access.
    try:
        dist.init_process_group("nccl", device_id=device)
    except TypeError:  # Compatibility with older PyTorch releases.
        dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if args.global_batch_size % args.micro_batch_size:
        raise ValueError("global batch must be divisible by global micro batch")
    if args.micro_batch_size % world_size:
        raise ValueError("global micro batch must be divisible by world size")
    accumulation_steps = args.global_batch_size // args.micro_batch_size
    per_gpu_batch = args.micro_batch_size // world_size

    expected_hidden_size = {
        "DiT-XL/2": 1152,
        "DiT-L/2": 1024,
        "DiT-B/2": 768,
        "DiT-S/2": 384,
    }[args.model]
    if args.hidden_size == 0:
        args.hidden_size = expected_hidden_size
    elif args.hidden_size != expected_hidden_size:
        raise ValueError(
            f"--hidden-size={args.hidden_size} does not match {args.model} "
            f"({expected_hidden_size})"
        )

    seed = args.seed * world_size + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(
            json.dumps({**vars(args), "world_size": world_size, "per_gpu_batch": per_gpu_batch,
                        "accumulation_steps": accumulation_steps}, indent=2),
            encoding="utf-8",
        )
    dist.barrier()

    dataset = ImageNetOriginal(args.data_path, args.image_size)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed
    )
    loader = DataLoader(
        dataset,
        batch_size=per_gpu_batch,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    teacher = load_frozen_teacher(
        Path(args.rdd_ckpt), device, args.model, args.teacher_family
    )
    vae = AutoencoderKL.from_pretrained(
        args.vae_path, local_files_only=True, torch_dtype=torch.bfloat16
    ).to(device).eval().requires_grad_(False)

    adapter = CSRAAdapterBank(
        args.layers,
        args.hidden_size,
        args.rank,
        architecture=args.adapter_architecture,
    ).to(device)
    if args.init_adapter:
        initial = torch.load(args.init_adapter, map_location="cpu", weights_only=False)
        if Path(initial["base_checkpoint"]).resolve() != Path(args.rdd_ckpt).resolve():
            raise ValueError("Warm-start must use the same frozen teacher checkpoint")
        initialize_adapter_weights(adapter, initial)
        if rank == 0:
            print(f"Initialized adapter weights from {args.init_adapter}; optimizer/update counter reset", flush=True)
        del initial
    adapter = DDP(adapter, device_ids=[local_rank], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    start_step = 0
    if args.resume:
        resume_path = output_dir / "latest.pt" if args.resume == "latest" else Path(args.resume)
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        unwrap(adapter).load_state_dict(checkpoint["adapter"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        if rank == 0:
            print(f"Resumed {resume_path} at step {start_step}", flush=True)

    if args.teacher_family == "rdd":
        fine_diffusion = create_diffusion_with_drift(
            timestep_respacing="", diffusion_steps=1000, c=0, T_c=0, T_c_plus_1=300
        )
        middle_diffusion = create_diffusion_with_drift(
            timestep_respacing="", diffusion_steps=1000, c=1, T_c=300, T_c_plus_1=600
        )
    else:
        dit_diffusion = create_diffusion(timestep_respacing="", diffusion_steps=1000)
    layer_set = set(args.layers)

    if rank == 0:
        trainable = sum(parameter.numel() for parameter in adapter.parameters())
        print(
            f"{args.teacher_family.upper()}-adapter | world={world_size} | per_gpu={per_gpu_batch} | "
            f"micro_global={args.micro_batch_size} | effective={args.global_batch_size} | "
            f"accum={accumulation_steps} | layers={args.layers} | rank={args.rank} | "
            f"trainable={trainable:,}",
            flush=True,
        )

    optimizer.zero_grad(set_to_none=True)
    step = start_step
    micro_step = 0
    epoch = 0
    running_loss = 0.0
    running_metrics = {"full": 0.0, "low": 0.0, "high": 0.0}
    log_start = time.perf_counter()

    if args.reseed_training_stream:
        # New architecture construction consumes a different amount of RNG.
        # Reset immediately before creating the first dataloader iterator so
        # matched arms use the same augmentations, VAE noise and diffusion t.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    while step < args.max_train_steps:
        sampler.set_epoch(epoch)
        for images, labels in loader:
            images = images.to(device, non_blocking=True, dtype=torch.bfloat16)
            labels = labels.to(device, non_blocking=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                latent32 = vae.encode(images).latent_dist.sample().mul_(0.18215)
                noise32, noise16 = aligned_noise(latent32)

                if args.teacher_family == "rdd":
                    latent16 = F.avg_pool2d(latent32, 2, 2)
                    source_local_t = torch.zeros(
                        per_gpu_batch, device=device, dtype=torch.long
                    )
                    source_global_t = torch.full(
                        (per_gpu_batch,), int(middle_diffusion.timestep_map[0]),
                        device=device, dtype=torch.long,
                    )
                    source_noisy = middle_diffusion.q_sample(
                        latent16, source_local_t,
                        d=stage_drift(middle_diffusion, latent16), noise=noise16,
                    )
                    source_trace = extract_block_traces(
                        teacher, source_noisy, source_global_t, labels, layer_set, True
                    )

                    fine_local_t = torch.randint(
                        0, 300, (per_gpu_batch,), device=device, dtype=torch.long
                    )
                    fine_global_t = torch.as_tensor(
                        [fine_diffusion.timestep_map[int(value)] for value in fine_local_t.tolist()],
                        device=device, dtype=torch.long,
                    )
                    fine_noisy = fine_diffusion.q_sample(
                        latent32, fine_local_t,
                        d=stage_drift(fine_diffusion, latent32), noise=noise32,
                    )
                    coarse_inputs = {
                        layer: upsample_tokens(source_trace[layer]["residual"], 16)
                        for layer in args.layers
                    }
                else:
                    # Ordinary full-resolution DiT control.  At 250 DDPM NFE,
                    # adjacent model evaluations are about four original
                    # timesteps apart.  The source is therefore a stale
                    # full-resolution residual at t+4, with identical clean
                    # latent and noise.
                    fine_local_t = torch.randint(
                        0, 300 - args.source_gap,
                        (per_gpu_batch,), device=device, dtype=torch.long,
                    )
                    source_global_t = fine_local_t + args.source_gap
                    source_noisy = dit_diffusion.q_sample(
                        latent32, source_global_t, noise=noise32
                    )
                    fine_noisy = dit_diffusion.q_sample(
                        latent32, fine_local_t, noise=noise32
                    )
                    source_trace = extract_block_traces(
                        teacher, source_noisy, source_global_t, labels, layer_set, False
                    )
                    fine_trace = extract_block_traces(
                        teacher, fine_noisy, fine_local_t, labels, layer_set, False
                    )
                    coarse_inputs = {
                        layer: source_trace[layer]["residual"] for layer in args.layers
                    }

                if args.training_mode == "teacher_forced" or args.teacher_family == "dit":
                    fine_trace = extract_block_traces(
                        teacher,
                        fine_noisy,
                        fine_global_t if args.teacher_family == "rdd" else fine_local_t,
                        labels,
                        layer_set,
                        args.teacher_family == "rdd",
                    )
                    fine_inputs = {
                        layer: fine_trace[layer]["input"] for layer in args.layers
                    }
                    targets = {
                        layer: fine_trace[layer]["residual"] for layer in args.layers
                    }
                else:
                    # Simulate the states visited when several fine blocks are
                    # replaced together.  Then rebuild a single DDP adapter
                    # graph below so gradient synchronization remains correct.
                    fine_inputs, targets = build_on_policy_examples(
                        teacher,
                        unwrap(adapter),
                        fine_noisy,
                        fine_global_t,
                        fine_local_t,
                        labels,
                        args.layers,
                        coarse_inputs,
                    )

            sync_gradients = (micro_step + 1) % accumulation_steps == 0
            sync_context = nullcontext() if sync_gradients else adapter.no_sync()
            # DDP.no_sync must cover both forward and backward.  Otherwise an
            # effective batch of 256 would launch one all-reduce per microbatch
            # instead of one all-reduce per optimizer step.
            with sync_context:
                with torch.autocast("cuda", enabled=False):
                    predictions = adapter(coarse_inputs, fine_inputs, fine_local_t)
                    total_loss = torch.zeros((), device=device, dtype=torch.float32)
                    batch_metrics = {"full": 0.0, "low": 0.0, "high": 0.0}
                    for layer in args.layers:
                        layer_loss, metrics = normalized_band_loss(predictions[layer], targets[layer])
                        total_loss = total_loss + layer_loss / len(args.layers)
                        for key in batch_metrics:
                            batch_metrics[key] += metrics[key] / len(args.layers)

                (total_loss / accumulation_steps).backward()
            running_loss += float(total_loss.detach()) / accumulation_steps
            for key in running_metrics:
                running_metrics[key] += batch_metrics[key] / accumulation_steps
            micro_step += 1

            if micro_step % accumulation_steps != 0:
                continue

            torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0:
                reduced = torch.tensor(
                    [running_loss, running_metrics["full"], running_metrics["low"], running_metrics["high"]],
                    device=device,
                )
                dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                reduced /= world_size * args.log_every
                elapsed = time.perf_counter() - log_start
                if rank == 0:
                    print(
                        f"step={step:07d} loss={reduced[0]:.5f} "
                        f"rel_full={reduced[1]:.5f} rel_low={reduced[2]:.5f} "
                        f"rel_high={reduced[3]:.5f} steps/s={args.log_every / elapsed:.3f} "
                        f"max_mem_gib={torch.cuda.max_memory_allocated(device) / 2**30:.2f}",
                        flush=True,
                    )
                running_loss = 0.0
                running_metrics = {"full": 0.0, "low": 0.0, "high": 0.0}
                log_start = time.perf_counter()

            if step % args.checkpoint_every == 0 or step == args.max_train_steps:
                dist.barrier()
                if rank == 0:
                    path = save_checkpoint(adapter, optimizer, args, step, output_dir)
                    print(f"Saved {path}", flush=True)
                dist.barrier()

            if step >= args.max_train_steps:
                break
        epoch += 1

    dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rdd-ckpt", required=True)
    parser.add_argument(
        "--teacher-family", choices=["rdd", "dit"], default="rdd",
        help="rdd uses cross-scale residuals; dit uses a stale full-resolution residual.",
    )
    parser.add_argument(
        "--model", default="DiT-XL/2",
        choices=["DiT-XL/2", "DiT-L/2", "DiT-B/2", "DiT-S/2"],
    )
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", type=lambda value: [int(x) for x in value.split(",")],
                        default=[1, 2, 3, 14, 17, 20])
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument(
        "--adapter-architecture",
        choices=["legacy_v1", "fine_context_v2"],
        default="legacy_v1",
        help="fine_context_v2 adds a cheap pointwise current-fine-hidden branch.",
    )
    parser.add_argument(
        "--training-mode",
        choices=["teacher_forced", "on_policy_layerwise"],
        default="teacher_forced",
        help="on_policy_layerwise exposes later adapters to upstream adapter predictions.",
    )
    parser.add_argument(
        "--source-gap", type=int, default=4,
        help="Original diffusion timestep gap for the full-resolution DiT control.",
    )
    parser.add_argument(
        "--hidden-size", type=int, default=0,
        help="0 derives the hidden width from --model.",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--micro-batch-size", type=int, default=32,
                        help="Global micro batch, divided across all DDP ranks.")
    parser.add_argument("--max-train-steps", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--resume", default="", help="Empty, 'latest', or checkpoint path.")
    parser.add_argument("--init-adapter", default="", help="Weights-only initialization; fresh optimizer and update counter.")
    parser.add_argument("--reseed-training-stream", action="store_true",
                        help="Reset training RNG after constructing models for matched architecture comparisons.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
