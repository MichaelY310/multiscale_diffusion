"""
RDD-Wan Sampling: multi-stage progressive video generation.

Generates video by solving ODE at progressively increasing resolutions:
  Stage k uses its training interval sigma in
  [1-s_{k+1}, 1-s_k], traversed from high to low noise.
  At a boundary, canonical noise filling upsamples the clean prediction and
  draws fresh full-space Gaussian noise, matching the established 2-D RDD
  evaluator and the next stage's RDD-FM start marginal.
  Finally: decode with Wan-VAE → pixel video

Euler, Heun, and variable-step Adams--Bashforth 2 (AB2) are provided.  These
generic ODE solvers must be compared fairly on both RDD and PyramidalWan; the
old code incorrectly called AB2 "DPM-Solver-2".

Wan 2.1 Convention:
  - Model input: x_list (List[C,F,H,W]), t (sigma*1000), context, seq_len
  - Model output: List[C,F,H,W] velocity predictions
  - v_pred = dx/dsigma (eps - x_0 for standard rectified flow)
  - RDD clean recovery includes an analytic projection-residual correction
  - ODE step: x_{s+ds} = x_s + v * ds (ds < 0 going toward clean)

Usage:
    python sample_rdd_wan.py \
        --checkpoint results_rdd_video/rdd_wan_1.3B/checkpoints/final.pt \
        --pretrained-dir /path/to/Wan2.1_weights \
        --prompt "A cat walking on the beach at sunset" \
        --size 832x480 --frame-num 81 \
        --steps-per-stage 20,20,10 --solver ab2
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import math
import argparse
import json
import os
import sys
import random
import time
from tqdm import tqdm
from typing import Optional, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "Wan2.1"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "related_repos", "Wan2.1"))

from flow_matching_video import RDDVideoSchedule


# ─── Sigma Schedule ──────────────────────────────────────────────────────────

def get_sigmas_shifted(num_steps: int, shift: float = 1.0,
                       sigma_max: float = 1.0, sigma_min: float = 0.0) -> torch.Tensor:
    """
    Endpoint-preserving shifted schedule for one RDD stage.

    We warp normalized progress rather than sigma itself, so stage endpoints
    remain exactly the training boundaries.  ``shift>1`` retains more points
    near the high-noise endpoint, matching Wan's sampling preference without
    moving an RDD boundary out of distribution.
    
    Args:
        num_steps: number of denoising steps
        shift: endpoint-preserving progress warp (1.0 is linear)
        sigma_max: starting sigma (1.0 = pure noise)
        sigma_min: ending sigma (0.0 = clean)
    
    Returns:
        [N+1] tensor of sigma values from high to low
    """
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    if shift <= 0:
        raise ValueError("shift must be positive")
    progress = torch.linspace(0.0, 1.0, num_steps + 1)
    warped = progress / (shift - (shift - 1.0) * progress)
    return sigma_max + (sigma_min - sigma_max) * warped


# ─── ODE Solvers ─────────────────────────────────────────────────────────────

def _model_forward_cfg(model_fn, model_fn_uncond, x_list, t, cfg_scale):
    """Run model with classifier-free guidance."""
    v_pred = model_fn(x_list, t)
    if isinstance(v_pred, list):
        v_pred = torch.stack(v_pred)

    if model_fn_uncond is not None and cfg_scale > 1.0:
        v_uncond = model_fn_uncond(x_list, t)
        if isinstance(v_uncond, list):
            v_uncond = torch.stack(v_uncond)
        v_pred = v_uncond + cfg_scale * (v_pred - v_uncond)

    return v_pred


@torch.no_grad()
def euler_sample(
    model_fn,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    cfg_scale: float = 1.0,
    model_fn_uncond=None,
    show_progress: bool = True,
) -> Tuple[torch.Tensor, int]:
    """
    Euler ODE solver for Flow Matching.
    
    At each step:
        v = model(x, sigma*1000)  [with CFG: v = v_uncond + scale*(v_cond - v_uncond)]
        x_next = x + v * (sigma_next - sigma)
    
    Args:
        model_fn: callable(x_list, t) -> List[Tensor] (conditioned)
        x: [B, C, F, H, W] current noisy latent
        sigmas: [N+1] decreasing sigma values
        cfg_scale: classifier-free guidance scale
        model_fn_uncond: unconditioned model for CFG
        show_progress: show tqdm progress bar
    
    Returns:
        (x_final, model_forward_calls): final sample and the actual number of
        conditioned/unconditioned model evaluations.
    """
    B = x.shape[0]
    steps = range(len(sigmas) - 1)
    if show_progress:
        steps = tqdm(steps, desc="  Euler", leave=False)

    for i in steps:
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        t = torch.full((B,), sigma.item() * 1000.0, device=x.device, dtype=x.dtype)
        x_list = [x[j] for j in range(B)]

        v_pred = _model_forward_cfg(model_fn, model_fn_uncond, x_list, t, cfg_scale)

        # Euler step: x_next = x + v * (sigma_next - sigma)
        x = x + v_pred * (sigma_next - sigma)

    calls_per_prediction = 2 if model_fn_uncond is not None and cfg_scale > 1.0 else 1
    return x, (len(sigmas) - 1) * calls_per_prediction


@torch.no_grad()
def distilled_euler_sample(
    model_fn,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    flow_matcher,
    cfg_scale: float = 1.0,
    model_fn_uncond=None,
    show_progress: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Euler sampling that reuses the final scheduled call for boundary clean.

    A 2/2/1 distilled model is trained to recover clean data from the state at
    the *start* of every scheduled call. Reusing that call avoids an accidental
    extra endpoint evaluation at every stage, so 2/2/1 really means five model
    evaluations (when CFG has been distilled and ``cfg_scale=1``).
    """

    B = x.shape[0]
    pred_x1 = None
    steps = range(len(sigmas) - 1)
    if show_progress:
        steps = tqdm(steps, desc="  Distilled Euler", leave=False)
    for index in steps:
        sigma = sigmas[index]
        sigma_next = sigmas[index + 1]
        time_input = torch.full(
            (B,), sigma.item() * 1000.0, device=x.device, dtype=x.dtype
        )
        velocity = _model_forward_cfg(
            model_fn,
            model_fn_uncond,
            [x[item] for item in range(B)],
            time_input,
            cfg_scale,
        )
        if index == len(sigmas) - 2:
            sigma_batch = torch.full(
                (B,), sigma.item(), device=x.device, dtype=x.dtype
            )
            pred_x1 = flow_matcher.predict_x1_from_velocity(
                x, velocity, sigma_batch
            )
        x = x + velocity * (sigma_next - sigma)
    assert pred_x1 is not None
    calls_per_prediction = 2 if model_fn_uncond is not None and cfg_scale > 1.0 else 1
    return x, pred_x1, (len(sigmas) - 1) * calls_per_prediction


@torch.no_grad()
def ab2_sample(
    model_fn,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    cfg_scale: float = 1.0,
    model_fn_uncond=None,
    show_progress: bool = True,
) -> Tuple[torch.Tensor, int]:
    """
    Variable-step, two-step Adams--Bashforth solver for a Flow Matching ODE.
    
    For flow matching with v = dx/dsigma, the ODE is:
        dx = v(x, sigma) * dsigma
    
    AB2 uses two previous function evaluations to achieve second order:
        - First step: Euler (1st order, since no history yet)
        - Subsequent steps: 2nd order multistep using current + previous v
    
    The 2nd order update:
        x_{n+1} = x_n + (sigma_{n+1} - sigma_n) * [
            (1 + r/(2*r_prev)) * v_n - (r/(2*r_prev)) * v_{n-1}
        ]
        where r = sigma_{n+1} - sigma_n, r_prev = sigma_n - sigma_{n-1}
    
    Args:
        model_fn: callable(x_list, t) -> List[Tensor] (conditioned)
        x: [B, C, F, H, W] current noisy latent
        sigmas: [N+1] decreasing sigma values
        cfg_scale: classifier-free guidance scale
        model_fn_uncond: unconditioned model for CFG
        show_progress: show tqdm progress bar
    
    Returns:
        (x_final, model_forward_calls): final sample and the actual number of
        conditioned/unconditioned model evaluations.
    """
    B = x.shape[0]
    num_steps = len(sigmas) - 1
    steps = range(num_steps)
    if show_progress:
        steps = tqdm(steps, desc="  AB2", leave=False)

    v_prev = None
    sigma_prev_prev = None

    for i in steps:
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        t = torch.full((B,), sigma.item() * 1000.0, device=x.device, dtype=x.dtype)
        x_list = [x[j] for j in range(B)]

        v_pred = _model_forward_cfg(model_fn, model_fn_uncond, x_list, t, cfg_scale)

        dsigma = sigma_next - sigma  # negative (going toward clean)

        if v_prev is None or sigma_prev_prev is None:
            # First step: plain Euler (no history for multistep)
            x = x + v_pred * dsigma
        else:
            # Variable-step AB2 extrapolation.
            # r = current step size, r_prev = previous step size
            r = dsigma
            r_prev = sigma - sigma_prev_prev  # sigma_{n} - sigma_{n-1} (negative)

            # Avoid division by zero
            if abs(r_prev.item()) < 1e-10:
                x = x + v_pred * dsigma
            else:
                coeff = r / (2.0 * r_prev)
                # Multistep: use linear extrapolation of velocity
                v_corrected = (1.0 + coeff) * v_pred - coeff * v_prev
                x = x + v_corrected * dsigma

        # Store for next step
        sigma_prev_prev = sigma
        v_prev = v_pred

    calls_per_prediction = 2 if model_fn_uncond is not None and cfg_scale > 1.0 else 1
    return x, num_steps * calls_per_prediction


@torch.no_grad()
def heun_sample(
    model_fn,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    cfg_scale: float = 1.0,
    model_fn_uncond=None,
    show_progress: bool = True,
) -> Tuple[torch.Tensor, int]:
    """Return a Heun sample and the actual number of model forward calls."""
    B = x.shape[0]
    steps = range(len(sigmas) - 1)
    if show_progress:
        steps = tqdm(steps, desc="  Heun", leave=False)
    for i in steps:
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        dsigma = sigma_next - sigma
        t = torch.full((B,), sigma.item() * 1000.0, device=x.device, dtype=x.dtype)
        v = _model_forward_cfg(
            model_fn, model_fn_uncond, [x[j] for j in range(B)], t, cfg_scale
        )
        x_euler = x + dsigma * v
        t_next = torch.full(
            (B,), sigma_next.item() * 1000.0, device=x.device, dtype=x.dtype
        )
        v_next = _model_forward_cfg(
            model_fn, model_fn_uncond, [x_euler[j] for j in range(B)], t_next, cfg_scale
        )
        x = x + 0.5 * dsigma * (v + v_next)
    calls_per_prediction = 2 if model_fn_uncond is not None and cfg_scale > 1.0 else 1
    return x, 2 * (len(sigmas) - 1) * calls_per_prediction


# ─── Multi-Stage Sampling ────────────────────────────────────────────────────

def compute_seq_len(F: int, H: int, W: int, patch_size: Tuple[int, int, int] = (1, 2, 2)) -> int:
    """
    Compute sequence length after Wan's patchification.
    seq_len = (F // p_t) * (H // p_h) * (W // p_w)
    """
    return (F // patch_size[0]) * (H // patch_size[1]) * (W // patch_size[2])


@torch.no_grad()
def sample_rdd_video(
    model,
    schedule: RDDVideoSchedule,
    context: List[torch.Tensor],
    context_uncond: Optional[List[torch.Tensor]] = None,
    full_shape: Tuple[int, ...] = (1, 16, 21, 60, 104),
    steps_per_stage: Tuple[int, ...] = (20, 20, 10),
    guidance_scale: float = 5.0,
    shift: float = 1.0,
    solver: str = "euler",
    seed: int = -1,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
    return_stats: bool = False,
    distilled_boundary_clean: bool = False,
    attention_controller=None,
) -> torch.Tensor:
    """
    Full RDD-Video multi-stage sampling pipeline.
    
    Stage 0 (coarsest): start from pure noise at low-res, denoise sigma=1→0
    Stage 1→N: training-matched lift of the previous clean estimate, then
    integrate only over that stage's recorded sigma interval.
    Each stage uses its own sigma schedule with the shifted transform.
    
    Args:
        model: Wan model (EMA weights, eval mode)
        schedule: RDDVideoSchedule with stage_configs
        context: conditioned text embeddings [List of [L, C_text]]
        context_uncond: unconditioned text embeddings for CFG
        full_shape: (B, C, F, H, W) target full-resolution latent shape
        steps_per_stage: num denoising steps per stage
        guidance_scale: CFG scale
        shift: sigma schedule shift factor
        solver: "euler", "ab2", or "heun"
        seed: random seed (-1 for random)
        device: computation device
        dtype: computation dtype
    
    Returns:
        x_0: [B, C, F, H, W] generated clean video latent at full resolution
    """
    B, C, F_full, H_full, W_full = full_shape
    num_stages = schedule.num_stages

    # Set seed
    if seed >= 0:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    # Select solver
    solve_fns = {"euler": euler_sample, "ab2": ab2_sample, "heun": heun_sample}
    if solver not in solve_fns:
        raise ValueError(f"Unknown solver {solver!r}; choose from {sorted(solve_fns)}")
    solve_fn = solve_fns[solver]

    def make_model_fn(ctx, sl, stage_idx, branch):
        """Create model closure with fixed context and seq_len."""
        def fn(x_list, t):
            if attention_controller is not None:
                attention_controller.begin_model_forward(stage_idx, t, branch)
            return model(x_list, t=t, context=ctx, seq_len=sl)
        return fn

    pred_x1 = None
    total_model_calls = 0
    stage_call_counts = []

    for stage_idx in range(num_stages):
        cfg = schedule.stage_configs[stage_idx]
        s_factor = cfg["spatial_factor"]
        t_factor = cfg["temporal_factor"]
        F_s = math.ceil(F_full / t_factor)
        H_s = math.ceil(H_full / s_factor)
        W_s = math.ceil(W_full / s_factor)
        seq_len = compute_seq_len(F_s, H_s, W_s)

        print(f"\n[Stage {stage_idx}/{num_stages-1}] "
              f"Resolution: F={F_s}, H={H_s}, W={W_s} "
              f"(factor: s={s_factor}, t={t_factor}), seq_len={seq_len}")

        # Prepare model functions for this stage's seq_len
        model_fn = make_model_fn(context, seq_len, stage_idx, "cond")
        model_fn_uncond = (
            make_model_fn(context_uncond, seq_len, stage_idx, "uncond")
            if context_uncond is not None else None
        )

        if stage_idx == 0:
            # Start from pure noise
            x = torch.randn(B, C, F_s, H_s, W_s, device=device, dtype=dtype)
        else:
            # Training-matched transition into the next stage at t=s_k.
            x = schedule.upsample_with_noise(
                pred_x1,
                stage_idx - 1,
                stage_idx,
                target_shape=(F_s, H_s, W_s),
            )
            x = x.to(device=device, dtype=dtype)

        # Stage [s_k,s_{k+1}] in data time is [1-s_k,1-s_{k+1}]
        # in Wan's decreasing noise-time convention.
        sigma_max = 1.0 - schedule.s_boundaries[stage_idx]
        sigma_min = 1.0 - schedule.s_boundaries[stage_idx + 1]
        num_steps = steps_per_stage[stage_idx] if stage_idx < len(steps_per_stage) else steps_per_stage[-1]
        sigmas = get_sigmas_shifted(
            num_steps, shift=shift, sigma_max=sigma_max, sigma_min=sigma_min
        )
        sigmas = sigmas.to(device=device, dtype=dtype)

        print(f"  Steps: {num_steps}, sigma: {sigmas[0].item():.4f} → {sigmas[-1].item():.4f}, "
              f"solver: {solver}")

        # Solve ODE for this stage. A distilled 2/2/1 checkpoint reuses its
        # final scheduled call for the clean estimate; ordinary checkpoints
        # retain the historical extra endpoint evaluation.
        if distilled_boundary_clean:
            if solver != "euler":
                raise ValueError("--distilled-boundary-clean requires --solver euler")
            x, pred_x1, solver_calls = distilled_euler_sample(
                model_fn,
                x,
                sigmas,
                schedule.flow_matchers[stage_idx],
                cfg_scale=guidance_scale,
                model_fn_uncond=model_fn_uncond,
                show_progress=True,
            )
            stage_calls = solver_calls
        else:
            x, solver_calls = solve_fn(
                model_fn, x, sigmas,
                cfg_scale=guidance_scale,
                model_fn_uncond=model_fn_uncond,
                show_progress=True,
            )
            sigma_end = torch.full((B,), sigma_min, device=device, dtype=dtype)
            if attention_controller is not None:
                attention_controller.set_capture_route(stage_idx < num_stages - 1)
            try:
                v_end = _model_forward_cfg(
                    model_fn, model_fn_uncond, [x[j] for j in range(B)],
                    sigma_end * 1000.0, guidance_scale,
                )
            finally:
                if attention_controller is not None:
                    attention_controller.set_capture_route(False)
            pred_x1 = schedule.flow_matchers[stage_idx].predict_x1_from_velocity(
                x, v_end, sigma_end
            )
            calls_per_prediction = (
                2 if model_fn_uncond is not None and guidance_scale > 1.0 else 1
            )
            stage_calls = solver_calls + calls_per_prediction
        stage_call_counts.append(stage_calls)
        total_model_calls += stage_calls
        suffix = "" if distilled_boundary_clean else " (including endpoint)"
        print(f"  Actual model forward calls{suffix}: {stage_calls}")
        print(f"  Done. pred_x1 range: [{pred_x1.min().item():.3f}, {pred_x1.max().item():.3f}]")

    stats = {
        "actual_model_forward_calls": total_model_calls,
        "stage_model_forward_calls": stage_call_counts,
        "scheduled_updates": int(sum(steps_per_stage[:num_stages])),
    }
    print(f"Total actual model forward calls: {total_model_calls}")
    return (pred_x1, stats) if return_stats else pred_x1


# ─── Video Saving ────────────────────────────────────────────────────────────

def save_video(video: torch.Tensor, path: str, fps: int = 16):
    """
    Save video tensor to .mp4 file.
    
    Args:
        video: [C, T, H, W] tensor in [-1, 1]
        path: output file path
        fps: frames per second
    """
    # [-1, 1] -> [0, 1] -> [0, 255] uint8
    video = (video + 1.0) / 2.0
    video = video.clamp(0, 1)
    video = (video * 255.0).byte()

    # [C, T, H, W] -> [T, H, W, C]
    video = video.permute(1, 2, 3, 0).contiguous().cpu()

    # torchvision>=0.24 removed write_video.  imageio-ffmpeg provides a small,
    # stable writer and is already part of the isolated RDD-video environment.
    import imageio_ffmpeg

    height, width = int(video.shape[1]), int(video.shape[2])
    writer = imageio_ffmpeg.write_frames(
        path,
        (width, height),
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        output_params=["-crf", "18"],
    )
    writer.send(None)
    try:
        for frame in video:
            writer.send(frame.numpy().tobytes())
    finally:
        writer.close()
    print(f"  Saved video: {path} ({video.shape[0]} frames, {fps} fps)")


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def parse_stage_configs_from_checkpoint(ckpt: dict) -> List[dict]:
    """Extract stage_configs from checkpoint."""
    if "stage_configs" in ckpt:
        return ckpt["stage_configs"]
    # Fallback: parse from args
    ckpt_args = ckpt.get("args", {})
    if "stage_configs" in ckpt_args:
        cfg_str = ckpt_args["stage_configs"]
        stages = []
        for part in cfg_str.split(";"):
            parts = part.strip().split(",")
            stages.append({
                "spatial_factor": int(parts[0]),
                "temporal_factor": int(parts[1]),
                "num_steps": int(parts[2]),
            })
        return stages
    # Default 3-stage config
    return [
        {"spatial_factor": 4, "temporal_factor": 4, "num_steps": 20},
        {"spatial_factor": 2, "temporal_factor": 2, "num_steps": 20},
        {"spatial_factor": 1, "temporal_factor": 1, "num_steps": 10},
    ]


def parse_stage_lengths_from_checkpoint(ckpt: dict, num_stages: int) -> List[int]:
    """Load the training partition; inference must use the same boundaries."""
    ckpt_args = ckpt.get("args", {})
    value = ckpt_args.get("stage_lengths", ckpt.get("stage_lengths"))
    if isinstance(value, str):
        lengths = [int(v) for v in value.split(",")]
    elif value is not None:
        lengths = [int(v) for v in value]
    elif num_stages == 3:
        lengths = [60, 350, 590]
    else:
        raise ValueError("Checkpoint does not record stage_lengths")
    if len(lengths) != num_stages or sum(lengths) != 1000:
        raise ValueError(
            f"Invalid stage_lengths={lengths}; expected {num_stages} values summing to 1000"
        )
    return lengths


def main(args):
    # Lazy imports keep solver and schedule tests usable without Wan weights/code.
    try:
        from wan.configs.wan_t2v_1_3B import t2v_1_3B as wan_config
        import wan.modules.model as wan_model_module
        from wan.modules.model import WanModel
        from wan.modules.attention import attention as safe_attention
        from wan.modules.t5 import T5EncoderModel
        from wan.modules.vae import WanVAE
    except ImportError as exc:
        raise RuntimeError(
            "Wan2.1 modules were not found. Clone the official Wan2.1 repository "
            "beside multiscale_diffusion or add it to PYTHONPATH."
        ) from exc

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    seed = args.seed if args.seed >= 0 else random.randint(0, 2**32 - 1)
    print(f"Using seed: {seed}")

    # ─── Parse size ──────────────────────────────────────────────────────
    W_pixel, H_pixel = [int(x) for x in args.size.split("x")]
    frame_num = args.frame_num

    # Wan-VAE compression: temporal 4x, spatial 8x
    # Latent frames: (frame_num - 1) / 4 + 1
    F_latent = (frame_num - 1) // 4 + 1
    H_latent = H_pixel // 8
    W_latent = W_pixel // 8
    C_latent = 16  # Wan-VAE channels

    print(f"Target: {W_pixel}x{H_pixel}, {frame_num} frames")
    print(f"Latent: C={C_latent}, F={F_latent}, H={H_latent}, W={W_latent}")

    # ─── Load RDD checkpoint ─────────────────────────────────────────────
    if args.use_pretrained_wan:
        if args.checkpoint:
            raise ValueError("--use-pretrained-wan and --checkpoint are mutually exclusive")
        print("\nExploratory mode: treating the original Wan weights as an RDD model")
        stage_configs = []
        for text in args.stage_configs.split(";"):
            spatial, temporal = (int(value) for value in text.split(","))
            stage_configs.append(
                {"spatial_factor": spatial, "temporal_factor": temporal, "num_steps": 1}
            )
        stage_lengths = [int(value) for value in args.stage_lengths.split(",")]
        ckpt_args = {
            "stage_lengths": args.stage_lengths,
            "drift_scale": args.drift_scale,
            "downsample_mode": "latent",
        }
    else:
        if not args.checkpoint:
            raise ValueError("provide --checkpoint or select --use-pretrained-wan")
        print(f"\nLoading RDD checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        stage_configs = parse_stage_configs_from_checkpoint(ckpt)
        stage_lengths = parse_stage_lengths_from_checkpoint(ckpt, len(stage_configs))
        ckpt_args = ckpt.get("args", {})
    print(f"Stage configs: {stage_configs}")
    print(f"Stage lengths: {stage_lengths}")

    # ─── Create schedule ─────────────────────────────────────────────────
    schedule = RDDVideoSchedule(
        stage_configs,
        stage_lengths=stage_lengths,
        diffusion_steps=1000,
        downsample_mode=ckpt_args.get("downsample_mode", "latent"),
        drift_scale=float(ckpt_args.get("drift_scale", 1.0)),
    )
    print(f"Drift scale: {schedule.drift_scale}")

    # ─── Create and load WanModel ────────────────────────────────────────
    dim = ckpt_args.get("dim", 1536)
    ffn_dim = ckpt_args.get("ffn_dim", 8960)
    num_heads = ckpt_args.get("num_heads", 12)
    num_layers = ckpt_args.get("num_layers", 30)

    print(f"\nCreating WanModel: dim={dim}, ffn_dim={ffn_dim}, "
          f"heads={num_heads}, layers={num_layers}")

    if args.use_pretrained_wan:
        model = WanModel.from_pretrained(
            args.pretrained_dir,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        print("  Loaded original Wan weights (not RDD-finetuned)")
    else:
        model = WanModel(
            model_type='t2v',
            patch_size=(1, 2, 2),
            text_len=512,
            in_dim=16, out_dim=16,
            dim=dim, ffn_dim=ffn_dim,
            freq_dim=256, text_dim=4096,
            num_heads=num_heads, num_layers=num_layers,
            window_size=(-1, -1),
            qk_norm=True, cross_attn_norm=True, eps=1e-6,
        )
        if "ema" in ckpt:
            model.load_state_dict(ckpt["ema"])
            print("  Loaded EMA weights")
        elif "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            print("  Loaded model weights (no EMA found)")
        else:
            raise ValueError("Checkpoint has neither 'ema' nor 'model' key")

    model = model.to(device=device, dtype=torch.bfloat16).eval()
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model loaded: {param_count:.1f}M params")

    attention_controller = None
    if args.attention_mode != "dense":
        from attention_analysis.rdd_wan_sampling_attention import RDDWanAttentionController

        attention_controller = RDDWanAttentionController(
            model,
            safe_attention=safe_attention,
            rope_apply=wan_model_module.rope_apply,
            mode=args.attention_mode,
            route_density=args.route_density,
            reuse_every=args.reuse_every,
            fastvideo_kernel_root=args.fastvideo_kernel_root,
        )
        print(
            f"  Attention experiment: {args.attention_mode}, "
            f"route density={args.route_density}, reuse every={args.reuse_every}"
        )

    # ─── Load T5 text encoder ────────────────────────────────────────────
    pretrained_dir = args.pretrained_dir
    print(f"\nLoading T5 encoder from: {pretrained_dir}")

    t5_checkpoint = os.path.join(pretrained_dir, "models_t5_umt5-xxl-enc-bf16.pth")
    t5_tokenizer = os.path.join(pretrained_dir, "google/umt5-xxl")

    # Fallback: check if tokenizer is at a different path
    if not os.path.exists(t5_tokenizer):
        t5_tokenizer = "google/umt5-xxl"

    text_encoder = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        checkpoint_path=t5_checkpoint,
        tokenizer_path=t5_tokenizer,
    )
    print("  T5 encoder loaded")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    text_encode_start = time.perf_counter()

    # Encode prompts
    print(f"  Encoding prompt: '{args.prompt[:80]}...' " if len(args.prompt) > 80
          else f"  Encoding prompt: '{args.prompt}'")

    text_encoder.model.to(device)
    context = text_encoder([args.prompt], device)
    context = [c.to(dtype=torch.bfloat16) for c in context]

    neg_prompt = args.negative_prompt
    if not neg_prompt:
        neg_prompt = wan_config.sample_neg_prompt
    context_uncond = text_encoder([neg_prompt], device)
    context_uncond = [c.to(dtype=torch.bfloat16) for c in context_uncond]

    # Offload T5 to CPU to save VRAM
    text_encoder.model.cpu()
    torch.cuda.empty_cache()
    print("  T5 offloaded to CPU")
    text_encode_seconds = time.perf_counter() - text_encode_start

    # ─── Multi-stage sampling ────────────────────────────────────────────
    steps_per_stage = tuple(int(s) for s in args.steps_per_stage.split(","))
    full_shape = (1, C_latent, F_latent, H_latent, W_latent)

    print(f"\n{'='*60}")
    print(f"Sampling Configuration:")
    print(f"  Full latent shape: {full_shape}")
    print(f"  Stages: {len(stage_configs)}")
    print(f"  Steps per stage: {steps_per_stage}")
    print(f"  Guidance scale: {args.guidance_scale}")
    print(f"  Shift: {args.shift}")
    print(f"  Solver: {args.solver}")
    print(f"  Seed: {seed}")
    print(f"{'='*60}")

    denoise_start = time.perf_counter()
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        video_latent, sample_stats = sample_rdd_video(
            model=model,
            schedule=schedule,
            context=context,
            context_uncond=context_uncond,
            full_shape=full_shape,
            steps_per_stage=steps_per_stage,
            guidance_scale=args.guidance_scale,
            shift=args.shift,
            solver=args.solver,
            seed=seed,
            device=device,
            dtype=torch.bfloat16,
            return_stats=True,
            distilled_boundary_clean=args.distilled_boundary_clean,
            attention_controller=attention_controller,
        )
    denoise_seconds = time.perf_counter() - denoise_start

    print(f"\nGenerated video latent: {video_latent.shape}")
    if args.latent_output:
        latent_output = os.path.abspath(args.latent_output)
        os.makedirs(os.path.dirname(latent_output), exist_ok=True)
        torch.save(video_latent.detach().cpu(), latent_output)
        print(f"Saved generated latent: {latent_output}")

    # ─── Offload DiT, decode with VAE ────────────────────────────────────
    model.cpu()
    torch.cuda.empty_cache()

    # Load the VAE only after the DiT is offloaded; keeping both resident is
    # unnecessary and can make the 1.3B path fail on smaller GPUs.
    print(f"\nLoading Wan-VAE from: {pretrained_dir}")
    vae_path = os.path.join(pretrained_dir, "Wan2.1_VAE.pth")
    vae = WanVAE(
        z_dim=16,
        vae_pth=vae_path,
        dtype=torch.float32,
        device=device,
    )
    print("  VAE loaded")
    print("\nDecoding with Wan-VAE...")

    # VAE expects list of [C, T, H, W] tensors
    decode_start = time.perf_counter()
    latent_list = [video_latent[i].float() for i in range(video_latent.shape[0])]
    videos = vae.decode(latent_list)
    decode_seconds = time.perf_counter() - decode_start

    # ─── Save output ─────────────────────────────────────────────────────
    output_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # videos[0] is [C, T, H, W] in [-1, 1]
    video_pixel = videos[0]
    print(f"  Decoded video: {video_pixel.shape}, "
          f"range: [{video_pixel.min().item():.3f}, {video_pixel.max().item():.3f}]")

    save_video(video_pixel, output_path, fps=16)
    end_to_end_seconds = text_encode_seconds + denoise_seconds + decode_seconds
    print(f"\nDone! Output saved to: {output_path}")

    metrics = {
        **sample_stats,
        "denoise_and_boundary_seconds": denoise_seconds,
        "text_encode_seconds": text_encode_seconds,
        "vae_decode_seconds": decode_seconds,
        "end_to_end_inference_seconds": end_to_end_seconds,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "solver": args.solver,
        "steps_per_stage": list(steps_per_stage),
        "guidance_scale": args.guidance_scale,
        "shift": args.shift,
        "seed": seed,
        "size": args.size,
        "frame_num": args.frame_num,
        "output": os.path.abspath(output_path),
        "latent_output": os.path.abspath(args.latent_output) if args.latent_output else None,
        "checkpoint_mode": "original_wan_as_faux_rdd" if args.use_pretrained_wan else "rdd_checkpoint",
        "stage_lengths": stage_lengths,
        "stage_configs": stage_configs,
        "drift_scale": schedule.drift_scale,
        "transition_mode": "fresh",
        "attention": attention_controller.summary() if attention_controller is not None else {
            "mode": "dense"
        },
    }
    metrics_path = args.metrics_json or os.path.splitext(output_path)[0] + ".metrics.json"
    os.makedirs(os.path.dirname(os.path.abspath(metrics_path)), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RDD-Wan Multi-Stage Video Sampling")

    # Model paths
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to RDD-Wan checkpoint (.pt)")
    parser.add_argument("--use-pretrained-wan", action="store_true",
                        help="Exploratory only: use original Wan weights as a faux RDD checkpoint")
    parser.add_argument("--pretrained-dir", type=str, required=True,
                        help="Path to original Wan 2.1 pretrained dir (for VAE + T5)")
    parser.add_argument("--stage-configs", type=str, default="4,4;2,2;1,1",
                        help="Faux-RDD stage spatial,temporal factors separated by semicolons")
    parser.add_argument("--stage-lengths", type=str, default="400,300,300",
                        help="Faux-RDD data-time allocation; must sum to 1000")
    parser.add_argument("--drift-scale", type=float, default=1.0,
                        help="Faux-RDD drift scale; 0 gives the boundary-only control")

    # Generation config
    parser.add_argument("--prompt", type=str, default="A beautiful sunset over the ocean",
                        help="Text prompt for video generation")
    parser.add_argument("--negative-prompt", type=str, default="",
                        help="Negative prompt for content exclusion")
    parser.add_argument("--size", type=str, default="832x448",
                        help="Video resolution WxH (default follows Wan/PyramidalWan: 832x448)")
    parser.add_argument("--frame-num", type=int, default=81,
                        help="Number of frames (should be 4n+1, e.g., 81)")

    # Sampling parameters
    parser.add_argument("--steps-per-stage", type=str, default="20,20,10",
                        help="Comma-separated denoising steps per stage")
    parser.add_argument("--guidance-scale", type=float, default=5.0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--shift", type=float, default=1.0,
                        help="Endpoint-preserving per-stage Wan schedule warp")
    parser.add_argument("--solver", type=str, default="euler",
                        choices=["euler", "ab2", "heun"],
                        help="ODE solver. AB2 uses one NFE/step; Heun uses two.")
    parser.add_argument(
        "--distilled-boundary-clean",
        action="store_true",
        help="Reuse the last scheduled Euler prediction at each boundary. Required "
             "for a true 2/2/1 five-NFE distilled checkpoint.",
    )

    # Misc
    parser.add_argument("--seed", type=int, default=-1,
                        help="Random seed (-1 for random)")
    parser.add_argument("--output", type=str, default="./output_rdd_video/sample.mp4",
                        help="Output video file path (.mp4)")
    parser.add_argument("--metrics-json", type=str, default=None,
                        help="Metrics JSON path (default: next to output video)")
    parser.add_argument("--latent-output", type=str, default=None,
                        help="Optional path for the generated clean latent tensor")
    parser.add_argument("--device", type=int, default=0,
                        help="CUDA device id")
    parser.add_argument("--attention-mode", choices=["dense", "reuse", "route", "route_reuse"],
                         default="dense", help="Training-free attention sanity-check mode")
    parser.add_argument("--route-density", type=float, default=0.75,
                        help="Fraction of previous-stage 64-token blocks retained")
    parser.add_argument("--reuse-every", type=int, default=1,
                        help="For reuse/route_reuse, recompute self-attention once every K solver steps")
    parser.add_argument("--fastvideo-kernel-root", type=str, default=None,
                        help="Path containing FastVideo fastvideo-kernel source")

    args = parser.parse_args()
    main(args)
