"""
Unified FID sampling script for multi-scale diffusion models.
Reads experiment configurations from a JSONL file and generates samples for each.

Supports model types:
  - "multistage" / "decouple": separate DiT models (two or three resolutions)
  - "unified": single DiTMultiRes model for all resolutions
  - "fm_unified": legacy RDD-FM v2 DiTMultiRes velocity checkpoint

Features:
  - Fully asynchronous image saving via ThreadPoolExecutor
  - Pipeline design: next experiment starts while previous I/O is still pending
  - Background NPZ packing + FID evaluation (runs in parallel with next experiment)
  - Validates PNGs before packing; deletes corrupt files so next run can regenerate them
  - 8-GPU parallel with rank-based allocation and skip/resume logic
  - python -u compatible (all prints flushed)

Usage:
  python -u sample_fid_unified.py \
      --config experiments.jsonl \
      --rank 0 --world-size 8 \
      --total-samples 50000 --batch-size 100 \
      --seed 0
"""
import builtins
import functools
builtins.print = functools.partial(print, flush=True)

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.nn.functional as F
import sys
import os
import gc
import time
import json
import argparse
import subprocess
import re
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, Future
from typing import List, Dict, Any, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Path setup
# ─────────────────────────────────────────────────────────────────────────────
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dit_path = os.path.join(parent_dir, 'DiT')
sys.path.insert(0, dit_path)
from models import DiT_models
from models_multires import DiTMultiRes_models
sys.path.remove(dit_path)

sys.path.insert(0, parent_dir)
from diffusion import create_diffusion_with_drift
from diffusion.flow_matching_with_drift import (
    FlowMatchingWithDrift,
    sample_rdd_fm_multistage,
)
from diffusion.respace import _WrappedModel
from diffusion.gaussian_diffusion_with_drift import _extract_into_tensor
from diffusion.rdd_improvements import (
    GuidanceConfig,
    apply_autoguidance_to_model_output,
    apply_cfg_to_model_output,
)
from diffusion.stage_transition import lift_noise_with_nullspace_2d
from diffusion.fdm_rescaling import (
    rescale_alpha_sigma,
    spatial_dimension_ratio,
    validate_fdm_rescaling,
)
from diffusers.models import AutoencoderKL


# ─────────────────────────────────────────────────────────────────────────────
# Global async I/O executor
# ─────────────────────────────────────────────────────────────────────────────
_io_executor = ThreadPoolExecutor(max_workers=8)
_io_futures: List[Future] = []


def _submit_save_task(samples_np: np.ndarray, indices: List[int], output_dir: str):
    """Submit async image saving task. Returns immediately."""
    def _save_batch():
        for i, img_idx in enumerate(indices):
            img = Image.fromarray(samples_np[i])
            img.save(os.path.join(output_dir, f"{img_idx:06d}.png"))
    fut = _io_executor.submit(_save_batch)
    _io_futures.append(fut)


def _drain_io_futures() -> List[Future]:
    """Take ownership of current I/O futures and reset the global list.
    Returns the list of futures that were pending."""
    global _io_futures
    taken = _io_futures
    _io_futures = []
    return taken


def _wait_all_io():
    """Wait for all pending I/O futures to complete."""
    for fut in _io_futures:
        fut.result()
    _io_futures.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Background NPZ packing + FID evaluation
# ─────────────────────────────────────────────────────────────────────────────
_eval_executor = ThreadPoolExecutor(max_workers=2)
_eval_futures: List[Future] = []

REF_NPZ = os.environ.get(
    "RDD_REF_NPZ",
    os.path.join(parent_dir, "eval_results", "reference", "VIRTUAL_imagenet256_labeled.npz"),
)
EVALUATOR = os.environ.get(
    "RDD_FID_EVALUATOR",
    os.path.join(parent_dir, "guided-diffusion", "evaluations", "evaluator.py"),
)
EVAL_LOG_DIR = os.environ.get(
    "RDD_EVAL_LOG_DIR",
    os.path.join(parent_dir, "eval_results", "fid_logs"),
)

_REQUIRED_EVAL_METRICS = {
    "inception_score": re.compile(r"(?m)^Inception Score:\s*([-+0-9.eE]+)\s*$"),
    "fid": re.compile(r"(?m)^FID:\s*([-+0-9.eE]+)\s*$"),
    "sfid": re.compile(r"(?m)^sFID:\s*([-+0-9.eE]+)\s*$"),
    "precision": re.compile(r"(?m)^Precision:\s*([-+0-9.eE]+)\s*$"),
    "recall": re.compile(r"(?m)^Recall:\s*([-+0-9.eE]+)\s*$"),
}


def _parse_complete_evaluation(content: str) -> Optional[Dict[str, float]]:
    """Return all five ADM metrics, or ``None`` for a partial/failed log."""
    metrics: Dict[str, float] = {}
    for name, pattern in _REQUIRED_EVAL_METRICS.items():
        match = pattern.search(content)
        if match is None:
            return None
        metrics[name] = float(match.group(1))
    return metrics


def _validate_and_pack_npz(output_dir: str, total_samples: int) -> Optional[str]:
    """Validate all PNGs by file size, delete corrupt ones, pack into NPZ.
    Returns npz path on success, None on failure."""
    print(f"  [NPZ] Validating {total_samples} PNGs in {output_dir}...")

    corrupt_count = 0
    missing_count = 0

    for i in range(total_samples):
        fpath = os.path.join(output_dir, f"{i:06d}.png")
        if not os.path.exists(fpath):
            missing_count += 1
            continue
        sz = os.path.getsize(fpath)
        if sz < 1024:
            print(f"  [NPZ] Corrupt file {fpath}: {sz} bytes, deleting")
            os.remove(fpath)
            corrupt_count += 1

    if corrupt_count > 0:
        print(f"  [NPZ] Deleted {corrupt_count} corrupt files. "
              f"They will be regenerated on next run.")

    # Re-check completeness after cleanup
    valid_count = sum(1 for i in range(total_samples)
                      if os.path.exists(os.path.join(output_dir, f"{i:06d}.png")))

    if valid_count < total_samples:
        print(f"  [NPZ] Only {valid_count}/{total_samples} valid files. "
              f"Skipping NPZ (will complete on next run).")
        return None

    # All files present — pack NPZ
    npz_path = os.path.join("/tmp", os.path.basename(output_dir) + "_samples.npz")
    print(f"  [NPZ] Packing {total_samples} images → {npz_path}")

    first_img = np.array(Image.open(os.path.join(output_dir, "000000.png")))
    h, w, c = first_img.shape
    all_arr = np.empty((total_samples, h, w, c), dtype=np.uint8)
    all_arr[0] = first_img

    def _load_img(args_tuple):
        idx, fp = args_tuple
        img = Image.open(fp)
        img.load()
        arr = np.array(img)
        return idx, arr

    tasks = [(i, os.path.join(output_dir, f"{i:06d}.png")) for i in range(1, total_samples)]
    with ThreadPoolExecutor(max_workers=16) as pool:
        for idx, arr in pool.map(_load_img, tasks):
            all_arr[idx] = arr

    np.savez(npz_path, all_arr)
    size_gb = os.path.getsize(npz_path) / 1e9
    print(f"  [NPZ] Done: shape={all_arr.shape}, size={size_gb:.2f} GB")
    del all_arr

    # An uncompressed np.savez payload is approximately N*H*W*C bytes.  The
    # previous hard-coded 9 GB threshold was valid only for 50K ImageNet-256
    # and incorrectly deleted every 10K screening or smoke-test archive.
    expected_bytes = total_samples * h * w * c
    actual_bytes = os.path.getsize(npz_path)
    if actual_bytes < 0.95 * expected_bytes:
        print(
            f"  [NPZ] WARNING: NPZ size {actual_bytes} bytes is below 95% of "
            f"the expected payload {expected_bytes}; deleting as likely corrupt."
        )
        os.remove(npz_path)
        return None

    return npz_path


def _run_fid_evaluation(npz_path: str, output_dir: str):
    """Run the ADM evaluator and require IS/FID/sFID/Precision/Recall."""
    os.makedirs(EVAL_LOG_DIR, exist_ok=True)
    log_name = os.path.basename(output_dir) + ".txt"
    log_path = os.path.join(EVAL_LOG_DIR, log_name)

    # A historical bug checked only for ``FID:``.  Logs where np.bool crashed
    # precision/recall were therefore mistaken for completed evaluations.
    if os.path.exists(log_path):
        try:
            with open(log_path) as f:
                content = f.read()
            metrics = _parse_complete_evaluation(content)
            if metrics is not None:
                print(f"  [EVAL] All five metrics already present: {log_path}")
                return
            print(f"  [EVAL] Partial log detected; recomputing all metrics: {log_path}")
        except Exception:
            pass

    print(f"  [EVAL] Running IS/FID/sFID/Precision/Recall: {os.path.basename(output_dir)}")
    temp_log_path = log_path + ".tmp"
    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        result = subprocess.run(
            [sys.executable, "-u", EVALUATOR, REF_NPZ, npz_path],
            capture_output=True, text=True,
            env=env,
        )
        with open(temp_log_path, "w") as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n--- STDERR ---\n")
                f.write(result.stderr)
        metrics = _parse_complete_evaluation(result.stdout)
        if result.returncode != 0 or metrics is None:
            failed_log_path = log_path + ".failed.txt"
            os.replace(temp_log_path, failed_log_path)
            missing = [
                name for name, pattern in _REQUIRED_EVAL_METRICS.items()
                if pattern.search(result.stdout) is None
            ]
            print(
                f"  [EVAL] FAILED (exit={result.returncode}, missing={missing}): "
                f"{failed_log_path}"
            )
            raise RuntimeError(
                "ADM evaluator did not produce every required metric; "
                f"inspect {failed_log_path}"
            )
        os.replace(temp_log_path, log_path)
        metrics_path = log_path + ".metrics.json"
        temp_metrics_path = metrics_path + ".tmp"
        with open(temp_metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(temp_metrics_path, metrics_path)
        print(
            "  [EVAL] Complete: "
            + ", ".join(f"{name}={value:.6g}" for name, value in metrics.items())
        )
        print(f"  [EVAL] Log: {log_path}")
    except Exception as e:
        if os.path.exists(temp_log_path):
            try:
                os.replace(temp_log_path, log_path + ".failed.txt")
            except Exception:
                pass
        print(f"  [EVAL] ERROR: {e}")
        raise


def _background_npz_and_eval(output_dir: str, total_samples: int, io_futures: List[Future]):
    """Background task: wait for I/O, validate, pack NPZ, run FID."""
    # First wait for all pending image saves of THIS rank to complete
    for fut in io_futures:
        fut.result()

    # Wait for all ranks to finish (poll until all files exist)
    poll_interval = 10
    wait_timeout = float(os.environ.get("RDD_EVAL_WAIT_TIMEOUT", "3600"))
    deadline = time.time() + wait_timeout
    while True:
        present = sum(1 for i in range(total_samples)
                      if os.path.exists(os.path.join(output_dir, f"{i:06d}.png"))
                      and os.path.getsize(os.path.join(output_dir, f"{i:06d}.png")) >= 1024)
        if present >= total_samples:
            break
        if time.time() >= deadline:
            raise TimeoutError(
                f"Timed out after {wait_timeout:.0f}s waiting for all ranks: "
                f"{present}/{total_samples} complete PNGs in {output_dir}"
            )
        print(f"  [NPZ] Waiting for all ranks: {present}/{total_samples} files present...")
        time.sleep(poll_interval)

    # Validate and pack
    npz_path = _validate_and_pack_npz(output_dir, total_samples)
    if npz_path is None:
        return

    # Run FID evaluation
    _run_fid_evaluation(npz_path, output_dir)

    # Clean up local NPZ to save disk space
    try:
        os.remove(npz_path)
        print(f"  [NPZ] Cleaned up: {npz_path}")
    except Exception:
        pass


def _submit_eval_task(output_dir: str, total_samples: int, io_futures: List[Future]):
    """Submit background NPZ packing + FID eval. Non-blocking."""
    fut = _eval_executor.submit(_background_npz_and_eval, output_dir, total_samples, io_futures)
    _eval_futures.append(fut)


def _wait_all_evals():
    """Wait for all background evaluations to complete."""
    for fut in _eval_futures:
        fut.result()
    _eval_futures.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def infer_num_classes(weights: dict) -> int:
    """Infer num_classes from the embedding table weight shape."""
    emb = weights.get("y_embedder.embedding_table.weight")
    if emb is None:
        return 0  # unconditional model
    return emb.shape[0] - 1


class _ScheduledCFGForward:
    """DiT CFG wrapper with a global-time interval and optional smooth ramp."""

    def __init__(self, model, config: GuidanceConfig, total_timesteps: int):
        self.model = model
        self.config = config
        self.total_timesteps = max(int(total_timesteps), 2)

    def __call__(self, x, t, y, cfg_scale=None):
        del cfg_scale  # The validated schedule owns the scale.
        half_size = x.shape[0] // 2
        half = x[:half_size]
        normalized_t = t[:half_size].float() / float(self.total_timesteps - 1)
        scales = self.config.scales(normalized_t).to(device=x.device, dtype=x.dtype)

        # Every sample in a diffusion update normally shares t.  If guidance
        # is inactive, skip the unconditional forward and duplicate the
        # conditional prediction, retaining the sampler's doubled batch shape.
        if bool(torch.all(scales == 1.0)):
            conditional = self.model.forward(half, t[:half_size], y[:half_size])
            return torch.cat([conditional, conditional], dim=0)

        combined = torch.cat([half, half], dim=0)
        model_output = self.model.forward(combined, t, y)
        return apply_cfg_to_model_output(model_output, scales, guided_channels=3)


class _AutoguidanceForward:
    """Guide a conditional RDD model with a weaker conditional RDD model."""

    def __init__(self, good_model, guide_model, config: GuidanceConfig, total_timesteps: int):
        self.good_model = good_model
        self.guide_model = guide_model
        self.config = config
        self.total_timesteps = max(int(total_timesteps), 2)

    def __call__(self, x, t, y):
        good = self.good_model.forward(x, t, y)
        normalized_t = t.float() / float(self.total_timesteps - 1)
        scales = self.config.scales(normalized_t).to(device=x.device, dtype=x.dtype)
        if bool(torch.all(scales == 1.0)):
            return good
        guide = self.guide_model.forward(x, t, y)
        # RDD predicts four latent epsilon channels followed by learned
        # variance.  Guide epsilon only and retain variance from the good model.
        return apply_autoguidance_to_model_output(
            good, guide, scales, guided_channels=self.good_model.in_channels
        )


def _load_dit_model(ckpt_path, arch, input_size, device, dtype=torch.float32):
    """Load a DiT checkpoint, auto-detecting num_classes and d_prediction."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    weights = sd.get("ema", sd.get("model"))
    num_classes = infer_num_classes(weights)
    d_prediction = "d_final_layer.linear.weight" in weights
    model = DiT_models[arch](input_size=input_size, num_classes=num_classes, d_prediction=d_prediction)
    model.load_state_dict(weights)
    model = model.to(dtype=dtype, device=device)
    model.eval()
    del sd, weights
    gc.collect()
    torch.cuda.empty_cache()
    return model, num_classes


def _load_unified_model(ckpt_path, arch, resolutions, device, dtype=torch.float32):
    """Load a DiTMultiRes checkpoint."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    weights = sd.get("ema", sd.get("model"))
    num_classes = infer_num_classes(weights)
    patch_size = int(arch.split("/")[1])
    base_size = resolutions[0] // patch_size
    repa_dim = None
    repa_projector_dim = 2048
    if "repa_projector.2.weight" in weights:
        repa_dim = int(weights["repa_projector.2.weight"].shape[0])
        repa_projector_dim = int(weights["repa_projector.2.weight"].shape[1])
    model = DiTMultiRes_models[arch](
        num_classes=num_classes,
        base_size=base_size,
        repa_dim=repa_dim,
        repa_projector_dim=repa_projector_dim,
    )
    model.load_state_dict(weights)
    model = model.to(dtype=dtype, device=device)
    model.eval()
    del sd, weights
    gc.collect()
    torch.cuda.empty_cache()
    return model, num_classes


def _redraw_upsample(pred_xstart, t, diffusion, target_size):
    """Canonical noise filling: upsample clean x0, then draw full IID noise.

    The private function name is retained for backward compatibility with old
    experiments that called the same operation ``redraw``.
    """
    current_size = pred_xstart.shape[-1]
    scale = target_size // current_size
    upsampled = pred_xstart.repeat_interleave(scale, dim=2).repeat_interleave(scale, dim=3)
    sqrt_alpha = _extract_into_tensor(diffusion.sqrt_alphas_cumprod, t, upsampled.shape)
    sqrt_one_minus_alpha = _extract_into_tensor(diffusion.sqrt_one_minus_alphas_cumprod, t, upsampled.shape)
    noise = torch.randn_like(upsampled)
    return sqrt_alpha * upsampled + sqrt_one_minus_alpha * noise, noise


def _fdm_rescaled_upsample_probe(
    pred_xstart,
    t,
    diffusion,
    target_size,
    mode,
    dimension_ratio,
):
    """Inference-only f-DM SP/VP probe for an old RDD checkpoint.

    This changes the coefficients at the stage boundary only.  A faithful f-DM
    implementation changes them throughout the stage and therefore requires a
    model trained with that forward process.  Keeping this helper separate makes
    it difficult to accidentally report this out-of-distribution probe as the
    fully trained method.
    """
    current_size = pred_xstart.shape[-1]
    scale = target_size // current_size
    upsampled = pred_xstart.repeat_interleave(scale, dim=2).repeat_interleave(
        scale, dim=3
    )
    alpha = _extract_into_tensor(diffusion.sqrt_alphas_cumprod, t, upsampled.shape)
    sigma = _extract_into_tensor(
        diffusion.sqrt_one_minus_alphas_cumprod, t, upsampled.shape
    )
    alpha, sigma = rescale_alpha_sigma(alpha, sigma, dimension_ratio, mode)
    noise = torch.randn_like(upsampled)
    return alpha * upsampled + sigma * noise, noise


def _noise_filling_upsample(
    pred_xstart,
    coarse_noise,
    t,
    diffusion,
    target_size,
    sigma_h=0.0,
):
    """Historical innovation-preserving coupling (not canonical noise filling).

    The isometric lift keeps the already sampled coarse innovation.  A fresh
    Gaussian contributes only its component orthogonal to block-constant
    tensors.  At ``sigma_h == 0`` the combined fine noise is still exactly
    standard Gaussian, so canonical noise filling and innovation preserving
    have identical one-stage marginals but different cross-stage couplings.

    ``sigma_h > 0`` retains the historical optional extra null-space variance.
    """
    if coarse_noise is None:
        raise ValueError("coarse_noise is required for innovation_preserving")
    current_size = pred_xstart.shape[-1]
    if coarse_noise.shape != pred_xstart.shape:
        raise ValueError(
            f"coarse_noise shape {tuple(coarse_noise.shape)} does not match "
            f"clean prediction {tuple(pred_xstart.shape)}"
        )
    scale = target_size // current_size
    upsampled = pred_xstart.repeat_interleave(scale, dim=2).repeat_interleave(scale, dim=3)
    sqrt_alpha = _extract_into_tensor(diffusion.sqrt_alphas_cumprod, t, upsampled.shape)
    sqrt_one_minus_alpha = _extract_into_tensor(diffusion.sqrt_one_minus_alphas_cumprod, t, upsampled.shape)

    fresh_noise = torch.randn_like(upsampled)
    filled_noise = lift_noise_with_nullspace_2d(
        coarse_noise, target_size, fresh_noise=fresh_noise
    )

    if sigma_h == 0.0:
        return sqrt_alpha * upsampled + sqrt_one_minus_alpha * filled_noise, filled_noise
    else:
        # Split the filled noise back into its carried row-space and new
        # null-space components, then apply the historical extra variance.
        row_noise = coarse_noise.repeat_interleave(scale, dim=2).repeat_interleave(
            scale, dim=3
        ) / float(scale)
        null_noise = filled_noise - row_noise
        alpha_bar = sqrt_alpha ** 2
        null_std = torch.sqrt(1.0 - alpha_bar + alpha_bar * sigma_h ** 2)
        state = (
            sqrt_alpha * upsampled
            + sqrt_one_minus_alpha * row_noise
            + null_std * null_noise
        )
        effective_noise = row_noise + (null_std / sqrt_one_minus_alpha) * null_noise
        return state, effective_noise


def _bilinear_upsample(pred_xstart, t, diffusion, target_size):
    """Bilinear upsample of clean x0 prediction, then q_sample to add noise."""
    current_size = pred_xstart.shape[-1]
    scale = target_size // current_size
    upsampled = F.interpolate(pred_xstart, scale_factor=scale, mode='bilinear', align_corners=False)
    noise = torch.randn_like(upsampled)
    sqrt_alpha = _extract_into_tensor(diffusion.sqrt_alphas_cumprod, t, upsampled.shape)
    sqrt_one_minus_alpha = _extract_into_tensor(diffusion.sqrt_one_minus_alphas_cumprod, t, upsampled.shape)
    return sqrt_alpha * upsampled + sqrt_one_minus_alpha * noise, noise


def _posterior_upsample(pred_xstart, t, diffusion, target_size):
    """Bayesian posterior upsample (equivalent to constant upsample).

    Derivation (see posterior_upsample_derivation.md for full proof):
      Let D = avg_pool, U = nearest-neighbor upsample, P = I - UD.
      Assume P x_0^{c-1} = 0 (high-res clean image = NN upsample of coarse).
      Then x_T^{c-1} = a * U(x0^c) + sqrt(b) * (UD eps + P eps)
                      = a * U(x0^c) + sqrt(b) * eps   (since UD + P = I)
      This is identical to constant upsample (q_sample of upsampled x0).

    The decomposition is kept explicit for clarity and future extension
    (e.g., if sigma_h > 0 prior on null-space high-frequency content).
    """
    current_size = pred_xstart.shape[-1]
    scale = target_size // current_size
    n = pred_xstart.shape[0]

    sqrt_alpha = _extract_into_tensor(diffusion.sqrt_alphas_cumprod, t, (n,)).view(n, 1, 1, 1)
    sqrt_1ma = _extract_into_tensor(diffusion.sqrt_one_minus_alphas_cumprod, t, (n,)).view(n, 1, 1, 1)

    # Upsample clean x0 prediction
    upsampled_x0 = pred_xstart.repeat_interleave(scale, dim=2).repeat_interleave(scale, dim=3)

    # Full-space i.i.d. noise (equivalent to row-space + null-space decomposition)
    eps = torch.randn(n, pred_xstart.shape[1], target_size, target_size,
                      device=pred_xstart.device, dtype=pred_xstart.dtype)

    # Standard q-sample: x_T = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * eps
    return sqrt_alpha * upsampled_x0 + sqrt_1ma * eps, eps


def _do_upsample(
    pred_xstart,
    t,
    diffusion,
    target_size,
    upsample_mode,
    sigma_h=0.0,
    coarse_noise=None,
    fdm_rescale_probe="none",
    fdm_dimension_ratio=1.0,
):
    """Dispatch to the correct upsample method."""
    fdm_rescale_probe = validate_fdm_rescaling(fdm_rescale_probe)
    if fdm_rescale_probe != "none":
        if upsample_mode not in ("noise_filling", "constant", "redraw"):
            raise ValueError(
                "fdm_rescale_probe is intentionally supported only with canonical "
                "noise_filling; combining inference probes would be ambiguous"
            )
        return _fdm_rescaled_upsample_probe(
            pred_xstart,
            t,
            diffusion,
            target_size,
            fdm_rescale_probe,
            fdm_dimension_ratio,
        )
    if upsample_mode in ("innovation_preserving", "noise_filling_preserve"):
        return _noise_filling_upsample(
            pred_xstart,
            coarse_noise,
            t,
            diffusion,
            target_size,
            sigma_h=sigma_h,
        )
    elif upsample_mode in ("noise_filling", "constant", "redraw"):
        # Canonical project-wide meaning.  "constant" and "redraw" remain
        # deprecated aliases so old experiment JSON files reproduce exactly.
        return _redraw_upsample(pred_xstart, t, diffusion, target_size)
    elif upsample_mode == "bilinear":
        return _bilinear_upsample(pred_xstart, t, diffusion, target_size)
    elif upsample_mode == "posterior":
        return _posterior_upsample(pred_xstart, t, diffusion, target_size)
    else:
        raise ValueError(f"Unsupported upsample_mode: {upsample_mode}")


def _run_stage(diffusion, model_forward, noise, model_kwargs, device, sampler, dpm_order=2):
    """Dispatch to the correct sampler for one stage."""
    common = dict(noise=noise, clip_denoised=False,
                  model_kwargs=model_kwargs, progress=False, device=device)
    if sampler == "ddpm":
        return diffusion.jump_step_sample_loop(
            model_forward, noise.shape, deterministic=False, **common)
    elif sampler == "dpm1":
        return diffusion.dpm_solver_sample_loop(
            model_forward, noise.shape, order=1, **common)
    elif sampler == "dpm2":
        return diffusion.dpm_solver_sample_loop(
            model_forward, noise.shape, order=2, **common)
    else:
        raise ValueError(f"Unknown sampler: {sampler}")


# ─────────────────────────────────────────────────────────────────────────────
# Multistage / Decouple sampling
# ─────────────────────────────────────────────────────────────────────────────

def _sample_batch_multistage(n, diffusions, models, resolutions_sampling_order,
                             num_classes, use_cfg, cfg_scale, sampler, upsample_mode,
                             num_steps_list, device, dtype):
    """
    Generate one batch with separate per-resolution models.

    ``diffusions``, ``models``, ``resolutions_sampling_order`` and
    ``num_steps_list`` are all ordered coarse -> fine.  This supports both the
    historical 8->16->32 topology and the direct 8->32 RDD ablation.
    """
    if not (
        len(diffusions) == len(models) == len(resolutions_sampling_order)
        == len(num_steps_list)
    ):
        raise ValueError("Multistage sampling lists must have the same length")

    # Class labels - random for FID evaluation
    if num_classes > 0:
        y_cond = torch.randint(0, num_classes, (n,), device=device)
    else:
        y_cond = torch.full((n,), 0, dtype=torch.long, device=device)

    if use_cfg:
        y_null = torch.full((n,), num_classes, dtype=torch.long, device=device)
        y = torch.cat([y_cond, y_null], dim=0)
        model_kwargs = dict(y=y, cfg_scale=cfg_scale)
        batch_mult = 2
        _fwd = lambda m: m.forward_with_cfg
    else:
        model_kwargs = dict(y=y_cond)
        batch_mult = 1
        _fwd = lambda m: m.forward

    pred_xstart = None
    samples = None
    stage_noise = None
    for stage_index, (diffusion, model, resolution, num_steps) in enumerate(zip(
        diffusions, models, resolutions_sampling_order, num_steps_list
    )):
        if stage_index == 0:
            stage_input = torch.randn(
                batch_mult * n, 4, resolution, resolution, device=device
            )
            stage_noise = stage_input[:n]
        else:
            t_bound = torch.full(
                (n,), num_steps - 1, dtype=torch.long, device=device
            )
            stage_input, stage_noise = _do_upsample(
                pred_xstart,
                t_bound,
                diffusion,
                resolution,
                upsample_mode,
                coarse_noise=stage_noise,
            )
            if use_cfg:
                stage_input = torch.cat([stage_input, stage_input], dim=0)

        with torch.autocast(
            device_type="cuda", dtype=dtype, enabled=(dtype == torch.float16)
        ):
            out = _run_stage(
                diffusion, _fwd(model), stage_input, model_kwargs, device, sampler
            )
        samples = out["sample"].float()
        pred_xstart = out["pred_xstart"].float()
        if use_cfg:
            samples = samples[:n]
            pred_xstart = pred_xstart[:n]

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Unified sampling
# ─────────────────────────────────────────────────────────────────────────────

def _sample_batch_unified(n, diffusions, resolutions, model, num_classes, use_cfg,
                          cfg_scale, sampler, upsample_mode, num_steps_list, device, dtype,
                          drift_scale=1.0, sigma_h=0.0, cfg_scale_per_stage=None,
                          guidance_config=None, guide_model=None,
                          autoguidance_config=None, total_timesteps=1000,
                          fdm_rescale_probe="none"):
    """
    Generate one batch of n samples using a single unified DiTMultiRes model.
    diffusions: list of diffusion objects per stage (index 0 = finest, last = coarsest)
    resolutions: list of resolutions (finest first, e.g. [32, 16, 8])
    num_steps_list: steps in sampling order (coarsest->finest)
    drift_scale: multiplier on drift correction (1.0 = default, 0.0 = no drift)
    sigma_h: extra null-space variance for noise filling upsample
    cfg_scale_per_stage: list of per-stage cfg scales in sampling order, or None
    """
    num_stages = len(resolutions)

    # Class labels
    if num_classes > 0:
        y_cond = torch.randint(0, num_classes, (n,), device=device)
    else:
        y_cond = torch.full((n,), 0, dtype=torch.long, device=device)

    if guide_model is not None:
        if use_cfg:
            raise ValueError(
                "This parity implementation keeps CFG and Autoguidance separate; "
                "set cfg_scale=1 when autoguidance_ckpt is used."
            )
        if autoguidance_config is None:
            raise ValueError("guide_model requires autoguidance_config")
        model_kwargs = dict(y=y_cond)
        batch_mult = 1
    elif use_cfg:
        y_null = torch.full((n,), num_classes, dtype=torch.long, device=device)
        y = torch.cat([y_cond, y_null], dim=0)
        model_kwargs = dict(y=y, cfg_scale=cfg_scale)
        batch_mult = 2
    else:
        model_kwargs = dict(y=y_cond)
        batch_mult = 1

    # Sampling order: coarsest -> finest (reverse of resolution list)
    sampling_order = list(range(num_stages - 1, -1, -1))  # e.g. [2, 1, 0] for 3 stages

    pred_xstart = None
    stage_noise = None

    for stage_idx, i in enumerate(sampling_order):
        res = resolutions[i]
        diffusion = diffusions[i]
        num_steps = num_steps_list[stage_idx]

        # Set drift_scale_multiplier for this stage's diffusion
        diffusion.drift_scale_multiplier = drift_scale

        # Per-stage CFG: update model_kwargs if needed
        if cfg_scale_per_stage is not None and use_cfg:
            stage_cfg = cfg_scale_per_stage[stage_idx] if stage_idx < len(cfg_scale_per_stage) else cfg_scale
            model_kwargs["cfg_scale"] = stage_cfg
        else:
            stage_cfg = cfg_scale

        if guide_model is not None:
            _fwd = _AutoguidanceForward(
                model, guide_model, autoguidance_config, total_timesteps
            )
        elif use_cfg and guidance_config is not None:
            stage_guidance = GuidanceConfig(
                scale=stage_cfg,
                low=guidance_config.low,
                high=guidance_config.high,
                schedule=guidance_config.schedule,
                schedule_power=guidance_config.schedule_power,
                min_scale=guidance_config.min_scale,
            )
            _fwd = _ScheduledCFGForward(model, stage_guidance, total_timesteps)
        elif use_cfg:
            # Exact historical forward path.
            _fwd = model.forward_with_cfg
        else:
            _fwd = model.forward

        if stage_idx == 0:
            # First stage (coarsest): sample from pure noise
            noise_shape = (batch_mult * n, 4, res, res)
            z = torch.randn(*noise_shape, device=device)
            if fdm_rescale_probe != "none":
                # At the finite top timestep, SP/VP can change the prior std.
                # This remains a boundary-only old-checkpoint probe; the reverse
                # transitions below deliberately remain the checkpoint's native
                # RDD transitions.
                top_t = torch.full(
                    (batch_mult * n,), num_steps - 1, dtype=torch.long, device=device
                )
                base_alpha = _extract_into_tensor(
                    diffusion.sqrt_alphas_cumprod, top_t, z.shape
                )
                base_sigma = _extract_into_tensor(
                    diffusion.sqrt_one_minus_alphas_cumprod, top_t, z.shape
                )
                _, probe_sigma = rescale_alpha_sigma(
                    base_alpha,
                    base_sigma,
                    spatial_dimension_ratio(resolutions[0], res),
                    fdm_rescale_probe,
                )
                z = z * (probe_sigma / base_sigma.clamp_min(1e-20))
            stage_noise = z[:n]
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=(dtype == torch.float16)):
                out = _run_stage(diffusion, _fwd, z, model_kwargs, device, sampler)
        else:
            # Upsample from previous stage
            t_bound = torch.full((n,), num_steps - 1, dtype=torch.long, device=device)
            init, stage_noise = _do_upsample(
                pred_xstart,
                t_bound,
                diffusion,
                res,
                upsample_mode,
                sigma_h=sigma_h,
                coarse_noise=stage_noise,
                fdm_rescale_probe=fdm_rescale_probe,
                fdm_dimension_ratio=spatial_dimension_ratio(resolutions[0], res),
            )
            if use_cfg:
                init = torch.cat([init, init], dim=0)
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=(dtype == torch.float16)):
                out = _run_stage(diffusion, _fwd, init, model_kwargs, device, sampler)

        samples = out["sample"].float()
        pred_xstart = out["pred_xstart"].float()
        if use_cfg:
            samples = samples[:n]
            pred_xstart = pred_xstart[:n]

    return samples


def _sample_batch_fm_unified(
    n,
    flow_matchers_sampling_order,
    resolutions,
    model,
    num_classes,
    cfg_scale,
    solver,
    upsample_mode,
    num_steps_list,
    device,
    dtype,
):
    """Sample the checkpoint-compatible RDD-FM v2 path.

    ``num_steps_list`` and the flow matchers are both in coarse-to-fine order.
    The returned NFE includes the two endpoint evaluations needed for exact
    drift-aware reconstruction at the 8→16 and 16→32 boundaries.
    """
    if num_classes > 0:
        y = torch.randint(0, num_classes, (n,), device=device)
    else:
        y = torch.zeros(n, dtype=torch.long, device=device)

    with torch.autocast(
        device_type="cuda", dtype=dtype, enabled=(dtype in {torch.float16, torch.bfloat16})
    ):
        latents, metrics = sample_rdd_fm_multistage(
            model=model,
            flow_matchers=flow_matchers_sampling_order,
            resolutions=list(reversed(resolutions)),
            num_steps_per_stage=num_steps_list,
            batch_size=n,
            device=torch.device(device),
            model_kwargs={"y": y},
            cfg_scale=cfg_scale,
            upsample_mode=upsample_mode,
            solver=solver,
            return_metrics=True,
        )
    return latents.float(), metrics


# ─────────────────────────────────────────────────────────────────────────────
# Experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(exp: Dict[str, Any], vae, rank: int, world_size: int,
                   total_samples: int, batch_size: int, vae_batch_size: int,
                   seed: int, device: str):
    """Run a single experiment from the JSONL config."""
    model_type = exp["model_type"]
    stage_lengths = [int(x) for x in exp["stage_lengths"].split(",")]
    resolutions = [int(x) for x in exp["resolutions"].split(",")]
    num_steps_list = [int(x) for x in exp["num_steps"].split(",")]
    sampler = exp.get("sampler", "ddpm")
    cfg_scale_raw = exp.get("cfg_scale", 1.0)
    upsample_mode = exp.get("upsample_mode", "noise_filling")
    fdm_rescale_probe = validate_fdm_rescaling(
        exp.get("fdm_rescale_probe", "none")
    )
    if fdm_rescale_probe != "none" and model_type != "unified":
        raise ValueError(
            "fdm_rescale_probe is currently implemented only for model_type=unified"
        )
    output_dir = exp["output_dir"]
    dtype = torch.float32

    # Per-stage CFG: "cfg_scale": "2.0,1.5,1.25" or "cfg_scale": 1.5
    if isinstance(cfg_scale_raw, str) and "," in cfg_scale_raw:
        cfg_scale_per_stage = [float(x) for x in cfg_scale_raw.split(",")]
        cfg_scale = cfg_scale_per_stage[0]  # use first for display / use_cfg check
    else:
        cfg_scale = float(cfg_scale_raw)
        cfg_scale_per_stage = None

    # Drift scale: multiplier on drift correction at inference (default 1.0)
    drift_scale = float(exp.get("drift_scale", 1.0))

    # Sigma_h: additional null-space variance in noise filling (default 0.0 = standard)
    sigma_h = float(exp.get("sigma_h", 0.0))

    # Optional limited-interval / scheduled CFG.  Leaving all fields absent
    # keeps the exact legacy model.forward_with_cfg path.
    guidance_low = float(exp.get("guidance_low", 0.0))
    guidance_high = float(exp.get("guidance_high", 1.0))
    guidance_schedule = str(exp.get("guidance_schedule", "constant"))
    guidance_power = float(exp.get("guidance_schedule_power", 1.0))
    guidance_min_scale = float(exp.get("guidance_min_scale", 1.0))
    custom_guidance = any(
        key in exp for key in (
            "guidance_low", "guidance_high", "guidance_schedule",
            "guidance_schedule_power", "guidance_min_scale",
        )
    )
    guidance_config = None
    if custom_guidance:
        guidance_config = GuidanceConfig(
            scale=cfg_scale,
            low=guidance_low,
            high=guidance_high,
            schedule=guidance_schedule,
            schedule_power=guidance_power,
            min_scale=guidance_min_scale,
        )
        guidance_config.validate()

    autoguidance_ckpt = exp.get("autoguidance_ckpt")
    autoguidance_config = None
    if autoguidance_ckpt:
        autoguidance_config = GuidanceConfig(
            scale=float(exp.get("autoguidance_scale", 2.0)),
            low=float(exp.get("autoguidance_low", 0.0)),
            high=float(exp.get("autoguidance_high", 1.0)),
            schedule=str(exp.get("autoguidance_schedule", "constant")),
            schedule_power=float(exp.get("autoguidance_schedule_power", 1.0)),
            min_scale=float(exp.get("autoguidance_min_scale", 1.0)),
        )
        autoguidance_config.validate()

    use_cfg = cfg_scale > 1.0
    num_stages = len(resolutions)
    if len(stage_lengths) != num_stages:
        raise ValueError(
            f"stage_lengths has {len(stage_lengths)} values but resolutions has {num_stages}"
        )
    if len(num_steps_list) != num_stages:
        raise ValueError(
            f"num_steps has {len(num_steps_list)} values but resolutions has {num_stages}"
        )
    if any(value <= 0 for value in stage_lengths + num_steps_list):
        raise ValueError("stage_lengths and num_steps must contain positive integers")
    for finer, coarser in zip(resolutions, resolutions[1:]):
        if finer <= coarser or finer % coarser != 0:
            raise ValueError(
                "resolutions must be strictly finest-to-coarsest with integer scale factors; "
                f"got {resolutions}"
            )
    stage_lengths_sampling_order = list(reversed(stage_lengths))
    for resolution, requested, available in zip(
        reversed(resolutions), num_steps_list, stage_lengths_sampling_order
    ):
        if requested > available:
            raise ValueError(
                f"Requested {requested} sampling steps at {resolution}x{resolution}, "
                f"but its training interval contains only {available} timesteps"
            )

    print(f"\n{'='*70}")
    print(f"  Experiment: model_type={model_type}, resolutions={resolutions}")
    print(f"  stage_lengths={stage_lengths}, num_steps={num_steps_list}, sampler={sampler}")
    print(f"  cfg_scale={cfg_scale_raw}, upsample_mode={upsample_mode}")
    print(f"  drift_scale={drift_scale}, sigma_h={sigma_h}")
    print(f"  fdm_rescale_probe={fdm_rescale_probe}")
    if fdm_rescale_probe != "none":
        print(
            "  WARNING: inference-only boundary probe on a checkpoint trained "
            "without f-DM rescaling; this is not a faithful trained f-DM model"
        )
    if guidance_config is not None:
        print(
            f"  guidance_interval=[{guidance_low}, {guidance_high}], "
            f"schedule={guidance_schedule}, min_scale={guidance_min_scale}"
        )
    if autoguidance_config is not None:
        print(
            f"  autoguidance_ckpt={autoguidance_ckpt}, "
            f"scale={autoguidance_config.scale}, "
            f"interval=[{autoguidance_config.low}, {autoguidance_config.high}]"
        )
    print(f"  output_dir={output_dir}")
    print(f"{'='*70}")

    os.makedirs(output_dir, exist_ok=True)

    # Compute sample indices for this rank
    my_indices = list(range(rank, total_samples, world_size))
    pending_indices = [i for i in my_indices if not os.path.exists(
        os.path.join(output_dir, f"{i:06d}.png"))]

    # Check last 100 files for corruption (OS-level size check, no PIL)
    corrupt_indices = []
    check_start = max(0, total_samples - 100)
    for i in range(check_start, total_samples):
        fpath = os.path.join(output_dir, f"{i:06d}.png")
        if os.path.exists(fpath) and os.path.getsize(fpath) < 1024:
            os.remove(fpath)
            corrupt_indices.append(i)

    if corrupt_indices:
        print(f"  [Rank {rank}] Deleted {len(corrupt_indices)} corrupt tail files "
              f"(indices {corrupt_indices[0]}..{corrupt_indices[-1]}), will regenerate")
        for i in corrupt_indices:
            if i in my_indices and i not in pending_indices:
                pending_indices.append(i)
        pending_indices.sort()

    existing_count = len(my_indices) - len(pending_indices)
    if existing_count > 0:
        print(f"  [Rank {rank}] Found {existing_count}/{len(my_indices)} existing, "
              f"{len(pending_indices)} remaining")

    if len(pending_indices) == 0:
        print(f"  [Rank {rank}] All samples already exist, skipping experiment.")
        return

    # ── Build diffusion objects ───────────────────────────────────────────────
    T_boundaries = [0]
    for length in stage_lengths:
        T_boundaries.append(T_boundaries[-1] + length)

    if model_type in ("multistage", "decouple"):
        if guidance_config is not None or autoguidance_config is not None:
            raise ValueError(
                "Scheduled CFG/Autoguidance parity is currently implemented for model_type=unified"
            )
        # Build in finest-first order to align with the global boundaries, then
        # reverse once so the sampler receives coarse->fine lists.
        diffusions_finest_first = []
        for i, resolution in enumerate(resolutions):
            sampling_position = num_stages - 1 - i
            projection_scale = (
                resolution // resolutions[i + 1] if i < num_stages - 1 else 2
            )
            diffusions_finest_first.append(create_diffusion_with_drift(
                f"{num_steps_list[sampling_position]}",
                c=i if i < num_stages - 1 else None,
                T_c=T_boundaries[i],
                T_c_plus_1=T_boundaries[i + 1],
                sample_scale=projection_scale,
            ))
        diffusions = list(reversed(diffusions_finest_first))

        # Load models
        ckpt_dir = exp["ckpt_dir"]
        ckpt_step = exp["ckpt_step"]
        default_architectures = {8: "DiT-B/2", 16: "DiT-L/2", 32: "DiT-XL/2"}
        models = []
        num_classes = None
        for resolution in reversed(resolutions):
            checkpoint = os.path.join(
                ckpt_dir, f"checkpoints_{resolution}x{resolution}", f"{ckpt_step}.pt"
            )
            architecture = exp.get(
                f"model_{resolution}", exp.get("model", default_architectures.get(resolution))
            )
            if architecture is None:
                raise ValueError(
                    f"No default model architecture for {resolution}x{resolution}; "
                    f"set model_{resolution} in the JSONL config"
                )
            print(f"  Loading {resolution}x{resolution} model: {architecture} from {checkpoint}")
            stage_model, stage_num_classes = _load_dit_model(
                checkpoint, architecture, resolution, device, dtype
            )
            if num_classes is None:
                num_classes = stage_num_classes
                print(f"    num_classes={num_classes}")
            elif stage_num_classes != num_classes:
                raise ValueError(
                    f"Class-count mismatch: expected {num_classes}, got {stage_num_classes} "
                    f"for {resolution}x{resolution}"
                )
            models.append(stage_model)

    elif model_type == "unified":
        # Unified model: single DiTMultiRes for all resolutions
        # Diffusions: per-stage, index 0 = finest, last = coarsest
        diffusions = []
        for i in range(num_stages):
            T_c = T_boundaries[i]
            T_c_plus_1 = T_boundaries[i + 1]
            c = i if i < num_stages - 1 else None
            # sampling position: stage i is sampled at position (num_stages - 1 - i)
            sampling_pos = (num_stages - 1) - i
            n_steps = num_steps_list[sampling_pos]
            diffusion = create_diffusion_with_drift(
                f"{n_steps}", c=c, T_c=T_c, T_c_plus_1=T_c_plus_1,
                sample_scale=(
                    resolutions[i] // resolutions[i + 1]
                    if i < num_stages - 1 else 2
                ))
            diffusions.append(diffusion)

        arch = exp["model"]
        ckpt_path = exp["ckpt"]
        print(f"  Loading unified model: {arch} from {ckpt_path}")
        model, num_classes = _load_unified_model(ckpt_path, arch, resolutions, device, dtype)
        print(f"    num_classes={num_classes}")
        guide_model = None
        if autoguidance_ckpt:
            guide_arch = exp.get("autoguidance_model", arch)
            print(f"  Loading Autoguidance model: {guide_arch} from {autoguidance_ckpt}")
            guide_model, guide_num_classes = _load_unified_model(
                autoguidance_ckpt, guide_arch, resolutions, device, dtype
            )
            if guide_num_classes != num_classes:
                raise ValueError(
                    f"Autoguidance class-count mismatch: good={num_classes}, guide={guide_num_classes}"
                )
    elif model_type == "fm_unified":
        if guidance_config is not None or autoguidance_config is not None:
            raise ValueError(
                "This sampler's scheduled guidance parity currently targets diffusion RDD, not RDD-FM"
            )
        # Reproduce train_RDD_FM_imagenet256_unified.py v2 exactly.  Its
        # normalized time boundaries are chronological (coarse→fine), while
        # the resolution list and model construction are finest→coarsest.
        total_time = sum(stage_lengths)
        s_boundaries = [0.0]
        for length in stage_lengths:
            s_boundaries.append(s_boundaries[-1] + length / total_time)

        flow_matchers_finest_first = []
        for i, res in enumerate(resolutions):
            stage_time_idx = num_stages - 1 - i
            c = i if i < num_stages - 1 else None
            scale = res // resolutions[i + 1] if i < num_stages - 1 else 2
            flow_matchers_finest_first.append(
                FlowMatchingWithDrift(
                    c=c,
                    sample_scale=scale,
                    s_k=s_boundaries[stage_time_idx],
                    s_k_plus_1=s_boundaries[stage_time_idx + 1],
                )
            )
        flow_matchers = list(reversed(flow_matchers_finest_first))

        arch = exp["model"]
        ckpt_path = exp["ckpt"]
        print(f"  Loading RDD-FM unified model: {arch} from {ckpt_path}")
        model, num_classes = _load_unified_model(ckpt_path, arch, resolutions, device, dtype)
        print(f"    num_classes={num_classes}")
        print(
            "    FM intervals coarse→fine="
            + str([(fm.s_k, fm.s_k_plus_1) for fm in flow_matchers])
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # ── Batch generation loop ─────────────────────────────────────────────────
    total_to_generate = len(pending_indices)
    num_generated = 0
    batch_idx = 0
    start_time = time.time()

    print(f"  [Rank {rank}] Generating {total_to_generate} samples in batches of {batch_size}")

    ptr = 0
    while ptr < len(pending_indices):
        current_batch = min(batch_size, len(pending_indices) - ptr)
        batch_indices = pending_indices[ptr:ptr + current_batch]

        # Seed based on first index for reproducibility
        torch.manual_seed(seed + batch_indices[0])

        if model_type in ("multistage", "decouple"):
            latents = _sample_batch_multistage(
                current_batch, diffusions, models, list(reversed(resolutions)),
                num_classes, use_cfg,
                cfg_scale, sampler, upsample_mode, num_steps_list, device, dtype)
        elif model_type == "unified":
            latents = _sample_batch_unified(
                current_batch, diffusions, resolutions, model, num_classes, use_cfg,
                cfg_scale, sampler, upsample_mode, num_steps_list, device, dtype,
                drift_scale=drift_scale, sigma_h=sigma_h,
                cfg_scale_per_stage=cfg_scale_per_stage,
                guidance_config=guidance_config,
                guide_model=guide_model,
                 autoguidance_config=autoguidance_config,
                 total_timesteps=sum(stage_lengths),
                 fdm_rescale_probe=fdm_rescale_probe)
        else:
            # Flow Matching has a generic velocity ODE.  Config aliases
            # ``ddpm``→Euler and ``dpm2``→midpoint are accepted for old launch
            # files, but result tables must use the mathematically correct
            # names Euler and midpoint RK2.
            fm_solver = {
                "ddpm": "euler",
                "euler": "euler",
                "dpm2": "midpoint",
                "midpoint": "midpoint",
                "rk2": "midpoint",
            }.get(sampler)
            if fm_solver is None:
                raise ValueError(f"Unsupported RDD-FM solver: {sampler}")
            latents, fm_metrics = _sample_batch_fm_unified(
                current_batch,
                flow_matchers,
                resolutions,
                model,
                num_classes,
                cfg_scale,
                fm_solver,
                upsample_mode,
                num_steps_list,
                device,
                dtype,
            )
            if batch_idx == 0:
                print(
                    f"  [RDD-FM] solver={fm_solver}, updates={sum(num_steps_list)}, "
                    f"actual NFE={fm_metrics['nfe']}, stage NFE={fm_metrics['stage_nfe']}"
                )

        # VAE decoding has a much higher activation-memory cost than latent
        # sampling.  Keep the requested generation batch intact, but decode it
        # in smaller chunks so large generation batches (for example 512) do
        # not OOM at the first 256x256 decoder upsampling block.
        decoded_chunks = []
        with torch.no_grad():
            for vae_start in range(0, current_batch, vae_batch_size):
                vae_end = min(vae_start + vae_batch_size, current_batch)
                images = vae.decode(latents[vae_start:vae_end] / 0.18215).sample
                decoded_chunks.append(
                    ((images.clamp(-1, 1) + 1) / 2 * 255)
                    .byte()
                    .permute(0, 2, 3, 1)
                    .cpu()
                )
                del images

        samples_np = torch.cat(decoded_chunks, dim=0).numpy()

        # Submit async save (returns immediately)
        _submit_save_task(samples_np, batch_indices, output_dir)

        num_generated += current_batch
        ptr += current_batch
        batch_idx += 1

        elapsed = time.time() - start_time
        avg_per_sample = elapsed / num_generated
        print(f"  [Rank {rank}] Batch {batch_idx}: {num_generated}/{total_to_generate} "
              f"({elapsed:.1f}s, {avg_per_sample:.3f}s/img)")

        # Free GPU memory
        del latents, decoded_chunks, samples_np
        gc.collect()
        torch.cuda.empty_cache()

    # ── Cleanup models (free GPU for next experiment) ─────────────────────────
    if model_type in ("multistage", "decouple"):
        del models
    else:
        del model
        if model_type == "unified" and guide_model is not None:
            del guide_model
    if model_type in ("multistage", "decouple", "unified"):
        del diffusions
    else:
        del flow_matchers
    gc.collect()
    torch.cuda.empty_cache()

    elapsed_total = time.time() - start_time
    print(f"  [Rank {rank}] Experiment done: {num_generated} samples in {elapsed_total:.1f}s "
          f"({elapsed_total/max(num_generated,1):.3f}s/img)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Unified FID sampling script for multi-scale diffusion models"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to JSONL config file (one experiment per line)")
    parser.add_argument("--rank", type=int, default=0,
                        help="Current GPU rank (0-indexed)")
    parser.add_argument("--world-size", type=int, default=8,
                        help="Total number of GPUs for parallel sampling")
    parser.add_argument("--total-samples", type=int, default=50000,
                        help="Total number of samples per experiment")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Batch size for generation")
    parser.add_argument("--vae-batch-size", type=int, default=32,
                        help="VAE decode chunk size (does not change the generation batch size)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base random seed")
    parser.add_argument("--vae", type=str, default="ema", choices=["ema", "mse"],
                        help="VAE variant")
    parser.add_argument("--vae-path", type=str, default=None,
                        help="Optional local VAE directory; takes precedence over the HF cache")
    parser.add_argument("--no-eval", action="store_true",
                        help="Disable automatic NPZ packing and FID evaluation")

    args = parser.parse_args()

    if args.vae_batch_size <= 0:
        parser.error("--vae-batch-size must be positive")

    torch.set_grad_enabled(False)

    # Device setup
    rank = args.rank
    world_size = args.world_size
    if world_size > 1:
        device = "cuda:0"
        torch.cuda.set_device(0)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"{'='*70}")
    print(f"  Unified FID Sampling Script")
    print(f"  rank={rank}/{world_size}, device={device}")
    print(f"  total_samples={args.total_samples}, batch_size={args.batch_size}, seed={args.seed}")
    print(f"  config={args.config}")
    print(f"{'='*70}")

    # Load VAE once (shared across all experiments)
    print("\n=== Loading VAE ===")
    if args.vae_path:
        local_vae_path = os.path.abspath(args.vae_path)
        if not os.path.isdir(local_vae_path):
            raise FileNotFoundError(f"--vae-path does not exist: {local_vae_path}")
        print(f"  Loading VAE from explicit local path: {local_vae_path}")
        vae = AutoencoderKL.from_pretrained(local_vae_path, local_files_only=True).to(device)
    else:
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        local_vae_path = os.path.join(
            hf_home, "hub", f"models--stabilityai--sd-vae-ft-{args.vae}", "snapshots"
        )
    if not args.vae_path and os.path.isdir(local_vae_path):
        # Use local snapshot directly to avoid any network access
        snapshots = os.listdir(local_vae_path)
        local_vae_path = os.path.join(local_vae_path, snapshots[0])
        print(f"  Loading VAE from local cache: {local_vae_path}")
        vae = AutoencoderKL.from_pretrained(local_vae_path, local_files_only=True).to(device)
    elif not args.vae_path:
        vae = AutoencoderKL.from_pretrained(
            f"stabilityai/sd-vae-ft-{args.vae}", local_files_only=True
        ).to(device)
    print("  VAE loaded")

    # Read JSONL config
    experiments = []
    with open(args.config, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                exp = json.loads(line)
                experiments.append(exp)
            except json.JSONDecodeError as e:
                print(f"  WARNING: Skipping malformed line {line_num}: {e}")

    print(f"\n  Loaded {len(experiments)} experiment(s) from {args.config}")

    # Run each experiment (pipeline: don't wait for previous I/O)
    failed_experiments = []
    for exp_idx, exp in enumerate(experiments):
        print(f"\n\n{'#'*70}")
        print(f"  EXPERIMENT {exp_idx + 1}/{len(experiments)}")
        print(f"{'#'*70}")

        try:
            run_experiment(
                exp=exp,
                vae=vae,
                rank=rank,
                world_size=world_size,
                total_samples=args.total_samples,
                batch_size=args.batch_size,
                vae_batch_size=args.vae_batch_size,
                seed=args.seed,
                device=device,
            )
        except Exception as e:
            import traceback
            print(f"\n  ERROR in experiment {exp_idx + 1}: {e}")
            traceback.print_exc()
            print("  Continuing to next experiment...")
            failed_experiments.append(exp_idx + 1)
            continue

        # Submit background NPZ + FID evaluation (only rank 0 does eval,
        # since it needs all files from all ranks to be present)
        if rank == 0 and not args.no_eval:
            output_dir = exp["output_dir"]
            # Drain this experiment's I/O futures so next experiment gets a fresh list
            exp_io_futures = _drain_io_futures()
            _submit_eval_task(output_dir, args.total_samples, exp_io_futures)
        elif not args.no_eval:
            # Non-rank-0: just drain futures (they'll complete in background)
            _drain_io_futures()

    # Wait for ALL remaining I/O and evaluations to finish before exit
    print(f"\n\n{'='*70}")
    if _io_futures:
        print(f"  Waiting for async I/O to complete ({len(_io_futures)} pending tasks)...")
        _wait_all_io()
    if _eval_futures:
        print(f"  Waiting for background evaluations to complete ({len(_eval_futures)} pending)...")
        _wait_all_evals()
    if failed_experiments:
        raise SystemExit(f"Experiment(s) failed: {failed_experiments}")
    print(f"  All done! Rank {rank} exiting.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
