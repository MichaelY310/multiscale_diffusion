#!/bin/bash
# exp08_inference_latency.sh
# ==============================
# 实验目的: 推理速度 benchmark，比较 DiT 和 RDD 在不同步数配置下的延迟。
# 对应论文 Section: Inference efficiency / Latency comparison
#
# 配置 (单GPU, batch_size=1, 100次重复):
#   1. DiT 250步
#   2. RDD 250步 (90/80/80)
#   3. RDD 125步 (45/40/40)
#   4. RDD  65步 (25/20/20)
#
# 输出格式: config, mean_ms, std_ms, img_per_sec
#
# 用法:
#   source /home/tiger/multiscale_diffusion/2d/2d/bin/activate
#   bash exp08_inference_latency.sh

set -e

# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────
source /home/tiger/multiscale_diffusion/2d/2d/bin/activate
export HF_HUB_OFFLINE=1
unset no_proxy NO_PROXY http_proxy https_proxy HTTP_PROXY HTTPS_PROXY HF_ENDPOINT
export HF_HOME="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/.cache/huggingface"
export TMPDIR="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/tmp"
mkdir -p "$TMPDIR"

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_DIR="/home/tiger/multiscale_diffusion"
RDD_CKPT="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/ImageNet_original/RDD-Unified-XL-300-300-400/imagenet256-unified/checkpoints/0400000.pt"
# TODO: 确认 DiT baseline checkpoint 路径
DIT_CKPT="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/DiT-XL-baseline/imagenet256-unified/checkpoints/PLACEHOLDER.pt"
RESULTS_DIR="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/inference_latency"

mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "[exp08] Inference Latency Benchmark"
echo "  RDD Checkpoint: ${RDD_CKPT}"
echo "  DiT Checkpoint: ${DIT_CKPT}"
echo "  Device: Single GPU (CUDA:0)"
echo "  Batch size: 1, Repeats: 100"
echo "============================================================"

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark script (inline Python)
# ─────────────────────────────────────────────────────────────────────────────
BENCHMARK_SCRIPT="${RESULTS_DIR}/benchmark_latency.py"

cat > "$BENCHMARK_SCRIPT" << 'PYTHON_SCRIPT'
"""
Inference latency benchmark for DiT and RDD models.
Measures wall-clock time for single-image generation on a single GPU.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import sys
import os
import time
import json
import numpy as np
import argparse

# Path setup
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
project_dir = "/home/tiger/multiscale_diffusion"
dit_path = os.path.join(project_dir, "DiT")
sys.path.insert(0, dit_path)
from models import DiT_models
from models_multires import DiTMultiRes_models
sys.path.remove(dit_path)
sys.path.insert(0, project_dir)
from diffusion import create_diffusion_with_drift
from diffusion.respace import _WrappedModel
from diffusion.gaussian_diffusion_with_drift import _extract_into_tensor
from diffusers.models import AutoencoderKL
import torch.nn.functional as F


def load_unified_model(ckpt_path, arch, resolutions, device):
    """Load a DiTMultiRes checkpoint."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    weights = sd.get("ema", sd.get("model"))
    # Infer num_classes
    emb = weights.get("y_embedder.embedding_table.weight")
    num_classes = emb.shape[0] - 1 if emb is not None else 0
    patch_size = int(arch.split("/")[1])
    base_size = resolutions[0] // patch_size
    model = DiTMultiRes_models[arch](num_classes=num_classes, base_size=base_size)
    model.load_state_dict(weights)
    model = model.to(device=device).eval()
    del sd, weights
    torch.cuda.empty_cache()
    return model, num_classes


def noise_filling_upsample(pred_xstart, t, diffusion, target_size):
    """Nearest-neighbor upsample + q_sample."""
    current_size = pred_xstart.shape[-1]
    scale = target_size // current_size
    upsampled = pred_xstart.repeat_interleave(scale, dim=2).repeat_interleave(scale, dim=3)
    noise = torch.randn_like(upsampled)
    sqrt_alpha = diffusion.sqrt_alphas_cumprod[t.item()]
    sqrt_1ma = diffusion.sqrt_one_minus_alphas_cumprod[t.item()]
    return sqrt_alpha * upsampled + sqrt_1ma * noise


def benchmark_dit(model, diffusion, num_classes, num_steps, device, num_repeats=100):
    """Benchmark standard DiT (single resolution 32x32 latent)."""
    # Warmup
    for _ in range(5):
        y = torch.randint(0, num_classes, (1,), device=device)
        y_null = torch.full((1,), num_classes, dtype=torch.long, device=device)
        y_cat = torch.cat([y, y_null], dim=0)
        model_kwargs = dict(y=y_cat, cfg_scale=1.5)
        z = torch.randn(2, 4, 32, 32, device=device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            diffusion.jump_step_sample_loop(
                model.forward_with_cfg, z.shape, noise=z,
                clip_denoised=False, model_kwargs=model_kwargs,
                progress=False, device=device)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_repeats):
        y = torch.randint(0, num_classes, (1,), device=device)
        y_null = torch.full((1,), num_classes, dtype=torch.long, device=device)
        y_cat = torch.cat([y, y_null], dim=0)
        model_kwargs = dict(y=y_cat, cfg_scale=1.5)
        z = torch.randn(2, 4, 32, 32, device=device)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            diffusion.jump_step_sample_loop(
                model.forward_with_cfg, z.shape, noise=z,
                clip_denoised=False, model_kwargs=model_kwargs,
                progress=False, device=device)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return times


def benchmark_rdd(model, diffusions, resolutions, num_classes, num_steps_list, device, num_repeats=100):
    """Benchmark RDD (multi-resolution)."""
    num_stages = len(resolutions)

    # Warmup
    for _ in range(3):
        _run_rdd_once(model, diffusions, resolutions, num_classes, num_steps_list, device)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _run_rdd_once(model, diffusions, resolutions, num_classes, num_steps_list, device)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return times


def _run_rdd_once(model, diffusions, resolutions, num_classes, num_steps_list, device):
    """Single RDD forward pass."""
    n = 1
    num_stages = len(resolutions)
    y = torch.randint(0, num_classes, (n,), device=device)
    y_null = torch.full((n,), num_classes, dtype=torch.long, device=device)
    y_cat = torch.cat([y, y_null], dim=0)
    model_kwargs = dict(y=y_cat, cfg_scale=1.5)

    # Sampling order: coarsest -> finest
    sampling_order = list(range(num_stages - 1, -1, -1))
    pred_xstart = None

    for stage_idx, i in enumerate(sampling_order):
        res = resolutions[i]
        diffusion = diffusions[i]
        num_steps = num_steps_list[stage_idx]

        if stage_idx == 0:
            z = torch.randn(2, 4, res, res, device=device)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                out = diffusion.jump_step_sample_loop(
                    model.forward_with_cfg, z.shape, noise=z,
                    clip_denoised=False, model_kwargs=model_kwargs,
                    progress=False, device=device)
        else:
            t_bound = torch.full((n,), num_steps - 1, dtype=torch.long, device=device)
            init = noise_filling_upsample(pred_xstart, t_bound, diffusion, res)
            init = torch.cat([init, init], dim=0)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                out = diffusion.jump_step_sample_loop(
                    model.forward_with_cfg, init.shape, noise=init,
                    clip_denoised=False, model_kwargs=model_kwargs,
                    progress=False, device=device)

        pred_xstart = out["pred_xstart"].float()[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rdd-ckpt", type=str, required=True)
    parser.add_argument("--dit-ckpt", type=str, default=None,
                        help="DiT checkpoint (if None, use same as RDD with single-res config)")
    parser.add_argument("--num-repeats", type=int, default=100)
    parser.add_argument("--output", type=str, default="latency_results.csv")
    args = parser.parse_args()

    device = "cuda:0"

    # Load model
    print("Loading RDD model...")
    resolutions = [32, 16, 8]
    model, num_classes = load_unified_model(args.rdd_ckpt, "DiT-XL/2", resolutions, device)
    print(f"  num_classes={num_classes}")

    results = []

    # ─── Config 1: DiT 250 steps ─────────────────────────────────────────────
    print("\n[1/4] DiT 250 steps (single resolution 32x32)...")
    dit_diffusion = create_diffusion_with_drift(
        timestep_respacing="250", diffusion_steps=1000,
        stage_lengths=[1000], resolutions=[32], stage_index=0
    )
    times = benchmark_dit(model, dit_diffusion, num_classes, 250, device, args.num_repeats)
    mean_ms = np.mean(times)
    std_ms = np.std(times)
    results.append(("DiT-250", mean_ms, std_ms, 1000.0/mean_ms))
    print(f"  → {mean_ms:.1f} ± {std_ms:.1f} ms, {1000.0/mean_ms:.2f} img/s")

    # ─── Config 2: RDD 250 steps (90/80/80) ──────────────────────────────────
    print("\n[2/4] RDD 250 steps (90/80/80)...")
    stage_lengths = [300, 300, 400]
    num_steps_list = [90, 80, 80]  # coarse to fine sampling order
    rdd_diffusions = []
    for i, (sl, res) in enumerate(zip(stage_lengths, resolutions)):
        d = create_diffusion_with_drift(
            timestep_respacing=str(num_steps_list[len(resolutions)-1-i]),
            diffusion_steps=1000,
            stage_lengths=stage_lengths, resolutions=resolutions,
            stage_index=i
        )
        rdd_diffusions.append(d)
    times = benchmark_rdd(model, rdd_diffusions, resolutions, num_classes, num_steps_list, device, args.num_repeats)
    mean_ms = np.mean(times)
    std_ms = np.std(times)
    results.append(("RDD-250(90/80/80)", mean_ms, std_ms, 1000.0/mean_ms))
    print(f"  → {mean_ms:.1f} ± {std_ms:.1f} ms, {1000.0/mean_ms:.2f} img/s")

    # ─── Config 3: RDD 125 steps (45/40/40) ──────────────────────────────────
    print("\n[3/4] RDD 125 steps (45/40/40)...")
    num_steps_list = [45, 40, 40]
    rdd_diffusions = []
    for i, (sl, res) in enumerate(zip(stage_lengths, resolutions)):
        d = create_diffusion_with_drift(
            timestep_respacing=str(num_steps_list[len(resolutions)-1-i]),
            diffusion_steps=1000,
            stage_lengths=stage_lengths, resolutions=resolutions,
            stage_index=i
        )
        rdd_diffusions.append(d)
    times = benchmark_rdd(model, rdd_diffusions, resolutions, num_classes, num_steps_list, device, args.num_repeats)
    mean_ms = np.mean(times)
    std_ms = np.std(times)
    results.append(("RDD-125(45/40/40)", mean_ms, std_ms, 1000.0/mean_ms))
    print(f"  → {mean_ms:.1f} ± {std_ms:.1f} ms, {1000.0/mean_ms:.2f} img/s")

    # ─── Config 4: RDD 65 steps (25/20/20) ───────────────────────────────────
    print("\n[4/4] RDD 65 steps (25/20/20)...")
    num_steps_list = [25, 20, 20]
    rdd_diffusions = []
    for i, (sl, res) in enumerate(zip(stage_lengths, resolutions)):
        d = create_diffusion_with_drift(
            timestep_respacing=str(num_steps_list[len(resolutions)-1-i]),
            diffusion_steps=1000,
            stage_lengths=stage_lengths, resolutions=resolutions,
            stage_index=i
        )
        rdd_diffusions.append(d)
    times = benchmark_rdd(model, rdd_diffusions, resolutions, num_classes, num_steps_list, device, args.num_repeats)
    mean_ms = np.mean(times)
    std_ms = np.std(times)
    results.append(("RDD-65(25/20/20)", mean_ms, std_ms, 1000.0/mean_ms))
    print(f"  → {mean_ms:.1f} ± {std_ms:.1f} ms, {1000.0/mean_ms:.2f} img/s")

    # ─── Save results ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"{'config':<25} {'mean_ms':>10} {'std_ms':>10} {'img_per_sec':>12}")
    print(f"{'-'*60}")
    with open(args.output, "w") as f:
        f.write("config,mean_ms,std_ms,img_per_sec\n")
        for config, mean, std, ips in results:
            print(f"{config:<25} {mean:>10.1f} {std:>10.1f} {ips:>12.2f}")
            f.write(f"{config},{mean:.1f},{std:.1f},{ips:.2f}\n")
    print(f"{'='*60}")
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
PYTHON_SCRIPT

echo "[exp08] Benchmark script written to: ${BENCHMARK_SCRIPT}"

# ─────────────────────────────────────────────────────────────────────────────
# Run benchmark
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_CSV="${RESULTS_DIR}/latency_results.csv"

echo "[exp08] Running inference latency benchmark..."
CUDA_VISIBLE_DEVICES=0 python -u "$BENCHMARK_SCRIPT" \
    --rdd-ckpt "$RDD_CKPT" \
    --num-repeats 100 \
    --output "$OUTPUT_CSV"

echo ""
echo "============================================================"
echo "[exp08] Results:"
if [ -f "$OUTPUT_CSV" ]; then
    cat "$OUTPUT_CSV"
fi
echo "============================================================"
echo "[exp08] All done. Results in: ${OUTPUT_CSV}"
echo "============================================================"
