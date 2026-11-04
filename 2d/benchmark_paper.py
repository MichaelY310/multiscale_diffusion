"""
论文用：一站式计算量 & 速度 Benchmark
═══════════════════════════════════════════════════════════════════════════════
6 个模型配置：
  1. Original DiT-XL/2
  2. Multi-Stage 300,300,400 (XL+L+B)
  3. Multi-Stage 100,300,600 (XL+L+B)
  4. Decouple (XL+L+B, 300,300,400 stage)
  5. Unified DiTMultiRes-XL/2 (300,300,400)
  6. Unified DiTMultiRes-L/2 (300,300,400)

每个模型测：训练速度 + 250步采样(80/80/90) + 1000步采样(300/300/400)
共 18 个实验，8张卡并行（每张卡串行跑分配到它的实验）。

250 步配置: num_steps_per_stage = [90, 80, 80] (32x32=90, 16x16=80, 8x8=80)
1000步配置: num_steps_per_stage = [300, 300, 400] (32x32=300, 16x16=300, 8x8=400)
Original DiT: 250 DDPM / 1000 DDPM

Usage:
    python 2d/benchmark_paper.py [--warmup 3] [--steps 10]
"""

import argparse
import json
import os
import sys
import time
import multiprocessing as mp

import torch
import torch.nn as nn
import numpy as np

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

sys.path.insert(0, "/home/tiger/multiscale_diffusion/DiT")
from models import DiT_models
from models_multires import DiTMultiRes_models

sys.path.insert(0, "/home/tiger/multiscale_diffusion")
from diffusion import create_diffusion_with_drift


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ═══════════════════════════════════════════════════════════════════════════════
# Training step functions
# ═══════════════════════════════════════════════════════════════════════════════

def train_step_original_dit(model, opt, diffusion, batch_size, device):
    x = torch.randn(batch_size, 4, 32, 32, device=device)
    y = torch.randint(0, 1000, (batch_size,), device=device)
    t = torch.randint(0, 1000, (batch_size,), device=device)
    opt.zero_grad()
    loss = diffusion.training_losses(model, x, t, dict(y=y))["loss"].mean()
    loss.backward()
    opt.step()


def train_step_multistage(models, opts, diffusions, batch_size, device, stage_lengths):
    T_0, T_1, T_2, T_3 = 0, stage_lengths[0], stage_lengths[0]+stage_lengths[1], sum(stage_lengths)
    y = torch.randint(0, 1000, (batch_size,), device=device)
    mk = dict(y=y)
    for (model, opt, diff, t_lo, t_hi, res) in [
        (models[0], opts[0], diffusions[0], T_0, T_1, 32),
        (models[1], opts[1], diffusions[1], T_1, T_2, 16),
        (models[2], opts[2], diffusions[2], T_2, T_3, 8),
    ]:
        x = torch.randn(batch_size, 4, res, res, device=device)
        t = torch.randint(0, t_hi - t_lo, (batch_size,), device=device)
        opt.zero_grad()
        loss = diff.training_losses(model, x, t, mk)["loss"].mean()
        loss.backward()
        opt.step()


def train_step_unified(model, opt, diffusions, batch_size, device):
    y = torch.randint(0, 1000, (batch_size,), device=device)
    mk = dict(y=y)
    opt.zero_grad()
    total_loss = 0
    for diff, res in zip(diffusions, [32, 16, 8]):
        x = torch.randn(batch_size, 4, res, res, device=device)
        t_range = diff.num_timesteps
        t = torch.randint(0, t_range, (batch_size,), device=device)
        total_loss = total_loss + diff.training_losses(model, x, t, mk)["loss"].mean()
    total_loss.backward()
    opt.step()


# ═══════════════════════════════════════════════════════════════════════════════
# Sampling functions
# ═══════════════════════════════════════════════════════════════════════════════

def sample_original_dit(model, num_steps, batch_size, device):
    diff = create_diffusion_with_drift(
        timestep_respacing=str(num_steps), diffusion_steps=1000, c=None, T_c=0, T_c_plus_1=1000
    )
    noise = torch.randn(batch_size, 4, 32, 32, device=device)
    y = torch.randint(0, 1000, (batch_size,), device=device)
    with torch.no_grad():
        sample = diff.p_sample_loop(
            model, noise.shape, noise=noise, clip_denoised=False,
            model_kwargs=dict(y=y), device=device
        )
    return sample


def sample_multistage_rdd(models, stage_lengths, num_steps_per_stage, batch_size, device):
    """num_steps_per_stage: [steps_32x32, steps_16x16, steps_8x8]"""
    T_0 = 0
    T_1 = stage_lengths[0]
    T_2 = stage_lengths[0] + stage_lengths[1]
    T_3 = sum(stage_lengths)
    y = torch.randint(0, 1000, (batch_size,), device=device)
    mk = dict(y=y)

    # Stage 8x8
    diff_8 = create_diffusion_with_drift(
        timestep_respacing=str(num_steps_per_stage[2]),
        diffusion_steps=1000, c=None, T_c=T_2, T_c_plus_1=T_3
    )
    noise_8 = torch.randn(batch_size, 4, 8, 8, device=device)
    with torch.no_grad():
        x_8 = diff_8.p_sample_loop(models[2], noise_8.shape, noise=noise_8,
                                    clip_denoised=False, model_kwargs=mk, device=device)["sample"]
    # Stage 16x16
    diff_16 = create_diffusion_with_drift(
        timestep_respacing=str(num_steps_per_stage[1]),
        diffusion_steps=1000, c=1, T_c=T_1, T_c_plus_1=T_2
    )
    x_16_init = torch.nn.functional.interpolate(x_8, size=16, mode='bilinear')
    x_16_init = x_16_init + torch.randn_like(x_16_init) * 0.5
    with torch.no_grad():
        x_16 = diff_16.p_sample_loop(models[1], x_16_init.shape, noise=x_16_init,
                                     clip_denoised=False, model_kwargs=mk, device=device)["sample"]
    # Stage 32x32
    diff_32 = create_diffusion_with_drift(
        timestep_respacing=str(num_steps_per_stage[0]),
        diffusion_steps=1000, c=0, T_c=T_0, T_c_plus_1=T_1
    )
    x_32_init = torch.nn.functional.interpolate(x_16, size=32, mode='bilinear')
    x_32_init = x_32_init + torch.randn_like(x_32_init) * 0.3
    with torch.no_grad():
        x_32 = diff_32.p_sample_loop(models[0], x_32_init.shape, noise=x_32_init,
                                     clip_denoised=False, model_kwargs=mk, device=device)["sample"]
    return x_32


def sample_unified_rdd(model, stage_lengths, num_steps_per_stage, batch_size, device):
    """num_steps_per_stage: [steps_32x32, steps_16x16, steps_8x8]"""
    T_0 = 0
    T_1 = stage_lengths[0]
    T_2 = stage_lengths[0] + stage_lengths[1]
    T_3 = sum(stage_lengths)
    y = torch.randint(0, 1000, (batch_size,), device=device)
    mk = dict(y=y)

    # Stage 8x8
    diff_8 = create_diffusion_with_drift(
        timestep_respacing=str(num_steps_per_stage[2]),
        diffusion_steps=1000, c=None, T_c=T_2, T_c_plus_1=T_3
    )
    noise_8 = torch.randn(batch_size, 4, 8, 8, device=device)
    with torch.no_grad():
        x_8 = diff_8.p_sample_loop(model, noise_8.shape, noise=noise_8,
                                   clip_denoised=False, model_kwargs=mk, device=device)["sample"]
    # Stage 16x16
    diff_16 = create_diffusion_with_drift(
        timestep_respacing=str(num_steps_per_stage[1]),
        diffusion_steps=1000, c=1, T_c=T_1, T_c_plus_1=T_2
    )
    x_16_init = torch.nn.functional.interpolate(x_8, size=16, mode='bilinear')
    x_16_init = x_16_init + torch.randn_like(x_16_init) * 0.5
    with torch.no_grad():
        x_16 = diff_16.p_sample_loop(model, x_16_init.shape, noise=x_16_init,
                                     clip_denoised=False, model_kwargs=mk, device=device)["sample"]
    # Stage 32x32
    diff_32 = create_diffusion_with_drift(
        timestep_respacing=str(num_steps_per_stage[0]),
        diffusion_steps=1000, c=0, T_c=T_0, T_c_plus_1=T_1
    )
    x_32_init = torch.nn.functional.interpolate(x_16, size=32, mode='bilinear')
    x_32_init = x_32_init + torch.randn_like(x_32_init) * 0.3
    with torch.no_grad():
        x_32 = diff_32.p_sample_loop(model, x_32_init.shape, noise=x_32_init,
                                     clip_denoised=False, model_kwargs=mk, device=device)["sample"]
    return x_32


# ═══════════════════════════════════════════════════════════════════════════════
# Timing helper
# ═══════════════════════════════════════════════════════════════════════════════

def timed_run(fn, warmup, steps):
    times = []
    for i in range(warmup + steps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
        if i >= warmup:
            times.append(elapsed)
    return {
        "mean_ms": round(float(np.mean(times)), 2),
        "std_ms": round(float(np.std(times)), 2),
        "min_ms": round(float(min(times)), 2),
        "max_ms": round(float(max(times)), 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Worker function (one GPU, multiple experiments)
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiments_on_gpu(gpu_id, experiment_configs, batch_size, sample_batch, warmup, steps):
    """Run a list of experiments on a single GPU, return results dict."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(f"cuda:{gpu_id}")
    results = {}

    for exp in experiment_configs:
        exp_name = exp["name"]
        exp_type = exp["type"]  # "train", "sample_250", "sample_1000"
        model_name = exp["model_name"]

        print(f"  [GPU {gpu_id}] Running: {exp_name}")

        if model_name == "original":
            model = DiT_models["DiT-XL/2"](input_size=32, num_classes=1000).to(device)
            if exp_type == "train":
                model.train()
                diff = create_diffusion_with_drift(timestep_respacing="", diffusion_steps=1000, c=None, T_c=0, T_c_plus_1=1000)
                opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
                t = timed_run(lambda: train_step_original_dit(model, opt, diff, batch_size, device), warmup, steps)
            else:
                model.eval()
                num_steps = 250 if exp_type == "sample_250" else 1000
                t = timed_run(lambda: sample_original_dit(model, num_steps, sample_batch, device), warmup, steps)

        elif model_name in ("ms_300_300_400", "ms_100_300_600", "decouple"):
            sl_map = {"ms_300_300_400": [300, 300, 400], "ms_100_300_600": [100, 300, 600], "decouple": [300, 300, 400]}
            sl = sl_map[model_name]
            T_0, T_1, T_2, T_3 = 0, sl[0], sl[0]+sl[1], sum(sl)

            m_xl = DiT_models["DiT-XL/2"](input_size=32, num_classes=1000).to(device)
            m_l = DiT_models["DiT-L/2"](input_size=16, num_classes=1000).to(device)
            m_b = DiT_models["DiT-B/2"](input_size=8, num_classes=1000).to(device)

            if exp_type == "train":
                m_xl.train(); m_l.train(); m_b.train()
                diffs = [
                    create_diffusion_with_drift(timestep_respacing="", diffusion_steps=1000, c=0, T_c=T_0, T_c_plus_1=T_1),
                    create_diffusion_with_drift(timestep_respacing="", diffusion_steps=1000, c=1, T_c=T_1, T_c_plus_1=T_2),
                    create_diffusion_with_drift(timestep_respacing="", diffusion_steps=1000, c=None, T_c=T_2, T_c_plus_1=T_3),
                ]
                opts = [
                    torch.optim.AdamW(m_xl.parameters(), lr=1e-4, weight_decay=0),
                    torch.optim.AdamW(m_l.parameters(), lr=1e-4, weight_decay=0),
                    torch.optim.AdamW(m_b.parameters(), lr=1e-4, weight_decay=0),
                ]
                t = timed_run(
                    lambda: train_step_multistage([m_xl, m_l, m_b], opts, diffs, batch_size, device, sl),
                    warmup, steps
                )
            else:
                m_xl.eval(); m_l.eval(); m_b.eval()
                nps = [90, 80, 80] if exp_type == "sample_250" else [300, 300, 400]
                t = timed_run(
                    lambda: sample_multistage_rdd([m_xl, m_l, m_b], sl, nps, sample_batch, device),
                    warmup, steps
                )
            del m_xl, m_l, m_b

        elif model_name in ("unified_xl", "unified_l"):
            dit_key = "DiT-XL/2" if model_name == "unified_xl" else "DiT-L/2"
            sl = [300, 300, 400]
            T_0, T_1, T_2, T_3 = 0, sl[0], sl[0]+sl[1], sum(sl)
            model = DiTMultiRes_models[dit_key](num_classes=1000, base_size=16).to(device)

            if exp_type == "train":
                model.train()
                diffs = [
                    create_diffusion_with_drift(timestep_respacing="", diffusion_steps=1000, c=0, T_c=T_0, T_c_plus_1=T_1),
                    create_diffusion_with_drift(timestep_respacing="", diffusion_steps=1000, c=1, T_c=T_1, T_c_plus_1=T_2),
                    create_diffusion_with_drift(timestep_respacing="", diffusion_steps=1000, c=None, T_c=T_2, T_c_plus_1=T_3),
                ]
                opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
                t = timed_run(lambda: train_step_unified(model, opt, diffs, batch_size, device), warmup, steps)
            else:
                model.eval()
                nps = [90, 80, 80] if exp_type == "sample_250" else [300, 300, 400]
                t = timed_run(
                    lambda: sample_unified_rdd(model, sl, nps, sample_batch, device),
                    warmup, steps
                )
            del model

        else:
            raise ValueError(f"Unknown model: {model_name}")

        results[exp_name] = t
        print(f"  [GPU {gpu_id}] Done: {exp_name} → {t['mean_ms']:.1f} ms")
        torch.cuda.empty_cache()

    return results


def worker_fn(gpu_id, experiment_configs, batch_size, sample_batch, warmup, steps, return_dict):
    """Wrapper for multiprocessing."""
    try:
        res = run_experiments_on_gpu(gpu_id, experiment_configs, batch_size, sample_batch, warmup, steps)
        return_dict[gpu_id] = res
    except Exception as e:
        return_dict[gpu_id] = {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Paper Benchmark: Training + Sampling Speed & FLOPs (8-GPU parallel)")
    parser.add_argument("--batch-size", type=int, default=32, help="Per-GPU batch size for training benchmark")
    parser.add_argument("--sample-batch", type=int, default=16, help="Batch size for sampling benchmark")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", type=str, default="2d/speed_benchmark/paper_benchmark.json")
    parser.add_argument("--only-250", action="store_true", help="Only run 250-step sampling speed tests (6 experiments)")
    args = parser.parse_args()

    # ─── Define experiments ───
    # Use --only-250 to run only 250-step sampling experiments (6 total)
    all_experiments_full = [
        # Original DiT-XL/2
        {"name": "Original DiT-XL/2 | train", "type": "train", "model_name": "original"},
        {"name": "Original DiT-XL/2 | sample 250 steps", "type": "sample_250", "model_name": "original"},
        {"name": "Original DiT-XL/2 | sample 1000 steps", "type": "sample_1000", "model_name": "original"},
        # Multi-Stage 300,300,400
        {"name": "Multi-Stage 300,300,400 (XL+L+B) | train", "type": "train", "model_name": "ms_300_300_400"},
        {"name": "Multi-Stage 300,300,400 (XL+L+B) | sample 250 steps [90/80/80]", "type": "sample_250", "model_name": "ms_300_300_400"},
        {"name": "Multi-Stage 300,300,400 (XL+L+B) | sample 1000 steps [300/300/400]", "type": "sample_1000", "model_name": "ms_300_300_400"},
        # Multi-Stage 100,300,600
        {"name": "Multi-Stage 100,300,600 (XL+L+B) | train", "type": "train", "model_name": "ms_100_300_600"},
        {"name": "Multi-Stage 100,300,600 (XL+L+B) | sample 250 steps [90/80/80]", "type": "sample_250", "model_name": "ms_100_300_600"},
        {"name": "Multi-Stage 100,300,600 (XL+L+B) | sample 1000 steps [300/300/400]", "type": "sample_1000", "model_name": "ms_100_300_600"},
        # Decouple (same architecture as multi-stage 300,300,400 but different training)
        {"name": "Decouple 300,300,400 (XL+L+B) | train", "type": "train", "model_name": "decouple"},
        {"name": "Decouple 300,300,400 (XL+L+B) | sample 250 steps [90/80/80]", "type": "sample_250", "model_name": "decouple"},
        {"name": "Decouple 300,300,400 (XL+L+B) | sample 1000 steps [300/300/400]", "type": "sample_1000", "model_name": "decouple"},
        # Unified XL/2
        {"name": "Unified XL/2 (300,300,400) | train", "type": "train", "model_name": "unified_xl"},
        {"name": "Unified XL/2 (300,300,400) | sample 250 steps [90/80/80]", "type": "sample_250", "model_name": "unified_xl"},
        {"name": "Unified XL/2 (300,300,400) | sample 1000 steps [300/300/400]", "type": "sample_1000", "model_name": "unified_xl"},
        # Unified L/2
        {"name": "Unified L/2 (300,300,400) | train", "type": "train", "model_name": "unified_l"},
        {"name": "Unified L/2 (300,300,400) | sample 250 steps [90/80/80]", "type": "sample_250", "model_name": "unified_l"},
        {"name": "Unified L/2 (300,300,400) | sample 1000 steps [300/300/400]", "type": "sample_1000", "model_name": "unified_l"},
    ]

    if args.only_250:
        all_experiments = [e for e in all_experiments_full if e["type"] == "sample_250"]
    else:
        all_experiments = all_experiments_full

    # ─── Distribute experiments to 8 GPUs (round-robin) ───
    num_gpus = 8
    gpu_experiments = [[] for _ in range(num_gpus)]
    for i, exp in enumerate(all_experiments):
        gpu_experiments[i % num_gpus].append(exp)

    print("=" * 80)
    print("PAPER BENCHMARK: 6 Models × 3 Experiments = 18 total, 8-GPU parallel")
    print(f"  Train BS: {args.batch_size}/GPU, Sample BS: {args.sample_batch}")
    print(f"  Warmup: {args.warmup}, Timed steps: {args.steps}")
    print("=" * 80)
    print("\nExperiment distribution:")
    for gpu_id in range(num_gpus):
        names = [e["name"] for e in gpu_experiments[gpu_id]]
        print(f"  GPU {gpu_id}: {names}")

    # ─── Launch 8 processes ───
    print("\n" + "=" * 80)
    print("Launching 8 parallel workers...")
    print("=" * 80 + "\n")

    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    for gpu_id in range(num_gpus):
        if not gpu_experiments[gpu_id]:
            continue
        p = mp.Process(
            target=worker_fn,
            args=(gpu_id, gpu_experiments[gpu_id], args.batch_size, args.sample_batch, args.warmup, args.steps, return_dict)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # ─── Collect results ───
    all_results = {}
    for gpu_id in range(num_gpus):
        if gpu_id in return_dict:
            gpu_res = return_dict[gpu_id]
            if "error" in gpu_res:
                print(f"  [GPU {gpu_id}] ERROR: {gpu_res['error']}")
            else:
                all_results.update(gpu_res)

    # ─── Compute FLOPs ───
    # Need param counts (compute on CPU to avoid GPU contention)
    p_xl = sum(p.numel() for p in DiT_models["DiT-XL/2"](input_size=32, num_classes=1000).parameters())
    p_l = sum(p.numel() for p in DiT_models["DiT-L/2"](input_size=16, num_classes=1000).parameters())
    p_b = sum(p.numel() for p in DiT_models["DiT-B/2"](input_size=8, num_classes=1000).parameters())
    p_uni_xl = sum(p.numel() for p in DiTMultiRes_models["DiT-XL/2"](num_classes=1000, base_size=16).parameters())
    p_uni_l = sum(p.numel() for p in DiTMultiRes_models["DiT-L/2"](num_classes=1000, base_size=16).parameters())

    seq_32, seq_16, seq_8 = 256, 64, 16

    param_summary = {
        "Original DiT-XL/2": p_xl,
        "Multi-Stage (XL+L+B) total": p_xl + p_l + p_b,
        "  XL (32x32)": p_xl,
        "  L (16x16)": p_l,
        "  B (8x8)": p_b,
        "Unified DiTMultiRes-XL/2": p_uni_xl,
        "Unified DiTMultiRes-L/2": p_uni_l,
    }

    flops_train = {
        "Original DiT-XL/2": 6 * p_xl * seq_32,
        "Multi-Stage 300,300,400": 6 * p_xl * seq_32 + 6 * p_l * seq_16 + 6 * p_b * seq_8,
        "Multi-Stage 100,300,600": 6 * p_xl * seq_32 + 6 * p_l * seq_16 + 6 * p_b * seq_8,
        "Decouple 300,300,400": 6 * p_xl * seq_32 + 6 * p_l * seq_16 + 6 * p_b * seq_8,
        "Unified XL/2": 6 * p_uni_xl * (seq_32 + seq_16 + seq_8),
        "Unified L/2": 6 * p_uni_l * (seq_32 + seq_16 + seq_8),
    }

    # 250步: [90, 80, 80]; 1000步: [300, 300, 400]
    flops_sample = {
        "Original DiT-XL/2 250 steps": 2 * p_xl * seq_32 * 250,
        "Original DiT-XL/2 1000 steps": 2 * p_xl * seq_32 * 1000,
        "Multi-Stage 300,300,400 250 steps": 2 * (p_xl * seq_32 * 90 + p_l * seq_16 * 80 + p_b * seq_8 * 80),
        "Multi-Stage 300,300,400 1000 steps": 2 * (p_xl * seq_32 * 300 + p_l * seq_16 * 300 + p_b * seq_8 * 400),
        "Multi-Stage 100,300,600 250 steps": 2 * (p_xl * seq_32 * 90 + p_l * seq_16 * 80 + p_b * seq_8 * 80),
        "Multi-Stage 100,300,600 1000 steps": 2 * (p_xl * seq_32 * 300 + p_l * seq_16 * 300 + p_b * seq_8 * 400),
        "Decouple 300,300,400 250 steps": 2 * (p_xl * seq_32 * 90 + p_l * seq_16 * 80 + p_b * seq_8 * 80),
        "Decouple 300,300,400 1000 steps": 2 * (p_xl * seq_32 * 300 + p_l * seq_16 * 300 + p_b * seq_8 * 400),
        "Unified XL/2 250 steps": 2 * p_uni_xl * (seq_32 * 90 + seq_16 * 80 + seq_8 * 80),
        "Unified XL/2 1000 steps": 2 * p_uni_xl * (seq_32 * 300 + seq_16 * 300 + seq_8 * 400),
        "Unified L/2 250 steps": 2 * p_uni_l * (seq_32 * 90 + seq_16 * 80 + seq_8 * 80),
        "Unified L/2 1000 steps": 2 * p_uni_l * (seq_32 * 300 + seq_16 * 300 + seq_8 * 400),
    }

    # ─── Print summary ───
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print("\n[Part C] Parameter Counts:")
    print("-" * 55)
    for k, v in param_summary.items():
        print(f"  {k:<35} {v:>12,}")

    print("\n[Part A] Training FLOPs per step:")
    print("-" * 70)
    base_f = flops_train["Original DiT-XL/2"]
    for k, v in flops_train.items():
        print(f"  {k:<35} {v/1e12:>8.3f} TFLOPs  ({v/base_f:.3f}x)")

    print("\n[Part B] Sampling FLOPs (total per image):")
    print("-" * 75)
    base_s = flops_sample["Original DiT-XL/2 250 steps"]
    for k, v in flops_sample.items():
        print(f"  {k:<45} {v/1e12:>8.3f} TFLOPs  ({v/base_s:.3f}x)")

    # ─── Wall-clock results ───
    print("\n[Part A] Training Speed (wall-clock):")
    print("-" * 70)
    train_results = {k: v for k, v in all_results.items() if "train" in k}
    baseline_ms = train_results.get("Original DiT-XL/2 | train", {}).get("mean_ms", 1)
    for k, v in sorted(train_results.items()):
        ratio = v["mean_ms"] / baseline_ms if baseline_ms > 0 else 0
        print(f"  {k:<55} {v['mean_ms']:>8.1f} ms  ({ratio:.3f}x)")

    print("\n[Part B] Sampling Speed (wall-clock):")
    print("-" * 80)
    sample_results = {k: v for k, v in all_results.items() if "sample" in k}
    sb = args.sample_batch
    baseline_250 = sample_results.get("Original DiT-XL/2 | sample 250 steps", {}).get("mean_ms", 1)
    for k, v in sorted(sample_results.items()):
        ratio = v["mean_ms"] / baseline_250 if baseline_250 > 0 else 0
        print(f"  {k:<60} {v['mean_ms']:>8.1f} ms  {v['mean_ms']/sb:>6.1f} ms/img  ({ratio:.3f}x)")

    # ─── Part D: Equivalent steps ───
    print("\n[Part D] Equivalent Training Steps:")
    print("-" * 70)
    for k, v in sorted(train_results.items()):
        ratio = v["mean_ms"] / baseline_ms if baseline_ms > 0 else 0
        eq_400k = int(400000 * ratio)
        eq_450k = int(450000 * ratio)
        print(f"  {k:<55}")
        print(f"    {ratio:.4f}x → 400k={eq_400k:,} | 450k={eq_450k:,} eq. DiT steps")

    # ─── Save JSON ───
    output = {
        "metadata": {
            "num_gpus_used": num_gpus,
            "train_batch_size": args.batch_size,
            "sample_batch_size": args.sample_batch,
            "warmup": args.warmup,
            "timed_steps": args.steps,
            "tf32": True,
        },
        "params": param_summary,
        "flops_training_per_step": flops_train,
        "flops_sampling_total": flops_sample,
        "wall_clock": all_results,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n✓ Results saved to {args.output}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
