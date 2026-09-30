"""
Benchmark per-step training time for all configurations used in the project.
Measures wall-clock time, throughput, GPU memory, and FLOPs estimation.

Configurations:
1. Original DiT-XL/2 (single model, 32x32 latent, 1000 diffusion steps)
2. Multi-Stage RDD 300,300,400 (DiT-XL/2 + DiT-L/2 + DiT-B/2, 3 separate backward)
3. Multi-Stage RDD 100,300,600 (DiT-XL/2 + DiT-L/2 + DiT-B/2, 3 separate backward)
4. Unified RDD XL/2 (DiTMultiRes-XL/2, 3 resolutions, single backward)
5. Unified RDD L/2 (DiTMultiRes-L/2, 3 resolutions, single backward)

All settings match real training:
- Per-GPU batch size = 32 (global 256 / 8 GPUs)
- TF32 enabled
- AdamW optimizer (lr=1e-4, weight_decay=0)
- learn_sigma=True
- No checkpoint saving

Usage:
    python 2d/benchmark_training_speed.py [--batch-size 32] [--warmup 5] [--steps 20]
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn

# Match real training settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(repo_root, "DiT"))
from models import DiT_models
from models_multires import DiTMultiRes_models

sys.path.insert(0, repo_root)
from diffusion import create_diffusion_with_drift


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def estimate_flops_per_step(model_configs):
    """
    Rough FLOPs estimation for transformer forward+backward.
    Forward FLOPs ≈ 2 * params * seq_len (for matmul-dominated models)
    Backward FLOPs ≈ 2x forward
    Total ≈ 6 * params * seq_len per step
    """
    total = 0
    for params, seq_len in model_configs:
        total += 6 * params * seq_len
    return total


def make_model(arch, input_size, device):
    model = DiT_models[arch](input_size=input_size, num_classes=1000)
    model = model.to(device).train()
    return model


def make_multires_model(arch, base_size, device):
    model = DiTMultiRes_models[arch](num_classes=1000, base_size=base_size)
    model = model.to(device).train()
    return model


def run_benchmark(name, step_fn, batch_size, device, warmup, steps):
    """Generic benchmark runner. Returns timing stats and peak memory."""
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    times = []
    for i in range(warmup + steps):
        elapsed = step_fn()
        if i >= warmup:
            times.append(elapsed)

    peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
    mean_ms = sum(times) / len(times)
    std_ms = (sum((t - mean_ms)**2 for t in times) / len(times)) ** 0.5
    min_ms = min(times)
    max_ms = max(times)
    throughput = batch_size / (mean_ms / 1000.0)  # samples/sec

    return {
        "mean_ms": round(mean_ms, 2),
        "std_ms": round(std_ms, 2),
        "min_ms": round(min_ms, 2),
        "max_ms": round(max_ms, 2),
        "throughput_samples_per_sec": round(throughput, 1),
        "peak_memory_gb": round(peak_mem_gb, 2),
        "times_ms": [round(t, 2) for t in times],
    }


# =============================================================================
# Benchmark configurations
# =============================================================================

def benchmark_original_dit(batch_size, device, warmup, steps):
    """Original DiT-XL/2: single model, single forward+backward on 32x32."""
    model = make_model("DiT-XL/2", 32, device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    diffusion = create_diffusion_with_drift(
        timestep_respacing="", diffusion_steps=1000, c=None, T_c=0, T_c_plus_1=1000
    )
    params = count_params(model)
    seq_len = (32 // 2) ** 2  # 256 tokens
    flops = estimate_flops_per_step([(params, seq_len)])

    def step_fn():
        x = torch.randn(batch_size, 4, 32, 32, device=device)
        y = torch.randint(0, 1000, (batch_size,), device=device)
        t = torch.randint(0, 1000, (batch_size,), device=device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        opt.zero_grad()
        loss = diffusion.training_losses(model, x, t, dict(y=y))["loss"].mean()
        loss.backward()
        opt.step()

        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    print(f"  Model: DiT-XL/2, Params: {params:,}, Tokens: {seq_len}")
    result = run_benchmark("Original DiT-XL/2", step_fn, batch_size, device, warmup, steps)
    result["params"] = params
    result["estimated_flops_per_step"] = flops
    return result


def benchmark_multistage_rdd(batch_size, device, warmup, steps, stage_lengths, label):
    """Multi-Stage RDD: XL(32) + L(16) + B(8), 3 separate backward+optimizer steps."""
    model_32 = make_model("DiT-XL/2", 32, device)
    model_16 = make_model("DiT-L/2", 16, device)
    model_8 = make_model("DiT-B/2", 8, device)

    opt_32 = torch.optim.AdamW(model_32.parameters(), lr=1e-4, weight_decay=0)
    opt_16 = torch.optim.AdamW(model_16.parameters(), lr=1e-4, weight_decay=0)
    opt_8 = torch.optim.AdamW(model_8.parameters(), lr=1e-4, weight_decay=0)

    T_0, T_1, T_2, T_3 = 0, stage_lengths[0], stage_lengths[0]+stage_lengths[1], sum(stage_lengths)

    diffusion_32 = create_diffusion_with_drift(
        timestep_respacing="", diffusion_steps=1000, c=0, T_c=T_0, T_c_plus_1=T_1
    )
    diffusion_16 = create_diffusion_with_drift(
        timestep_respacing="", diffusion_steps=1000, c=1, T_c=T_1, T_c_plus_1=T_2
    )
    diffusion_8 = create_diffusion_with_drift(
        timestep_respacing="", diffusion_steps=1000, c=None, T_c=T_2, T_c_plus_1=T_3
    )

    p32, p16, p8 = count_params(model_32), count_params(model_16), count_params(model_8)
    total_params = p32 + p16 + p8
    seq_32, seq_16, seq_8 = (32//2)**2, (16//2)**2, (8//2)**2
    flops = estimate_flops_per_step([(p32, seq_32), (p16, seq_16), (p8, seq_8)])

    def step_fn():
        x_32 = torch.randn(batch_size, 4, 32, 32, device=device)
        x_16 = torch.randn(batch_size, 4, 16, 16, device=device)
        x_8 = torch.randn(batch_size, 4, 8, 8, device=device)
        y = torch.randint(0, 1000, (batch_size,), device=device)
        model_kwargs = dict(y=y)

        t_32 = torch.randint(0, T_1 - T_0, (batch_size,), device=device)
        t_16 = torch.randint(0, T_2 - T_1, (batch_size,), device=device)
        t_8 = torch.randint(0, T_3 - T_2, (batch_size,), device=device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        # Stage 32x32 (XL)
        opt_32.zero_grad()
        loss_32 = diffusion_32.training_losses(model_32, x_32, t_32, model_kwargs)["loss"].mean()
        loss_32.backward()
        opt_32.step()

        # Stage 16x16 (L)
        opt_16.zero_grad()
        loss_16 = diffusion_16.training_losses(model_16, x_16, t_16, model_kwargs)["loss"].mean()
        loss_16.backward()
        opt_16.step()

        # Stage 8x8 (B)
        opt_8.zero_grad()
        loss_8 = diffusion_8.training_losses(model_8, x_8, t_8, model_kwargs)["loss"].mean()
        loss_8.backward()
        opt_8.step()

        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    print(f"  Models: XL({p32:,}) + L({p16:,}) + B({p8:,}) = {total_params:,} total")
    print(f"  Tokens: 32x32→{seq_32}, 16x16→{seq_16}, 8x8→{seq_8}")
    print(f"  Stage lengths: {stage_lengths}")
    result = run_benchmark(label, step_fn, batch_size, device, warmup, steps)
    result["params"] = total_params
    result["params_breakdown"] = {"XL_32": p32, "L_16": p16, "B_8": p8}
    result["estimated_flops_per_step"] = flops
    result["stage_lengths"] = stage_lengths
    return result


def benchmark_unified_rdd(batch_size, device, warmup, steps, arch, label):
    """Unified RDD: single DiTMultiRes model, 3 resolutions summed loss, one backward."""
    patch_size = int(arch.split("/")[1])
    base_size = 32 // patch_size  # 16 for patch_size=2
    model = make_multires_model(arch, base_size, device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    stage_lengths = [300, 300, 400]
    T_0, T_1, T_2, T_3 = 0, 300, 600, 1000

    diffusion_32 = create_diffusion_with_drift(
        timestep_respacing="", diffusion_steps=1000, c=0, T_c=T_0, T_c_plus_1=T_1
    )
    diffusion_16 = create_diffusion_with_drift(
        timestep_respacing="", diffusion_steps=1000, c=1, T_c=T_1, T_c_plus_1=T_2
    )
    diffusion_8 = create_diffusion_with_drift(
        timestep_respacing="", diffusion_steps=1000, c=None, T_c=T_2, T_c_plus_1=T_3
    )

    params = count_params(model)
    seq_32, seq_16, seq_8 = (32//patch_size)**2, (16//patch_size)**2, (8//patch_size)**2
    # Unified: same model params, but 3 forward passes → sum of seq_lens
    flops = estimate_flops_per_step([(params, seq_32), (params, seq_16), (params, seq_8)])

    def step_fn():
        x_32 = torch.randn(batch_size, 4, 32, 32, device=device)
        x_16 = torch.randn(batch_size, 4, 16, 16, device=device)
        x_8 = torch.randn(batch_size, 4, 8, 8, device=device)
        y = torch.randint(0, 1000, (batch_size,), device=device)
        model_kwargs = dict(y=y)

        t_32 = torch.randint(0, T_1 - T_0, (batch_size,), device=device)
        t_16 = torch.randint(0, T_2 - T_1, (batch_size,), device=device)
        t_8 = torch.randint(0, T_3 - T_2, (batch_size,), device=device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        opt.zero_grad()
        loss_32 = diffusion_32.training_losses(model, x_32, t_32, model_kwargs)["loss"].mean()
        loss_16 = diffusion_16.training_losses(model, x_16, t_16, model_kwargs)["loss"].mean()
        loss_8 = diffusion_8.training_losses(model, x_8, t_8, model_kwargs)["loss"].mean()
        total_loss = loss_32 + loss_16 + loss_8
        total_loss.backward()
        opt.step()

        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    print(f"  Model: DiTMultiRes-{arch}, Params: {params:,}, base_size={base_size}")
    print(f"  Tokens: 32x32→{seq_32}, 16x16→{seq_16}, 8x8→{seq_8}")
    result = run_benchmark(label, step_fn, batch_size, device, warmup, steps)
    result["params"] = params
    result["estimated_flops_per_step"] = flops
    result["stage_lengths"] = stage_lengths
    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark training speed for all configurations")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Per-GPU batch size (default 32 = global 256 / 8 GPUs)")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Number of warmup steps (not timed)")
    parser.add_argument("--steps", type=int, default=20,
                        help="Number of timed steps")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default="2d/speed_benchmark/train_speed_comparison.json")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Get GPU info
    gpu_name = torch.cuda.get_device_name(device)
    gpu_mem_total = torch.cuda.get_device_properties(device).total_memory / (1024**3)

    print("=" * 70)
    print("TRAINING SPEED BENCHMARK")
    print("=" * 70)
    print(f"  GPU: {gpu_name} ({gpu_mem_total:.1f} GB)")
    print(f"  Per-GPU batch size: {args.batch_size}")
    print(f"  Global batch size (8 GPUs): {args.batch_size * 8}")
    print(f"  TF32: enabled")
    print(f"  Warmup steps: {args.warmup}, Timed steps: {args.steps}")
    print(f"  Optimizer: AdamW (lr=1e-4, weight_decay=0)")
    print("=" * 70)

    results = {
        "metadata": {
            "gpu": gpu_name,
            "gpu_memory_gb": round(gpu_mem_total, 1),
            "per_gpu_batch_size": args.batch_size,
            "global_batch_size": args.batch_size * 8,
            "num_gpus": 8,
            "tf32": True,
            "optimizer": "AdamW (lr=1e-4, wd=0)",
            "warmup_steps": args.warmup,
            "timed_steps": args.steps,
        },
        "benchmarks": {}
    }

    configs = [
        ("original_dit_xl2", "Original DiT-XL/2 (32x32, 1000 steps)"),
        ("multistage_300_300_400", "Multi-Stage RDD [300,300,400]"),
        ("multistage_100_300_600", "Multi-Stage RDD [100,300,600]"),
        ("unified_xl2", "Unified RDD DiTMultiRes-XL/2"),
        ("unified_l2", "Unified RDD DiTMultiRes-L/2"),
    ]

    # 1. Original DiT-XL/2
    print(f"\n[1/5] {configs[0][1]}...")
    results["benchmarks"]["original_dit_xl2"] = benchmark_original_dit(
        args.batch_size, device, args.warmup, args.steps
    )
    torch.cuda.empty_cache()

    # 2. Multi-Stage RDD 300,300,400
    print(f"\n[2/5] {configs[1][1]}...")
    results["benchmarks"]["multistage_300_300_400"] = benchmark_multistage_rdd(
        args.batch_size, device, args.warmup, args.steps,
        stage_lengths=[300, 300, 400], label=configs[1][1]
    )
    torch.cuda.empty_cache()

    # 3. Multi-Stage RDD 100,300,600
    print(f"\n[3/5] {configs[2][1]}...")
    results["benchmarks"]["multistage_100_300_600"] = benchmark_multistage_rdd(
        args.batch_size, device, args.warmup, args.steps,
        stage_lengths=[100, 300, 600], label=configs[2][1]
    )
    torch.cuda.empty_cache()

    # 4. Unified RDD XL/2
    print(f"\n[4/5] {configs[3][1]}...")
    results["benchmarks"]["unified_xl2"] = benchmark_unified_rdd(
        args.batch_size, device, args.warmup, args.steps,
        arch="DiT-XL/2", label=configs[3][1]
    )
    torch.cuda.empty_cache()

    # 5. Unified RDD L/2
    print(f"\n[5/5] {configs[4][1]}...")
    results["benchmarks"]["unified_l2"] = benchmark_unified_rdd(
        args.batch_size, device, args.warmup, args.steps,
        arch="DiT-L/2", label=configs[4][1]
    )
    torch.cuda.empty_cache()

    # ==========================================================================
    # Summary
    # ==========================================================================
    baseline_ms = results["benchmarks"]["original_dit_xl2"]["mean_ms"]
    baseline_flops = results["benchmarks"]["original_dit_xl2"]["estimated_flops_per_step"]

    print("\n")
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Configuration':<40} {'ms/step':>8} {'Ratio':>7} {'Samples/s':>10} {'Peak Mem':>9} {'Est. FLOPs':>12}")
    print("-" * 70)

    equivalents = {}
    for key, label in configs:
        b = results["benchmarks"][key]
        ratio = b["mean_ms"] / baseline_ms
        flops_ratio = b["estimated_flops_per_step"] / baseline_flops
        print(f"  {label:<38} {b['mean_ms']:>7.1f} {ratio:>6.2f}x {b['throughput_samples_per_sec']:>9.1f} {b['peak_memory_gb']:>7.1f}GB {b['estimated_flops_per_step']/1e12:>9.2f}T")
        equivalents[key] = {
            "time_ratio": round(ratio, 3),
            "flops_ratio": round(flops_ratio, 3),
        }

    print()
    print("EQUIVALENT ORIGINAL DiT STEPS (for fair wall-clock comparison):")
    print("-" * 70)
    for key, label in configs[1:]:
        ratio = equivalents[key]["time_ratio"]
        eq_400k = int(400000 * ratio)
        eq_450k = int(450000 * ratio)
        print(f"  {label:<38}")
        print(f"    400k steps → equivalent to {eq_400k:,} Original DiT steps (~{round(eq_400k/1000)*1000:,})")
        print(f"    450k steps → equivalent to {eq_450k:,} Original DiT steps (~{round(eq_450k/1000)*1000:,})")
        equivalents[key]["eq_original_steps_400k"] = eq_400k
        equivalents[key]["eq_original_steps_450k"] = eq_450k

    results["equivalents"] = equivalents

    # Also compute: how long would 400k steps take in hours (per-GPU)
    print()
    print("ESTIMATED WALL-CLOCK TIME (single GPU, 400k steps):")
    print("-" * 70)
    for key, label in configs:
        b = results["benchmarks"][key]
        hours_400k = (b["mean_ms"] * 400000) / (1000 * 3600)
        hours_450k = (b["mean_ms"] * 450000) / (1000 * 3600)
        print(f"  {label:<38} 400k: {hours_400k:>5.1f}h | 450k: {hours_450k:>5.1f}h")
        results["benchmarks"][key]["estimated_hours_400k"] = round(hours_400k, 1)
        results["benchmarks"][key]["estimated_hours_450k"] = round(hours_450k, 1)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
