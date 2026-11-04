"""
Sampling script for the unified DiTMultiRes model (single model for all resolutions).

Instead of loading 3 separate DiT models for 3 resolutions, this script loads ONE
DiTMultiRes model and uses it for ALL stages. The model naturally handles different
resolution inputs thanks to 2D RoPE (Centralized Interpolative Positional Encoding).

Sampling goes coarse → fine: e.g. 8x8 → 16x16 → 32x32.

Example:
  python RDD_sample_unified.py \
    --model DiT-XL/2 \
    --ckpt results_imagenet256/000-imagenet256-unified/checkpoints/0100000.pt \
    --vae ema \
    --stage-lengths 300,300,400 \
    --resolutions 32,16,8 \
    --num-steps-per-stage 300,300,400 \
    --total-samples 16 \
    --batch-size 16 \
    --class-label random \
    --cfg-scale 1.5 \
    --use-high-t-bound \
    --save-grid \
    --output-dir samples_unified_preview
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
import torch.nn.functional as F
import sys
import os
import gc
import argparse
import time
import statistics
import json

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dit_path = os.path.join(parent_dir, "DiT")
sys.path.insert(0, dit_path)
from models_multires import DiTMultiRes_models
sys.path.remove(dit_path)
sys.path.insert(0, parent_dir)

from diffusion import create_diffusion_with_drift, FlowMatchingWithDrift
from diffusion.gaussian_diffusion_with_drift import _extract_into_tensor
from diffusion.flow_matching_with_drift import upsample_with_noise_fm, noise_filling_transition_fm
from diffusers.models import AutoencoderKL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def infer_num_classes(weights: dict) -> int:
    """Infer num_classes from the embedding table weight shape."""
    emb = weights.get("y_embedder.embedding_table.weight")
    if emb is None:
        return 0  # unconditional model
    # embedding_table has shape (num_classes + 1, hidden_size); the +1 is the null class
    return emb.shape[0] - 1


def upsample_with_noise(pred_xstart, t_next, diffusion_next, target_size, mode="fresh",
                        previous_state=None, previous_diffusion=None):
    """Fresh discards old noise; noise_filling recovers it from the old endpoint."""
    from sample_fid_unified import _do_upsample
    state, _ = _do_upsample(pred_xstart, t_next, diffusion_next, target_size, mode,
                            previous_state=previous_state, previous_diffusion=previous_diffusion)
    return state


def _run_stage(diffusion, model_forward, noise, model_kwargs, device, args):
    """Dispatch to the correct sampler for one stage (DDPM mode)."""
    common = dict(noise=noise, clip_denoised=False,
                  model_kwargs=model_kwargs, progress=True, device=device)
    if args.use_ddim:
        return diffusion.ddim_sample_loop(
            model_forward, noise.shape, eta=0.0, **common)
    elif getattr(args, 'use_dpm_solver', False):
        return diffusion.dpm_solver_sample_loop(
            model_forward, noise.shape, order=args.dpm_solver_order, **common)
    elif args.use_ddpm_jump_step:
        return diffusion.jump_step_sample_loop(
            model_forward, noise.shape,
            deterministic=not args.jump_step_stochastic, **common)
    else:
        return diffusion.p_sample_loop(model_forward, noise.shape, **common)


def _sample_one_batch_fm(n, flow_matchers_sampling, resolutions_sampling,
                         num_steps_list, model, vae, num_classes, use_cfg, args, device):
    """Generate one batch of n samples using Flow Matching ODE. Returns decoded images."""
    num_stages = len(resolutions_sampling)

    # Class labels
    class_label = args.class_label
    if class_label == "random":
        y_cond = torch.randint(0, num_classes, (n,), device=device)
    else:
        y_cond = torch.full((n,), int(class_label), dtype=torch.long, device=device)

    cfg_scale = args.cfg_scale if use_cfg else 1.0

    # Sampling goes coarsest → finest
    pred_x1 = None
    x = None

    for stage_idx in range(num_stages):
        fm = flow_matchers_sampling[stage_idx]
        res = resolutions_sampling[stage_idx]
        n_steps = num_steps_list[stage_idx]

        if stage_idx == 0:
            # Coarsest stage: pure noise at t=0
            x_init = torch.randn(n, 4, res, res, device=device)
        else:
            # Upsample pred_x1 from previous stage + add noise at t=s_k
            mode = getattr(args, "upsample_mode", "fresh")
            if mode == "noise_filling":
                x_init = noise_filling_transition_fm(x, pred_x1, fm.s_k, res)
            else:
                x_init = upsample_with_noise_fm(pred_x1, fm.s_k, res,
                    mode="bilinear" if mode == "bilinear" else "nearest")

        # ODE integrate within this stage
        out = fm.sample_ode(
            model, x_init, num_steps=n_steps,
            model_kwargs=dict(y=y_cond), cfg_scale=cfg_scale,
        )
        pred_x1 = out["pred_x1"]
        x = out["sample"]

    # pred_x1 is the final generated sample at finest resolution
    samples = pred_x1

    # Decode with VAE
    decoded = vae.decode(samples / 0.18215).sample
    return decoded


# ---------------------------------------------------------------------------
# Main sampling
# ---------------------------------------------------------------------------

def _sample_one_batch(n, diffusions, resolutions, model, vae,
                      num_classes, use_cfg, args, device):
    """Generate one batch of n samples. Returns decoded images [n, 3, H, W]."""
    if getattr(args, 'upsample_mode', 'fresh') == 'noise_filling' and (
        getattr(args, 'use_ddim', False) or not (
            getattr(args, 'use_dpm_solver', False) or getattr(args, 'use_ddpm_jump_step', False)
        )
    ):
        raise ValueError('Recovered noise_filling requires the jump-step or DPM endpoint; '
                         'DDIM/ancestral terminal states use a different clock. Use fresh, '
                         'or select --use-ddpm-jump-step / --use-dpm-solver.')
    num_stages = len(resolutions)

    # Class labels
    class_label = args.class_label
    if class_label == "random":
        y_cond = torch.randint(0, num_classes, (n,), device=device)
    else:
        y_cond = torch.full((n,), int(class_label), dtype=torch.long, device=device)

    if use_cfg:
        y_null = torch.full((n,), num_classes, dtype=torch.long, device=device)
        y = torch.cat([y_cond, y_null], dim=0)
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
        batch_mult = 2
        _fwd = model.forward_with_cfg
    else:
        model_kwargs = dict(y=y_cond)
        batch_mult = 1
        _fwd = model.forward

    # Sampling order: coarsest → finest
    # resolutions is [32, 16, 8] (finest first), sampling goes reversed
    sampling_order = list(range(num_stages - 1, -1, -1))  # [2, 1, 0]

    # num_steps_per_stage: sampling steps for each stage in sampling order (coarse→fine)
    num_steps_list = [int(x) for x in args.num_steps_per_stage.split(",")]
    assert len(num_steps_list) == num_stages

    samples = None
    pred_xstart = None

    for stage_idx, i in enumerate(sampling_order):
        res = resolutions[i]
        diffusion = diffusions[i]
        num_steps = num_steps_list[stage_idx]

        if stage_idx == 0:
            # First stage (coarsest): sample from pure noise
            noise_shape = (batch_mult * n, 4, res, res)
            z = torch.randn(*noise_shape, device=device)
            out = _run_stage(diffusion, _fwd, z, model_kwargs, device, args)
        else:
            # Upsample from previous stage
            if args.use_high_t_bound:
                t_bound = torch.full((n,), num_steps - 1, dtype=torch.long, device=device)
            else:
                t_bound = torch.zeros(n, dtype=torch.long, device=device)

            init = upsample_with_noise(pred_xstart, t_bound, diffusion, res,
                                       mode=getattr(args, 'upsample_mode', 'fresh'),
                                       previous_state=samples, previous_diffusion=diffusions[sampling_order[stage_idx-1]])
            if use_cfg:
                init = torch.cat([init, init], dim=0)
            out = _run_stage(diffusion, _fwd, init, model_kwargs, device, args)

        samples = out["sample"].float()
        pred_xstart = out["pred_xstart"].float()
        if use_cfg:
            samples = samples[:n]
            pred_xstart = pred_xstart[:n]

    # Decode with VAE (final stage is the finest resolution)
    decoded = vae.decode(samples / 0.18215).sample
    return decoded


def main(args):
    from PIL import Image
    import numpy as np

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    # Multi-GPU: select device by rank
    rank = args.rank
    world_size = args.world_size
    if world_size > 1:
        # CUDA_VISIBLE_DEVICES is set externally, so always use device 0
        device = "cuda:0"
        torch.cuda.set_device(0)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    total_samples = args.total_samples
    batch_size = args.batch_size
    use_cfg = args.cfg_scale > 1.0

    # Parse resolutions and stage lengths
    resolutions = [int(x) for x in args.resolutions.split(",")]
    stage_lengths = [int(x) for x in args.stage_lengths.split(",")]
    num_stages = len(resolutions)
    assert len(stage_lengths) == num_stages, \
        f"--stage-lengths must have {num_stages} values, got {len(stage_lengths)}"

    print(f"device={device}  total_samples={total_samples}  batch_size={batch_size}  rank={rank}/{world_size}")
    print(f"resolutions={resolutions}  stage_lengths={stage_lengths}")
    print(f"class_label={args.class_label}  cfg_scale={args.cfg_scale}")
    print(f"output_dir={args.output_dir}")

    # ── Diffusion objects (one per stage) ─────────────────────────────────────
    # T_boundaries: [0, stage_lengths[0], stage_lengths[0]+stage_lengths[1], ...]
    T_boundaries = [0]
    for length in stage_lengths:
        T_boundaries.append(T_boundaries[-1] + length)

    # FM time boundaries
    diffusion_steps = sum(stage_lengths)
    s_boundaries = [T / diffusion_steps for T in T_boundaries]

    # Stages: index 0 is finest (e.g. 32x32), last is coarsest (e.g. 8x8)
    # Stage i: c=i if i < num_stages-1 else None, T_c=T_boundaries[i], T_c_plus_1=T_boundaries[i+1]
    num_steps_list = [int(x) for x in args.num_steps_per_stage.split(",")]
    assert len(num_steps_list) == num_stages

    use_fm = getattr(args, 'use_fm', False)

    if use_fm:
        # Flow Matching mode: create FlowMatchingWithDrift objects
        # Sampling order: coarsest → finest
        # resolutions = [32, 16, 8], sampling goes [8, 16, 32]
        # FM time intervals (same as training):
        #   resolutions[0]=32 (finest) → time interval [s_{N-1}, 1.0) (near data)
        #   resolutions[-1]=8 (coarsest) → time interval [0, s_1) (near noise)

        flow_matchers_sampling = []  # in sampling order (coarsest first)
        resolutions_sampling = []

        for stage_idx in range(num_stages):
            # sampling_order[stage_idx] = index in resolutions list
            i = num_stages - 1 - stage_idx  # coarsest first
            res = resolutions[i]

            # Time interval for resolution resolutions[i]:
            # Same mapping as training: stage_time_idx = num_stages - 1 - i
            stage_time_idx = num_stages - 1 - i
            s_k = s_boundaries[stage_time_idx]
            s_k_plus_1 = s_boundaries[stage_time_idx + 1]

            # Drift: coarsest has no drift
            if i < num_stages - 1:
                c = i
                scale = resolutions[i] // resolutions[i + 1]
            else:
                c = None
                scale = 2

            fm = FlowMatchingWithDrift(
                c=c,
                sample_scale=scale,
                s_k=s_k,
                s_k_plus_1=s_k_plus_1,
            )
            flow_matchers_sampling.append(fm)
            resolutions_sampling.append(res)
            print(f"  FM Stage {stage_idx} (sampling order): res={res}x{res}, c={c}, "
                  f"t∈[{s_k:.3f}, {s_k_plus_1:.3f}), steps={num_steps_list[stage_idx]}")

        diffusions = None  # not used in FM mode
    else:
        # DDPM mode (original)
        # Sampling goes coarse→fine, so sampling_order = [num_stages-1, ..., 0]
        # Diffusions are created per stage index (same as training)
        diffusions = []
        for i in range(num_stages):
            T_c = T_boundaries[i]
            T_c_plus_1 = T_boundaries[i + 1]
            c = i if i < num_stages - 1 else None
            sampling_pos = (num_stages - 1) - i
            n_steps = num_steps_list[sampling_pos]
            diffusion = create_diffusion_with_drift(
                f"{n_steps}",
                c=c,
                T_c=T_c,
                T_c_plus_1=T_c_plus_1,
            )
            diffusions.append(diffusion)
            print(f"  Stage {i}: res={resolutions[i]}x{resolutions[i]}, c={c}, "
                  f"T=[{T_c},{T_c_plus_1}), sample_steps={n_steps}")
        flow_matchers_sampling = None
        resolutions_sampling = None

    # ── Load unified model ────────────────────────────────────────────────────
    print("\n=== Loading Unified DiTMultiRes Model ===")
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    weights = sd.get("ema", sd.get("model"))
    num_classes = infer_num_classes(weights)

    patch_size = int(args.model.split("/")[1])
    base_size = resolutions[0] // patch_size

    model = DiTMultiRes_models[args.model](
        num_classes=num_classes,
        base_size=base_size,
    )
    model.load_state_dict(weights)
    model = model.to(device)
    model.eval()
    del sd, weights
    gc.collect()
    torch.cuda.empty_cache()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model: {args.model}, base_size={base_size}, "
          f"num_classes={num_classes}, params={total_params:,}")

    # ── Load VAE ──────────────────────────────────────────────────────────────
    print("\n=== Loading VAE ===")
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    print("  ✓ VAE loaded")

    # ── Output directory ──────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Compute sample indices for this rank ──────────────────────────────────
    my_indices = list(range(rank, total_samples, world_size))
    pending_indices = [i for i in my_indices if not os.path.exists(
        os.path.join(args.output_dir, f"{i:06d}.png"))]
    existing_count = len(my_indices) - len(pending_indices)
    if existing_count > 0:
        print(f"\n  [Rank {rank}] Found {existing_count}/{len(my_indices)} existing samples, "
              f"{len(pending_indices)} remaining")

    # ── Batch loop ────────────────────────────────────────────────────────────
    benchmark_mode = getattr(args, 'benchmark', False)
    benchmark_warmup = getattr(args, 'benchmark_warmup', 1)
    batch_times_ms = []

    total_to_generate = len(pending_indices)
    num_generated = 0
    batch_idx = 0
    print(f"\n=== [Rank {rank}] Generating {total_to_generate} remaining samples "
          f"(of {len(my_indices)} assigned) in batches of {batch_size} ===")
    _start_time = time.time()

    ptr = 0
    while ptr < len(pending_indices):
        current_batch = min(batch_size, len(pending_indices) - ptr)
        batch_indices = pending_indices[ptr:ptr + current_batch]
        print(f"\n--- [Rank {rank}] Batch {batch_idx} ({num_generated}/{total_to_generate}) "
              f"generating {current_batch} samples ---")

        torch.manual_seed(args.seed + batch_indices[0])

        if benchmark_mode:
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        if use_fm:
            samples = _sample_one_batch_fm(
                current_batch, flow_matchers_sampling, resolutions_sampling,
                num_steps_list, model, vae, num_classes, use_cfg, args, device
            )
        else:
            samples = _sample_one_batch(
                current_batch, diffusions, resolutions, model, vae,
                num_classes, use_cfg, args, device
            )

        if benchmark_mode:
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            if batch_idx >= benchmark_warmup:
                batch_times_ms.append(elapsed_ms)
                print(f"  [benchmark] batch {batch_idx}: {elapsed_ms:.2f} ms "
                      f"({elapsed_ms/current_batch:.2f} ms/sample)")
            else:
                print(f"  [benchmark] batch {batch_idx}: {elapsed_ms:.2f} ms (warmup, discarded)")
        else:
            # [N, 3, H, W] float [-1, 1] -> [N, H, W, 3] uint8 [0, 255]
            samples_np = ((samples.clamp(-1, 1) + 1) / 2 * 255).byte() \
                .permute(0, 2, 3, 1).cpu().numpy()

            for i, img_idx in enumerate(batch_indices):
                img = Image.fromarray(samples_np[i])
                img.save(os.path.join(args.output_dir, f"{img_idx:06d}.png"))

            print(f"  ✓ saved {current_batch} images ({num_generated + current_batch}/{total_to_generate} for rank {rank})")
            del samples_np

        num_generated += current_batch
        ptr += current_batch
        batch_idx += 1

        del samples
        gc.collect()
        torch.cuda.empty_cache()

    # ── Benchmark report ──────────────────────────────────────────────────────
    if benchmark_mode and batch_times_ms:
        per_sample_sec = [t / 1000.0 / batch_size for t in batch_times_ms]
        mean_lat = statistics.mean(per_sample_sec)
        std_lat = statistics.stdev(per_sample_sec) if len(per_sample_sec) > 1 else 0.0
        total_samples_measured = len(batch_times_ms) * batch_size
        total_time_sec = sum(batch_times_ms) / 1000.0
        throughput = total_samples_measured / total_time_sec if total_time_sec > 0 else 0.0

        print(f"\n{'='*60}")
        print(f"  BENCHMARK REPORT")
        print(f"{'='*60}")
        print(f"  Warmup batches:          {benchmark_warmup}")
        print(f"  Measured batches:         {len(batch_times_ms)}")
        print(f"  Batch size:              {batch_size}")
        print(f"  Total samples measured:  {total_samples_measured}")
        print(f"  Mean latency/sample:     {mean_lat:.4f} s")
        print(f"  Std latency/sample:      {std_lat:.4f} s")
        print(f"  Throughput:              {throughput:.2f} samples/s")
        print(f"  Total measured time:     {total_time_sec:.3f} s")
        print(f"{'='*60}")

        if args.benchmark_log:
            num_steps_list_report = [int(x) for x in args.num_steps_per_stage.split(",")]
            log_data = {
                "name": "RDD_sample_unified",
                "num_warmup_batches": benchmark_warmup,
                "num_measured_batches": len(batch_times_ms),
                "batch_size": batch_size,
                "total_samples_measured": total_samples_measured,
                "mean_latency_per_sample": mean_lat,
                "std_latency_per_sample": std_lat,
                "throughput_samples_per_sec": throughput,
                "total_time_sec": total_time_sec,
                "per_batch_times_ms": batch_times_ms,
                "cfg_scale": args.cfg_scale,
                "stage_lengths": [int(x) for x in args.stage_lengths.split(",")],
                "num_steps": num_steps_list_report,
                "sampler": "ddim" if args.use_ddim else ("jump_step" if args.use_ddpm_jump_step else "ddpm"),
            }
            os.makedirs(os.path.dirname(args.benchmark_log) if os.path.dirname(args.benchmark_log) else ".", exist_ok=True)
            with open(args.benchmark_log, "w") as f:
                json.dump(log_data, f, indent=2)
            print(f"  Benchmark log saved → {args.benchmark_log}")

        return

    # ── Optional grid (rank 0 only, single-GPU mode) ──────────────────────────
    if args.save_grid and total_samples >= 16 and world_size == 1:
        print("\n=== Generating 4x4 grid preview ===")
        torch.manual_seed(args.seed)
        if use_fm:
            grid_samples = _sample_one_batch_fm(
                min(16, total_samples), flow_matchers_sampling, resolutions_sampling,
                num_steps_list, model, vae, num_classes, use_cfg, args, device
            )
        else:
            grid_samples = _sample_one_batch(
                min(16, total_samples), diffusions, resolutions, model, vae,
                num_classes, use_cfg, args, device
            )
        grid_path = os.path.join(args.output_dir, "grid_preview.png")
        save_image(grid_samples, grid_path, nrow=4, normalize=True, value_range=(-1, 1))
        print(f"  ✓ Grid saved → {grid_path}")

    _total_time = time.time() - _start_time
    print(f"\n{'='*60}")
    print(f"✓ [Rank {rank}] Done! {num_generated} samples saved to {args.output_dir}/")
    print(f"  Total time: {_total_time:.1f}s ({_total_time/60:.1f}min)")
    print(f"  Avg per sample: {_total_time/max(num_generated, 1):.3f}s")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sampling for unified DiTMultiRes model (single model, all resolutions)"
    )

    parser.add_argument("--model", type=str, default="DiT-XL/2",
                        choices=list(DiTMultiRes_models.keys()),
                        help="Model architecture name (e.g. DiT-XL/2, DiT-L/2)")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to single unified checkpoint file")
    parser.add_argument("--vae", type=str, default="ema", choices=["ema", "mse"],
                        help="VAE variant (stabilityai/sd-vae-ft-{vae})")
    parser.add_argument("--stage-lengths", type=str, default="300,300,400",
                        help="Comma-separated training stage lengths (finest→coarsest, e.g. 300,300,400)")
    parser.add_argument("--resolutions", type=str, default="32,16,8",
                        help="Comma-separated resolutions (finest→coarsest, e.g. 32,16,8)")
    parser.add_argument("--num-steps-per-stage", type=str, default="300,300,400",
                        help="Comma-separated sampling steps per stage in sampling order "
                             "(coarsest→finest, e.g. 400,300,300)")
    parser.add_argument("--total-samples", type=int, default=16,
                        help="Total number of images to generate")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--class-label", type=str, default="random",
                        help="Class label: int or 'random'")
    parser.add_argument("--cfg-scale", type=float, default=1.5,
                        help="Classifier-free guidance scale (>1.0 enables CFG)")
    parser.add_argument("--use-high-t-bound", action="store_true",
                        help="Use t=num_steps-1 at scale transitions")
    parser.add_argument("--upsample-mode", type=str, default="fresh",
                        choices=["fresh", "refresh", "bilinear", "noise_filling", "legacy_ddpm_noise_filling"],
                        help="Boundary: fresh/refresh discards old noise; noise_filling recovers endpoint noise (default: fresh)")
    parser.add_argument("--use-ddpm-jump-step", action="store_true",
                        help="Use jump-step sampler")
    parser.add_argument("--jump-step-stochastic", action="store_true",
                        help="Stochastic jump-step")
    parser.add_argument("--use-ddim", action="store_true",
                        help="Use DDIM sampler")
    parser.add_argument("--use-fm", action="store_true",
                        help="Use Flow Matching ODE sampler (for FM-trained models)")
    parser.add_argument("--use-dpm-solver", action="store_true",
                        help="Use DPM-Solver sampler")
    parser.add_argument("--dpm-solver-order", type=int, default=2, choices=[1, 2],
                        help="DPM-Solver order (1 or 2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--save-grid", action="store_true",
                        help="Save a 4x4 grid preview image")
    parser.add_argument("--output-dir", type=str, default="samples_unified_output",
                        help="Output directory for generated images")

    # ── Multi-GPU (no DDP, fully independent) ─────────────────────────────────
    parser.add_argument("--rank", type=int, default=0,
                        help="Rank of this process (0-indexed)")
    parser.add_argument("--world-size", type=int, default=1,
                        help="Total number of parallel processes")

    # ── Benchmark mode ────────────────────────────────────────────────────────
    parser.add_argument("--benchmark", action="store_true",
                        help="Enable benchmark mode: use torch.cuda.Event for precise GPU timing, "
                             "skip saving images, report mean/std/throughput")
    parser.add_argument("--benchmark-warmup", type=int, default=1,
                        help="Number of warmup batches to discard (default: 1)")
    parser.add_argument("--benchmark-log", type=str, default=None,
                        help="Path to save benchmark results as JSON")

    args = parser.parse_args()
    main(args)
