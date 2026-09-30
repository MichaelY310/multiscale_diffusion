"""
JSONL-driven FID sampling for Original DiT.
Reads experiment configurations from a JSONL file and runs them sequentially.

Features:
  - Fully asynchronous image saving via ThreadPoolExecutor
  - Background NPZ packing + FID evaluation (runs in parallel with next experiment)
  - 8-GPU parallel with rank-based allocation and skip/resume logic
  - VAE loaded once, shared across all experiments
  - Models loaded per-experiment (freed after each to save GPU memory)

Usage:
  # 8-GPU parallel launch
  for RANK in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$RANK python -u sample_fid_jsonl.py \
        --config dit_experiments.jsonl \
        --rank $RANK --world-size 8 \
        --total-samples 50000 --batch-size 64 \
        --seed 0 &
  done
  wait

JSONL format (one JSON per line):
  {
    "name": "dit500k_250step_ddpm_cfg1",
    "ckpt": "/path/to/0500000.pt",
    "model": "DiT-XL/2",
    "image_size": 256,
    "num_sampling_steps": 250,
    "sampling_method": "ddpm",
    "cfg_scale": 1.0,
    "output_dir": "/path/to/output"
  }
"""
import builtins
import functools
builtins.print = functools.partial(print, flush=True)

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import os
import gc
import sys
import time
import json
import argparse
import subprocess
import numpy as np
from PIL import Image
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, Future
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import DiT_models
from diffusion import create_diffusion
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
    """Take ownership of current I/O futures and reset the global list."""
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

REF_NPZ = "/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/VIRTUAL_imagenet256_labeled.npz"
EVALUATOR = "/home/tiger/multiscale_diffusion/guided-diffusion/evaluations/evaluator.py"
EVAL_LOG_DIR = "/home/tiger/multiscale_diffusion/2d/eval_results_unified"


def _validate_and_pack_npz(output_dir: str, total_samples: int) -> Optional[str]:
    """Validate all PNGs by file size, delete corrupt ones, pack into NPZ."""
    print(f"  [NPZ] Validating {total_samples} PNGs in {output_dir}...")

    corrupt_count = 0
    for i in range(total_samples):
        fpath = os.path.join(output_dir, f"{i:06d}.png")
        if not os.path.exists(fpath):
            continue
        sz = os.path.getsize(fpath)
        if sz < 1024:
            print(f"  [NPZ] Corrupt file {fpath}: {sz} bytes, deleting")
            os.remove(fpath)
            corrupt_count += 1

    if corrupt_count > 0:
        print(f"  [NPZ] Deleted {corrupt_count} corrupt files.")

    # Check completeness
    valid_count = sum(1 for i in range(total_samples)
                      if os.path.exists(os.path.join(output_dir, f"{i:06d}.png")))
    if valid_count < total_samples:
        print(f"  [NPZ] Only {valid_count}/{total_samples} valid files. Skipping NPZ.")
        return None

    # Pack NPZ
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

    if size_gb < 9.0:
        print(f"  [NPZ] WARNING: NPZ size {size_gb:.2f} GB < 9.0 GB, likely corrupt. Deleting.")
        os.remove(npz_path)
        return None

    return npz_path


def _run_fid_evaluation(npz_path: str, output_dir: str):
    """Run FID evaluator and save results."""
    os.makedirs(EVAL_LOG_DIR, exist_ok=True)
    log_name = os.path.basename(output_dir) + ".txt"
    log_path = os.path.join(EVAL_LOG_DIR, log_name)

    if os.path.exists(log_path):
        try:
            with open(log_path) as f:
                content = f.read()
            if "FID:" in content:
                print(f"  [FID] Already evaluated: {log_path}")
                return
        except Exception:
            pass

    print(f"  [FID] Running evaluation: {os.path.basename(output_dir)}")
    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        result = subprocess.run(
            ["python", "-u", EVALUATOR, REF_NPZ, npz_path],
            capture_output=True, text=True,
            env=env,
        )
        with open(log_path, "w") as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n--- STDERR ---\n")
                f.write(result.stderr)
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "FID:" in line or "fid:" in line.lower():
                    print(f"  [FID] {os.path.basename(output_dir)}: {line.strip()}")
                    break
            else:
                print(f"  [FID] Done: {log_path}")
        else:
            print(f"  [FID] FAILED (exit code {result.returncode}): {log_path}")
    except Exception as e:
        print(f"  [FID] ERROR: {e}")


def _background_npz_and_eval(output_dir: str, total_samples: int, io_futures: List[Future]):
    """Background task: wait for I/O, validate, pack NPZ, run FID."""
    for fut in io_futures:
        fut.result()

    # Wait for all ranks to finish
    poll_interval = 10
    while True:
        present = sum(1 for i in range(total_samples)
                      if os.path.exists(os.path.join(output_dir, f"{i:06d}.png")))
        if present >= total_samples:
            break
        print(f"  [NPZ] Waiting for all ranks: {present}/{total_samples} files present...")
        time.sleep(poll_interval)

    npz_path = _validate_and_pack_npz(output_dir, total_samples)
    if npz_path is None:
        return

    _run_fid_evaluation(npz_path, output_dir)

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
# Experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(exp: dict, vae, rank: int, world_size: int,
                   total_samples: int, batch_size: int, seed: int, device: str):
    """Run a single DiT experiment from the JSONL config."""
    name = exp.get("name", "unnamed")
    ckpt_path = exp["ckpt"]
    model_arch = exp.get("model", "DiT-XL/2")
    image_size = exp.get("image_size", 256)
    num_sampling_steps = exp.get("num_sampling_steps", 250)
    sampling_method = exp.get("sampling_method", "ddpm")
    cfg_scale = float(exp.get("cfg_scale", 1.0))
    num_classes = exp.get("num_classes", 1000)
    output_dir = exp["output_dir"]

    use_cfg = cfg_scale > 1.0
    latent_size = image_size // 8

    print(f"\n{'='*70}")
    print(f"  Experiment: {name}")
    print(f"  ckpt={ckpt_path}")
    print(f"  model={model_arch}, image_size={image_size}, latent_size={latent_size}")
    print(f"  steps={num_sampling_steps}, method={sampling_method}, cfg_scale={cfg_scale}")
    print(f"  output_dir={output_dir}")
    print(f"{'='*70}")

    os.makedirs(output_dir, exist_ok=True)

    # Skip logic
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

    # Load model
    print(f"  Loading model: {model_arch} from {ckpt_path}")
    model = DiT_models[model_arch](
        input_size=latent_size,
        num_classes=num_classes,
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "ema" in checkpoint:
        model.load_state_dict(checkpoint["ema"])
        print("  Loaded EMA weights.")
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        print("  Loaded model weights (no EMA found).")
    else:
        model.load_state_dict(checkpoint)
        print("  Loaded raw state dict.")
    model.eval()
    del checkpoint
    gc.collect()
    torch.cuda.empty_cache()

    # Create diffusion
    diffusion = create_diffusion(str(num_sampling_steps))

    # Batch generation
    total_to_generate = len(pending_indices)
    num_generated = 0
    start_time = time.time()

    print(f"  [Rank {rank}] Generating {total_to_generate} samples in batches of {batch_size}")

    ptr = 0
    while ptr < len(pending_indices):
        current_batch = min(batch_size, len(pending_indices) - ptr)
        batch_indices = pending_indices[ptr:ptr + current_batch]

        # Seed for reproducibility
        torch.manual_seed(seed + batch_indices[0])
        y = torch.randint(0, num_classes, (current_batch,), device=device)
        z = torch.randn(current_batch, 4, latent_size, latent_size, device=device)

        if use_cfg:
            z_input = torch.cat([z, z], dim=0)
            y_null = torch.tensor([num_classes] * current_batch, device=device)
            y_input = torch.cat([y, y_null], dim=0)
            model_kwargs = dict(y=y_input, cfg_scale=cfg_scale)
            model_fn = model.forward_with_cfg
        else:
            z_input = z
            model_kwargs = dict(y=y)
            model_fn = model.forward

        # Sample
        if sampling_method == "dpm_solver_1":
            samples = diffusion.dpm_solver_sample_loop(
                model_fn, z_input.shape, noise=z_input,
                clip_denoised=False, model_kwargs=model_kwargs,
                progress=False, device=device, order=1,
            )
        elif sampling_method == "dpm_solver_2":
            samples = diffusion.dpm_solver_sample_loop(
                model_fn, z_input.shape, noise=z_input,
                clip_denoised=False, model_kwargs=model_kwargs,
                progress=False, device=device, order=2,
            )
        else:
            # DDPM
            samples = diffusion.p_sample_loop(
                model_fn, z_input.shape, z_input,
                clip_denoised=False, model_kwargs=model_kwargs,
                progress=False, device=device,
            )

        if use_cfg:
            samples, _ = samples.chunk(2, dim=0)

        # VAE decode
        with torch.no_grad():
            images = vae.decode(samples / 0.18215).sample

        # Convert to uint8 numpy
        samples_np = torch.clamp(127.5 * images + 128.0, 0, 255)
        samples_np = samples_np.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        # Submit async save
        _submit_save_task(samples_np, batch_indices, output_dir)

        num_generated += current_batch
        ptr += current_batch

        elapsed = time.time() - start_time
        avg_per_sample = elapsed / num_generated
        print(f"  [Rank {rank}] {num_generated}/{total_to_generate} "
              f"({elapsed:.1f}s, {avg_per_sample:.3f}s/img)")

        del z, z_input, samples, images, samples_np
        gc.collect()
        torch.cuda.empty_cache()

    # Free model
    del model, diffusion
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
        description="JSONL-driven FID sampling for Original DiT"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to JSONL config file (one experiment per line)")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--total-samples", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vae", type=str, default="ema", choices=["ema", "mse"])
    parser.add_argument("--no-eval", action="store_true",
                        help="Disable automatic NPZ packing and FID evaluation")

    args = parser.parse_args()
    torch.set_grad_enabled(False)

    rank = args.rank
    world_size = args.world_size
    if world_size > 1:
        device = "cuda:0"
        torch.cuda.set_device(0)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"{'='*70}")
    print(f"  DiT JSONL FID Sampling")
    print(f"  rank={rank}/{world_size}, device={device}")
    print(f"  total_samples={args.total_samples}, batch_size={args.batch_size}, seed={args.seed}")
    print(f"  vae={args.vae}, config={args.config}")
    print(f"{'='*70}")

    # Load VAE once
    print("\n=== Loading VAE ===")
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    local_vae_path = os.path.join(
        hf_home, "hub", f"models--stabilityai--sd-vae-ft-{args.vae}", "snapshots"
    )
    if os.path.isdir(local_vae_path):
        snapshots = os.listdir(local_vae_path)
        local_vae_path = os.path.join(local_vae_path, snapshots[0])
        print(f"  Loading VAE from local cache: {local_vae_path}")
        vae = AutoencoderKL.from_pretrained(local_vae_path, local_files_only=True).to(device)
    else:
        vae = AutoencoderKL.from_pretrained(
            f"stabilityai/sd-vae-ft-{args.vae}", local_files_only=True
        ).to(device)
    print("  VAE loaded")

    # Read JSONL config
    experiments = []
    with open(args.config, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            try:
                exp = json.loads(line)
                experiments.append(exp)
            except json.JSONDecodeError as e:
                print(f"  WARNING: Skipping malformed line {line_num}: {e}")

    print(f"\n  Loaded {len(experiments)} experiment(s) from {args.config}")

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
                seed=args.seed,
                device=device,
            )
        except Exception as e:
            import traceback
            print(f"\n  ERROR in experiment {exp_idx + 1}: {e}")
            traceback.print_exc()
            print("  Continuing to next experiment...")
            continue

        # Submit background NPZ + FID (only rank 0 does eval)
        if rank == 0 and not args.no_eval:
            output_dir = exp["output_dir"]
            exp_io_futures = _drain_io_futures()
            _submit_eval_task(output_dir, args.total_samples, exp_io_futures)
        elif not args.no_eval:
            _drain_io_futures()

    # Wait for all remaining I/O and evaluations
    print(f"\n\n{'='*70}")
    if _io_futures:
        print(f"  Waiting for async I/O to complete ({len(_io_futures)} pending tasks)...")
        _wait_all_io()
    if _eval_futures:
        print(f"  Waiting for background evaluations to complete ({len(_eval_futures)} pending)...")
        _wait_all_evals()
    print(f"  All done! Rank {rank} exiting.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
