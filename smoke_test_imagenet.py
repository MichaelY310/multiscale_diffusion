#!/usr/bin/env python
"""
Smoke test: verify all 6 training configs can load original ImageNet data
and complete at least one training step without errors.

Usage:
  python smoke_test_imagenet.py
"""

import os
import sys
import subprocess
import time

DATA_PATH = "/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/data/imagenet/ILSVRC/Data/CLS-LOC/train"

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"

def run_smoke(name, script, extra_args=""):
    """Run a training script with --epochs 1 --ckpt-every 999999 to do minimal work."""
    cmd = (
        f"cd /home/tiger/multiscale_diffusion && "
        f"CUDA_VISIBLE_DEVICES=0 python -u {script} "
        f"--data-source imagenet --data-path {DATA_PATH} "
        f"--epochs 1 --ckpt-every 999999 --log-every 1 "
        f"--results-dir /tmp/smoke_test_{name.replace(' ', '_')} "
        f"{extra_args}"
    )
    print(f"{YELLOW}[TEST] {name}{RESET}")
    print(f"  CMD: {cmd[:120]}...")
    start = time.time()
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=300,
            env={**os.environ, "HF_HUB_OFFLINE": "1"}
        )
        elapsed = time.time() - start
        if result.returncode == 0 and "Train Loss" in result.stdout:
            # Extract the first loss line
            for line in result.stdout.split("\n"):
                if "Train Loss" in line or "Loss[" in line:
                    print(f"  {GREEN}PASS{RESET} ({elapsed:.1f}s) — {line.strip()}")
                    return True
            print(f"  {GREEN}PASS{RESET} ({elapsed:.1f}s) — no loss line but exit 0")
            return True
        else:
            print(f"  {RED}FAIL{RESET} ({elapsed:.1f}s) — exit={result.returncode}")
            # Print last 20 lines of stderr
            stderr_lines = result.stderr.strip().split("\n")[-20:]
            for l in stderr_lines:
                print(f"  STDERR: {l}")
            if result.stdout.strip():
                stdout_lines = result.stdout.strip().split("\n")[-10:]
                for l in stdout_lines:
                    print(f"  STDOUT: {l}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  {RED}TIMEOUT{RESET} (300s)")
        return False
    except Exception as e:
        print(f"  {RED}ERROR{RESET}: {e}")
        return False


def main():
    print(f"Data path: {DATA_PATH}")
    n_classes = len(os.listdir(DATA_PATH))
    n_images = sum(len(os.listdir(os.path.join(DATA_PATH, d))) for d in os.listdir(DATA_PATH))
    print(f"Available: {n_classes} classes, {n_images} images\n")

    if n_classes == 0:
        print(f"{RED}No data found! Is unzip still running?{RESET}")
        sys.exit(1)

    tests = [
        # (name, script, extra_args)
        ("DiT-XL/2", "DiT/train_imagenet256.py",
         "--model DiT-XL/2 --global-batch-size 4 --num-workers 0 --no-pin-memory"),
        ("RDD Multistage 300-300-400", "2d/train_RDD_imagenet256.py",
         "--model-32 DiT-XL/2 --model-16 DiT-L/2 --model-8 DiT-B/2 "
         "--stage-lengths 300,300,400 --global-batch-size 4 --num-workers 0"),
        ("RDD Multistage 100-300-600", "2d/train_RDD_imagenet256.py",
         "--model-32 DiT-XL/2 --model-16 DiT-L/2 --model-8 DiT-B/2 "
         "--stage-lengths 100,300,600 --global-batch-size 4 --num-workers 0"),
        ("RDD Unified XL/2", "2d/train_RDD_imagenet256_unified.py",
         "--model DiT-XL/2 --resolutions 32,16,8 --stage-lengths 300,300,400 "
         "--global-batch-size 4 --num-workers 0 --no-pin-memory"),
        ("RDD Unified L/2", "2d/train_RDD_imagenet256_unified.py",
         "--model DiT-L/2 --resolutions 32,16,8 --stage-lengths 300,300,400 "
         "--global-batch-size 4 --num-workers 0 --no-pin-memory"),
        ("RDD Decouple 300-300-400", "2d/train_RDD_imagenet256_decouple.py",
         "--model-32 DiT-XL/2 --model-16 DiT-L/2 --model-8 DiT-B/2 "
         "--stage-lengths 300,300,400 --global-batch-size 4 --num-workers 0"),
    ]

    results = {}
    for name, script, extra_args in tests:
        results[name] = run_smoke(name, script, extra_args)
        print()

    # Summary
    print("=" * 60)
    passed = sum(results.values())
    total = len(results)
    print(f"Results: {passed}/{total} passed")
    for name, ok in results.items():
        status = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        print(f"  {status}  {name}")

    if passed == total:
        print(f"\n{GREEN}All smoke tests passed!{RESET}")
    else:
        print(f"\n{RED}{total - passed} test(s) failed.{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
