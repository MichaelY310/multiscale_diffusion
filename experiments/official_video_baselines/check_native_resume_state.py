"""Fresh-process CPU audit of native PF AR state and actual input preparation.

No DiT, VAE, text encoder, optimizer, CUDA context or quality evaluation.
The official methods execute unchanged on a synthetic, tiny latent fixture.
This is a state-contract subcheck, NOT a reduced-model checkpoint smoke.
"""
import argparse
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

from pyramid_smoke import PIN, check_source


CASES = {
    "nonaligned": {"max_frames": 16, "sync_group": 8, "prefix_calls": 3, "max_units": 16},
    "aligned": {"max_frames": 16, "sync_group": 8, "prefix_calls": 4, "max_units": 16},
    "clipped_smoke": {"max_frames": 8, "sync_group": 1, "prefix_calls": 3, "max_units": 4},
}


def digest(tensor):
    return {
        "shape": list(tensor.shape), "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest(),
    }


def flatten_tensors(value):
    if isinstance(value, (list, tuple)):
        return [flatten_tensors(item) for item in value]
    return digest(value)


def worker(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU audit requires CUDA_VISIBLE_DEVICES empty")
    import numpy as np
    import torch
    from accelerate import Accelerator
    from accelerate.checkpointing import RNG_STATE_NAME
    from pyramid_runner_state import (
        apply_runner_state, capture_runner_state, register_runner_save_hook,
        restore_runner_state_after_load,
    )

    hashes = check_source(args.source)
    sys.path.insert(0, str(args.source.resolve()))
    from pyramid_dit import PyramidDiTForVideoGeneration
    from diffusion_schedulers import PyramidFlowMatchEulerDiscreteScheduler

    case = CASES[args.case]
    # Construct only the state needed by the official methods. Load no model.
    runner = PyramidDiTForVideoGeneration.__new__(PyramidDiTForVideoGeneration)
    runner.max_temporal_length = case["max_frames"]
    runner.frame_per_unit = 1
    runner.video_sync_group = case["sync_group"]
    runner.stages = [1, 2, 4]
    runner.sample_ratios = [1, 2, 1]
    runner.corrupt_ratio = 1 / 3
    runner.cfg_rate = 0.1
    runner.accumulate_steps = 0
    runner.scheduler = PyramidFlowMatchEulerDiscreteScheduler(
        shift=1.0, stages=3, stage_range=[0, 1 / 3, 2 / 3, 1], gamma=1 / 3,
    )
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    assert accelerator.num_processes == 1
    frames = case["max_units"]
    latent = torch.arange(4 * 2 * frames * 8 * 8, dtype=torch.float32).reshape(4, 2, frames, 8, 8) / 10000
    pyramid = runner.get_pyramid_latent(latent, 2)

    def trace():
        results = []
        for _ in range(4):
            before = runner.accumulate_steps
            # Same dropout draw as the official __call__, followed by its actual
            # temporal noise/history/timestep/target preparation.
            dropout = torch.rand((4,)) <= runner.cfg_rate
            prepared = runner.add_pyramid_noise_with_temporal_pyramid(pyramid, runner.sample_ratios)
            results.append({
                "counter_before": before, "counter_after": runner.accumulate_steps,
                "cfg_dropout": dropout.tolist(), "prepared": flatten_tensors(prepared),
                "python_draw": random.random(), "numpy_draw": float(np.random.rand()),
            })
        return results

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if args.mode == "save":
        for _ in range(case["prefix_calls"]):
            torch.rand((4,))
            runner.add_pyramid_noise_with_temporal_pyramid(pyramid, runner.sample_ratios)
            random.random()
            np.random.rand()
        register_runner_save_hook(accelerator, runner)
        accelerator.save_state(str(args.checkpoint))
        saved_counter = runner.accumulate_steps
        # Exercise rejection paths while verifying the live streams are intact.
        valid = capture_runner_state(runner, accelerator)
        rejected = []
        for key, value in [("schema_version", 2), ("source_commit", "wrong"),
                           ("rank", 1), ("world_size", 2), ("accumulate_steps", -1),
                           ("device_type", "cuda"), ("recipe", {})]:
            invalid = copy.deepcopy(valid)
            invalid[key] = value
            try:
                apply_runner_state(invalid, runner, accelerator)
            except ValueError:
                rejected.append(key)
            else:
                raise AssertionError(f"Accepted incompatible state: {key}")
        assert runner.accumulate_steps == saved_counter
        result = {"saved_counter": saved_counter, "rejected_metadata": rejected, "trace": trace()}
    else:
        random.seed(987)
        np.random.seed(987)
        torch.manual_seed(987)
        accelerator.load_state(str(args.checkpoint))
        rng_load_error = None
        try:
            torch.load(args.checkpoint / f"{RNG_STATE_NAME}_0.pkl", weights_only=True)
        except Exception as exc:
            rng_load_error = {"type": type(exc).__name__, "message": str(exc)}
        if args.mode == "counter_only":
            sidecar = torch.load(args.checkpoint / "pyramid_runner_rank_0.pt", weights_only=True)
            runner.accumulate_steps = sidecar["accumulate_steps"]
        elif args.mode == "sidecar":
            restore_runner_state_after_load(accelerator, runner, args.checkpoint)
        result = {"rng_load_error": rng_load_error, "trace": trace()}
    assert not torch.cuda.is_initialized()
    result.update({"pid": os.getpid(), "mode": args.mode, "case": args.case,
                   "source_hashes": hashes, "cuda_context_initialized": False})
    args.result.write_text(json.dumps(result, indent=2) + "\n")


def compare(reference, actual):
    return {
        "counter_equal": all(a["counter_before"] == b["counter_before"] and a["counter_after"] == b["counter_after"]
                             for a, b in zip(reference, actual)),
        "history_shapes_equal": all([[t["shape"] for t in stage] for stage in a["prepared"][0]] ==
                                    [[t["shape"] for t in stage] for stage in b["prepared"][0]]
                                    for a, b in zip(reference, actual)),
        "dropout_equal": all(a["cfg_dropout"] == b["cfg_dropout"] for a, b in zip(reference, actual)),
        "prepared_tensors_bitwise_equal": all(a["prepared"] == b["prepared"] for a, b in zip(reference, actual)),
        "python_rng_equal": all(a["python_draw"] == b["python_draw"] for a, b in zip(reference, actual)),
        "numpy_rng_equal": all(a["numpy_draw"] == b["numpy_draw"] for a, b in zip(reference, actual)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["save", "bare", "counter_only", "sidecar"])
    parser.add_argument("--case", choices=list(CASES))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.mode:
        worker(args)
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("Set CUDA_VISIBLE_DEVICES= for this CPU-only audit")
    hashes = check_source(args.source)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "source_commit": PIN, "source_hashes": hashes,
              "scope": "native AR input-preparation and counter/CPU RNG recovery only; no model/optimizer checkpoint",
              "device": "CPU", "gpu_jobs_launched": 0, "cases": {}}
    for name, config in CASES.items():
        checkpoint = args.output / f"checkpoint_{name}"
        results = {}
        for mode in ["save", "bare", "counter_only", "sidecar"]:
            result = args.output / f"{name}_{mode}.json"
            command = [sys.executable, str(Path(__file__).resolve()), "--source", str(args.source.resolve()),
                       "--output", str(args.output.resolve()), "--mode", mode, "--case", name,
                       "--checkpoint", str(checkpoint.resolve()), "--result", str(result.resolve())]
            completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, timeout=45, check=False)
            (args.output / f"{name}_{mode}.log").write_text(completed.stdout)
            if completed.returncode:
                raise RuntimeError(f"CPU worker {name}/{mode} failed; see its log")
            results[mode] = json.loads(result.read_text())
        checks = {mode: compare(results["save"]["trace"], results[mode]["trace"])
                  for mode in ["bare", "counter_only", "sidecar"]}
        assert all(checks["sidecar"].values()), checks
        assert not checks["bare"]["counter_equal"]
        assert checks["counter_only"]["counter_equal"]
        assert not checks["counter_only"]["prepared_tensors_bitwise_equal"]
        assert not checks["bare"]["python_rng_equal"]
        assert not checks["bare"]["numpy_rng_equal"]
        if name == "nonaligned":
            assert not checks["bare"]["history_shapes_equal"]
        if name == "aligned":
            assert checks["bare"]["history_shapes_equal"]
        assert len({r["pid"] for r in results.values()}) == 4
        report["cases"][name] = {"config": config, "saved_counter": results["save"]["saved_counter"],
                                 "comparisons": checks, "rejected_metadata": results["save"]["rejected_metadata"],
                                 "fresh_process_pids": {mode: r["pid"] for mode, r in results.items()}}
        print(json.dumps({"case": name, "comparisons": checks}), flush=True)
    now = dt.datetime.now(dt.timezone.utc)
    free = shutil.disk_usage(args.output).free / 2**30
    report.update({"status": "passed", "utc": now.isoformat(),
                   "la": now.astimezone(ZoneInfo("America/Los_Angeles")).isoformat(),
                   "source_hashes_after": check_source(args.source),
                   "cuda_context_initialized": False,
                   "disk_free_gib": free, "smoke_reserve_gib": 80,
                   "full_native_checkpoint_min_gib": 1972059200 * 12 / 2**30,
                   "checkpoint_files_bytes": sum(p.stat().st_size for p in args.output.glob("checkpoint_*/*") if p.is_file()),
                   "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                       [Path(__file__), Path(__file__).with_name("pyramid_runner_state.py")]},
                   "remaining": ["CUDA RNG", "multi-rank/FSDP", "full native model/AdamW/EMA save and update parity",
                                 "data sampler/bucket cursor and worker RNG", "LR/global-step and accumulation position",
                                 "real caches and empty-prompt embeddings", "decoded sampling/quality", "FastVideo native bridge"]})
    assert report["source_hashes_after"] == hashes
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "summary": str(args.output / "summary.json")}), flush=True)


if __name__ == "__main__":
    main()
