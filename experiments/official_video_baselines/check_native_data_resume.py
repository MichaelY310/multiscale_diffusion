"""CPU native PF data-cursor recovery, fresh-process positive/negative checks.

Actual official cached dataset and temporal bucketeer, small synthetic caches.
No DiT/optimizer, GPU, training, quality evaluation or dataset download.
"""
import argparse
import copy
import datetime as dt
import faulthandler
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch.utils.data import Dataset

from pyramid_smoke import PIN, check_source
from check_native_resume_state import flatten_tensors, digest

if "--mode" in sys.argv:
    faulthandler.dump_traceback_later(30, repeat=True)


PREFIX = 2
CONTINUATION = 8
MODES = ["native", "save", "full", "rng_only", "skip_count", "no_buckets", "no_worker_rng"]


class ProbeDataset(Dataset):
    """Diagnostic fields only; official cached video/text values unchanged."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        result = self.dataset[index]
        info = torch.utils.data.get_worker_info()
        return {**result, "row_id": torch.tensor([index]),
                "worker_draws": torch.tensor([[random.random(), np.random.rand(), torch.rand(()).item()]], dtype=torch.float64),
                "worker_id": torch.tensor([info.id if info is not None else -1]),
                "worker_cuda": torch.tensor([torch.cuda.is_initialized()])}


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def fixture(directory):
    directory.mkdir()
    text = directory / "text.pt"
    torch.save({"prompt_embed": torch.arange(16, dtype=torch.float32).reshape(1, 2, 8) / 100,
                "prompt_attention_mask": torch.ones(1, 2, dtype=torch.long),
                "pooled_prompt_embed": torch.zeros(1, 4)}, text)
    rows = []
    hashes = {text.name: hashlib.sha256(text.read_bytes()).hexdigest()}
    for index in range(24):
        frames = [1, 3, 5][index % 3]
        path = directory / f"latent_{index}.pt"
        # One known-invalid synthetic cache exercises the official random
        # retry path. It is not an accidental or silently accepted data error.
        width = 79 if index == 11 else 80
        latent = torch.full((1, 16, frames, 48, width), (index + 1) / 100)
        torch.save(latent, path)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        per_row_text = directory / f"text_{index}.pt"
        torch.save({"prompt_embed": torch.full((1, 2, 8), (index + 1) / 100),
                    "prompt_attention_mask": torch.ones(1, 2, dtype=torch.long),
                    "pooled_prompt_embed": torch.full((1, 4), (index + 1) / 100)}, per_row_text)
        hashes[per_row_text.name] = hashlib.sha256(per_row_text.read_bytes()).hexdigest()
        rows.append({"latent": str(path), "text_fea": str(per_row_text), "text": "synthetic cached cursor fixture"})
    annotation = directory / "annotation.jsonl"
    annotation.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    hashes[annotation.name] = hashlib.sha256(annotation.read_bytes()).hexdigest()
    manifest = {"files": hashes, "samples": len(rows), "synthetic": True,
                "intentional_invalid_cache_rows": [11],
                "invalid_reason": "width79 versus official width80; exercises unchanged random retry"}
    manifest["fingerprint"] = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def remove_dataset_rng(value):
    if isinstance(value, dict):
        for key in value:
            if key == "dataset_state":
                value[key] = None
            else:
                remove_dataset_rng(value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            remove_dataset_rng(item)


def worker(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be empty")
    print(json.dumps({"phase": "worker_begin", "mode": args.mode, "workers": args.workers}), flush=True)
    source_hashes = check_source(args.source)
    sys.path.insert(0, str(args.source.resolve()))
    from dataset import LengthGroupedVideoTextDataset
    from dataset.bucket_loader import TemporalLengthBucketeer
    from dataset.dataloaders import identity
    from pyramid_data_state import build_stateful_native_loader
    from pyramid_runner_state import capture_runner_state, apply_runner_state
    from pyramid_dit import PyramidDiTForVideoGeneration
    from diffusion_schedulers import PyramidFlowMatchEulerDiscreteScheduler
    from accelerate import Accelerator
    from torch.utils.data import DataLoader, DistributedSampler

    manifest = json.loads((args.fixture / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((args.fixture / name).read_bytes()).hexdigest() == expected
    seed(42)
    dataset = ProbeDataset(LengthGroupedVideoTextDataset(str(args.fixture / "annotation.jsonl"), max_frames=5))
    if args.mode == "native":
        sampler = DistributedSampler(dataset, shuffle=True, num_replicas=1, rank=0, seed=42)
        kwargs = {"multiprocessing_context": "spawn"} if args.workers else {}
        reader = DataLoader(dataset, batch_size=4, num_workers=args.workers, sampler=sampler,
                            pin_memory=True, collate_fn=identity, drop_last=True, **kwargs)
        loader = TemporalLengthBucketeer(reader, max_frames=5, epoch=42)
    else:
        loader = build_stateful_native_loader(dataset, batch_size=4, max_frames=5, num_workers=args.workers,
                                             rank=0, world_size=1, epoch=42, dataset_fingerprint=manifest["fingerprint"])
    runner = PyramidDiTForVideoGeneration.__new__(PyramidDiTForVideoGeneration)
    runner.max_temporal_length = 16
    runner.frame_per_unit = runner.video_sync_group = 1
    runner.stages = [1, 2, 4]
    runner.sample_ratios = [1, 2, 1]
    runner.corrupt_ratio = 1 / 3
    runner.cfg_rate = 0.1
    runner.accumulate_steps = 0
    runner.scheduler = PyramidFlowMatchEulerDiscreteScheduler(
        shift=1.0, stages=3, stage_range=[0, 1 / 3, 2 / 3, 1], gamma=1 / 3)
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    print(json.dumps({"phase": "loader_ready", "mode": args.mode, "workers": args.workers}), flush=True)

    def step():
        print(json.dumps({"phase": "next_batch", "counter": runner.accumulate_steps}), flush=True)
        batch = next(loader)
        assert not bool(batch["worker_cuda"].any())
        dropout = torch.rand((4,)) <= runner.cfg_rate
        prepared = runner.add_pyramid_noise_with_temporal_pyramid(runner.get_pyramid_latent(batch["video"], 2), runner.sample_ratios)
        return {
            "row_ids": batch["row_id"].tolist(), "video": digest(batch["video"]),
            "text": {key: digest(value) for key, value in batch["text"].items()},
            "worker_draws": batch["worker_draws"].tolist(), "dropout": dropout.tolist(),
            "worker_ids": batch["worker_id"].tolist(), "worker_cuda_initialized": batch["worker_cuda"].tolist(),
            "prepared": flatten_tensors(prepared), "ar_counter": runner.accumulate_steps,
            "bucket_epoch": loader._epoch, "sampler_epoch": loader._dataloader.sampler.epoch,
            "main_python": random.random(), "main_numpy": float(np.random.rand()),
        }

    try:
        result = {"mode": args.mode, "workers": args.workers, "pid": os.getpid()}
        if args.mode in ("native", "save"):
            result["prefix"] = [step() for _ in range(PREFIX)]
            if args.mode == "save":
                data = loader.state_dict()
                result["saved_state"] = {"bucket_epoch": data["bucket_epoch"], "sampler_epoch": data["sampler_epoch"],
                    "emitted_batches": data["emitted_batches"], "bucket_sizes": {k: len(v) for k, v in data["buckets"].items()}}
                assert sum(result["saved_state"]["bucket_sizes"].values()) > 0
                torch.save({"data": data, "runner": capture_runner_state(runner, accelerator)}, args.checkpoint)
                result["rejected_metadata"] = []
                for key, replacement in [("source_commit", "wrong"), ("schema_version", 2), ("config", {}), ("emitted_batches", -1)]:
                    invalid = dict(data)
                    invalid[key] = replacement
                    try:
                        loader.load_state_dict(invalid)
                    except ValueError:
                        result["rejected_metadata"].append(key)
                    else:
                        raise AssertionError(f"Accepted invalid {key}")
        else:
            saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            seed(987)
            if args.mode in ("full", "no_buckets", "no_worker_rng"):
                data = saved["data"]
                if args.mode == "no_buckets":
                    data["buckets"] = {key: [] for key in data["buckets"]}
                if args.mode == "no_worker_rng":
                    remove_dataset_rng(data["reader"])
                loader.load_state_dict(data)
            apply_runner_state(saved["runner"], runner, accelerator)
            if args.mode == "skip_count":
                for _ in range(PREFIX):
                    next(loader)
                # Re-restore even the main RNG to isolate cursor insufficiency.
                apply_runner_state(saved["runner"], runner, accelerator)
        result["trace"] = [step() for _ in range(CONTINUATION)]
        assert not torch.cuda.is_initialized()
        result.update({"cuda_context_initialized": False, "source_hashes": source_hashes})
        args.result.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        faulthandler.cancel_dump_traceback_later()
        if hasattr(loader, "close"):
            loader.close()
        elif hasattr(loader.iterator, "_shutdown_workers"):
            loader.iterator._shutdown_workers()


def comparison(reference, actual):
    return {key: all(a[key] == b[key] for a, b in zip(reference, actual)) for key in reference[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--workers", type=int, choices=[0, 2])
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--aggregate-existing", action="store_true", help="Validate existing completed worker evidence without rerunning it")
    args = parser.parse_args()
    if args.mode:
        worker(args)
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("Set CUDA_VISIBLE_DEVICES= for CPU audit")
    before = check_source(args.source)
    if args.aggregate_existing:
        report = json.loads((args.output / "summary.json").read_text())
        scratch = Path(report["scratch_directory"])
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        scratch = Path(tempfile.mkdtemp(prefix="rdd-native-data-cursor-20261005-", dir="/tmp"))
        manifest = fixture(scratch / "fixture")
        (args.output / "fixture_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        report = {"status": "running", "source_commit": PIN, "source_hashes": before,
                  "scope": "native cached-data / AR input-preparation recovery only; no model or optimizer checkpoint",
                  "scratch_directory": str(scratch), "gpu_jobs_launched": 0, "cases": {}}
    for workers in [0, 2]:
        checkpoint = scratch / f"data_workers{workers}.pt"
        results = {}
        for mode in MODES:
            # Main-thread RNG is restored after the reader. In workers=0 the
            # same dataset RNG lives in that main stream, so this ablation is
            # meaningful only for actual multiprocessing workers.
            if mode == "no_worker_rng" and workers == 0:
                continue
            result = args.output / f"workers{workers}_{mode}.json"
            if args.aggregate_existing:
                results[mode] = json.loads(result.read_text())
                continue
            command = [sys.executable, str(Path(__file__).resolve()), "--source", str(args.source.resolve()),
                       "--output", str(args.output.resolve()), "--mode", mode, "--workers", str(workers),
                       "--fixture", str(scratch / "fixture"), "--checkpoint", str(checkpoint), "--result", str(result.resolve())]
            try:
                completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=90)
            except subprocess.TimeoutExpired as exc:
                output = exc.stdout or ""
                if isinstance(output, bytes):
                    output = output.decode(errors="replace")
                (args.output / f"workers{workers}_{mode}.log").write_text(output)
                report.update({"status": "failed", "failure": {"workers": workers, "mode": mode, "reason": "CPU worker deadline90s"}})
                (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
                raise
            (args.output / f"workers{workers}_{mode}.log").write_text(completed.stdout)
            if completed.returncode:
                raise RuntimeError(f"CPU worker workers={workers} mode={mode} failed; see log")
            results[mode] = json.loads(result.read_text())
            print(json.dumps({"workers": workers, "mode": mode, "completed": True}), flush=True)
        for value in results.values():
            assert len(value["trace"]) == CONTINUATION
            assert value["source_hashes"] == before and not value["cuda_context_initialized"]
        checks = {mode: comparison(results["save"]["trace"], results[mode]["trace"])
                  for mode in results if mode != "save"}
        assert results["native"]["prefix"] == results["save"]["prefix"]
        assert all(checks["native"].values()), checks
        assert all(checks["full"].values()), checks
        for mode in ["rng_only", "no_buckets"]:
            # A matching requested index list alone is not recovery evidence.
            assert not all(checks[mode].values()), (mode, checks)
        if workers:
            assert not checks["no_worker_rng"]["worker_draws"], checks
        worker_ids = {identity for step in results["save"]["prefix"] + results["full"]["trace"] for identity in step["worker_ids"]}
        assert worker_ids == ({0, 1} if workers else {-1})
        assert max(item["bucket_epoch"] for item in results["full"]["trace"]) > results["save"]["saved_state"]["bucket_epoch"]
        pids = {mode: value["pid"] for mode, value in results.items()}
        assert len(set(pids.values())) == len(pids)
        report["cases"][str(workers)] = {"comparisons": checks, "saved_state": results["save"]["saved_state"],
            "rejected_metadata": results["save"]["rejected_metadata"], "fresh_process_pids": pids,
            "worker_ids_seen": sorted(worker_ids), "all_worker_cuda_contexts_false": True,
            "count_replay_matches_this_fixture": all(checks["skip_count"].values()),
            "checkpoint_bytes": checkpoint.stat().st_size, "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    now = dt.datetime.now(dt.timezone.utc)
    data_hashes = {name: hashlib.sha256((args.source / name).read_bytes()).hexdigest()
                   for name in ["dataset/dataset_cls.py", "dataset/dataloaders.py", "dataset/bucket_loader.py"]}
    report.update({"status": "passed", "utc": now.isoformat(),
        "la": now.astimezone(ZoneInfo("America/Los_Angeles")).isoformat(), "source_hashes_after": check_source(args.source),
        "data_source_hashes": data_hashes, "cuda_context_initialized": False,
        "packages": {name: md.version(name) for name in ["torch", "torchdata", "accelerate"]},
        "prefix_batches": PREFIX, "continuation_batches": CONTINUATION,
        "disk_free_gib": shutil.disk_usage(args.output).free / 2**30, "smoke_reserve_gib": 80,
        "scratch_bytes": sum(path.stat().st_size for path in scratch.rglob("*") if path.is_file()),
        "code_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in [Path(__file__), Path(__file__).with_name("pyramid_data_state.py")]},
        "worker_execution_driver_sha256": hashlib.sha256((args.output / "check_native_data_resume_v4_executed.py").read_bytes()).hexdigest()
            if (args.output / "check_native_data_resume_v4_executed.py").exists() else hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "aggregation_reuses_completed_worker_evidence": args.aggregate_existing,
        "remaining": ["official main integration", "LR/global step and gradient-accumulation recovery", "EMA cadence",
                      "full native DiT/AdamW checkpoint", "multi-rank/SP/FSDP and CUDA RNG", "real native caches",
                      "FastVideo model/method/callback parity", "decoded quality evaluation"]})
    assert report["source_hashes_after"] == before and not torch.cuda.is_initialized()
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "passed", "summary": str(args.output / "summary.json")}), flush=True)


if __name__ == "__main__":
    main()
