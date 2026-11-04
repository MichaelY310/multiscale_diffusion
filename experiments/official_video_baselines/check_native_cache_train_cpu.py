"""Combine actual native cached dataset/buckets, tiny DiT and FastVideo DCP."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def worker(args):
    from check_fastvideo_native_pyramid import prepare, seed, trace, hashed, make_trainer
    args.case = "accum2"
    compatibility, cfg, method = prepare(args)
    import torch
    from dataset import LengthGroupedVideoTextDataset
    from pyramid_data_state import build_stateful_native_loader
    from pyramid_dcp_state import NativeDCPDataLoader
    from fastvideo_native_checkpoint import NativePyramidCheckpointManager
    from fastvideo.train.utils.checkpoint import CheckpointConfig
    fixture = args.fixture
    manifest = json.loads((fixture / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((fixture / name).read_bytes()).hexdigest() == expected
    dataset = LengthGroupedVideoTextDataset(str(fixture / "annotation.jsonl"), max_frames=5)
    loader = NativeDCPDataLoader(build_stateful_native_loader(dataset,
        batch_size=4, max_frames=5, num_workers=args.workers, rank=0, world_size=1,
        epoch=42, dataset_fingerprint=manifest["fingerprint"]))
    # These are explicitly synthetic cached embeddings, matching this tiny
    # architecture. Latents retain the official 48x80 cache shape: no crop/resize.
    method.student.runner.null_text_embeds = torch.load(fixture / "text.pt", weights_only=True)
    method.student.dataloader = loader
    trainer = make_trainer(cfg)
    cfg.training.checkpoint.output_dir = str(args.scratch / ("resume_runs" if args.mode == "resume" else "runs"))
    resume = str(args.scratch / "runs/checkpoint-2") if args.mode == "resume" else ""
    cfg.training.checkpoint.resume_from_checkpoint = resume
    manager = NativePyramidCheckpointManager(method=method, dataloader=loader,
        output_dir=cfg.training.checkpoint.output_dir, config=CheckpointConfig(save_steps=2, keep_last=0),
        callbacks=trainer.callbacks, raw_config=cfg.raw)
    record = trace(method.student.runner, method.optimizer)
    seed()
    try:
        trainer.run(method, dataloader=loader, max_steps=3, checkpoint_manager=manager)
        # Only the next outer update after checkpoint-2 is compared.
        continuation = {key: value[-(2 if key in ("inputs", "predictions", "losses") else 1):]
                        for key, value in record.items()}
        result = {"continuation": continuation,
            "weights": hashed(method.student.transformer.state_dict()),
            "optimizer": hashed(method.optimizer.state_dict()),
            "ema": hashed(trainer.callbacks.state_dict()), "control": method.control_state.state_dict(),
            "data_counter": loader.emitted_batches, "bucket_epoch": loader._epoch,
            "compatibility": compatibility, "parameters": 254208, "workers": args.workers,
            "synthetic": True, "full_pretrained_model": False, "cuda_initialized": torch.cuda.is_initialized()}
        assert not result["cuda_initialized"]
        args.result.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        loader.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--fastvideo", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--workers", type=int, choices=[0, 2], default=0)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["continuous", "resume"])
    p.add_argument("--scratch", type=Path)
    p.add_argument("--result", type=Path)
    args = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU audit requires CUDA_VISIBLE_DEVICES empty")
    if args.mode:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    scratch = Path(tempfile.mkdtemp(prefix="rdd-native-cache-train-cpu-"))
    results = {}
    for mode in ("continuous", "resume"):
        cmd = [sys.executable, str(Path(__file__).resolve()), "--source", str(args.source),
            "--fastvideo", str(args.fastvideo), "--fixture", str(args.fixture), "--workers", str(args.workers),
            "--output", str(args.output), "--mode", mode, "--scratch", str(scratch),
            "--result", str(args.output / f"{mode}.json")]
        with (args.output / f"{mode}.log").open("w") as log:
            done = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=90)
        if done.returncode:
            raise RuntimeError(f"Native cached tiny-model CPU worker failed: {mode}")
        results[mode] = json.loads((args.output / f"{mode}.json").read_text())
        print(f"passed {mode} workers={args.workers}", flush=True)
    keys = ("continuation", "weights", "optimizer", "ema", "control", "data_counter", "bucket_epoch")
    checks = {key: results["continuous"][key] == results["resume"][key] for key in keys}
    summary = {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
               "workers": args.workers, "scratch": str(scratch), "component_only": True,
               "gpu_used": False, "full_pretrained_checkpoint": False,
               "native_cached_shape": [16, "1/3/5", 48, 80],
               "intentional_invalid_cache_rows": manifest_rows(args.fixture),
               "checkpoint_manager": "native RNG hooks on actual FastVideo CheckpointManager"}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    assert all(checks.values()), checks
    print(json.dumps(summary))


def manifest_rows(fixture):
    return json.loads((fixture / "manifest.json").read_text())["intentional_invalid_cache_rows"]


if __name__ == "__main__":
    main()
