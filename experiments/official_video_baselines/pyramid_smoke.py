"""Short plumbing test of an unchanged official Pyramid Flow training entrypoint.

Synthetic cached features are NOT a real dataset or a quality evaluation.
External cache initialization/device placement, telemetry and checkpoint
suppression differ; the official objective, schedule and AR code are unchanged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PIN = "a012faa1dc4d71301a7a153c7f9554c081947ea2"


def check_source(source):
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True)
    if head != PIN or dirty:
        raise RuntimeError("Official source must match the pinned commit with no tracked modifications")
    return {p: hashlib.sha256((source / p).read_bytes()).hexdigest() for p in (
        "train/train_pyramid_flow.py", "trainer_misc/fsdp_trainer.py",
        "pyramid_dit/pyramid_dit_for_video_gen_pipeline.py",
        "pyramid_dit/flux_modules/modeling_pyramid_flux.py")}


def fixture(root, frames):
    import torch
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    rows = []
    for i in range(8):
        latent, text = root / f"latent_{i}.pt", root / f"text_{i}.pt"
        torch.save(torch.randn(1, 16, frames, 48, 80) * .1, latent)
        torch.save({"prompt_embed": torch.randn(1, 16, 4096) * .01,
                    "prompt_attention_mask": torch.ones(1, 16, dtype=torch.long),
                    "pooled_prompt_embed": torch.randn(1, 768) * .01}, text)
        rows.append({"text": "synthetic plumbing fixture", "latent": str(latent.resolve()),
                     "text_fea": str(text.resolve())})
    torch.save({"prompt_embed": torch.zeros(1, 16, 4096),
                "prompt_attention_mask": torch.ones(1, 16, dtype=torch.long),
                "pooled_prompt_embed": torch.zeros(1, 768)}, root / "synthetic_null.pt")
    (root / "annotation.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--fixture-only", action="store_true")
    parser.add_argument("--temporal-pyramid", action="store_true")
    parser.add_argument("--attention-backend", choices=["sdpa", "flash"], default="sdpa")
    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument("--video-sync-group", type=int, default=1)
    parser.add_argument("--sync-video-input", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    hashes = check_source(source)
    if args.fixture_only:
        fixture(args.output.resolve(), args.frames)
        print(json.dumps({"fixture": str(args.output), "source_commit": PIN, "hashes": hashes}))
        return
    if not args.model:
        parser.error("--model is required for a GPU smoke")
    args.model = args.model.resolve()
    if not (args.model / "diffusion_transformer_384p/config.json").is_file():
        raise RuntimeError("Local official checkpoint config is missing")
    import torch
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    caches = args.output.parent / "fixtures"
    if not (caches / "annotation.jsonl").exists():
        raise RuntimeError("Generate CPU fixtures before starting the GPU deadline")
    sys.path.insert(0, str(source))
    os.chdir(source)
    os.environ["FSDP_USE_ORIG_PARAMS"] = "true"
    from train import train_pyramid_flow as official
    from accelerate import Accelerator
    telemetry = {"source_commit": PIN, "source_hashes": hashes, "fixture": "synthetic",
                 "null_text": "synthetic zeros; not real empty-prompt embeddings",
                 "optimizer_steps": 0, "temporal_pyramid": args.temporal_pyramid,
                 "attention_backend": args.attention_backend, "fsdp": args.fsdp,
                 "checkpoint_saving": "suppressed only for storage-limited smoke"}
    original_builder = official.build_model_runner
    original_optimizer = official.create_optimizer
    original_epoch = official.train_one_epoch_with_fsdp

    def build(opts):
        runner = original_builder(opts)
        runner.null_text_embeds = torch.load(caches / "synthetic_null.pt", map_location="cpu")
        # Official trainer moves video but leaves cached text on the CPU.
        # Device placement only; preserve official dropout and embedding values.
        original_text = runner.get_text_embeddings
        def text_embeddings(text, rand_idx, device):
            text = {key: value.to(device) for key, value in text.items()}
            return original_text(text, rand_idx, device)
        runner.get_text_embeddings = text_embeddings
        telemetry["compatibility_changes"] = ["cached text tensors moved to training device"]
        telemetry["parameters"] = sum(p.numel() for p in runner.dit.parameters())
        original_latent = runner.get_vae_latent
        def latent(*pos, **kw):
            result = original_latent(*pos, **kw)
            telemetry["stage_inputs"] = [[list(x.shape) for x in stage] for stage in result[0]]
            telemetry["stage_targets"] = [list(x.shape) for x in result[3]]
            return result
        runner.get_vae_latent = latent
        original_lengths = runner.sample_stage_length
        def lengths(*pos, **kw):
            result = original_lengths(*pos, **kw)
            telemetry["ar_stage_units"] = result
            return result
        runner.sample_stage_length = lengths
        return runner

    def optimizer(opts, model):
        opt = original_optimizer(opts, model)
        snapshot = {}
        def before_step(*unused):
            # FSDP original parameters are local shards outside forward.
            probe = next(p for group in opt.param_groups for p in group["params"]
                         if p.requires_grad and p.numel() and p.grad is not None)
            snapshot["probe"] = probe
            snapshot["before"] = probe.detach().flatten()[:1024].cpu().clone()
        def after_step(*unused):
            telemetry["optimizer_steps"] += 1
            telemetry["probe_max_abs_update"] = float((snapshot["probe"].detach().flatten()[:1024].cpu() - snapshot["before"]).abs().max())
        opt.register_step_pre_hook(before_step)
        opt.register_step_post_hook(after_step)
        return opt

    def epoch(*pos, **kw):
        telemetry["distributed_type"] = str(pos[2].distributed_type)
        telemetry["world_size"] = pos[2].num_processes
        telemetry["prepared_model_type"] = type(pos[0].dit).__name__
        stats = original_epoch(*pos, **kw)
        telemetry["train_stats"] = stats
        return stats

    def no_checkpoint(*pos, **kw):
        print("Smoke only: skip large checkpoint serialization", flush=True)

    official.build_model_runner = build
    official.create_optimizer = optimizer
    official.train_one_epoch_with_fsdp = epoch
    Accelerator.save_state = no_checkpoint
    sys.argv = ["official", "--task", "t2v", "--model_path", str(args.model.resolve()),
                "--model_name", "pyramid_flux", "--model_variant", "diffusion_transformer_384p",
                "--model_dtype", "bf16", "--batch_size", "4", "--max_frames", str(args.frames),
                "--resolution", "384p", "--anno_file", str(caches / "annotation.jsonl"),
                "--num_workers", "0", "--epochs", "1", "--iters_per_epoch", "1",
                "--warmup_steps", "1", "--lr", "1e-4", "--opt_beta2", ".95",
                "--clip_grad", "1", "--seed", "42", "--gradient_checkpointing",
                "--gradient_checkpointing_ratio", "1",
                "--output_dir", str(args.output), "--print_freq", "1"]
    if args.attention_backend == "flash":
        sys.argv += ["--use_flash_attn"]
    if args.fsdp:
        sys.argv += ["--use_fsdp", "--fsdp_shard_strategy", "zero3"]
    if args.temporal_pyramid:
        sys.argv += ["--use_temporal_pyramid", "--video_sync_group", str(args.video_sync_group)]
    if args.sync_video_input:
        sys.argv += ["--sync_video_input"]
    start = time.monotonic()
    try:
        official.main(official.get_args())
        telemetry["status"] = "passed" if telemetry["optimizer_steps"] > 0 else "no_update"
    except BaseException as exc:
        telemetry.update(status="failed", error=repr(exc))
        raise
    finally:
        telemetry["seconds"] = time.monotonic() - start
        telemetry["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        telemetry["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
        (args.output / f"metrics_rank{os.getenv('RANK', '0')}.json").write_text(json.dumps(telemetry, indent=2))
        check_source(source)


if __name__ == "__main__":
    main()
