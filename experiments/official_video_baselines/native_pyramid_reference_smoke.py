"""Run the author's fixed training main with the same prepared native smoke YAML.

Compatibility only: install provided empty-prompt cache, cached-text device move,
telemetry and explicit checkpoint suppression for this <=3-update smoke. This
does not modify official tracked files or replace its objective/training loop.
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import socket
import sys
import time

import yaml

from launch_smoke import bounded_deadline
from prepare_fair_suite import ROOT
from pyramid_smoke import check_source


def native_argv(cfg):
    tc, student, method = cfg["training"], cfg["models"]["student"], cfg["method"]
    recipe = student["recipe"]
    oc = tc["optimizer"]
    args = ["official", "--task", "t2v", "--model_path", student["init_from"],
        "--model_name", recipe["model_name"], "--model_variant", recipe["model_variant"],
        "--model_dtype", recipe["model_dtype"], "--batch_size", str(tc["data"]["train_batch_size"]),
        "--max_frames", str(recipe["max_temporal_length"]), "--frame_per_unit", str(recipe["frame_per_unit"]),
        "--video_sync_group", str(recipe["video_sync_group"]), "--resolution", "384p",
        "--anno_file", student["annotation"], "--num_workers", str(tc["data"]["dataloader_num_workers"]),
        "--epochs", "1", "--iters_per_epoch", str(tc["loop"]["max_train_steps"]),
        "--gradient_accumulation_steps", str(tc["loop"]["gradient_accumulation_steps"]),
        "--seed", str(tc["data"]["seed"]), "--lr", str(oc["learning_rate"]),
        "--weight_decay", str(oc["weight_decay"]), "--opt_beta1", str(oc["betas"][0]),
        "--opt_beta2", str(oc["betas"][1]), "--warmup_steps", str(oc["lr_warmup_steps"]),
        "--clip_grad", str(method["clip_grad"]), "--output_dir", tc["checkpoint"]["output_dir"],
        "--print_freq", "100000"]
    if method["use_temporal_pyramid"]:
        args.append("--use_temporal_pyramid")
    if recipe["use_gradient_checkpointing"]:
        args += ["--gradient_checkpointing", "--gradient_checkpointing_ratio", str(recipe["gradient_checkpointing_ratio"])]
    return args


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--trace-file", type=Path)
    p.add_argument("--deterministic", action="store_true")
    args = p.parse_args()
    if socket.gethostname().split(".")[0] != "surfi1" or Path.cwd().resolve() != ROOT:
        p.error("GPU reference requires surfi1 and the official runtime working directory")
    remaining = float(os.environ.get("RDD_DEADLINE_UNIX", "0")) - time.time()
    if not os.environ.get("RDD_JOB_TOKEN") or remaining <= 0:
        p.error("independent deadline guard required")
    bounded_deadline(dt.datetime.now(dt.timezone.utc), remaining)
    cfg = yaml.safe_load(args.config.read_text())
    tc, student = cfg["training"], cfg["models"]["student"]
    if tc["loop"]["max_train_steps"] > 3 or tc["checkpoint"]["training_state_checkpointing_steps"]:
        raise ValueError("This reference smoke requires <=3 updates and explicit no-checkpoint policy")
    output = Path(tc["checkpoint"]["output_dir"])
    if list(output.glob("checkpoint-*")):
        raise ValueError("Official main auto-resumes; use a fresh reference output directory")
    # The author main creates output_dir on rank zero without a barrier,
    # while all ranks immediately list it during auto_resume.
    output.mkdir(parents=True,exist_ok=True)
    source = Path(student["source"])
    hashes = check_source(source)
    sys.path.insert(0, str(source))
    import torch
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    from accelerate import Accelerator
    from train import train_pyramid_flow as official
    original_builder, original_optimizer = official.build_model_runner, official.create_optimizer
    original_save = Accelerator.save_state
    updates = [0]
    runners = []
    traces = []
    def build(opts):
        runner = original_builder(opts)
        runner.null_text_embeds = torch.load(student["empty_prompt"], map_location="cpu", weights_only=True)
        original_text = runner.get_text_embeddings
        def text(text, dropout, device):
            return original_text({key: value.to(device) for key, value in text.items()}, dropout, device)
        runner.get_text_embeddings = text
        runners.append(runner)
        return runner
    def optimizer(opts, model):
        result = original_optimizer(opts, model)
        def updated(*unused):
            updates[0] += 1
        result.register_step_post_hook(updated)
        if args.trace_file:
            from full_model_trace import trace_native
            traces.append(trace_native(runners[-1], result))
        return result
    def skip_checkpoint(*unused, **kwargs):
        print("Explicit <=3-update reference smoke: checkpoint writing disabled", flush=True)
    official.build_model_runner, official.create_optimizer = build, optimizer
    Accelerator.save_state = skip_checkpoint
    try:
        sys.argv = native_argv(cfg)
        official.main(official.get_args())
        assert updates[0] == tc["loop"]["max_train_steps"]
        report = {"status": "passed", "source_hashes": hashes, "engine": "author_native_main",
                  "optimizer_updates": updates[0], "checkpoint_written": False,
                  "decoded_sample": False, "full_model_fastvideo_parity": False,
                  "cache_fingerprint": student["cache_fingerprint"],
                  "compatibility": ["provided empty prompt", "cached text device placement", "explicit smoke checkpoint suppression"]}
        output.mkdir(parents=True, exist_ok=True)
        (output / f"reference_{os.environ['RDD_JOB_TOKEN']}_rank{os.environ.get('RANK', '0')}.json").write_text(json.dumps(report, indent=2))
        if args.trace_file:
            trace_path=args.trace_file
            if int(os.environ.get('WORLD_SIZE','1'))>1:
                trace_path=trace_path.with_name(trace_path.stem+'_rank'+os.environ['RANK']+trace_path.suffix)
            trace_path.write_text(json.dumps(traces[0]))
    finally:
        Accelerator.save_state = original_save
        check_source(source)


if __name__ == "__main__":
    main()
