"""Actual native tiny-DiT/FastVideo CPU component parity; never full-model smoke."""
import argparse
import copy
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile


def hashed(value):
    import torch
    if isinstance(value, torch.Tensor):
        t = value.detach().cpu().contiguous()
        return {"shape": list(t.shape), "dtype": str(t.dtype),
                "sha256": hashlib.sha256(t.numpy().tobytes()).hexdigest()}
    if isinstance(value, dict):
        return {str(k): hashed(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [hashed(v) for v in value]
    return value


class FixtureStream:
    """Explicit synthetic 8x8 component inputs, independent of native cache tests."""
    def __init__(self, amplitude=0.1):
        import torch
        self.index = 0
        self.amplitude = amplitude
        g = torch.Generator().manual_seed(111)
        self.rows = [{"video": torch.randn(4, 16, 4, 8, 8, generator=g) * amplitude,
                      "text": {"prompt_embeds": torch.randn(4, 3, 8, generator=g),
                               "prompt_attention_mask": torch.ones(4, 3, dtype=torch.long),
                               "pooled_prompt_embeds": torch.randn(4, 4, generator=g)},
                      "identifier": ["video"] * 4} for _ in range(8)]

    def __iter__(self):
        return self

    def __next__(self):
        row = copy.deepcopy(self.rows[self.index % len(self.rows)])
        self.index += 1
        return row

    def state_dict(self):
        return {"index": self.index, "amplitude": self.amplitude}

    def load_state_dict(self, state):
        if state["amplitude"] != self.amplitude:
            raise ValueError("Fixture amplitude changed")
        self.index = state["index"]


def seed():
    import torch
    import numpy as np
    random.seed(20261005)
    np.random.seed(20261005)
    torch.manual_seed(20261005)


def trace(runner, optimizer):
    import torch
    result = {"inputs": [], "predictions": [], "losses": [], "gradients": [], "updates": []}
    def before(module, unused, kwargs):
        result["inputs"].append(hashed(kwargs))
    runner.dit.register_forward_pre_hook(before, with_kwargs=True)
    runner.dit.register_forward_hook(lambda module, unused, out: result["predictions"].append(hashed(out)))
    loss_impl = runner.calculate_loss
    def loss(pred, targets):
        out = loss_impl(pred, targets)
        result["losses"].append({"loss": hashed(out[0]), "targets": hashed(targets)})
        return out
    runner.calculate_loss = loss
    raw_opt = getattr(optimizer, "optimizer", optimizer)
    def before_step(*unused):
        result["gradients"].append(hashed({k: p.grad for k, p in runner.dit.named_parameters()}))
    def after_step(*unused):
        result["updates"].append({"weights": hashed(runner.dit.state_dict()),
                                   "adamw": hashed(optimizer.state_dict())})
    raw_opt.register_step_pre_hook(before_step)
    raw_opt.register_step_post_hook(after_step)
    return result


class Tracker:
    def __init__(self):
        self.metrics = []
    def log(self, metrics, iteration):
        self.metrics.append({"step": iteration, "metrics": metrics})
    def finish(self):
        pass


def make_trainer(cfg):
    from fastvideo.train.trainer import Trainer
    from fastvideo.train.callbacks.callback import CallbackDict
    # Distributed CUDA group/tracker construction is intentionally excluded.
    # The real, unmodified Trainer.run and CallbackDict execute on CPU.
    trainer = Trainer.__new__(Trainer)
    trainer.training_config = cfg.training
    trainer.global_rank = trainer.local_rank = 0
    trainer.tracker = Tracker()
    trainer.callbacks = CallbackDict(cfg.callbacks, cfg.training)
    return trainer


def prepare(args):
    from fastvideo_cpu_import import bootstrap
    compatibility = bootstrap(args.fastvideo)
    import torch
    import numpy as np
    from fastvideo.train.utils.config import load_run_config
    from fastvideo.train.utils.builder import build_from_config
    from pyramid_smoke import check_source
    check_source(args.source)
    sys.path.insert(0, str(args.source))
    from pyramid_dit.flux_modules import PyramidFluxTransformer
    from trainer_misc import setup_for_distributed
    setup_for_distributed(True)  # Native print(force=...) compatibility, stdout only.
    torch.set_num_threads(2)
    root = args.scratch
    root.mkdir(parents=True, exist_ok=True)
    weights = root / "model/diffusion_transformer_384p"
    if not weights.exists():
        seed()
        dit = PyramidFluxTransformer(num_layers=1, num_single_layers=1,
            attention_head_dim=64, num_attention_heads=1, joint_attention_dim=8,
            pooled_projection_dim=4, axes_dims_rope=[16, 24, 24],
            use_flash_attn=False, use_temporal_causal=True)
        dit.save_pretrained(weights)
        cache = root / "cache"
        cache.mkdir()
        torch.save(torch.zeros(1, 16, 4, 48, 80), cache / "latent.pt")
        null = {"prompt_embed": torch.zeros(1, 3, 8),
                "prompt_attention_mask": torch.ones(1, 3, dtype=torch.long),
                "pooled_prompt_embed": torch.zeros(1, 4)}
        torch.save(null, cache / "text.pt")
        row = {"text": "SYNTHETIC component fixture", "latent": str(cache / "latent.pt"),
               "text_fea": str(cache / "text.pt")}
        (cache / "annotation.jsonl").write_text("\n".join(json.dumps(row) for _ in range(8)) + "\n")
    recipe = {"model_name": "pyramid_flux", "model_variant": "diffusion_transformer_384p",
              "model_dtype": "float32", "max_temporal_length": 8, "frame_per_unit": 1,
              "video_sync_group": 1, "sample_ratios": [1, 2, 1],
              "use_temporal_causal": True, "load_vae": False, "load_text_encoder": False}
    config = {"models": {"student": {"_target_": "fastvideo_native_pyramid.NativePyramidModel",
        "source": str(args.source), "init_from": str(root / "model"), "recipe": recipe,
        "annotation": str(root / "cache/annotation.jsonl"), "empty_prompt": str(root / "cache/text.pt"),
        "cache_fingerprint": hashlib.sha256((root / "cache/annotation.jsonl").read_bytes()).hexdigest(),
        "component_cpu": True}},
        "method": {"_target_": "fastvideo_native_pyramid.NativePyramidMethod",
                   "use_temporal_pyramid": args.case != "spatial", "clip_grad": 1.0},
        "training": {"distributed": {"num_gpus": 1, "sp_size": 1, "tp_size": 1},
            "data": {"data_path": str(root / "cache"), "train_batch_size": 4,
                     "dataloader_num_workers": 0, "seed": 42},
            "optimizer": {"learning_rate": 1e-4, "betas": [0.9, 0.95],
                          "weight_decay": 0.01, "lr_warmup_steps": 2},
            "loop": {"max_train_steps": 3, "gradient_accumulation_steps": 1 if args.case == "accum1" else 2},
            "checkpoint": {"output_dir": str(root / "checkpoint"), "checkpoints_total_limit": 0},
            "tracker": {"trackers": ["none"]}, "dit_precision": "fp32"},
        "callbacks": {"native_ema": {"_target_": "fastvideo_native_pyramid.NativePyramidEMACallback", "decay": 0.9}}}
    import yaml
    yaml_path = root / f"{args.case}.yaml"
    yaml_path.write_text(yaml.safe_dump(config))
    cfg = load_run_config(str(yaml_path))
    _, method, built_loader, _ = build_from_config(cfg)
    built_loader.close()
    return compatibility, cfg, method


def worker(args):
    compatibility, cfg, method = prepare(args)
    import torch
    import torch.distributed.checkpoint as dcp
    from accelerate import Accelerator
    from pyramid_dit import PyramidDiTForVideoGeneration
    from trainer_misc import create_optimizer, train_one_epoch_with_fsdp
    from fastvideo.train.utils.checkpoint import _CallbackStateWrapper
    from pyramid_runner_state import capture_runner_state, apply_runner_state
    stream = FixtureStream(100.0 if args.case == "high_loss" else 0.1)
    trainer = make_trainer(cfg)
    actual_trace = trace(method.student.runner, method.optimizer)
    if args.mode == "parity":
        reference = PyramidDiTForVideoGeneration(str(args.scratch / "model"), **method.student.recipe)
        reference.null_text_embeds = torch.load(args.scratch / "cache/text.pt", weights_only=True)
        accelerator = Accelerator(cpu=True, mixed_precision="no",
                                  gradient_accumulation_steps=cfg.training.loop.gradient_accumulation_steps)
        opt = create_optimizer(method.native_args, reference.dit)
        reference.dit, opt = accelerator.prepare(reference.dit, opt)
        ema = copy.deepcopy(reference.dit).eval().requires_grad_(False)
        ref_trace = trace(reference, opt)
        seed()
        train_one_epoch_with_fsdp(reference, ema, accelerator, "float32", FixtureStream(stream.amplitude),
            opt, method.lr_values, torch.device("cpu"), epoch=0, start_steps=0,
            args=method.native_args, iters_per_epoch=3, print_freq=100000,
            ema_decay=0.9, use_temporal_pyramid=method.temporal)
        seed()
        trainer.run(method, dataloader=stream, max_steps=3)
        comparisons = {key: actual_trace[key] == ref_trace[key] for key in actual_trace}
        comparisons["final_weights"] = hashed(reference.dit.state_dict()) == hashed(method.student.transformer.state_dict())
        comparisons["final_optimizer"] = hashed(opt.state_dict()) == hashed(method.optimizer.state_dict())
        comparisons["ema"] = hashed(ema.state_dict()) == hashed(trainer.callbacks._callbacks["native_ema"].model_ema.state_dict())
        assert all(comparisons.values()), comparisons
        assert len(actual_trace["updates"]) == (0 if args.case == "high_loss" else 3)
        result = {"checks": comparisons, "optimizer_updates": len(actual_trace["updates"]),
                  "microbatches": len(actual_trace["inputs"]), "trace": actual_trace,
                  "ar_counter": method.student.runner.accumulate_steps}
    else:
        checkpoint = args.scratch / "dcp"
        if args.mode == "save":
            seed()
            trainer.run(method, dataloader=stream, max_steps=2)
            states = method.checkpoint_state()
            states.update(callbacks=_CallbackStateWrapper(trainer.callbacks), data=stream)
            dcp.save(states, checkpoint_id=str(checkpoint))
            torch.save(capture_runner_state(method.student.runner, method.student.accelerator), args.scratch / "rng.pt")
            actual_trace = trace(method.student.runner, method.optimizer)
            # Continue without recreating callbacks; actual Trainer.run would
            # reinitialize EMA if called again. One managed native step + normal
            # callback dispatch is the continuation boundary under test.
            losses, outputs, metrics = method.managed_train_step(iter(stream), 3)
            trainer.callbacks.on_training_step_end(method, losses, iteration=3)
        else:
            method.on_train_start()
            trainer.callbacks.on_train_start(method, iteration=0)
            method.seed_optimizer_state_for_resume()
            states = method.checkpoint_state()
            states.update(callbacks=_CallbackStateWrapper(trainer.callbacks), data=stream)
            dcp.load(states, checkpoint_id=str(checkpoint))
            # Reader restoration precedes final global RNG restoration.
            iter(stream)
            apply_runner_state(torch.load(args.scratch / "rng.pt", weights_only=True),
                               method.student.runner, method.student.accelerator)
            losses, outputs, metrics = method.managed_train_step(iter(stream), 3)
            trainer.callbacks.on_training_step_end(method, losses, iteration=3)
        result = {"trace": actual_trace, "weights": hashed(method.student.transformer.state_dict()),
                  "optimizer": hashed(method.optimizer.state_dict()),
                  "ema": hashed(trainer.callbacks.state_dict()), "control": method.control_state.state_dict(),
                  "data": stream.state_dict()}
    assert not torch.cuda.is_initialized()
    result.update(mode=args.mode, case=args.case, cuda_initialized=False,
        compatibility=compatibility, parameters=sum(p.numel() for p in method.student.transformer.parameters()),
        component_only=True, full_native_checkpoint=False,
        packages={name: md.version(name) for name in ("torch", "accelerate", "diffusers", "transformers", "torchdata")})
    args.result.write_text(json.dumps(result, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--fastvideo", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["parity", "save", "resume"])
    p.add_argument("--case", default="accum2", choices=["accum1", "accum2", "spatial", "high_loss"])
    p.add_argument("--scratch", type=Path)
    p.add_argument("--result", type=Path)
    args = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU check requires CUDA_VISIBLE_DEVICES empty")
    if args.mode:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    scratch = Path(tempfile.mkdtemp(prefix="rdd-fastvideo-native-cpu-"))
    commands = [(case, "parity") for case in ("accum1", "accum2", "spatial", "high_loss")]
    commands += [("accum2", "save"), ("accum2", "resume")]
    reports = {}
    for case, mode in commands:
        key = f"{case}_{mode}"
        cmd = [sys.executable, str(Path(__file__).resolve()), "--source", str(args.source),
            "--fastvideo", str(args.fastvideo), "--output", str(args.output), "--scratch", str(scratch / case),
            "--mode", mode, "--case", case, "--result", str(args.output / f"{key}.json")]
        with (args.output / f"{key}.log").open("w") as log:
            completed = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=90)
        if completed.returncode:
            raise RuntimeError(f"CPU worker failed: {key}; see its log")
        reports[key] = json.loads((args.output / f"{key}.json").read_text())
        print(f"passed {key}", flush=True)
    reference = reports["accum2_save"]
    restored = reports["accum2_resume"]
    keys = ("trace", "weights", "optimizer", "ema", "control", "data")
    checks = {key: reference[key] == restored[key] for key in keys}
    assert all(checks.values()), checks
    summary = {"status": "passed", "component_only": True, "tiny_native_dit_not_full_weights": True,
               "gpu_used": False, "full_native_checkpoint": False, "scratch": str(scratch),
               "cases": {k: {x: v[x] for x in ("parameters", "component_only", "cuda_initialized", "packages")}
                         | {"checks": v.get("checks"), "optimizer_updates": v.get("optimizer_updates")}
                         for k, v in reports.items()}, "fresh_process_dcp_resume": checks}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
