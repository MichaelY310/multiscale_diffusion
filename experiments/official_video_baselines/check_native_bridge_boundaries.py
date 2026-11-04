"""Native 0/100 EMA cadence, cosine LR and checkpoint rejection on CPU."""
import argparse
import copy
import json
import os
from pathlib import Path
import tempfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--fastvideo", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU audit requires CUDA_VISIBLE_DEVICES empty")
    from check_fastvideo_native_pyramid import prepare, seed, FixtureStream, make_trainer, hashed
    args.scratch = Path(tempfile.mkdtemp(prefix="rdd-native-boundary-cpu-"))
    args.case = "accum1"
    compatibility, cfg, method = prepare(args)
    import torch
    from accelerate import Accelerator
    from pyramid_dit import PyramidDiTForVideoGeneration
    from trainer_misc import create_optimizer, train_one_epoch_with_fsdp, cosine_scheduler
    method.lr_values = cosine_scheduler(base_value=1e-4, final_value=1e-6,
        epochs=1, niter_per_ep=101, warmup_epochs=0, warmup_steps=0)
    reference = PyramidDiTForVideoGeneration(str(args.scratch / "model"), **method.student.recipe)
    reference.null_text_embeds = torch.load(args.scratch / "cache/text.pt", weights_only=True)
    accelerator = Accelerator(cpu=True, mixed_precision="no", gradient_accumulation_steps=1)
    opt = create_optimizer(method.native_args, reference.dit)
    reference.dit, opt = accelerator.prepare(reference.dit, opt)
    ema = copy.deepcopy(reference.dit).eval().requires_grad_(False)
    seed()
    train_one_epoch_with_fsdp(reference, ema, accelerator, "float32", FixtureStream(), opt,
        method.lr_values, torch.device("cpu"), epoch=0, start_steps=0,
        args=method.native_args, iters_per_epoch=101, print_freq=100000,
        ema_decay=0.9, use_temporal_pyramid=True)
    trainer = make_trainer(cfg)
    seed()
    trainer.run(method, dataloader=FixtureStream(), max_steps=101)
    checks = {"weights": hashed(reference.dit.state_dict()) == hashed(method.student.transformer.state_dict()),
              "optimizer": hashed(opt.state_dict()) == hashed(method.optimizer.state_dict()),
              "ema": hashed(ema.state_dict()) == hashed(trainer.callbacks._callbacks["native_ema"].model_ema.state_dict()),
              "ema_steps_0_100": trainer.callbacks._callbacks["native_ema"].update_steps == [0, 100],
              "global_step_101": method.global_step == 101}
    assert all(checks.values()), checks
    valid = method.control_state.state_dict()
    rejected = []
    for key, value in [("schema", 2), ("source_commit", "bad"), ("recipe", {}),
                       ("lr_sha256", "bad"), ("gradient_accumulation_steps", 2),
                       ("clip_grad", 2.0), ("temporal", False),
                       ("global_step", -1), ("global_step", 102), ("accumulate_steps", -1)]:
        invalid = copy.deepcopy(valid)
        invalid[key] = value
        try:
            method.control_state.load_state_dict(invalid)
        except ValueError:
            rejected.append(key)
        else:
            raise AssertionError(f"Accepted incompatible checkpoint: {key}")
    assert method.control_state.state_dict() == valid
    method.student.accelerator.gradient_state._set_sync_gradients(False)
    try:
        try:
            method.control_state.state_dict()
        except ValueError:
            rejected.append("partial_accumulation_boundary")
        else:
            raise AssertionError("Accepted a partial accumulation checkpoint")
    finally:
        method.student.accelerator.gradient_state._set_sync_gradients(True)
    assert not torch.cuda.is_initialized()
    args.output.write_text(json.dumps({"status": "passed", "checks": checks,
        "rejected": rejected, "component_only": True, "parameters": 254208,
        "synthetic": True, "gpu_used": False, "compatibility": compatibility,
        "scratch": str(args.scratch), "scope": "101 tiny CPU updates only; no long/full-model training"}, indent=2) + "\n")
    print(json.dumps(checks))


if __name__ == "__main__":
    main()
