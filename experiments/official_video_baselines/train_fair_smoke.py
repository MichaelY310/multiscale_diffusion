"""Guard-required worker composing the unchanged current FastVideo Trainer."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import socket
import time

from launch_smoke import bounded_deadline
from prepare_fair_suite import ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--trace-file", type=Path)
    p.add_argument("--deterministic", action="store_true")
    args = p.parse_args()
    if socket.gethostname().split(".")[0] != "surfi1" or Path.cwd().resolve() != ROOT:
        p.error("GPU worker requires surfi1 and the official runtime working directory")
    deadline = float(os.environ.get("RDD_DEADLINE_UNIX", "0"))
    if not os.environ.get("RDD_JOB_TOKEN") or deadline <= time.time():
        p.error("launch through the independent deadline guard")
    end = bounded_deadline(dt.datetime.now(dt.timezone.utc), deadline - time.time())
    if deadline > end.timestamp() + 1:
        p.error("job deadline exceeds the authorized GPU cutoff")
    if args.deterministic:
        import torch
        torch.use_deterministic_algorithms(True)
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
    from fastvideo.train.trainer import Trainer
    from fastvideo.train.utils.builder import build_from_config
    from fastvideo.train.utils.config import load_run_config
    from fastvideo.train.utils.checkpoint import CheckpointManager, CheckpointConfig
    cfg = load_run_config(args.config)
    tc = cfg.training
    if tc.loop.max_train_steps > 3 or tc.checkpoint.checkpoints_total_limit:
        raise ValueError("Only <=3-update smoke and no checkpoint deletion are allowed")
    maybe_init_distributed_environment_and_model_parallel(tc.distributed.tp_size, tc.distributed.sp_size)
    _, method, loader, start = build_from_config(cfg)
    trace = None
    if args.trace_file:
        from full_model_trace import trace_native, trace_method
        trace = (trace_native(method.student.runner, method.optimizer)
                 if hasattr(method.student,'runner') else trace_method(method))
    trainer = Trainer(tc, config=cfg.resolved_config(), callback_configs=cfg.callbacks)
    from exact_checkpoint import ExactCheckpointManager
    manager_class = ExactCheckpointManager
    if cfg.method["_target_"] == "fastvideo_native_pyramid.NativePyramidMethod":
        from fastvideo_native_checkpoint import NativePyramidCheckpointManager
        manager_class = NativePyramidCheckpointManager
    manager = manager_class(method=method, dataloader=loader, output_dir=tc.checkpoint.output_dir,
        config=CheckpointConfig(save_steps=tc.checkpoint.training_state_checkpointing_steps, keep_last=0),
        callbacks=trainer.callbacks, raw_config=cfg.raw)
    # Count actual AdamW calls; a deadline/skip may not become a successful smoke.
    updates = [0]
    handles = []
    for optimizer in method.get_optimizers(0):
        raw = getattr(optimizer, "optimizer", optimizer)
        def updated(*unused):
            updates[0] += 1
        handles.append(raw.register_step_post_hook(updated))
    trainer.run(method, dataloader=loader, max_steps=tc.loop.max_train_steps,
                start_step=start, checkpoint_manager=manager)
    resumed_start = start
    if tc.checkpoint.resume_from_checkpoint:
        from fastvideo.train.utils.checkpoint import _resolve_resume_checkpoint
        resumed_start = int(_resolve_resume_checkpoint(tc.checkpoint.resume_from_checkpoint,
            output_dir=tc.checkpoint.output_dir).name.split('-')[-1])
    if updates[0] != tc.loop.max_train_steps - resumed_start:
        raise AssertionError(f"Expected complete AdamW updates, observed {updates[0]}")
    result = {"status": "passed", "method": cfg.method["_target_"], "optimizer_updates": updates[0],
              "world_size": tc.distributed.num_gpus, "config": args.config,
              "full_model_parity": False, "quality_evaluation": False,
              "token": os.environ["RDD_JOB_TOKEN"]}
    out = Path(tc.checkpoint.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"smoke_{result['token']}_rank{os.environ.get('RANK', '0')}.json").write_text(json.dumps(result, indent=2))
    if trace is not None:
        from full_model_trace import hashed
        trace['final_callbacks'] = hashed(trainer.callbacks.state_dict())
        control=getattr(method,'control_state',getattr(method,'control',None))
        trace['final_control'] = control.state_dict() if control else None
        trace['final_data_counter'] = getattr(loader,'emitted_batches',None)
        trace['final_scheduler'] = hashed({str(i):s.state_dict() for i,s in enumerate(method.get_lr_schedulers(0))})
        trace_path=args.trace_file
        if tc.distributed.num_gpus>1:
            trace_path=trace_path.with_name(trace_path.stem+'_rank'+os.environ['RANK']+trace_path.suffix)
        trace_path.write_text(json.dumps(trace))


if __name__ == "__main__":
    main()
