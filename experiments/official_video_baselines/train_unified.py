"""Guarded training entry: original modules, current FastVideo Trainer, exact DCP.

Supports ordinary runs and explicitly bounded integration runs. A deadline is
checked at completed optimizer boundaries; stop/save before the independent
hard guard. No existing checkpoint is overwritten or automatically deleted.
"""
from typing import Any, cast
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import socket
import time


class CompletedBoundaryStop(Exception):
    pass


def main() -> None:
    worker_started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--run-steps', type=int, help='Limit this invocation, retain the complete configured LR recipe')
    p.add_argument('--deterministic', action='store_true')
    p.add_argument('--trace-file', type=Path)
    p.add_argument('--save-margin-seconds', type=float, default=120.)
    p.add_argument('--initial-step-seconds', type=float, default=90.)
    args = p.parse_args()
    from launch_smoke import active_policy, bounded_deadline
    from prepare_fair_suite import ROOT
    policy = active_policy()
    deadline = float(os.environ.get('RDD_DEADLINE_UNIX', '0'))
    if socket.gethostname().split('.')[0] != policy['host'] or Path.cwd().resolve() != ROOT:
        p.error('Run in the pinned official runtime on surfi1')
    if not os.environ.get('RDD_JOB_TOKEN') or deadline <= time.time():
        p.error('Independent guarded launcher required')
    end = bounded_deadline(dt.datetime.now(dt.timezone.utc), deadline - time.time())
    if deadline > end.timestamp() + 1:
        p.error('Deadline exceeds the authorized absolute cutoff')
    if args.save_margin_seconds < 60 or args.initial_step_seconds < 1:
        p.error('Use >=60 seconds for full checkpoint writes')
    if args.run_steps is not None and args.run_steps < 1:
        p.error('Positive invocation step limit required')
    if deadline - time.time() < args.save_margin_seconds + args.initial_step_seconds + 60:
        raise RuntimeError('Insufficient authorized time to load, update and save a complete checkpoint')
    import torch
    import torch.distributed as dist
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
    from fastvideo.train.trainer import Trainer
    from fastvideo.train.utils.builder import build_from_config
    from fastvideo.train.utils.config import load_run_config
    from fastvideo.train.utils.checkpoint import CheckpointConfig, _resolve_resume_checkpoint
    from exact_checkpoint import ExactCheckpointManager
    cfg = load_run_config(args.config)
    tc = cfg.training
    if tc.checkpoint.checkpoints_total_limit:
        raise ValueError('Automatic checkpoint deletion is disabled')
    maybe_init_distributed_environment_and_model_parallel(tc.distributed.tp_size, tc.distributed.sp_size)
    _, method, loader, start = build_from_config(cfg)
    if tc.checkpoint.resume_from_checkpoint:
        start = int(
            _resolve_resume_checkpoint(tc.checkpoint.resume_from_checkpoint,
                                       output_dir=tc.checkpoint.output_dir).name.split('-')[-1])
    maximum = min(tc.loop.max_train_steps, start + args.run_steps) if args.run_steps else tc.loop.max_train_steps
    if maximum <= start:
        raise ValueError('Requested training has no remaining updates')
    trainer = Trainer(tc, config=cfg.resolved_config(), callback_configs=cfg.callbacks)
    manager_class = ExactCheckpointManager
    if hasattr(method.student, 'runner'):
        from fastvideo_native_checkpoint import NativePyramidCheckpointManager
        manager_class = NativePyramidCheckpointManager
    manager = manager_class(method=method,
                            dataloader=loader,
                            output_dir=tc.checkpoint.output_dir,
                            config=CheckpointConfig(save_steps=tc.checkpoint.training_state_checkpointing_steps,
                                                    keep_last=0),
                            callbacks=trainer.callbacks,
                            raw_config=cfg.raw)
    # Explicit invocation cap and final state must save even when an interval
    # was omitted in an integration config. No deletion/overwrite is enabled.
    completed = [start]
    update_seconds = [args.initial_step_seconds]
    save_seconds = [args.save_margin_seconds]
    actual_updates = [0]
    optimizer_handles = []
    post_clip_norm: list[float | None] = [None]
    stage_work = {'student_tokens': 0, 'teacher_tokens': 0, 'student_forwards': 0, 'teacher_forwards': 0}
    forward_handles = []

    def volume(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.shape[0] * value.shape[2] * value.shape[3] * value.shape[4] // 4 if value.ndim == 5 else 0
        if isinstance(value, list | tuple):
            return sum(volume(x) for x in value)
        return 0

    for role, model in method._role_models.items():

        def count(module: Any, pos: Any, kw: Any, role: Any = role) -> Any:
            sample = kw.get('sample', kw.get('hidden_states', pos[0] if pos else None))
            name = 'student' if role == 'student' else 'teacher'
            stage_work[name + '_tokens'] += volume(sample)
            stage_work[name + '_forwards'] += 1

        forward_handles.append(model.transformer.register_forward_pre_hook(count, with_kwargs=True))
    for optimizer in method.get_optimizers(0):
        raw = getattr(optimizer, 'optimizer', optimizer)

        def norm(*unused: Any) -> Any:
            values = [
                torch.linalg.vector_norm(p.grad.detach().float()) for p in method.student.transformer.parameters()
                if p.grad is not None
            ]
            post_clip_norm[0] = float(torch.linalg.vector_norm(torch.stack(values))) if values else None

        optimizer_handles.append(raw.register_step_pre_hook(norm))
        optimizer_handles.append(
            raw.register_step_post_hook(lambda *unused: actual_updates.__setitem__(0, actual_updates[0] + 1)))
    training_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    raw_save = manager.save

    def save(step: Any) -> Any:
        before = time.perf_counter()
        raw_save(step)
        save_seconds[0] = max(save_seconds[0], 1.5 * (time.perf_counter() - before))

    cast(Any, manager).save = save
    trace = None
    if args.trace_file:
        from full_model_trace import trace_method, trace_native
        trace = trace_native(method.student.runner, method.optimizer) if hasattr(method.student,
                                                                                 'runner') else trace_method(method)
    original_end = trainer.callbacks.on_training_step_end
    out = Path(tc.checkpoint.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def boundary(method: Any, metrics: Any, iteration: Any = 0) -> Any:
        original_end(method, metrics, iteration=iteration)
        completed[0] = iteration
        update_seconds[0] = max(update_seconds[0], 1.5 * float(metrics['step_time_sec']))
        metrics = dict(metrics,
                       actual_optimizer_updates=actual_updates[0],
                       post_clip_gradient_norm=post_clip_norm[0],
                       **stage_work,
                       cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                       elapsed_seconds_including_teacher=time.perf_counter() - training_started,
                       world_size=dist.get_world_size())
        if dist.get_rank() == 0:
            with (out / 'training_metrics.jsonl').open('a') as f:
                f.write(json.dumps({'step': iteration, **metrics}) + '\n')
        stop = torch.tensor(time.time() + save_seconds[0] + update_seconds[0] + 15 >= deadline,
                            device=method.student.device,
                            dtype=torch.int32)
        dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        if stop.item() and iteration < maximum:
            manager.save(iteration)
            raise CompletedBoundaryStop('Saved before the independent GPU deadline')

    trainer.callbacks.on_training_step_end = boundary
    status = 'completed'
    try:
        trainer.run(method, dataloader=loader, max_steps=maximum, start_step=start, checkpoint_manager=manager)
        manager.save(completed[0])
    except CompletedBoundaryStop:
        status = 'stopped_at_completed_boundary'
        trainer.callbacks.on_train_end(method, iteration=completed[0])
        trainer.tracker.finish()
    if trace is not None:
        from full_model_trace import hashed
        trace['final_callbacks'] = hashed(trainer.callbacks.state_dict())
        control = getattr(method, 'control_state', getattr(method, 'control', None))
        trace['final_control'] = control.state_dict() if control else None
        trace['final_scheduler'] = hashed({str(i): s.state_dict() for i, s in enumerate(method.get_lr_schedulers(0))})
        target = args.trace_file
        if dist.get_world_size() > 1:
            target = target.with_name(target.stem + '_rank' + str(dist.get_rank()) + target.suffix)
        target.write_text(json.dumps(trace))
    (out / ('run_' + os.environ['RDD_JOB_TOKEN'] + '_rank' + str(dist.get_rank()) + '.json')).write_text(
        json.dumps(
            {
                'status': status,
                'start_step': start,
                'completed_step': completed[0],
                'optimizer_updates': actual_updates[0],
                'processed_steps': completed[0] - start,
                'skipped_optimizer_steps': completed[0] - start - actual_updates[0],
                'configured_max_steps': tc.loop.max_train_steps,
                'invocation_target': maximum,
                'checkpoint': str(out / f'checkpoint-{completed[0]}'),
                'method': cfg.method['_target_'],
                'world_size': dist.get_world_size(),
                'exact_checkpoint_protocol': 'full roles + rank-local data/RNG + contract + atomic commit',
                'quality_training_complete': False,
                'worker_seconds_including_model_loading': time.perf_counter() - worker_started,
                'worker_gpu_hours': (time.perf_counter() - worker_started) * dist.get_world_size() / 3600,
                'elapsed_seconds_including_teacher_and_checkpoint': time.perf_counter() - training_started,
                'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved()
            },
            indent=2))
    for handle in optimizer_handles + forward_handles:
        handle.remove()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
