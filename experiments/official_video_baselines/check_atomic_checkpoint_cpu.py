"""Real two-rank Gloo/DCP regression: empty DDP roles, skewed save, contracts.

Tiny CPU network tests the storage protocol, not full video model parity.
"""
from typing import Any
import argparse
import copy
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU only')
    from fastvideo_cpu_import import bootstrap
    bootstrap('/scr/yuheng/bugswarm/official_video_baselines_20261004/reference/FastVideo-37dd31d5')
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torchdata.stateful_dataloader import StatefulDataLoader
    from fastvideo.training.checkpointing_utils import ModelWrapper, OptimizerWrapper, SchedulerWrapper
    from fastvideo.train.utils.checkpoint import CheckpointConfig, _find_latest_checkpoint
    from exact_checkpoint import ExactCheckpointManager
    from full_model_trace import hashed
    dist.init_process_group('gloo')
    rank = dist.get_rank()
    torch.manual_seed(37)
    model = DistributedDataParallel(torch.nn.Linear(8, 4))
    opt = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda n: 1.)
    loader = StatefulDataLoader(torch.arange(24).reshape(3, 8).float() + 100 * rank,
                                batch_size=1,
                                generator=torch.Generator().manual_seed(100 + rank))
    stream = iter(loader)
    x = next(stream)
    model(x).square().mean().backward()
    opt.step()
    scheduler.step()
    opt.zero_grad()
    original_bug_reproduced = not ModelWrapper(model).state_dict()
    role = SimpleNamespace(_trainable=True, transformer=model)
    method = SimpleNamespace(student=role,
                             _role_models={'student': role},
                             checkpoint_state=lambda: {
                                 'roles.student.transformer': ModelWrapper(model),
                                 'optimizer': OptimizerWrapper(model, opt),
                                 'scheduler': SchedulerWrapper(scheduler)
                             })

    class CPUManager(ExactCheckpointManager):

        def _save_rng_snapshot(self: Any, path: Any) -> Any:
            torch.save({'cpu': torch.get_rng_state()}, path / f'rng_state_rank{rank}.pt')

        def load_rng_snapshot(self: Any, path: Any) -> Any:
            torch.set_rng_state(torch.load(Path(path) / f'rng_state_rank{rank}.pt', weights_only=True)['cpu'])

    raw = {
        'models': {
            'student': {
                'base': 'fixed'
            }
        },
        'method': {
            'name': 'test'
        },
        'training': {
            'optimizer': {
                'lr': .01
            },
            'data': {
                'split': 'train'
            },
            'loop': {
                'gradient_accumulation_steps': 1,
                'max_train_steps': 5000
            }
        }
    }
    manager = CPUManager(method=method,
                         dataloader=loader,
                         output_dir=str(args.output),
                         config=CheckpointConfig(1, 0),
                         raw_config=raw)
    os.environ['RDD_JOB_TOKEN'] = 'cpu-gloo-atomic-regression'
    if rank:
        time.sleep(.4)  # Prior pending-directory race was rank-timing dependent.
    manager.save(1)
    cp = args.output / 'checkpoint-1'
    expected = {
        'weights': hashed(model.state_dict()),
        'optimizer': hashed(opt.state_dict()),
        'scheduler': hashed(scheduler.state_dict()),
        'next_data': hashed(next(stream))
    }
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(99)
    opt.state.clear()
    loader2 = StatefulDataLoader(loader.dataset, batch_size=1, generator=torch.Generator().manual_seed(900 + rank))
    restored = CPUManager(method=method,
                          dataloader=loader2,
                          output_dir=str(args.output / 'resume'),
                          config=CheckpointConfig(1, 0),
                          raw_config=raw)
    restored.maybe_resume(resume_from_checkpoint=str(cp))
    actual = {
        'weights': hashed(model.state_dict()),
        'optimizer': hashed(opt.state_dict()),
        'scheduler': hashed(scheduler.state_dict()),
        'next_data': hashed(next(iter(loader2)))
    }
    checks = {k: expected[k] == actual[k] for k in expected}
    rejected = []
    for label, key in [('optimizer', 'optimizer'), ('data', 'data'), ('lr_schedule', 'loop')]:
        changed: Any = copy.deepcopy(raw)
        changed['training'][key] = {'changed': True}
        restored._raw_config = changed
        try:
            restored.maybe_resume(resume_from_checkpoint=str(cp))
        except ValueError:
            rejected.append(label)
        else:
            raise AssertionError('Incompatible resume accepted')
    restored._raw_config = raw
    try:
        restored.save(1)
    except FileExistsError:
        pass
    else:
        # This manager writes a separate output; probe protection in original.
        manager._last_saved_step = None
        try:
            manager.save(1)
        except FileExistsError:
            rejected.append('overwrite')
        else:
            raise AssertionError('Existing checkpoint overwritten')
    if rank == 0:
        pending = args.output / '.checkpoint-999-interrupted.pending'
        (pending / 'dcp').mkdir(parents=True)
    dist.barrier()
    checks['latest_ignores_interrupted'] = _find_latest_checkpoint(args.output) == cp
    checks['committed'] = (cp / 'complete.json').is_file()
    checks['original_empty_role_bug_reproduced'] = original_bug_reproduced
    assert all(checks.values()) and len(rejected) == 4, (checks, rejected)
    result = {
        'status': 'passed',
        'checks': checks,
        'rejected': rejected,
        'cuda_context_initialized': torch.cuda.is_initialized()
    }
    (args.output / f'regression_rank{rank}.json').write_text(json.dumps(result, indent=2))
    assert not torch.cuda.is_initialized()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
