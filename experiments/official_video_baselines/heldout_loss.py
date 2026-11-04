"""Deterministic held-out path loss diagnostics, not cross-objective quality."""
import argparse
import json
import os
from pathlib import Path
import time


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', default='base')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-prompts', type=int, default=2)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not os.environ.get('RDD_JOB_TOKEN') or time.time() >= float(os.environ.get('RDD_DEADLINE_UNIX', '0')):
        raise RuntimeError('Independent GPU guard required')
    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict, StateDictOptions
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
    from fastvideo.train.utils.config import load_run_config
    from fastvideo.train.utils.builder import build_from_config
    cfg = load_run_config(args.config)
    cfg.training.distributed.num_gpus = 1
    paper = 'PaperPyramidalWanMethod' in cfg.method['_target_']
    native = 'NativePyramidMethod' in cfg.method['_target_']
    if native:
        raise ValueError('Native PF uses original training loss; run the native validation recipe separately')
    if not paper:
        cfg.training.data.data_path = str(args.data)
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    _, method, loader, _ = build_from_config(cfg)
    method.on_train_start()
    method.student.transformer.eval()
    if args.checkpoint != 'base':
        state = {'roles.student.transformer': get_model_state_dict(method.student.transformer)}
        reader = dcp.FileSystemReader(str(Path(args.checkpoint) / 'dcp'))
        prefix = 'roles.student.transformer.'
        keys = {k[len(prefix):] for k in reader.read_metadata().state_dict_metadata if k.startswith(prefix)}
        if keys != set(state['roles.student.transformer']):
            raise ValueError('Incomplete model checkpoint')
        dcp.load(state, storage_reader=reader)
        set_model_state_dict(method.student.transformer,
                             state['roles.student.transformer'],
                             options=StateDictOptions(strict=True))
    results = []
    if paper:
        from fastvideo_pyramidal_wan_paper import PaperCacheDataset
        dataset = PaperCacheDataset(args.data, split='validation', shape='paper')
        for n in range(min(len(dataset), args.max_prompts)):
            # One real held-out clip repeated across stage endpoints solely to
            # report the three different targets of this same clip.
            batch = [dataset[n] for _ in range(3 * method.per_stage)]
            method.cuda_generator.manual_seed(7000 + n)
            with torch.no_grad():
                loss, _, metrics = method.single_train_step(batch, n + 1)
            results.append(
                dict(id=dataset[n]['id'],
                     loss={
                         k: float(v)
                         for k, v in loss.items()
                     },
                     diagnostics={
                         k: float(v)
                         for k, v in metrics.items()
                     }))
    else:
        for n, batch in enumerate(loader):
            if n >= args.max_prompts:
                break
            for stage in range(len(method.student.rdd_path.factors)):
                method.cuda_generator.manual_seed(7000 + n)
                # Fix only the stage for diagnostic coverage; objective and
                # student/teacher math are the same as training.
                method.method_config['smoke_all_stages'] = True
                with torch.no_grad():
                    loss, _, metrics = method.single_train_step(batch, stage + 1)
                results.append(
                    dict(batch=n,
                         stage=stage,
                         loss={
                             k: float(v)
                             for k, v in loss.items()
                         },
                         diagnostics={
                             k: float(v)
                             for k, v in metrics.items()
                         }))
    if not results:
        raise ValueError('No held-out clips')
    args.output.mkdir(parents=True)
    (args.output / 'metrics.json').write_text(
        json.dumps(dict(status='passed',
                        checkpoint=args.checkpoint,
                        data=str(args.data.resolve()),
                        synthetic=False,
                        optimizer_updates=0,
                        path_loss_diagnostic_only=True,
                        warning='Different path targets/teacher terms make raw loss values incomparable across methods',
                        results=results),
                   indent=2))
    (args.output / 'complete.json').write_text(json.dumps(dict(completed=True)))
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
