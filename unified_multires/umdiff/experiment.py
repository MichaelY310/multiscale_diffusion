"""Resolve a complete experiment file into the existing offline training CLI.

No downloading, no new GPU authorization, no changes to legacy RDD code.
This is a single-process frontend, not a claim to duplicate old eight-rank RNG.
"""
import argparse
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace


def load(path):
    value = json.loads(Path(path).read_text())
    if value.get('format') != 'umdiff_experiment_v1':
        raise ValueError('Expected umdiff_experiment_v1')
    allowed = {'format', 'id', 'suite', 'method', 'run', 'assets', 'sampling', 'comparison'}
    if set(value)-allowed:
        raise ValueError(f'Unknown experiment fields: {set(value)-allowed}')
    run = value['run']
    if run['global_batch'] % run['micro_batch']:
        raise ValueError('Global batch must divide micro batch exactly')
    if run['world_size'] != 1:
        raise ValueError('This frontend supports one process; legacy eight-rank commands are separately recorded')
    if min(run['updates'], run['global_batch'], run['micro_batch']) < 1:
        raise ValueError('Invalid run budget')
    return value


def resolve(spec, vae=None):
    config = copy.deepcopy(spec['method'])
    run = spec['run']
    config['training']['optimizer']['accumulation_steps'] = run['global_batch']//run['micro_batch']
    def replace(value):
        if isinstance(value, str) and value == '@sd_vae_ema@':
            return vae or value
        if isinstance(value, dict):
            return {k: replace(v) for k, v in value.items()}
        if isinstance(value, list):
            return [replace(v) for v in value]
        return value
    return replace(config)


def validate_math(spec):
    from .cli import make_method
    return make_method(resolve(spec))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'export', 'train', 'sample'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--data', help='Original ImageNet train directory with class subdirectories')
    parser.add_argument('--vae', help='Local stabilityai/sd-vae-ft-ema Diffusers directory')
    parser.add_argument('--output')
    parser.add_argument('--resume')
    parser.add_argument('--checkpoint', help='Checkpoint for the sample action; EMA is used')
    parser.add_argument('--count', type=int, default=4)
    parser.add_argument('--labels', type=int, nargs='+')
    parser.add_argument('--sample-seed', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    spec = load(args.config)
    vae = args.vae or os.environ.get('SD_VAE_EMA_DIR')
    data = args.data or os.environ.get('IMAGENET_TRAIN_DIR')
    config = resolve(spec, vae)
    from .cli import make_method
    make_method(config)
    if args.action == 'plan':
        print(json.dumps({'id': spec['id'], 'suite': spec['suite'], 'run': spec['run'],
                          'assets': {'data': data, 'vae': vae}, 'sampling': spec['sampling'],
                          'comparison': spec['comparison'],
                          'status': 'mathematical configuration valid; no network or data loaded'}, indent=2, ensure_ascii=False))
        return
    if not args.output:
        parser.error('--output is required for export/train/sample')
    output = Path(args.output)
    if args.action == 'export':
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(config, indent=2)+'\n')
        return
    # Enforce the existing device guard before allocating a large network.
    from . import cli
    cli.choose_device(args.device)
    if args.action == 'sample':
        if not args.checkpoint or not args.labels:
            parser.error('sample requires --checkpoint and --labels')
        if output.exists():
            raise FileExistsError(output)
        import torch
        torch.set_num_threads(args.threads)
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        # Use the recorded codec path when --vae was omitted. Sampling itself
        # does not need to allocate the VAE; decoding is an explicit CLI step.
        bound_vae = checkpoint['config']['training']['input_transform']['args']['model']['args']['path']
        if checkpoint['config'] != resolve(spec, vae or bound_vae):
            raise ValueError('Checkpoint method differs from the requested experiment')
        del checkpoint
        cli.sample(SimpleNamespace(checkpoint=args.checkpoint, output=str(output),
                                   count=args.count, labels=args.labels, seed=args.sample_seed,
                                   steps=spec['sampling'].get('sample_cli_steps'),
                                   raw_weights=False, conditioning=None, device=args.device))
        return
    if not data or not Path(data).is_dir():
        raise FileNotFoundError('Provide --data or IMAGENET_TRAIN_DIR: original ImageNet train directory')
    if not vae or not (Path(vae)/'config.json').is_file():
        raise FileNotFoundError('Provide --vae or SD_VAE_EMA_DIR: local sd-vae-ft-ema; no downloads')
    output.mkdir(parents=True, exist_ok=True)
    resolved = output/'resolved_config.json'
    if resolved.exists() and json.loads(resolved.read_text()) != config:
        raise ValueError('Existing output belongs to a different resolved configuration')
    if (output/'checkpoint.pt').exists() and not args.resume:
        raise FileExistsError('Existing checkpoint: explicitly resume or select a new output')
    resolved.write_text(json.dumps(config, indent=2)+'\n')
    (output/'experiment.json').write_text(json.dumps(spec, indent=2, ensure_ascii=False)+'\n')
    import torch
    torch.set_num_threads(args.threads)
    run = spec['run']
    torch.backends.cuda.matmul.allow_tf32=run.get('allow_tf32',False)
    torch.backends.cudnn.allow_tf32=run.get('allow_tf32',False)
    remaining = run['updates']
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        remaining -= ckpt['step']
        del ckpt
        if remaining <= 0:
            print('Checkpoint already reaches the configured total update budget.')
            return
    cli.train(SimpleNamespace(config=str(resolved), data=data, synthetic=False,
                              output=str(output), resume=args.resume, steps=remaining,
                              batch_size=run['global_batch'], seed=run['seed'], device=args.device,
                              save_every=run['save_every'], log_every=run['log_every']))


if __name__ == '__main__':
    main()
