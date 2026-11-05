#!/usr/bin/env python3
"""Portable, restartable FM fresh sampling and matched PF training.

Only the explicit assets action downloads weights. Training/sampling are offline.
The production training worker delegates all mathematics to umdiff.cli.train.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
UNIFIED = HERE.parents[1]
REPO = UNIFIED.parent
sys.path.insert(0, str(UNIFIED))
ASSETS = REPO/'checkpoints/fm_pf_assets'
RUNS = UNIFIED/'runs/fm_pf'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path):
    import torch
    # The pinned project checkpoint is trusted. mmap avoids eagerly allocating
    # all historical optimizer tensors just to inspect the EMA/step.
    try:
        return torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    except RuntimeError as error:
        if 'mmap' not in str(error):
            raise
        return torch.load(path, map_location='cpu', weights_only=False)


def environment():
    import importlib.metadata
    versions = {}
    for name in ('torch', 'torchvision', 'numpy', 'scipy', 'timm', 'diffusers',
                 'transformers', 'huggingface-hub', 'Pillow'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'python': sys.version, 'platform': platform.platform(), 'packages': versions}


def offline():
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'


def verify_asset(path, entry):
    if path.stat().st_size != entry['size']:
        raise ValueError(f'Wrong asset size: {path}')
    if 'sha256' in entry:
        digest = sha256(path)
        if digest != entry['sha256']:
            raise ValueError(f'Asset SHA256 mismatch: {path}')
    else:
        body = path.read_bytes()
        blob = hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()
        if blob != entry['git_blob']:
            raise ValueError(f'Config Git blob mismatch: {path}')
        digest = hashlib.sha256(body).hexdigest()
    return {'path': str(path.resolve()), 'sha256': digest, 'size': entry['size']}


def assets(args):
    # Explicit download action, independent of offline activation scripts.
    os.environ['HF_HUB_OFFLINE'] = '0'
    os.environ['TRANSFORMERS_OFFLINE'] = '0'
    from huggingface_hub import hf_hub_download
    lock = read(HERE/'assets.lock.json')
    root = args.assets.resolve()
    root.mkdir(parents=True, exist_ok=True)
    selected = [args.only] if args.only else ['vae', 'fm']
    missing = sum(f['size'] for name in selected for f in lock[name]['files']
                  if not (root/f['local']).exists())
    if not args.verify and shutil.disk_usage(root).free < missing+2*1024**3:
        raise OSError('Insufficient space for requested assets plus download headroom')
    verified = []
    for name in selected:
        entry = lock[name]
        for file in entry['files']:
            target = root/file['local']
            if not target.exists():
                if args.verify:
                    raise FileNotFoundError(target)
                downloaded = hf_hub_download(entry['repo'], filename=file['remote'],
                    revision=entry['revision'], local_dir=root/name)
                if Path(downloaded).resolve() != target.resolve():
                    raise ValueError('Downloaded file is outside the selected asset layout')
            print(f'Verifying {target}', flush=True)
            verified.append(verify_asset(target, file))
    write(root/'verified_assets.json', {'lock': lock, 'files': verified})
    print(json.dumps({'assets': str(root), 'verified': len(verified)}))


def ensure_fm_envelope(root):
    raw = root/'fm/checkpoints/RDD-FM-Unified-XL-v2/0400000.pt'
    dest = root/'fm_ema_inference.pt'
    if dest.exists():
        sidecar = read(dest.with_suffix('.pt.import.json'))
        if sidecar['source_sha256'] != read(HERE/'assets.lock.json')['fm']['files'][0]['sha256']:
            raise ValueError('Existing FM envelope belongs to a different source')
        if sidecar.get('selected_weights') != 'ema':
            raise ValueError('Existing FM envelope must contain the original EMA weights; raw model weights are a different experiment')
        if sidecar.get('source_step') != 400000:
            raise ValueError('Existing FM envelope must come from the selected 400K checkpoint')
        if sha256(dest) != sidecar['output_sha256']:
            raise ValueError('Existing FM envelope hash differs from its import record')
        return dest
    verify_asset(raw, read(HERE/'assets.lock.json')['fm']['files'][0])
    spec = importlib.util.spec_from_file_location('fm_import', UNIFIED/'tools/import_legacy_rdd_checkpoint.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.import_checkpoint(HERE/'rdd_fm_fresh.json', raw, dest, weights='ema')
    return dest


def sample(args):
    offline()
    from sampling import sample_directory
    checkpoint = args.checkpoint
    if checkpoint is None:
        if args.action == 'fm-sample':
            checkpoint = ensure_fm_envelope(args.assets.resolve())
        else:
            checkpoint = RUNS/'pf_train/checkpoint.pt'
    vae = args.vae or args.assets/'vae'
    output = args.output or RUNS/('fm_fresh_cfg'+str(args.cfg) if args.action == 'fm-sample' else 'pf_samples')
    sample_directory(checkpoint, vae, output, count=args.count, batch_size=args.batch_size,
        decode_batch=args.decode_batch, seed=args.seed, cfg=args.cfg,
        kind='fm' if args.action == 'fm-sample' else 'pf', device=args.device,
        threads=args.threads)


def inspect_data(path):
    from umdiff.cli import ImageFolder
    from PIL import Image
    path = Path(path).resolve()
    if not path.is_dir():
        raise FileNotFoundError('Set IMAGENET_TRAIN_DIR to the extracted ImageNet train directory')
    dataset = ImageFolder(path, 256, {'crop': 'adm_center'})
    classes = sorted(dataset.labels)
    numeric = classes == [f'{i:04d}' for i in range(1000)]
    if numeric:
        provenance = read(path/'HF_EXPORT_COMPLETE.json')
        if provenance.get('repo') != 'ILSVRC/imagenet-1k' or provenance.get('count') != 1281167:
            raise ValueError('Numeric folders require the supplied ImageNet HF export provenance')
    elif len(classes) != 1000 or not all(re.fullmatch(r'n\d{8}', c) for c in classes):
        raise ValueError('Expected1000 WordNet class folders (n########), not flat JPEGs/parquet/latents')
    if len(dataset) != 1281167:
        raise ValueError(f'Expected1281167 original training images; found{len(dataset)}. Do not silently train a subset.')
    paths = dataset.images.paths
    for index in [round(i*(len(paths)-1)/31) for i in range(32)]:
        with Image.open(paths[index]) as image:
            image.convert('RGB').load()
    return {'path': str(path), 'classes': classes, 'images': len(dataset),
            'class_order_sha256': hashlib.sha256('\n'.join(classes).encode()).hexdigest(),
            'decoded_checks': 32, 'validation_scope': 'inventory and32decoded images, not full content authentication'}


def configure_training(args, output):
    from umdiff.experiment import load, resolve, validate_math
    spec = load(args.config)
    if args.micro_batch is not None:
        spec['run']['micro_batch'] = args.micro_batch
    if spec['run']['global_batch'] % spec['run']['micro_batch']:
        raise ValueError('micro_batch must divide global256')
    if args.updates is not None:
        spec['run']['updates'] = args.updates
    if spec['run']['updates'] < 1:
        raise ValueError('Updates must be positive')
    validate_math(spec)
    vae = (args.vae or args.assets/'vae').resolve()
    if not (vae/'config.json').is_file():
        raise FileNotFoundError('Prepare the local VAE with the assets action')
    config = resolve(spec, str(vae))
    method_file = output/'resolved_config.json'
    if method_file.exists() and read(method_file) != config:
        raise ValueError('Existing run has another configuration/microbatch; use a new output directory')
    write(method_file, config)
    write(output/'experiment.json', spec)
    return spec, method_file


def train_worker(args):
    offline()
    import torch
    from umdiff import cli
    torch.set_num_threads(args.threads)
    device = cli.choose_device(args.device)
    spec = read(args.output/'experiment.json')
    torch.backends.cuda.matmul.allow_tf32 = spec['run'].get('allow_tf32', False)
    torch.backends.cudnn.allow_tf32 = spec['run'].get('allow_tf32', False)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    checkpoint = args.output/'checkpoint.pt'
    start = load_checkpoint(checkpoint)['step'] if checkpoint.exists() else 0
    remaining = args.target_step-start
    if remaining < 1:
        return
    began = time.monotonic()
    cli.train(SimpleNamespace(config=str(args.output/'resolved_config.json'),
        data=str(args.data), synthetic=False, output=str(args.output),
        resume=str(checkpoint) if checkpoint.exists() else None, steps=remaining,
        batch_size=spec['run']['global_batch'], seed=spec['run']['seed'], device=args.device,
        save_every=spec['run']['save_every'], log_every=spec['run']['log_every']))
    result = {'start_step': start, 'end_step': args.target_step,
              'elapsed_seconds': time.monotonic()-began, 'environment': environment(),
              'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated() if device.type=='cuda' else 0,
              'device': str(device), 'trace': read(args.output/'trace.json')}
    write(args.output/f'progress/{start:07d}_{args.target_step:07d}.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('trace', 'environment')}), flush=True)


def train(args):
    offline()
    output = (args.output or RUNS/'pf_train').resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Prevent two agents from simultaneously writing the same training run.
    import fcntl
    with (output/'.writer.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec, method_file = configure_training(args, output)
        data = Path(args.data or os.environ.get('IMAGENET_TRAIN_DIR', '')).resolve()
        inventory = inspect_data(data)
        write(output/'dataset_inventory.json', inventory)
        checkpoint = output/'checkpoint.pt'
        current = load_checkpoint(checkpoint)['step'] if checkpoint.exists() else 0
        total = spec['run']['updates']
        if current >= total:
            print(f'Already at step{current}, requested total{total}.');return
        if shutil.disk_usage(output).free < 32*1024**3:
            raise OSError('Need32GiB free checkpoint headroom before full-model preflight')
        def worker(target):
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '_train-worker',
                '--output', str(output), '--data', str(data), '--device', args.device,
                '--threads', str(args.threads), '--target-step', str(target)], check=True)
        # A real full-size optimizer update is preflight AND the first update
        # of the actual run. It is resumed, never counted twice or discarded.
        if not (output/'PREFLIGHT.json').exists():
            target = min(current+1, total)
            worker(target)
            write(output/'PREFLIGHT.json', {'passed': True, 'completed_step': target,
                  'config_sha256': sha256(method_file), 'data': inventory['path'],
                  'note': 'Full selected network, real images, VAE, forward/backward, optimizer/EMA/checkpoint.'})
            current = target
        if args.preflight_only:
            print('Real-data preflight passed; rerun without --preflight-only to resume.');return
        # Keep latest + selected research milestones. Temporary checkpoint
        # files are atomically replaced by Trainer.save.
        while current < total:
            target = min(((current//5000)+1)*5000, total)
            if shutil.disk_usage(output).free < 32*1024**3:
                raise OSError('Need32GiB free checkpoint headroom; free space, then rerun to resume')
            worker(target)
            current = target
            if current in (50000, 100000, 200000, 400000) or current == total:
                snapshots = output/'milestones'
                snapshots.mkdir(exist_ok=True)
                dest = snapshots/f'{current:07d}.pt'
                if not dest.exists():
                    # Subsequent atomic checkpoint replacement preserves the inode.
                    try:os.link(checkpoint, dest)
                    except OSError:shutil.copy2(checkpoint, dest)
        write(output/'TRAINING_COMPLETE.json', {'step': current,
              'checkpoint': str(checkpoint), 'experiment': spec,
              'quality_metrics': 'Not automatically computed; sample/ADM evaluation are separate.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('assets')
    p.add_argument('--assets', type=Path, default=ASSETS)
    p.add_argument('--only', choices=['fm', 'vae'])
    p.add_argument('--verify', action='store_true')
    p.set_defaults(run=assets)
    for name in ('fm-sample', 'pf-sample'):
        p = sub.add_parser(name)
        p.add_argument('--assets', type=Path, default=ASSETS)
        p.add_argument('--checkpoint', type=Path)
        p.add_argument('--vae', type=Path)
        p.add_argument('--output', type=Path)
        p.add_argument('--count', type=int, default=50000)
        p.add_argument('--batch-size', type=int, default=4)
        p.add_argument('--decode-batch', type=int, default=1)
        p.add_argument('--seed', type=int, default=0)
        p.add_argument('--cfg', type=float, default=1.)
        p.add_argument('--device', default='cuda')
        p.add_argument('--threads', type=int, default=4)
        p.set_defaults(run=sample)
    p = sub.add_parser('pf-train')
    p.add_argument('--assets', type=Path, default=ASSETS)
    p.add_argument('--config', type=Path, default=HERE/'pyramid_flow_train.json')
    p.add_argument('--vae', type=Path)
    p.add_argument('--data', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--micro-batch', type=int)
    p.add_argument('--updates', type=int)
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--device', default='cuda')
    p.add_argument('--threads', type=int, default=4)
    p.set_defaults(run=train)
    p = sub.add_parser('_train-worker')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--target-step', type=int, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=4)
    p.set_defaults(run=train_worker)
    args = parser.parse_args()
    for key in ('count', 'batch_size', 'decode_batch', 'threads', 'micro_batch', 'updates'):
        if getattr(args, key, None) is not None and getattr(args, key)<1:
            parser.error(f'{key} must be positive')
    args.run(args)


if __name__ == '__main__':
    main()
