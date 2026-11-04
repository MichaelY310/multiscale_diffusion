#!/usr/bin/env python3
"""CPU CLI/relocation checks; temporary tensors/checkpoints are removed on exit.

Run with the project's existing Python environment. No model/data downloads.
This tests packaging and small random models, not formal image quality.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def equal_tree(a, b):
    import torch
    if torch.is_tensor(a):
        assert a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_tree(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal_tree(x, y)
    else:
        assert a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', UMDIFF_CPU_ONLY='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env.pop('UMDIFF_RDD_ROOT', None)
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    commands = []
    checks = []

    def run(root, *argv):
        child_env = dict(env, PYTHONPATH=os.pathsep.join([
            str(root/'unified_multires'), str(ROOT/'_deps')]))
        command = [sys.executable, *map(str, argv)]
        result = subprocess.run(command, cwd=root, env=child_env,
                                text=True, capture_output=True, timeout=180)
        record = {'cwd': str(root), 'command': command,
                  'exit_code': result.returncode}
        commands.append(record)
        (output/f'command_{len(commands):02d}.log').write_text(
            json.dumps(record)+'\n'+result.stdout+result.stderr)
        if result.returncode:
            raise RuntimeError(f'Command {len(commands)} failed: {result.stderr[-3000:]}')
        return result.stdout

    import torch
    torch.set_num_threads(1)

    def load(path):
        return torch.load(path, map_location='cpu', weights_only=False)

    def sample(root, checkpoint, dest, count=1, labels=False):
        extra = ['--labels', '17'] if labels else []
        run(root, '-m', 'umdiff', 'sample', '--checkpoint', checkpoint,
            '--output', dest, '--count', count, '--seed', 0, '--device', 'cpu', *extra)
        value = load(dest)
        assert torch.is_tensor(value) and torch.isfinite(value).all()
        return value

    with tempfile.TemporaryDirectory(prefix='umdiff-layout-') as temporary:
        temp = Path(temporary)
        relocated = temp/'arbitrary_checkout_name'
        for name in ('DiT', 'diffusion'):
            shutil.copytree(ROOT.parent/name, relocated/name,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        target = relocated/'unified_multires'
        target.mkdir()
        for name in ('umdiff', 'configs'):
            shutil.copytree(ROOT/name, target/name,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        found = run(relocated, '-c',
                    'from umdiff.locations import rdd_root; print(rdd_root())').strip()
        assert Path(found) == relocated

        for mode, shape, grids, batch in [
            ('image', '3,16,16', '[[8,8],[16,16]]', 2),
            ('video', '4,3,16,16', '[[3,8,8],[3,16,16]]', 1),
        ]:
            config = temp/f'{mode}.json'
            run(relocated, '-m', 'umdiff', 'configure', '--method', 'rdd',
                '--shape', shape, '--backbone', 'grid_dit', '--width', 24,
                '--depth', 1, '--attention-heads', 3, '--patch', '1,2,2',
                '--grids', grids, '--sampling-steps', 4, '--output', config)
            trained = temp/mode/'train'
            base = ['-m', 'umdiff', 'train', '--config', config, '--synthetic',
                    '--batch-size', batch, '--device', 'cpu']
            run(relocated, *base, '--steps', 2, '--output', trained)
            value = sample(relocated, trained/'checkpoint.pt', temp/f'{mode}.pt', batch)
            expected = [batch, *map(int, shape.split(','))]
            assert list(value.shape) == expected
            checks.append({'name': mode+'_cli_train_sample', 'pass': True,
                           'shape': expected, 'updates': 2})
            if mode == 'image':
                resumed, full = temp/'resumed', temp/'full'
                run(relocated, *base, '--steps', 2, '--resume', trained/'checkpoint.pt',
                    '--output', resumed)
                run(relocated, *base, '--steps', 4, '--output', full)
                a, b = load(resumed/'checkpoint.pt'), load(full/'checkpoint.pt')
                assert a['step'] == b['step'] == 4
                for key in ('model', 'ema', 'optimizer'):
                    equal_tree(a[key], b[key])
                sa = sample(relocated, resumed/'checkpoint.pt', temp/'resumed.pt')
                sb = sample(relocated, full/'checkpoint.pt', temp/'full.pt')
                equal_tree(sa, sb)
                checks.append({'name': 'resume_2_plus_2_equals_4', 'pass': True,
                               'comparison': 'model/EMA/optimizer/samples torch.equal'})

        recipes = {
            'ddpm': 'rdd_ddpm_xl_300_300_400_ddpm_cfg1p0_fresh.json',
            'fm': 'rdd_fm_xl_v2_euler_cfg1p5_noise_filling.json',
        }
        for mode, name in recipes.items():
            cfg = json.loads((ROOT/'configs/rdd_reproduction'/name).read_text())
            cfg['model']['args'].update(hidden_size=32, depth=2, num_heads=4)
            cfg['training'].pop('input_transform')
            cfg['training']['optimizer']['accumulation_steps'] = 1
            if mode == 'fm':
                cfg['process']['steps'] = [3, 3, 3]
            else:
                cfg['sampler']['args']['steps'] = [3, 3, 3]
            path = temp/f'native_{mode}.json'
            path.write_text(json.dumps(cfg))
            states, samples = [], []
            for label, root in [('canonical', ROOT.parent), ('relocated', relocated)]:
                dest = temp/f'{mode}_{label}'
                run(root, '-m', 'umdiff', 'train', '--config', path, '--synthetic',
                    '--steps', 2, '--batch-size', 2, '--device', 'cpu', '--output', dest)
                states.append(load(dest/'checkpoint.pt'))
                samples.append(sample(root, dest/'checkpoint.pt', dest/'sample.pt', labels=True))
            for key in ('model', 'ema', 'optimizer'):
                equal_tree(states[0][key], states[1][key])
            equal_tree(samples[0], samples[1])
            digest = hashlib.sha256(samples[0].contiguous().numpy().tobytes()).hexdigest()
            checks.append({'name': 'native_'+mode+'_relocation', 'pass': True,
                           'comparison': 'model/EMA/optimizer/samples torch.equal',
                           'sample_sha256': digest, 'shape': list(samples[0].shape),
                           'configuration': 'random width32 depth2; grids32/16/8; steps3/3/3'})

        for family in ('main', 'controlled'):
            for path in sorted((ROOT/'configs/comparison_imagenet256'/family).glob('*.json')):
                result = json.loads(run(ROOT.parent, '-m', 'umdiff.experiment', 'plan', '--config', path))
                checks.append({'name': family+'/'+path.stem, 'pass': True,
                               'status': result['status']})

    report = {'all_pass': all(row['pass'] for row in checks), 'checks': checks,
              'commands': commands, 'temporary_payloads_removed': True,
              'gpu_used': False, 'model_or_dataset_downloads': False,
              'scope': 'Relocation/CLI smoke; not a new author-parity or formal quality certificate'}
    (output/'CLI_RELOCATION.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'all_pass': report['all_pass'], 'checks': len(checks),
                      'commands': len(commands), 'report': str(output/'CLI_RELOCATION.json')}))


if __name__ == '__main__':
    main()
