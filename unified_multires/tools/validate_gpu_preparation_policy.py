#!/usr/bin/env python3
"""CPU-only policy tests for the preparation wrapper; no model tests."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/gpu_preparation_20261010/POLICY_CHECK.json'


def main():
    with tempfile.TemporaryDirectory(prefix='umdiff_gpu_plan_policy_') as directory:
        sandbox = Path(directory)
        # A child process audit hook makes an accidental Torch/CUDA import,
        # networking or nested training subprocess fail this preparation test.
        (sandbox / 'sitecustomize.py').write_text('''import os, sys
def forbidden(event, args):
    if event == 'import' and args[0].split('.')[0] in {'torch', 'pynvml', 'cupy'}:
        raise RuntimeError('Preparation imported a GPU/model runtime: ' + args[0])
    if event in {'subprocess.Popen', 'os.system', 'os.posix_spawn', 'socket.connect', 'socket.bind'}:
        raise RuntimeError('Preparation attempted process/network activity: ' + event)
    if event == 'open' and isinstance(args[0], str) and (args[0].startswith('/dev/nvidia') or args[0].startswith('/proc/driver/nvidia')):
        raise RuntimeError('Preparation accessed a CUDA device')
sys.addaudithook(forbidden)
if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise RuntimeError('Wrapper did not enforce CPU-only environment')
''')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='7', UMDIFF_PYTHON=sys.executable,
                   PYTHONPATH=str(sandbox))
        command = ['bash', str(ROOT / 'prepare_gpu_checks.sh')]
        good = subprocess.run(command + ['--action', 'check'], env=env, text=True,
                              capture_output=True, timeout=30)
        if good.returncode or good.stderr:
            raise RuntimeError('Guarded CPU audit failed: ' + good.stderr + good.stdout)
        result = json.loads(good.stdout)
        if result['gpu_used'] is not False:
            raise RuntimeError('Preparation claims GPU use')
        forbidden = subprocess.run(command + ['--action', 'run'], env=env, text=True,
                                   capture_output=True, timeout=30)
        if forbidden.returncode != 2 or 'invalid choice' not in forbidden.stderr:
            raise RuntimeError('Unexpectedly accepted an execution action')
        report = dict(status='passed', device='cpu', gpu_used=False, model_test_executed=False,
            cases=['Caller CUDA_VISIBLE_DEVICES=7 overridden to empty before Python startup.',
                   'Plan audit passes with Torch/CUDA imports, nested subprocesses and network connections forbidden.',
                   'GPU execution action run rejected by the CLI.'],
            audit=result, scope='CPU preparation policy only; no CUDA numerical certification.')
        OUT.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
