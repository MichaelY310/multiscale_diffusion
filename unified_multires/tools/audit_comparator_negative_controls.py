#!/usr/bin/env python3
"""Ensure complex differences and signed zeros cannot become false exact claims."""
import argparse, hashlib, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tools'), str(ROOT), str(ROOT/'_deps')]
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
import conformance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        conformance.compare('imaginary_only', torch.tensor([1+2j]), torch.tensor([1+1j]), atol=0, rtol=0)
    except AssertionError:
        pass
    else:
        raise AssertionError('Imaginary difference was accepted')
    imaginary = conformance.CHECKS[-1]
    assert imaginary['max_abs'] == 1. and not imaginary['passed'] and not imaginary['bitwise']
    conformance.compare('signed_zero', torch.tensor([0.]), torch.tensor([-0.]), atol=0, rtol=0)
    zero = conformance.CHECKS[-1]
    assert zero['passed'] and zero['values_exact'] and not zero['bitwise']
    assert not torch.cuda.is_initialized()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(status='passed', checks=conformance.CHECKS, gpu_used=False,
        meaning='Intentional mismatches correctly detected; signed zero is allclose but fails the byte gate.',
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2)+'\n')
    print('Comparator negative controls passed')


if __name__ == '__main__':
    main()
