"""CPU negative tests against real committed 14-clip caches, without mutations."""
from typing import Any
import argparse
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

from preprocess_dataset import merge


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    w = Path(__file__).resolve().parents[3]
    checks = {}
    with tempfile.TemporaryDirectory(prefix='rdd-cache-regression-') as name:
        root = Path(name)
        for family in ('wan', 'pf'):
            original = w / 'fair_data_20261006' / ('fixture14_' + family + '_workers')
            inp = root / family
            inp.mkdir()
            shutil.copytree(original / 'rank0', inp / 'rank0')
            args = SimpleNamespace(input=inp,
                                   manifest=w / 'assets_20261006/small_mixkit/manifest14.jsonl',
                                   family=family,
                                   output=root / (family + '_good'),
                                   allow_partial=False)
            merge(args)
            checks[family + '_actual_merge'] = True
            markers = sorted((inp / 'rank0/clips').glob('*/complete.json'))
            marker = markers[0]
            saved = marker.read_text()
            cases: dict[str, Any] = {
                'caption_changed':
                lambda row: row['identity'].__setitem__('caption', 'changed'),
                'split_changed':
                lambda row: row['identity'].__setitem__(
                    'split', 'validation' if row['identity']['split'] == 'train' else 'train'),
                'cache_hash_changed':
                lambda row: row['file_sha256'].__setitem__(next(iter(row['file_sha256'])), '0' * 64)
            }
            for label, change in cases.items():
                row = json.loads(saved)
                change(row)
                marker.write_text(json.dumps(row))
                args.output = root / (family + '_' + label)
                try:
                    merge(args)
                except ValueError:
                    checks[family + '_' + label] = True
                else:
                    raise AssertionError('Invalid cache accepted: ' + label)
                marker.write_text(saved)
            marker.rename(marker.with_suffix('.hidden'))
            args.output = root / (family + '_missing')
            try:
                merge(args)
            except ValueError:
                checks[family + '_missing_rejected'] = True
            else:
                raise AssertionError('Incomplete formal cache accepted')
            marker.with_suffix('.hidden').rename(marker)
    import torch
    assert not torch.cuda.is_initialized()
    a.output.write_text(
        json.dumps(dict(status='passed', checks=checks, cpu_only=True, original_caches_unchanged=True), indent=2))


if __name__ == '__main__':
    main()
