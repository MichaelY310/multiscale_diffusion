"""Fixed-snapshot student-state replay, NOT continuously refreshed on-policy KD.

Two arms differ only in training states. Same seeds/labels/t/cache and teacher
queries on the matching state. No inference architecture changes or new claims
of RDD exclusivity. Keep this protocol separate from immutable old checkpoints.
"""
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

PROTOCOL = 'csra_fixed2400_snapshot_replay400_v1'
ROOT = Path('eval_results/csra_state_replay_20260914')
PARENT = 'checkpoints/RDD-CSRA-neighbor-off-aligned-L400K-20260914/latest.pt'
PARENT_SHA = '510e6925feff22a4e24a78162331c433299122e4fabbe2de648fd8d4ca5fcadd'
BASE_SHA = 'cade56f41317b2d04045431c139b2dd0097346e6392142a39f189274187a0c9f'
SEED = 20289000
DATA_SEED = 20960914
QUALITY_SEED = 21060914
TIMES = (299, 259, 215, 171, 128, 84, 40, 0)


def role_for(index, epoch, arm):
    if arm not in ('teacher_only', 'mixed50'):
        raise ValueError('Unknown replay arm')
    return 'student' if arm == 'mixed50' and (index + epoch) % 2 else 'teacher'


def check_record(data, split, seed, count):
    if (data.get('protocol') != PROTOCOL or data.get('parent_sha256') != PARENT_SHA
            or data.get('base_sha256') != BASE_SHA or data.get('split') != split
            or data.get('seed') != seed or data.get('coarse_endpoint_max_abs') != 0
            or data.get('fine_calls') != 90 or not data.get('rng_unchanged')):
        raise ValueError('Wrong state replay provenance')
    if data['labels'].shape != (count,) or data['coarse'].shape != (count, 64, 1024):
        raise ValueError('Wrong labels/native coarse cache')
    for role in ('teacher', 'student'):
        rec = data[role]
        if rec['x'].shape != (8, count, 4, 32, 32) or rec['target'].shape != (8, count, 8, 32, 32):
            raise ValueError('Wrong matched state/teacher target shape')
        if rec['t'].shape != (8, count) or not torch.equal(rec['t'], torch.tensor(TIMES)[:, None].expand(-1, count)):
            raise ValueError('Wrong capture timesteps')
        if not torch.isfinite(rec['x']).all() or not torch.isfinite(rec['target']).all():
            raise ValueError('Nonfinite replay data')
    if not torch.equal(data['teacher']['x'][0], data['student']['x'][0]):
        raise ValueError('First fine state must match before any student action')


class PairedReplayDataset(Dataset):
    """Memory-map immutable shard files; paired index addresses the same image/t."""
    def __init__(self, root, split, arm='teacher_only', fixed_role=None):
        if split not in ('train', 'validation') or fixed_role not in (None, 'teacher', 'student'):
            raise ValueError('Bad split/role')
        self.arm, self.fixed_role, self.epoch = arm, fixed_role, 0
        role_for(0, 0, arm)
        self.records, self.addresses, self.seeds = [], [], set()
        manifests = sorted(Path(root).glob('shard_*/manifest.json'))
        if len(manifests) != 4:
            raise ValueError('Need all four completed data shards')
        for shard, path in enumerate(manifests):
            manifest = json.loads(path.read_text())
            if (manifest.get('protocol') != PROTOCOL or not manifest.get('complete')
                    or manifest.get('shard') != shard or manifest.get('parent_sha256') != PARENT_SHA
                    or manifest.get('base_sha256') != BASE_SHA
                    or manifest.get('collector_parity_max_abs') != 0):
                raise ValueError('Wrong or unverified shard')
            items = [f for f in manifest['files'] if f['split'] == split]
            if len(items) != (16 if split == 'train' else 1):
                raise ValueError('Wrong fixed split size')
            for item in items:
                if Path(item['name']).name != item['name'] or item['seed'] in self.seeds:
                    raise ValueError('Unsafe filename or duplicated seed')
                self.seeds.add(item['seed'])
                data = torch.load(path.parent/item['name'], mmap=True, map_location='cpu', weights_only=True)
                expected_n = 64 if split == 'train' else 32
                check_record(data, split, item['seed'], expected_n)
                ri = len(self.records)
                self.records.append(data)
                self.addresses.extend((ri, ti, i) for ti in range(8) for i in range(expected_n))
        self.images = sum(len(r['labels']) for r in self.records)

    def __len__(self):
        return len(self.addresses)

    def __getitem__(self, index):
        ri, ti, i = self.addresses[index]
        data = self.records[ri]
        role = self.fixed_role or role_for(index, self.epoch, self.arm)
        return dict(x=data[role]['x'][ti, i], t=data[role]['t'][ti, i],
                    target=data[role]['target'][ti, i], labels=data['labels'][i],
                    coarse={13: data['coarse'][i]}, index=torch.tensor(index),
                    state_role=torch.tensor(int(role == 'student')))
