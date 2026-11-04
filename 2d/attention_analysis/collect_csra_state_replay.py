"""Matched fresh teacher/snapshot-student trajectories and same-state targets.

Original8/16 are frozen and bitwise matched. Diagnostic calls consume no RNG.
The snapshot is fixed2400; this is a one-round replay pilot, not online training.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import torch

from csra_state_replay_data import PROTOCOL, ROOT, PARENT, PARENT_SHA, BASE_SHA, DATA_SEED, TIMES, check_record
from sample_csra_neighbor_quality import checked_loader
from sample_csra_prefix_preview import PrefixRollout
from sample_csra_adapter_preview import sample_rdd, manual_forward, relative_l2
from train_rdd_csra_adapter import load_frozen_teacher, atomic_save
from probe_csra_front_keep import BASE, file_sha, write_json


class ReplayCollector:
    def __init__(self, teacher, student, role):
        if role not in ('teacher', 'student'):
            raise ValueError(role)
        self.teacher, self.student, self.role = teacher, student, role
        self.states, self.rows, self.endpoints = [], [], {}
        self.source, self.labels, self.fine_calls = None, None, 0

    @torch.inference_mode()
    def forward(self, x, t, y):
        r, tv = x.shape[-1], int(t[0])
        if not torch.all(t == tv):
            raise ValueError('Mixed timestep in rollout')
        if self.labels is None:
            self.labels = y.cpu().clone()
        elif not torch.equal(self.labels, y.cpu()):
            raise ValueError('Changed labels')
        cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state(x.device)
        if r != 32:
            capture = {13} if r == 16 and tv == 300 else set()
            out, trace = manual_forward(self.teacher, x, t, y, multires=True, capture_layers=capture)
            if capture:
                self.source = trace[13]['output'].detach()
        else:
            self.fine_calls += 1
            if self.source is None:
                raise ValueError('No true coarse source')
            selected = tv in TIMES
            pred = target = None
            if self.role == 'student' or selected:
                pred = self.student(dict(x=x, t=t, labels=y, coarse={13: self.source}))
            if self.role == 'teacher' or selected:
                target, _ = manual_forward(self.teacher, x, t, y, multires=True)
            out = target if self.role == 'teacher' else pred
            if selected:
                self.states.append(dict(x=x.cpu(), t=t.cpu(), target=target.cpu()))
                err = (pred[:, :4].float()-target[:, :4].float()).square().mean((1, 2, 3))
                self.rows.extend(dict(image=i, t=tv, role=self.role, main_mse=float(v)) for i, v in enumerate(err))
        if not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(gpu_rng, torch.cuda.get_rng_state(x.device)):
            raise RuntimeError('Extra prediction changed sampling RNG')
        if (r, tv) in ((8, 600), (16, 300)):
            self.endpoints[r] = (x.cpu(), out.cpu())
        return out

    def payload(self):
        if self.fine_calls != 90 or [int(d['t'][0]) for d in self.states] != list(TIMES):
            raise ValueError('Missing/duplicated fine states')
        return {k: torch.stack([d[k] for d in self.states]) for k in ('x', 't', 'target')}


@torch.inference_mode()
def run(args):
    torch.set_num_threads(2)
    dev = torch.device('cuda:0')
    if torch.cuda.mem_get_info(dev)[0] < 18*2**30 or shutil.disk_usage('.').free < 20*2**30:
        raise RuntimeError('Need18GiB GPU and20GiB disk; do not delete/stop other jobs')
    torch.cuda.set_per_process_memory_fraction(.18, dev)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if file_sha(BASE) != BASE_SHA or file_sha(PARENT) != PARENT_SHA:
        raise ValueError('Immutable parent/base mismatch')
    out = ROOT/('smoke_data' if args.smoke else 'data')/f'shard_{args.shard}'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'manifest.json').exists():
        raise FileExistsError('Read completed data, never regenerate silently')
    teacher = load_frozen_teacher(BASE, dev, 'DiT-L/2', 'rdd')
    student, config = checked_loader('aligned', dev, PARENT, 6, 2400)
    files, start, parity = [], time.perf_counter(), None
    old_seeds = set()
    for mp in Path('eval_results/csra_rollout_data4k_20260912').glob('shard_*/manifest.json'):
        old_seeds.update(f['seed'] for f in json.loads(mp.read_text())['files'])
    count = 1 if args.smoke else 17
    for batch in range(count):
        split = 'train' if batch < 16 else 'validation'
        n = 4 if args.smoke else (64 if split == 'train' else 32)
        seed = DATA_SEED + args.shard*1000 + batch + (500000 if args.smoke else 0)
        if seed in old_seeds:
            raise ValueError('Seed overlap with prior train/dev corpus')
        path = out/f'{split}_{batch:03d}.pt'
        if path.exists():
            data = torch.load(path, map_location='cpu', weights_only=True)
            check_record(data, split, seed, n)
            if batch == 0:
                parity = data['collector_parity_max_abs']
            files.append(dict(name=path.name, split=split, seed=seed, images=n, sha256=file_sha(path)))
            continue
        records, probes, latents = {}, {}, {}
        for role in ('teacher', 'student'):
            probe = ReplayCollector(teacher, student, role)
            latents[role], _ = sample_rdd(probe, n, [80,80,90], dev, seed)
            records[role], probes[role] = probe.payload(), probe
        for r in (8,16):
            for a, b in zip(probes['teacher'].endpoints[r], probes['student'].endpoints[r]):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
        torch.testing.assert_close(probes['teacher'].source, probes['student'].source, atol=0, rtol=0)
        if batch == 0:
            for role in ('teacher', 'student'):
                wrapper = PrefixRollout(teacher, student if role == 'student' else None)
                plain, _ = sample_rdd(wrapper, n, [80,80,90], dev, seed)
                torch.testing.assert_close(plain, latents[role], atol=0, rtol=0)
                del plain
            parity = 0.
        data = dict(protocol=PROTOCOL, parent_sha256=PARENT_SHA, base_sha256=BASE_SHA,
                    seed=seed, split=split, labels=probes['teacher'].labels,
                    coarse=probes['teacher'].source.cpu(), coarse_endpoint_max_abs=0., fine_calls=90,
                    rng_unchanged=True, collector_parity_max_abs=parity, **records,
                    rows=probes['teacher'].rows+probes['student'].rows,
                    final_relative_latent_l2=relative_l2(latents['student'], latents['teacher']))
        check_record(data, split, seed, n)
        if shutil.disk_usage('.').free < 18*2**30:
            raise RuntimeError('Disk headroom reduced; stop only this collector')
        atomic_save(data, path)
        files.append(dict(name=path.name, split=split, seed=seed, images=n, sha256=file_sha(path)))
        print(json.dumps(dict(event='paired_data_saved', shard=args.shard, batch=batch,
                              images=n, split=split, elapsed=time.perf_counter()-start,
                              main_mse={role:sum(r['main_mse'] for r in probes[role].rows)/(n*8)
                                        for role in ('teacher','student')},
                              peak_gib=torch.cuda.max_memory_allocated()/2**30)), flush=True)
        del probes, records, latents, data, probe
    write_json(out/'manifest.json', dict(complete=True, smoke=args.smoke, protocol=PROTOCOL,
        shard=args.shard, files=files, base_sha256=BASE_SHA, parent_sha256=PARENT_SHA,
        collector_parity_max_abs=parity, config=config,
        note='Fixed parent2400 student snapshot; same-state teacher queries; no continuous on-policy refresh.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shard', type=int, choices=range(4), required=True)
    p.add_argument('--smoke', action='store_true')
    run(p.parse_args())
