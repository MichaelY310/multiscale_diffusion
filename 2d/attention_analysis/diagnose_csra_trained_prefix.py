#!/usr/bin/env python3
"""Read-only paired trained-prefix diagnosis on fixed development teacher states.

Matched trained-zero is the control. Removing/rolling the aligned model's source
is only a dependency intervention, NOT a fairly retrained alternative.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
import torch.nn.functional as F
from csra_prefix_student import PrefixLiftStudent, check_prefix_initializer
from probe_csra_prefix_information import BASE, TIMES, records, load_record, write_json
from train_rdd_csra_adapter import load_frozen_teacher


def output_bands(x):
    """Orthogonal 2x2-cell average and within-cell difference, NOT ideal FFT bands."""
    if x.ndim != 4 or x.shape[-1] % 2 or x.shape[-2] % 2:
        raise ValueError('Expected BCHW with even spatial dimensions')
    low = F.avg_pool2d(x.float(), 2).repeat_interleave(2, -2).repeat_interleave(2, -1)
    return low, x.float() - low


def scores(pred, target):
    error = pred[:, :4].float() - target[:, :4].float()
    low, high = output_bands(error)
    tl, th = output_bands(target[:, :4])
    mean = lambda x: x.square().mean((1, 2, 3))
    result = dict(main_mse=mean(error), low_mse=mean(low), high_mse=mean(high),
                  low_target_power=mean(tl), high_target_power=mean(th),
                  variance_mse=mean(pred[:, 4:].float()-target[:, 4:].float()))
    torch.testing.assert_close(result['main_mse'], result['low_mse']+result['high_mse'],
                               atol=1e-6, rtol=1e-5)
    return result


def summarize(rows):
    result = {}
    keys = ('main_mse', 'low_mse', 'high_mse', 'low_target_power',
            'high_target_power', 'variance_mse')
    for mode in sorted({r['mode'] for r in rows}):
        selected = [r for r in rows if r['mode'] == mode]
        result[mode] = {}
        for t in ['all'] + TIMES:
            group = selected if t == 'all' else [r for r in selected if r['t'] == t]
            if not group:
                raise ValueError('Missing time bin')
            result[mode][str(t)] = {k: sum(r[k] for r in group)/len(group) for k in keys}
            result[mode][str(t)]['states'] = len(group)
    return result


@torch.inference_mode()
def run(args):
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    if torch.cuda.mem_get_info(device)[0] < 12*2**30:
        raise RuntimeError('Need 12GiB headroom; do not stop another job')
    root, out = Path(args.capture_root), Path(args.output_root)
    target = out / f'shard_{args.shard}.json'
    if target.exists():
        raise FileExistsError('Read existing diagnostic; do not overwrite/repeat it')
    entries, fingerprints, counts = records(root)
    entries = [e for e in entries if e[1]['role'] == 'validation'
               and e[0].parent.name == f'shard_{args.shard}']
    if len(entries) != 4:
        raise ValueError('Expected four development batches per shard')
    torch.backends.cuda.matmul.allow_tf32 = False
    rows, configs = [], {}
    start = time.perf_counter()
    for arm in ('aligned', 'zero'):
        path = Path(args.checkpoint_template.format(mode=arm))
        ck = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
        if (ck['format'] != 'csra_prefix_linear_v7' or ck['args']['coarse_source'] != arm
            or Path(ck['base_checkpoint']).resolve() != BASE.resolve()
            or ck['prefix_depth'] != args.prefix_depth or ck['source_layer'] != 13):
            raise ValueError('Wrong trained prefix checkpoint')
        ip = Path(ck['args']['initializer'])
        if hashlib.sha256(ip.read_bytes()).hexdigest() != ck['initializer_sha256']:
            raise ValueError('Initializer identity mismatch')
        init = torch.load(ip, map_location='cpu', weights_only=True)
        check_prefix_initializer(init, args.prefix_depth)
        model = PrefixLiftStudent(load_frozen_teacher(BASE, device, 'DiT-L/2', 'rdd'),
            init['fits']['fine_only'], arm, prefix_depth=args.prefix_depth,
            checkpoint_blocks=False).to(device)
        model.load_state_dict(ck['student'])
        model.eval().requires_grad_(False)
        configs[arm] = dict(checkpoint=str(path), step=ck['step'], args=ck['args'],
                            initializer_sha256=ck['initializer_sha256'])
        # mmap checkpoint/optimizer stay on CPU; discard before processing states.
        del ck, init
        for entry in entries:
            rec = load_record(entry)
            ref = torch.load(Path(args.data_root)/entry[0].parent.name/entry[0].name,
                             map_location='cpu', weights_only=True)
            if not torch.equal(ref['labels'], rec['labels']):
                raise ValueError('Label mismatch')
            source = rec['coarse'].to(device)
            modes = ('aligned', 'aligned_source_zero', 'aligned_source_roll') if arm == 'aligned' else ('trained_zero',)
            rng = torch.cuda.get_rng_state(device)
            for ti, tv in enumerate(TIMES):
                batch = dict(x=ref['x'][ti].to(device), t=ref['t'][ti].to(device),
                             labels=ref['labels'].to(device), coarse={13:source})
                if not torch.all(batch['t'] == tv):
                    raise ValueError('Mixed time bin')
                target_pred = ref['target'][ti].to(device)
                for mode in modes:
                    if mode == 'aligned_source_zero':
                        batch['coarse'] = {13:torch.zeros_like(source)}
                    elif mode == 'aligned_source_roll':
                        grid = source.reshape(len(source), 8, 8, -1)
                        batch['coarse'] = {13:torch.roll(grid, (1, 1), (1, 2)).reshape_as(source)}
                    else:
                        batch['coarse'] = {13:source}
                    pred = model(batch)
                    measured = scores(pred, target_pred)
                    if not all(torch.isfinite(v).all() for v in measured.values()):
                        raise ValueError('Nonfinite result')
                    rows.extend(dict(seed=rec['seed'], image=j, label=int(rec['labels'][j]),
                                     t=tv, mode=mode,
                                     **{k:float(v[j]) for k,v in measured.items()})
                                for j in range(len(source)))
            if not torch.equal(rng, torch.cuda.get_rng_state(device)):
                raise RuntimeError('Read-only diagnostic consumed CUDA RNG')
            print(json.dumps(dict(arm=arm, seed=rec['seed'], seconds=time.perf_counter()-start)), flush=True)
        del model
        torch.cuda.empty_cache()
    if len(rows) != 32*8*4:
        raise ValueError('Incomplete diagnostic')
    write_json(target, dict(complete=True, shard=args.shard, rows=rows, summaries=summarize(rows),
        configs=configs, fingerprints=fingerprints, counts=counts, prefix_depth=args.prefix_depth,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        warning='Development teacher states, not student rollout or FID. Output epsilon and variance errors; 2x2 average/difference are orthogonal cell subspaces, not ideal Fourier low/high pass. Only aligned versus independently trained_zero is a matched training comparison. Source-zero/roll are OOD dependency interventions. No latent-quality/speed claim. Hidden coordinate MSE after fine backbone training is deliberately not used.'))


def aggregate(args):
    out = Path(args.output_root)
    all_rows, shards = [], []
    for i in range(4):
        d = json.loads((out/f'shard_{i}.json').read_text())
        if not d['complete'] or d['shard'] != i:
            raise ValueError('Wrong shard')
        if shards and any(d[k] != shards[0][k] for k in ('configs', 'fingerprints', 'prefix_depth')):
            raise ValueError('Mixed checkpoints/corpus/depth')
        all_rows.extend(d['rows']); shards.append(d)
    if len({(r['seed'],r['image'],r['t'],r['mode']) for r in all_rows}) != 128*8*4:
        raise ValueError('Duplicates/missing states')
    write_json(out/'metrics.json', dict(complete=True, rows=all_rows, summaries=summarize(all_rows),
        configs=shards[0]['configs'], fingerprints=shards[0]['fingerprints'],
        prefix_depth=shards[0]['prefix_depth'], warning=shards[0]['warning'],
        states_per_mode=1024, images=128, max_peak_allocated_gib=max(s['peak_allocated_gib'] for s in shards)))
    print(json.dumps(summarize(all_rows), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('run', 'aggregate'))
    p.add_argument('--shard', type=int, choices=range(4))
    p.add_argument('--data-root', default='eval_results/csra_rollout_data4k_20260912')
    p.add_argument('--capture-root', default='eval_results/csra_prefix8_information_20260913')
    p.add_argument('--checkpoint-template', default='checkpoints/RDD-CSRA-prefix8-{mode}-L400K-20260913/best.pt')
    p.add_argument('--prefix-depth', type=int, default=8)
    p.add_argument('--output-root', default='eval_results/csra_prefix8_trained_diagnostic_20260913')
    a = p.parse_args()
    if a.action == 'run':
        if a.shard is None: p.error('--shard required')
        run(a)
    else: aggregate(a)
