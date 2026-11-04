#!/usr/bin/env python3
"""Read-only ridge probe of coarse information BEFORE expensive fine blocks.

No original model updates, no accelerated sampling, no quality or latency claim.
Four feature capture shards then one globally pooled fit/calibration/validation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from csra_prefix_probe_math import MODES, Moments, select_records, features, split_error, fit_ridge, predict
from train_rdd_csra_adapter import atomic_save, load_frozen_teacher
from probe_rdd_vs_dit_block_reuse import manual_forward

VERSION = 'prefix_information_v1'
BASE = Path('checkpoints/RDD-Unified-L/0400000.pt')
ROOT = Path('eval_results/csra_prefix_information_20260913')
TIMES = [299, 259, 215, 171, 128, 84, 40, 0]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def gpu():
    dev = torch.device('cuda:0')
    if torch.cuda.mem_get_info(dev)[0] < 8*2**30:
        raise RuntimeError('Need8GiB free; never stop another process')
    return dev


@torch.inference_mode()
def capture(args):
    dev = gpu()
    folder = Path(args.data_root)/f'shard_{args.shard}'
    mp = folder/'manifest.json'
    digest = hashlib.sha256(mp.read_bytes()).hexdigest()
    manifest = json.loads(mp.read_text())
    hidden_folder = Path(args.hidden_root)/folder.name
    side = json.loads((hidden_folder/'manifest.json').read_text())
    if (not manifest['complete'] or manifest['parity_max_abs'] != 0
        or not side['complete'] or side['feature'] != 'output'
        or side['source_manifest_sha256'] != digest):
        raise ValueError('Corpus/source manifest or parity mismatch')
    if Path(manifest['base_checkpoint']).resolve() != BASE.resolve():
        raise ValueError('Wrong base checkpoint')
    selected = select_records(manifest)
    out = Path(args.output_root)/folder.name
    out.mkdir(parents=True, exist_ok=True)
    target_layer = args.prefix_depth - 1
    if not 1 <= args.prefix_depth < 24:
        raise ValueError('Prefix depth must leave at least one fine block')
    if (out/'manifest.json').exists():
        existing = json.loads((out/'manifest.json').read_text())
        if (existing['complete'] and existing['version'] == VERSION and existing['source_sha256'] == digest
            and existing.get('prefix_depth',14) == args.prefix_depth):
            print(f'{folder.name} already complete', flush=True)
            return
        raise ValueError('Refuse to overwrite another feature capture')
    model = load_frozen_teacher(BASE, dev, 'DiT-L/2', 'rdd')
    start = time.time()
    for i, item in enumerate(selected):
        name = item['name']
        if Path(name).name != name or item['images'] != 8 or item['states_per_image'] != 8:
            raise ValueError('Unexpected record layout')
        ref = torch.load(folder/name, map_location='cpu', weights_only=True)
        cached = torch.load(hidden_folder/name, map_location='cpu', weights_only=True)
        if (cached['seed'] != item['seed'] or cached['source_manifest_sha256'] != digest
            or cached['feature'] != 'output' or cached['parity_max_abs'] != 0
            or cached['split'] != item['split'] or not torch.equal(cached['labels'], ref['labels'])):
            raise ValueError('Hidden-source mismatch')
        if cached['coarse'][13].shape != (8,64,1024):
            raise ValueError('Expected native coarse h_after13, not upsampled delta')
        rng = torch.cuda.get_rng_state(dev)
        h0s, targets, conditions = [], [], []
        labels = ref['labels'].to(dev)
        if [int(t[0]) for t in ref['t']] != TIMES:
            raise ValueError('Unexpected fine times')
        for k in range(8):
            x, t, y = (ref[key][k].to(dev) for key in ('x','t','target'))
            if not torch.all(t == TIMES[k]):
                raise ValueError('Mixed per-image time')
            pred, trace = manual_forward(model, x, t, labels, multires=True, capture_layers={0,target_layer})
            torch.testing.assert_close(pred, y, atol=0, rtol=0)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                h0 = model.x_embedder(x)
                c = model.t_embedder(t) + model.y_embedder(labels, False)
            torch.testing.assert_close(h0, trace[0]['input'], atol=0, rtol=0)
            h0s.append(h0.cpu())
            targets.append(trace[target_layer]['output'].cpu())
            conditions.append(c.cpu())
        if not torch.equal(rng, torch.cuda.get_rng_state(dev)):
            raise RuntimeError('Capture changed CUDA RNG')
        record = {'version': VERSION, 'seed':item['seed'], 'role':item['role'],
                  'source_sha256':digest, 'labels':ref['labels'], 'times':TIMES,
                  'h0':torch.stack(h0s), 'target':torch.stack(targets),
                  'condition':torch.stack(conditions), 'coarse':cached['coarse'][13],
                  'parity_max_abs':0., 'prefix_depth':args.prefix_depth, 'source_layer':13}
        for key in ('h0','target','condition','coarse'):
            if not torch.isfinite(record[key]).all(): raise ValueError('Nonfinite features')
        if (out/name).exists():
            old = torch.load(out/name, map_location='cpu', weights_only=True)
            if old.get('prefix_depth',14) != args.prefix_depth:
                raise ValueError('Existing feature target belongs to another prefix')
            for key in ('h0','target','condition','coarse'):
                torch.testing.assert_close(old[key], record[key], atol=0, rtol=0)
        atomic_save(record, out/name)
        print(json.dumps({'shard':args.shard,'files':i+1,'total':len(selected),
                          'role':item['role'],'seconds':time.time()-start}), flush=True)
    write_json(out/'manifest.json', {'complete':True, 'version':VERSION,
        'source_sha256':digest, 'files':selected, 'parity_max_abs':0.,
        'source':'native16 t300 h_after13', 'target':f'fine h_after{target_layer}, zero-based',
        'prefix_depth':args.prefix_depth, 'source_layer':13,
        'peak_memory_gib':torch.cuda.max_memory_allocated()/2**30})


def records(root):
    result = []
    fingerprints = {}
    depths = set()
    for shard in range(4):
        folder = root/f'shard_{shard}'
        mp = folder/'manifest.json'
        manifest = json.loads(mp.read_text())
        if not manifest['complete'] or manifest['version'] != VERSION or manifest['parity_max_abs'] != 0:
            raise ValueError('Incomplete or incompatible capture')
        depths.add(manifest.get('prefix_depth',14))
        if manifest.get('source_layer',13) != 13:
            raise ValueError('Wrong source layer')
        fingerprints[folder.name] = hashlib.sha256(mp.read_bytes()).hexdigest()
        for item in manifest['files']:
            if Path(item['name']).name != item['name']: raise ValueError('Unsafe filename')
            result.append((folder/item['name'], {**item,'prefix_depth':manifest.get('prefix_depth',14)}, manifest['source_sha256']))
    if len(depths) != 1:
        raise ValueError('Mixed prefix depths across capture shards')
    if len({r[1]['seed'] for r in result}) != len(result):
        raise ValueError('Seed collision across shards/splits')
    counts = {role:sum(r[1]['images'] for r in result if r[1]['role'] == role)
              for role in ('fit','calibration','validation')}
    if counts != {'fit':192, 'calibration':64, 'validation':128}:
        raise ValueError(f'Wrong split counts: {counts}')
    return result, fingerprints, counts


def load_record(entry):
    path, item, digest = entry
    rec = torch.load(path, map_location='cpu', weights_only=True)
    if (rec['seed'] != item['seed'] or rec['role'] != item['role'] or rec['version'] != VERSION
        or rec['source_sha256'] != digest or rec['parity_max_abs'] != 0 or rec['times'] != TIMES
        or rec.get('prefix_depth',14) != item.get('prefix_depth',14) or rec.get('source_layer',13) != 13):
        raise ValueError('Feature record does not match selected split')
    return rec


def on_gpu(fit, dev):
    return {k:v.to(device=dev, dtype=torch.float32) if torch.is_tensor(v) else v for k,v in fit.items()}


@torch.inference_mode()
def evaluate(entries, role, fits, dev, save_rows=False):
    sums = {key:0. for key in fits}
    n = 0
    rows = []
    active = {k:on_gpu(v, dev) for k,v in fits.items()}
    for entry in entries:
        if entry[1]['role'] != role: continue
        rec = load_record(entry)
        source = rec['coarse'].to(dev).float()
        for ti,t in enumerate(TIMES):
            h0, target, c = (rec[key][ti].to(dev).float() for key in ('h0','target','condition'))
            n += len(h0)
            for mode in MODES:
                x = features(h0, c, source, mode)
                for key, fit in active.items():
                    if key.split('/')[0] != mode: continue
                    pred_h = h0 + predict(x, fit).reshape_as(h0)
                    error = pred_h-target
                    mse = error.square().mean((1,2))
                    sums[key] += float(mse.double().sum())
                    if save_rows:
                        low, high = split_error(error)
                        lo, hi = low.square().mean((1,2)), high.square().mean((1,2))
                        torch.testing.assert_close(mse, lo+hi, atol=1e-6, rtol=1e-5)
                        power = target.square().mean((1,2))
                        for j in range(len(h0)):
                            rows.append({'seed':rec['seed'],'image':j,'label':int(rec['labels'][j]),
                                'mode':mode,'t':t,'mse':float(mse[j]),'low_mse':float(lo[j]),
                                'high_mse':float(hi[j]),'target_power':float(power[j])})
    return {k:v/n for k,v in sums.items()}, rows


@torch.inference_mode()
def fit_and_evaluate(args):
    root = Path(args.output_root)
    if (root/'metrics.json').exists():
        if json.loads((root/'metrics.json').read_text())['complete']: return
        raise ValueError('Existing incomplete metrics')
    dev = gpu()
    # Disable TF32 for regression products, not teacher feature collection.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(4)
    entries, fingerprints, counts = records(root)
    prefix_depth = json.loads((root/'shard_0'/'manifest.json').read_text()).get('prefix_depth',14)
    stats_path = root/'fit_statistics.pt'
    if stats_path.exists():
        stored = torch.load(stats_path, map_location='cpu', weights_only=True)
        if stored['fingerprints'] != fingerprints or stored['version'] != VERSION:
            raise ValueError('Regression statistics belong to another corpus')
        stats = stored['stats']
    else:
        moments = {mode:Moments(2048 if mode == 'fine_only' else 3072,1024,dev) for mode in MODES}
        for i,entry in enumerate(entries):
            if entry[1]['role'] != 'fit': continue
            rec = load_record(entry)
            source = rec['coarse'].to(dev).float()
            for ti in range(8):
                h0, target, c = (rec[key][ti].to(dev).float() for key in ('h0','target','condition'))
                delta = (target-h0).flatten(0,1)
                for mode in MODES:
                    moments[mode].add(features(h0,c,source,mode), delta)
            print(json.dumps({'fit_file':i,'seed':rec['seed']}),flush=True)
        stats = {k:v.cpu() for k,v in moments.items()}
        del moments
        atomic_save({'version':VERSION,'fingerprints':fingerprints,'stats':stats}, stats_path)
    all_fits = {}
    for mode in MODES:
        for ridge in (.01,.1,1.):
            key = f'{mode}/{ridge}'
            all_fits[key] = fit_ridge(stats[mode],ridge)
            print(json.dumps({'fit':key,'solve_residual':all_fits[key]['solve_relative_residual']}),flush=True)
    del stats
    calibration, _ = evaluate(entries,'calibration',all_fits,dev)
    chosen = {mode:min((k for k in calibration if k.split('/')[0] == mode),key=calibration.get) for mode in MODES}
    fits = {mode:all_fits[key] for mode,key in chosen.items()}
    atomic_save({'version':VERSION,'fingerprints':fingerprints,'fits':fits,'chosen':chosen,
                 'calibration_mse':calibration,'prefix_depth':prefix_depth,'source_layer':13},root/'selected_ridge.pt')
    del all_fits
    print(json.dumps({'selected':chosen,'calibration_mse':calibration}),flush=True)
    validation, rows = evaluate(entries,'validation',fits,dev,save_rows=True)
    by_time = {mode:{str(t):sum(r['mse'] for r in rows if r['mode']==mode and r['t']==t)/128
                    for t in TIMES} for mode in MODES}
    write_json(root/'metrics.json',{'complete':True,'version':VERSION,'images':counts,
        'fingerprints':fingerprints,'source':'native16 t300 h_after13','target':f'fine h_after{prefix_depth-1}',
        'prefix_depth':prefix_depth,'source_layer':13,
        'selected':chosen,'calibration_mse':calibration,'validation_mse':validation,
        'validation_by_time':by_time,'rows':rows,'parity_max_abs':0.,
        'solve_relative_residual':{k:v['solve_relative_residual'] for k,v in fits.items()},
        'probe_parameters':{mode:(2048 if mode=='fine_only' else 3072)*1024+1024 for mode in MODES},
        'peak_memory_gib':torch.cuda.max_memory_allocated()/2**30,
        'warning':f'Read-only linear information probe, not quality/FID/latency or trained suffix. Uses full h0, time/class condition, optional existing coarse h_after13. Aligned adds1,048,576 linear weights versus fine_only. Roll/wrong/mean match parameter tensors but not effective rank; wrong image changes class and spatial content, mean retains own-image class/global information. Separate192fit/64calibration/128validation trajectories; validation repeatedly used in prior experiments, developmental only. Ridge chosen on calibration only. Fits delta from h0 to h_after{prefix_depth-1}, then adds h0. Other pyramidal models can also supply coarse features; no RDD-only proof.'})
    print(json.dumps({'complete':True,'validation_mse':validation,'by_time':by_time}),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action',choices=('capture','fit'))
    p.add_argument('--output-root',default=str(ROOT))
    p.add_argument('--data-root',default='eval_results/csra_rollout_data4k_20260912')
    p.add_argument('--hidden-root',default='eval_results/csra_hidden_sources_20260913')
    p.add_argument('--shard',type=int)
    p.add_argument('--prefix-depth',type=int,default=14)
    args = p.parse_args()
    if args.action == 'capture':
        if args.shard not in range(4): p.error('Capture requires shard0–3')
        capture(args)
    else:
        fit_and_evaluate(args)
