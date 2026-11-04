#!/usr/bin/env python3
"""Matched six-block replacement placement probe, NOT training or sampling.

front0 skips blocks0..5; front1 executes0 and skips1..6. Both execute18/24
fine blocks. Fits only a linear readout on training captures, chooses ridge on
calibration, and measures BOTH hidden error and the same original output head
after the unchanged suffix. New captures/checkpoints cannot initialize the old
prefix student silently: their version/target contracts are deliberately distinct.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import torch
from csra_prefix_probe_math import MODES, Moments, select_records, features, split_error, fit_ridge, predict
from probe_csra_prefix_information import TIMES, BASE, write_json, on_gpu
from train_rdd_csra_adapter import atomic_save, load_frozen_teacher
from probe_rdd_vs_dit_block_reuse import manual_forward

VERSION = 'front_keep_six_block_information_v1'
ROOT = 'eval_results/csra_front_keep6_information_20260914'
BASE_SHA = 'cade56f41317b2d04045431c139b2dd0097346e6392142a39f189274187a0c9f'


def layout(front):
    if type(front) is not int or front not in (0, 1):
        raise ValueError('This predeclared pilot tests front0/front1 only')
    return {'front_keep': front, 'skip_count': 6, 'target_layer': front+5,
            'skipped_layers': list(range(front, front+6)),
            'executed_layers': list(range(front))+list(range(front+6, 24))}


def check_contract(rec, front=None):
    if (rec.get('version') != VERSION or rec.get('base_sha256') != BASE_SHA
            or rec.get('source_layer') != 13 or rec.get('skip_count') != 6
            or rec.get('fronts') != [0, 1] or rec.get('times') != TIMES
            or rec.get('parity_max_abs') != 0):
        raise ValueError('Incompatible front-keep source/target/time contract')
    if front is not None:
        layout(front)


def file_sha(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def device(args):
    if not .05 <= args.memory_fraction <= .15:
        raise ValueError('Read-only probe allocator cap must be 5%-15%')
    dev = torch.device('cuda:0')
    if torch.cuda.mem_get_info(dev)[0] < 12*2**30:
        raise RuntimeError('Need12GiB free; never stop another process')
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, dev)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    return dev


def common():
    return dict(version=VERSION, base_sha256=BASE_SHA, source_layer=13,
                skip_count=6, fronts=[0, 1], times=TIMES, parity_max_abs=0.)


@torch.inference_mode()
def capture(args):
    dev = device(args)
    if shutil.disk_usage('.').free < 25*2**30:
        raise RuntimeError('Need25GiB free disk for bounded captures')
    folder = Path(args.data_root)/f'shard_{args.shard}'
    source_folder = Path(args.hidden_root)/folder.name
    baseline_folder = Path(args.baseline_root)/folder.name
    mp = folder/'manifest.json'
    digest = file_sha(mp)
    manifest = json.loads(mp.read_text())
    hidden_manifest = json.loads((source_folder/'manifest.json').read_text())
    if (not manifest['complete'] or manifest['parity_max_abs'] != 0
            or not hidden_manifest['complete'] or hidden_manifest['feature'] != 'output'
            or hidden_manifest['source_manifest_sha256'] != digest
            or Path(manifest['base_checkpoint']).resolve() != BASE.resolve()):
        raise ValueError('Wrong/incomplete source corpus')
    if file_sha(BASE) != BASE_SHA:
        raise ValueError('Wrong base checkpoint SHA')
    selected = select_records(manifest)
    out = Path(args.output_root)/folder.name
    out.mkdir(parents=True, exist_ok=True)
    if (out/'manifest.json').exists():
        old = json.loads((out/'manifest.json').read_text())
        check_contract(old)
        if old['complete'] and old['source_sha256'] == digest and old['files'] == selected:
            print(f'{out} already complete', flush=True)
            return
        raise ValueError('Refuse to overwrite another capture')
    model = load_frozen_teacher(BASE, dev, 'DiT-L/2', 'rdd')
    start = time.time()
    baseline_verified = 0
    for index, item in enumerate(selected):
        name = item['name']
        if Path(name).name != name or item['images'] != 8 or item['states_per_image'] != 8:
            raise ValueError('Unexpected record layout')
        ref = torch.load(folder/name, map_location='cpu', weights_only=True)
        coarse = torch.load(source_folder/name, map_location='cpu', weights_only=True)
        if (coarse['seed'] != item['seed'] or coarse['source_manifest_sha256'] != digest
                or coarse['feature'] != 'output' or coarse['parity_max_abs'] != 0
                or coarse['split'] != item['split'] or not torch.equal(coarse['labels'], ref['labels'])
                or coarse['coarse'][13].shape != (8,64,1024)
                or [int(t[0]) for t in ref['t']] != TIMES):
            raise ValueError('Wrong coarse source/labels/times')
        tensors = {key: [] for key in ('input0','input1','target0','target1','condition')}
        rng = torch.cuda.get_rng_state(dev)
        labels = ref['labels'].to(dev)
        for ti, tval in enumerate(TIMES):
            x, t, target = (ref[key][ti].to(dev) for key in ('x','t','target'))
            if not torch.all(t == tval):
                raise ValueError('Mixed times')
            pred, trace = manual_forward(model,x,t,labels,multires=True,capture_layers={0,5,6})
            torch.testing.assert_close(pred,target,atol=0,rtol=0)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                c = model.t_embedder(t)+model.y_embedder(labels,False)
            for front in (0,1):
                inp = trace[0]['input'] if front == 0 else trace[0]['output']
                tensors[f'input{front}'].append(inp.cpu())
                tensors[f'target{front}'].append(trace[layout(front)['target_layer']]['output'].cpu())
            tensors['condition'].append(c.cpu())
        if not torch.equal(rng,torch.cuda.get_rng_state(dev)):
            raise RuntimeError('Teacher capture changed CUDA RNG')
        rec = {**common(), 'source_sha256':digest, 'seed':item['seed'], 'role':item['role'],
               'labels':ref['labels'], 'coarse':coarse['coarse'][13],
               **{key:torch.stack(values) for key,values in tensors.items()}}
        for key,value in rec.items():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise ValueError(f'Nonfinite {key}')
        # This ties the new front0 arm to the existing six-block information probe.
        old = torch.load(baseline_folder/name,map_location='cpu',weights_only=True)
        if (old['version'] != 'prefix_information_v1' or old['prefix_depth'] != 6
                or old['seed'] != rec['seed'] or old['role'] != rec['role']
                or old['source_sha256'] != digest):
            raise ValueError('Existing front0 baseline is not matched')
        for newkey, oldkey in [('input0','h0'),('target0','target'),('condition','condition'),('coarse','coarse')]:
            torch.testing.assert_close(rec[newkey],old[oldkey],atol=0,rtol=0)
        baseline_verified += 1
        if (out/name).exists():
            previous = torch.load(out/name,map_location='cpu',weights_only=True)
            check_contract(previous)
            for key,value in rec.items():
                if torch.is_tensor(value):
                    torch.testing.assert_close(value,previous[key],atol=0,rtol=0)
                elif previous[key] != value:
                    raise ValueError('Existing capture belongs to another record')
        atomic_save(rec,out/name)
        print(json.dumps({'shard':args.shard,'files':index+1,'total':len(selected),
                          'seconds':time.time()-start}),flush=True)
    write_json(out/'manifest.json',{**common(),'complete':True,'source_sha256':digest,
        'files':selected,'baseline_front0_bitwise_files':baseline_verified,
        'peak_memory_gib':torch.cuda.max_memory_allocated()/2**30})


def entries(root):
    result, fingerprints = [], {}
    for shard in range(4):
        folder = root/f'shard_{shard}'
        manifest = json.loads((folder/'manifest.json').read_text())
        check_contract(manifest)
        if not manifest['complete'] or manifest['baseline_front0_bitwise_files'] != 12:
            raise ValueError('Incomplete parity audit')
        fingerprints[folder.name] = file_sha(folder/'manifest.json')
        for item in manifest['files']:
            if Path(item['name']).name != item['name']:
                raise ValueError('Unsafe record path')
            result.append((folder/item['name'],item,manifest['source_sha256']))
    counts = {role:sum(item['images'] for _,item,_ in result if item['role']==role)
              for role in ('fit','calibration','validation')}
    if counts != {'fit':192,'calibration':64,'validation':128} or len({r[1]['seed'] for r in result}) != 48:
        raise ValueError('Wrong split counts or overlapping seeds')
    return result, fingerprints, counts


def load(entry):
    path,item,digest = entry
    rec = torch.load(path,map_location='cpu',weights_only=True)
    check_contract(rec)
    if rec['seed'] != item['seed'] or rec['role'] != item['role'] or rec['source_sha256'] != digest:
        raise ValueError('Record metadata mismatch')
    return rec


def hidden_error_row(pred, target, inp):
    error = pred-target
    low,high = split_error(error)
    mse = error.square().mean((1,2))
    lo,hi = low.square().mean((1,2)),high.square().mean((1,2))
    torch.testing.assert_close(mse,lo+hi,atol=1e-6,rtol=1e-5)
    target_power = target.square().mean((1,2))
    delta_power = (target-inp).square().mean((1,2))
    return {'hidden_mse':mse,'low_mse':lo,'high_mse':hi,'target_power':target_power,
            'delta_power':delta_power,'target_nmse':mse/target_power.clamp_min(1e-12),
            'delta_nmse':mse/delta_power.clamp_min(1e-12)}


@torch.inference_mode()
def run_suffix(model,h,c,front):
    with torch.autocast('cuda',dtype=torch.bfloat16):
        cos,sin = model.get_rope(16,16,h.device)
        for block in model.blocks[front+6:]:
            h = block(h,c,cos,sin)
        return model.unpatchify(model.final_layer(h,c),16,16)


@torch.inference_mode()
def evaluate(items,role,fits,dev,front,data_root,model=None):
    sums,n,rows = {k:0. for k in fits},0,[]
    active = {key:on_gpu(value,dev) for key,value in fits.items()}
    for entry in items:
        if entry[1]['role'] != role: continue
        rec = load(entry)
        source = rec['coarse'].to(dev).float()
        ref = None
        if model is not None:
            ref = torch.load(Path(data_root)/entry[0].parent.name/entry[0].name,map_location='cpu',weights_only=True)
            if not torch.equal(ref['labels'],rec['labels']): raise ValueError('Output target labels mismatch')
        for ti,t in enumerate(TIMES):
            inp,target,c = (rec[key][ti].to(dev).float() for key in (f'input{front}',f'target{front}','condition'))
            n += len(inp)
            original_output = None
            if model is not None:
                original_output = ref['target'][ti].to(dev)
                # Suffix alone with exact teacher hidden must reproduce original output bitwise.
                exact = run_suffix(model,target.bfloat16(),c.bfloat16(),front)
                torch.testing.assert_close(exact,original_output,atol=0,rtol=0)
            for key,fit in active.items():
                mode = key.split('/')[0]
                pred_h = inp+predict(features(inp,c,source,mode),fit).reshape_as(inp)
                values = hidden_error_row(pred_h,target,inp)
                sums[key] += float(values['hidden_mse'].double().sum())
                if model is None: continue
                rng = torch.cuda.get_rng_state(dev)
                out = run_suffix(model,pred_h.bfloat16(),c.bfloat16(),front)
                if not torch.equal(rng,torch.cuda.get_rng_state(dev)):
                    raise RuntimeError('Suffix evaluation changed RNG')
                main,var = (out.float()-original_output.float()).chunk(2,dim=1)
                values.update(main_mse=main.square().mean((1,2,3)),variance_mse=var.square().mean((1,2,3)))
                # Both arms target the SAME final head; different hidden depths alone are not comparable.
                for j in range(len(inp)):
                    rows.append({'seed':rec['seed'],'image':j,'label':int(rec['labels'][j]),'t':t,
                                 'front_keep':front,'mode':mode,**{k:float(v[j]) for k,v in values.items()}})
    return {k:v/n for k,v in sums.items()},rows


@torch.inference_mode()
def fit(args):
    front = args.front_keep
    layout(front)
    dev = device(args)
    root = Path(args.output_root)
    out = root/f'front{front}'
    out.mkdir(exist_ok=True)
    items,fingerprints,counts = entries(root)
    if (out/'metrics.json').exists():
        old = json.loads((out/'metrics.json').read_text())
        check_contract(old,front)
        if old['complete'] and old['fingerprints']==fingerprints and old['front_keep']==front:
            print(f'{out} already complete',flush=True); return
        raise ValueError('Existing result mismatch')
    stats_path = out/'fit_statistics.pt'
    if stats_path.exists():
        stored = torch.load(stats_path,map_location='cpu',weights_only=True)
        check_contract(stored,front)
        if stored['fingerprints']!=fingerprints or stored['front_keep']!=front:
            raise ValueError('Wrong cached fit statistics')
        stats = stored['stats']
    else:
        moments = {mode:Moments(2048 if mode=='fine_only' else 3072,1024,dev) for mode in MODES}
        for index,entry in enumerate(items):
            if entry[1]['role']!='fit': continue
            rec = load(entry)
            source = rec['coarse'].to(dev).float()
            for ti in range(8):
                inp,target,c = (rec[key][ti].to(dev).float() for key in (f'input{front}',f'target{front}','condition'))
                for mode in MODES:
                    moments[mode].add(features(inp,c,source,mode),(target-inp).flatten(0,1))
            print(json.dumps({'front':front,'fit_file':index,'seed':rec['seed']}),flush=True)
        stats = {key:value.cpu() for key,value in moments.items()}
        del moments
        atomic_save({**common(),'front_keep':front,'fingerprints':fingerprints,'stats':stats},stats_path)
    all_fits = {f'{mode}/{ridge}':fit_ridge(stats[mode],ridge) for mode in MODES for ridge in (.01,.1,1.)}
    del stats
    calibration,_ = evaluate(items,'calibration',all_fits,dev,front,args.data_root)
    chosen = {mode:min((k for k in calibration if k.split('/')[0]==mode),key=calibration.get) for mode in MODES}
    fits = {mode:all_fits[key] for mode,key in chosen.items()}
    atomic_save({**common(),**layout(front),'fingerprints':fingerprints,'fits':fits,'chosen':chosen,
                 'calibration_mse':calibration},out/'selected_ridge.pt')
    del all_fits
    print(json.dumps({'front':front,'chosen':chosen}),flush=True)
    model = load_frozen_teacher(BASE,dev,'DiT-L/2','rdd')
    _,rows = evaluate(items,'validation',fits,dev,front,args.data_root,model)
    fields = ('hidden_mse','low_mse','high_mse','target_power','delta_power','target_nmse','delta_nmse','main_mse','variance_mse')
    means = {mode:{field:sum(row[field] for row in rows if row['mode']==mode)/1024 for field in fields} for mode in MODES}
    write_json(out/'metrics.json',{**common(),**layout(front),'complete':True,'images':counts,
        'fingerprints':fingerprints,'chosen':chosen,'calibration_mse':calibration,'means':means,'rows':rows,
        'peak_memory_gib':torch.cuda.max_memory_allocated()/2**30,
        'suffix_teacher_parity_max_abs':0.,
        'warning':'Read-only linear probe, not conditional trained bridge, rollout/FID or speed. Both replace six and execute18 fine blocks, but hidden targets differ: use normalized hidden errors and SAME final-output teacher target. Fine-only has fewer linear parameters; aligned/roll/wrong/mean tensors match, effective ranks do not. Teacher states reused development data,192fit/64calibration/128validation; ridge chosen by calibration hidden MSE, never reported validation. Source is existing16t300block13 hidden. Front1 is a hypothesis, not proof of RDD-exclusive improvement.'})
    print(json.dumps({'front':front,'complete':True,'means':means}),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('capture','fit'))
    p.add_argument('--output-root',default=ROOT)
    p.add_argument('--data-root',default='eval_results/csra_rollout_data4k_20260912')
    p.add_argument('--hidden-root',default='eval_results/csra_hidden_sources_20260913')
    p.add_argument('--baseline-root',default='eval_results/csra_prefix6_information_20260913')
    p.add_argument('--shard',type=int)
    p.add_argument('--front-keep',type=int,choices=(0,1),default=0)
    p.add_argument('--memory-fraction',type=float,default=.12)
    args = p.parse_args()
    if args.action=='capture':
        if args.shard not in range(4): p.error('Capture needs shard0-3')
        capture(args)
    else: fit(args)


if __name__=='__main__': main()
