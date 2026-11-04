#!/usr/bin/env python3
"""Streaming ridge probe of new six-layer skip layouts, with same-head checks.

No giant hidden capture files. Fit192/calibration64/validation128 images from
the pre-existing teacher corpus. Fits ridge on train, selects on calibration;
validation runs the actual composed skipped path (including both split spans).
This is a structural diagnostic, NOT trained rollout quality or video speed.
"""
import argparse
import json
from pathlib import Path
import shutil
import torch
from csra_layer_layouts import LAYOUTS, layout
from csra_prefix_probe_math import Moments, select_records, features, fit_ridge, predict
from probe_csra_prefix_information import BASE, TIMES, write_json, on_gpu
from probe_csra_front_keep import BASE_SHA, file_sha, hidden_error_row
from probe_rdd_vs_dit_block_reuse import manual_forward
from train_rdd_csra_adapter import atomic_save, load_frozen_teacher

VERSION = 'csra_six_layer_layout_stream_probe_v1'
MODES = ('fine_only', 'coarse_aligned', 'coarse_wrong')
ROOT = '/home/yuheng/rdd_experiments/csra_skip_layout_20260915'


def entries(args):
    items, provenance = [], {}
    for shard in range(4):
        folder = Path(args.data_root)/f'shard_{shard}'
        hidden = Path(args.hidden_root)/folder.name
        manifest = json.loads((folder/'manifest.json').read_text())
        side = json.loads((hidden/'manifest.json').read_text())
        digest = file_sha(folder/'manifest.json')
        if (not manifest['complete'] or manifest['parity_max_abs'] != 0
            or not side['complete'] or side['feature'] != 'output'
            or side['source_manifest_sha256'] != digest
            or Path(manifest['base_checkpoint']).resolve() != BASE.resolve()):
            raise ValueError('Wrong teacher corpus/source contract')
        provenance[str(folder/'manifest.json')] = digest
        provenance[str(hidden/'manifest.json')] = file_sha(hidden/'manifest.json')
        for item in select_records(manifest):
            if Path(item['name']).name != item['name'] or item['images'] != 8:
                raise ValueError('Unexpected source record')
            items.append((folder/item['name'], hidden/item['name'], item, digest))
    counts = {role:sum(e[2]['images'] for e in items if e[2]['role']==role)
        for role in ('fit','calibration','validation')}
    if counts != dict(fit=192,calibration=64,validation=128) or len({e[2]['seed'] for e in items}) != 48:
        raise ValueError('Counts or seed split mismatch')
    return items, provenance, counts


def records(items, role, model, spans, device):
    capture = {i for s,e in spans for i in (s,e-1)} | {0}
    for rp,hp,item,digest in items:
        if item['role'] != role: continue
        ref = torch.load(rp, map_location='cpu', weights_only=True)
        side = torch.load(hp, map_location='cpu', weights_only=True)
        if (side['seed'] != item['seed'] or side['source_manifest_sha256'] != digest
            or side['feature'] != 'output' or side['parity_max_abs'] != 0
            or not torch.equal(side['labels'],ref['labels'])
            or side['coarse'][13].shape != (8,64,1024)
            or [int(t[0]) for t in ref['t']] != TIMES):
            raise ValueError('Source/labels/times mismatch')
        source,labels = side['coarse'][13].to(device).float(),ref['labels'].to(device)
        rng = torch.cuda.get_rng_state(device)
        for ti,tval in enumerate(TIMES):
            x,t,target = (ref[k][ti].to(device) for k in ('x','t','target'))
            if not bool((t==tval).all()): raise ValueError('Mixed global timesteps')
            output,trace = manual_forward(model,x,t,labels,multires=True,capture_layers=capture)
            torch.testing.assert_close(output,target,atol=0,rtol=0)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                c = model.t_embedder(t)+model.y_embedder(labels,False)
            yield dict(trace=trace,condition=c,source=source,target=target,
                seed=item['seed'],labels=ref['labels'],t=tval)
        if not torch.equal(rng,torch.cuda.get_rng_state(device)):
            raise ValueError('Probe changed CUDA RNG')
        print(json.dumps(dict(role=role,seed=item['seed'],file=rp.name)),flush=True)


def composed(model, rec, spans, selected=None, mode=None, oracle=False):
    h = rec['trace'][0]['input']; c=rec['condition']
    starts = {s:(n,e) for n,(s,e) in enumerate(spans)}
    with torch.autocast('cuda',dtype=torch.bfloat16):
        cos,sin = model.get_rope(16,16,h.device)
        i=0
        while i < 24:
            if i in starts:
                n,end=starts[i]
                if oracle:
                    h=rec['trace'][end-1]['output']
                else:
                    # Same FP32 affine policy as the existing reader, followed
                    # by explicit BF16 rounding at each span boundary.
                    with torch.autocast('cuda',enabled=False):
                        delta=predict(features(h.float(),c.float(),rec['source'],mode),selected[str(n)][mode])
                        h=(h.float()+delta.reshape_as(h)).bfloat16()
                i=end
            else:
                h=model.blocks[i](h,c,cos,sin); i+=1
        return model.unpatchify(model.final_layer(h,c),16,16)


@torch.inference_mode()
def main(args):
    spec=layout(args.layout); spans=LAYOUTS[args.layout]
    root=Path(args.root)/('layout_smoke' if args.smoke else 'layouts')/args.layout
    root.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(root).free < 12*2**30: raise RuntimeError('Need12GiB free disk; do not delete')
    dev=torch.device('cuda:0'); torch.set_num_threads(4)
    if torch.cuda.mem_get_info(dev)[0] < 16*2**30: raise RuntimeError('Need16GiB headroom')
    torch.cuda.set_per_process_memory_fraction(.15,dev)
    torch.backends.cuda.matmul.allow_tf32=False
    items,provenance,counts=entries(args)
    if args.smoke:
        items=[next(e for e in items if e[2]['role']==role) for role in ('fit','calibration','validation')]
        counts=dict(fit=8,calibration=8,validation=8)
    if file_sha(BASE) != BASE_SHA: raise ValueError('Wrong base checkpoint')
    contract=dict(version=VERSION,layout=spec,base_sha256=BASE_SHA,times=TIMES,
        source_layer=13,source_feature='full_hidden',source_t=300,counts=counts,smoke=args.smoke,
        provenance=provenance,code_sha256={str(p):file_sha(p) for p in [Path(__file__),
            Path(__file__).with_name('csra_layer_layouts.py'), Path(__file__).with_name('csra_prefix_probe_math.py')]})
    if (root/'protocol.json').exists() and json.loads((root/'protocol.json').read_text())!=contract:
        raise ValueError('Protocol mismatch; never overwrite other experiments')
    write_json(root/'protocol.json',contract)
    if (root/'metrics.json').exists():
        print(f'{root} already complete',flush=True); return
    model=load_frozen_teacher(BASE,dev,'DiT-L/2','rdd')
    stats_file=root/'statistics.pt'
    if stats_file.exists():
        payload=torch.load(stats_file,map_location='cpu',weights_only=True)
        if payload['contract'] != contract: raise ValueError('Wrong cached statistics')
        stats=payload['stats']; del payload
    else:
        moments={str(n):{m:Moments(2048 if m=='fine_only' else 3072,1024,dev)
            for m in MODES} for n in range(len(spans))}
        for rec in records(items,'fit',model,spans,dev):
            for n,(s,e) in enumerate(spans):
                h=rec['trace'][s]['input'].float(); target=rec['trace'][e-1]['output'].float()
                for mode in MODES:
                    moments[str(n)][mode].add(features(h,rec['condition'].float(),rec['source'],mode),
                        (target-h).flatten(0,1))
        stats={n:{m:obj.cpu() for m,obj in ms.items()} for n,ms in moments.items()}
        del moments
        atomic_save(dict(contract=contract,stats=stats),stats_file)
    fits={n:{f'{m}/{r}':on_gpu(fit_ridge(st,r),dev) for m,st in ms.items() for r in (.01,.1,1.)}
        for n,ms in stats.items()}
    del stats
    calibration={n:{k:0. for k in ms} for n,ms in fits.items()}; images=0
    for rec in records(items,'calibration',model,spans,dev):
        images+=len(rec['labels'])
        for n,(s,e) in enumerate(spans):
            h=rec['trace'][s]['input'].float(); target=rec['trace'][e-1]['output'].float()
            for key,fit in fits[str(n)].items():
                mode=key.split('/')[0]
                pred=h+predict(features(h,rec['condition'].float(),rec['source'],mode),fit).reshape_as(h)
                calibration[str(n)][key]+=float((pred-target).square().mean((1,2)).double().sum())
    if images!=counts['calibration']*8: raise ValueError('Calibration count mismatch')
    chosen={n:{m:min((k for k in scores if k.split('/')[0]==m),key=scores.get)
        for m in MODES} for n,scores in calibration.items()}
    selected={n:{m:fits[n][k] for m,k in ms.items()} for n,ms in chosen.items()}
    atomic_save(dict(**contract,fits={n:{m:{k:v.cpu() if torch.is_tensor(v) else v
        for k,v in fit.items()} for m,fit in ms.items()} for n,ms in selected.items()},
        chosen=chosen,calibration_sum=calibration),root/'selected_ridge.pt')
    del fits
    rows=[]
    for rec in records(items,'validation',model,spans,dev):
        exact=composed(model,rec,spans,oracle=True)
        torch.testing.assert_close(exact,rec['target'],atol=0,rtol=0)
        for mode in MODES:
            pred=composed(model,rec,spans,selected,mode)
            eps,var=(pred.float()-rec['target'].float()).chunk(2,1)
            values=dict(main_mse=eps.square().mean((1,2,3)),variance_mse=var.square().mean((1,2,3)))
            for n,(s,e) in enumerate(spans):
                h=rec['trace'][s]['input'].float(); target=rec['trace'][e-1]['output'].float()
                independent=h+predict(features(h,rec['condition'].float(),rec['source'],mode),selected[str(n)][mode]).reshape_as(h)
                for k,v in hidden_error_row(independent,target,h).items(): values[f'span{n}_{k}']=v
            for j in range(len(rec['labels'])):
                rows.append(dict(mode=mode,seed=rec['seed'],image=j,t=rec['t'],label=int(rec['labels'][j]),
                    **{k:float(v[j]) for k,v in values.items()}))
    fields=[k for k in rows[0] if k not in ('mode','seed','image','t','label')]
    means={m:{f:sum(r[f] for r in rows if r['mode']==m)/(counts['validation']*8) for f in fields} for m in MODES}
    if len(rows)!=counts['validation']*8*3: raise ValueError('Validation count mismatch')
    write_json(root/'metrics.json',dict(**contract,complete=True,means=means,rows=rows,chosen=chosen,
        suffix_oracle_bitwise_parity=True,teacher_output_bitwise_parity=True,
        peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,
        warning='Teacher-forced development linear diagnostic, not trained reader, FID or video speed. Compare SAME final head across layouts; raw hidden MSE across depths is not comparable. Split3+3 has TWO readers and must pay both costs. Coarse-wrong uses shuffled image within batch, equal parameter count; fine-only has fewer parameters. No data was selected using reported validation.'))
    print(json.dumps(dict(layout=args.layout,complete=True,means=means)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--layout',choices=tuple(LAYOUTS),required=True)
    p.add_argument('--root',default=ROOT)
    p.add_argument('--data-root',default='eval_results/csra_rollout_data4k_20260912')
    p.add_argument('--hidden-root',default='eval_results/csra_hidden_sources_20260913')
    p.add_argument('--smoke',action='store_true')
    main(p.parse_args())
