#!/usr/bin/env python3
"""Read-only original teacher suffix after a fitted prefix surrogate.

Oracle hidden insertion must exactly reproduce the original full model.
No new sampling, training, or latency claims.
"""
import argparse
import json
from pathlib import Path
import torch
from probe_csra_prefix_information import (BASE, ROOT, TIMES, gpu, records, load_record,
                                         on_gpu, write_json)
from csra_prefix_probe_math import features, predict
from train_rdd_csra_adapter import load_frozen_teacher
from probe_rdd_vs_dit_block_reuse import manual_forward


@torch.inference_mode()
def main(args):
    root = Path(args.root)
    out = root/'suffix_probe.json'
    if out.exists():
        if json.loads(out.read_text())['complete']: return
        raise ValueError('Existing incomplete suffix results')
    dev = gpu()
    model = load_frozen_teacher(BASE,dev,'DiT-L/2','rdd')
    torch.backends.cuda.matmul.allow_tf32 = False
    entries, fingerprints, counts = records(root)
    saved = torch.load(root/'selected_ridge.pt',map_location='cpu',weights_only=True)
    if saved['fingerprints'] != fingerprints: raise ValueError('Ridge from another capture')
    depth = saved.get('prefix_depth',14)
    if saved.get('source_layer',13) != 13 or not 1 <= depth < 24:
        raise ValueError('Invalid prefix metadata')
    endpoint_layer = depth-1
    modes = ('fine_only','coarse_aligned','coarse_mean')
    fits = {m:on_gpu(saved['fits'][m],dev) for m in modes}
    rows = []
    for entry in entries:
        if entry[1]['role'] != 'validation': continue
        rec = load_record(entry)
        original = Path(args.data_root)/entry[0].parent.name/entry[0].name
        ref = torch.load(original,map_location='cpu',weights_only=True)
        if not torch.equal(rec['labels'],ref['labels']): raise ValueError('Labels mismatch')
        labels = rec['labels'].to(dev)
        source = rec['coarse'].to(dev).float()
        for ti,tv in enumerate(TIMES):
            x,t,target = (ref[key][ti].to(dev) for key in ('x','t','target'))
            if not torch.all(t == tv): raise ValueError('Fine time mismatch')
            dense,trace = manual_forward(model,x,t,labels,multires=True,capture_layers={0,endpoint_layer})
            torch.testing.assert_close(dense,target,atol=0,rtol=0)
            torch.testing.assert_close(trace[endpoint_layer]['output'].cpu(),rec['target'][ti],atol=0,rtol=0)
            torch.testing.assert_close(trace[0]['input'].cpu(),rec['h0'][ti],atol=0,rtol=0)
            rng = torch.cuda.get_rng_state(dev)
            for mode in ('oracle',)+modes:
                # Carry current h0 unchanged, then replace the prefix endpoint;
                # do not accumulate old residuals from the skipped blocks.
                replacements = {l:(lambda h,c:h) for l in range(endpoint_layer)}
                if mode == 'oracle':
                    replacements[endpoint_layer] = lambda h,c:trace[endpoint_layer]['output']
                else:
                    def endpoint(h,c,mode=mode):
                        # Explicit FP32 regression matches the offline probe;
                        # enclosing teacher AMP must not silently make this BF16.
                        with torch.autocast('cuda',enabled=False):
                            return h.float()+predict(features(h.float(),c.float(),source,mode),fits[mode]).reshape_as(h)
                    replacements[endpoint_layer] = endpoint
                pred,_ = manual_forward(model,x,t,labels,multires=True,replacements=replacements,
                    conditioned_replacements=True,replacement_mode='hidden')
                if mode == 'oracle':
                    torch.testing.assert_close(pred,dense,atol=0,rtol=0)
                    continue
                mse = (pred[:,:4].float()-target[:,:4].float()).square().mean((1,2,3))
                var = (pred[:,4:].float()-target[:,4:].float()).square().mean((1,2,3))
                power = target[:,:4].float().square().mean((1,2,3))
                rows.extend({'seed':rec['seed'],'image':j,'t':tv,'mode':mode,
                    'main_mse':float(mse[j]),'variance_mse':float(var[j]),'target_power':float(power[j])}
                    for j in range(len(labels)))
            if not torch.equal(rng,torch.cuda.get_rng_state(dev)): raise RuntimeError('Changed CUDA RNG')
        print(json.dumps({'done_seed':rec['seed']}),flush=True)
    if len(rows) != 128*8*3: raise ValueError('Incomplete suffix states')
    means = {m:{key:sum(r[key] for r in rows if r['mode']==m)/1024
                for key in ('main_mse','variance_mse','target_power')} for m in modes}
    by_time = {m:{str(t):sum(r['main_mse'] for r in rows if r['mode']==m and r['t']==t)/128
                 for t in TIMES} for m in modes}
    write_json(out,{'complete':True,'states_per_mode':1024,'images':counts,'mean':means,'by_time':by_time,
        'rows':rows,'oracle_suffix_parity_max_abs':0.,'fingerprints':fingerprints,'prefix_depth':depth,'source_layer':13,
        'peak_memory_gib':torch.cuda.max_memory_allocated()/2**30,
        'warning':f'Frozen original last{24-depth} blocks after a linear{depth}-block prefix surrogate. Coarse/zero ridge fitted192train and calibrated64, this128validation developmental. Teacher states, NOT student rollout/image/FID or latency. Oracle exact endpoint matches dense bitwise. No fine suffix finetuning yet; no claim10% speed.'})
    print(json.dumps({'complete':True,'mean':means,'by_time':by_time}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',default=str(ROOT))
    p.add_argument('--data-root',default='eval_results/csra_rollout_data4k_20260912')
    main(p.parse_args())
