#!/usr/bin/env python3
"""Record original coarse block OUTPUT h, not h_after-h_before.

One existing saved 16-stage source forward per trajectory, not new sampling.
Checks the entire output and every old delta cache exactly before saving h.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from probe_rdd_vs_dit_block_reuse import manual_forward
from train_rdd_csra_adapter import atomic_save, load_frozen_teacher


@torch.inference_mode()
def main(args):
    src = Path(args.data_root) / f'shard_{args.shard}'
    replay = Path(args.source_root) / src.name
    out = Path(args.output_root) / src.name
    manifest_path = src / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    side = json.loads((replay/'manifest.json').read_text())
    if (not manifest['complete'] or manifest['parity_max_abs'] != 0
        or not side['complete'] or side['source_manifest_sha256'] != digest):
        raise ValueError('Original corpus/source replay validation failed')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():
        old = json.loads((out/'manifest.json').read_text())
        if old.get('complete') and old.get('source_manifest_sha256') == digest and old.get('feature') == 'output':
            return
        raise ValueError('Refusing to overwrite another sidecar')
    device = torch.device('cuda:0')
    if torch.cuda.mem_get_info(device)[0] < 8*2**30:
        raise RuntimeError('Need 8GiB free; not stopping other processes')
    teacher = load_frozen_teacher(Path(manifest['base_checkpoint']),device,'DiT-L/2','rdd')
    for index,item in enumerate(manifest['files']):
        name = item['name']
        if Path(name).name != name:
            raise ValueError('Unsafe filename')
        ref = torch.load(src/name,map_location='cpu',weights_only=True)
        saved = torch.load(replay/name,map_location='cpu',weights_only=True)
        if saved['seed'] != item['seed'] or saved['cache_parity_max_abs'] != 0 or saved['source_manifest_sha256'] != digest:
            raise ValueError('Source record mismatch')
        source = saved['sources']['16']
        if not torch.equal(source['labels'],ref['labels']) or not torch.all(source['t'] == 300):
            raise ValueError('Source labels/time mismatch')
        result,trace = manual_forward(teacher,source['x'].to(device),source['t'].to(device),
            source['labels'].to(device),multires=True,capture_layers=set(manifest['layers']))
        torch.testing.assert_close(result.cpu(),source['target'],atol=0,rtol=0)
        for l in manifest['layers']:
            torch.testing.assert_close(trace[l]['residual'].cpu(),ref['coarse'][l],atol=0,rtol=0)
            if trace[l]['output'].shape != ref['coarse'][l].shape or not torch.isfinite(trace[l]['output']).all():
                raise ValueError('Invalid full hidden output')
        dest = out/name
        if dest.exists():
            existing = torch.load(dest,map_location='cpu',weights_only=True)
            if existing['seed'] != item['seed'] or existing['source_manifest_sha256'] != digest or existing['feature'] != 'output':
                raise ValueError('Existing hidden sidecar mismatch')
            for l in manifest['layers']:
                torch.testing.assert_close(existing['coarse'][l],trace[l]['output'].cpu(),atol=0,rtol=0)
        else:
            atomic_save({'seed':item['seed'],'split':item['split'],'labels':ref['labels'],
                'feature':'output','source_manifest_sha256':digest,'parity_max_abs':0.,
                'coarse':{l:trace[l]['output'].cpu().clone() for l in manifest['layers']}},dest)
        if index % 16 == 0:
            print(json.dumps({'shard':args.shard,'files':index+1,'total':len(manifest['files']),
                              'peak_gib':torch.cuda.max_memory_allocated()/2**30}),flush=True)
    temp = out/'manifest.json.tmp'
    temp.write_text(json.dumps({'complete':True,'feature':'output','source_manifest_sha256':digest,
                               'files':manifest['files'],'layers':manifest['layers']},indent=2))
    temp.replace(out/'manifest.json')
    print(f'Hidden-source shard {args.shard} complete with exact delta/output parity',flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',required=True)
    p.add_argument('--source-root',required=True)
    p.add_argument('--output-root',required=True)
    p.add_argument('--shard',required=True,type=int)
    main(p.parse_args())
