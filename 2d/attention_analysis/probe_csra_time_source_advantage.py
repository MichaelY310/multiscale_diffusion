#!/usr/bin/env python3
"""Read-only per-time, per-trajectory matched trained-source comparison.

Errors are main effective-epsilon prediction errors, NOT clean-image spectra,
FID, SNR or speed. Validation trajectories were used for checkpoint selection.
"""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from csra_joint_distillation import PreservedCoarseStudent
from train_rdd_csra_adapter import CSRAAdapterBank, load_frozen_teacher

ARMS = {
    'v2_aligned': ('RDD-CSRA-preserved_joint-continue1600-L400K-20260913', 'aligned', 'fine_context_v2'),
    'v2_zero': ('RDD-CSRA-preserved_joint-no_coarse-L400K-20260913', 'zero', 'fine_context_v2'),
    'v5_aligned': ('RDD-CSRA-memory_v5-aligned-L400K-20260913', 'aligned', 'coarse_memory_v5'),
    'v5_zero': ('RDD-CSRA-memory_v5-zero-L400K-20260913', 'zero', 'coarse_memory_v5'),
}


def band_errors(prediction, target):
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 8:
        raise ValueError('Expected matching main+variance B,8,H,W outputs')
    if any(n % 2 for n in prediction.shape[-2:]):
        raise ValueError('Even spatial dimensions required for orthogonal block-average split')
    delta = prediction[:, :4].float() - target[:, :4].float()
    low = F.interpolate(F.avg_pool2d(delta, 2, 2), scale_factor=2, mode='nearest')
    power = lambda x: x.square().mean((1,2,3))
    full, lp, hp = power(delta), power(low), power(delta-low)
    torch.testing.assert_close(lp+hp, full, atol=1e-8, rtol=2e-5)
    return {'main_mse':full, 'low_error_mse':lp, 'high_error_mse':hp,
            'variance_mse':power(prediction[:,4:].float()-target[:,4:].float())}


@torch.inference_mode()
def main(args):
    out = Path(args.output_dir)
    if (out/'metrics.json').exists():
        return
    device = torch.device('cuda:0')
    if torch.cuda.mem_get_info(device)[0] < 8*2**30:
        raise RuntimeError('Need 8GiB free; will not stop another process')
    base = Path('checkpoints/RDD-Unified-L/0400000.pt')
    folder, source, architecture = ARMS[args.arm]
    path = Path('checkpoints')/folder/'best.pt'
    ck = torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    if ck['args'].get('coarse_source','aligned') != source:
        raise ValueError('Checkpoint source mode mismatch')
    meta = ck['adapter_metadata']
    if meta['adapter_architecture'] != architecture or Path(ck['base_checkpoint']).resolve() != base.resolve():
        raise ValueError('Checkpoint architecture/base mismatch')
    model = load_frozen_teacher(base,device,'DiT-L/2','rdd')
    adapter = CSRAAdapterBank(meta['layers'],meta['hidden_size'],meta['rank'],architecture).to(device)
    net = PreservedCoarseStudent(model,adapter,False,False,source)
    net.load_state_dict(ck['student'])
    net.eval().requires_grad_(False)
    rows = []
    seen = set()
    for manifest_path in sorted(Path(args.data_root).glob('shard_*/manifest.json')):
        manifest = json.loads(manifest_path.read_text())
        if not manifest['complete'] or manifest['parity_max_abs'] != 0 or Path(manifest['base_checkpoint']).resolve() != base.resolve():
            raise ValueError('Invalid reference corpus')
        for item in manifest['files']:
            if item['split'] != 'validation':
                continue
            if Path(item['name']).name != item['name']:
                raise ValueError('Unsafe reference filename')
            ref = torch.load(manifest_path.parent/item['name'],map_location='cpu',weights_only=True)
            if ref['split'] != 'validation' or ref['layers'] != adapter.layers or ref['seed'] != item['seed']:
                raise ValueError('Reference metadata mismatch')
            labels = ref['labels'].to(device)
            cache = {layer: values.to(device) for layer,values in ref['coarse'].items()}
            rng = torch.cuda.get_rng_state(device)
            for i in range(len(ref['t'])):
                x,t,target = (ref[key][i].to(device) for key in ('x','t','target'))
                pred = net.stage(x,t,labels,cache=cache)[0]
                metrics = {k:v.cpu().tolist() for k,v in band_errors(pred,target).items()}
                for image in range(len(labels)):
                    key = (item['seed'],image,int(t[image]))
                    if key in seen:
                        raise ValueError('Duplicate validation state')
                    seen.add(key)
                    rows.append({'seed':key[0],'image':image,'t':key[2],
                                 **{k:v[image] for k,v in metrics.items()}})
            if not torch.equal(rng,torch.cuda.get_rng_state(device)):
                raise RuntimeError('Probe consumed CUDA RNG')
    if len(rows) != 1024 or len({(r['seed'],r['image']) for r in rows}) != 128:
        raise ValueError('Expected exactly 128 validation trajectories and 1024 states')
    summary = {}
    for t in sorted({r['t'] for r in rows}, reverse=True):
        subset = [r for r in rows if r['t'] == t]
        summary[str(t)] = {k:sum(r[k] for r in subset)/len(subset)
                          for k in ('main_mse','low_error_mse','high_error_mse','variance_mse')}
        summary[str(t)]['states'] = len(subset)
    out.mkdir(parents=True,exist_ok=True)
    payload = {'complete':True,'arm':args.arm,'checkpoint':str(path),'step':ck['step'],
               'architecture':architecture,'coarse_source':source,'config':ck['args'],
               'summary':summary,'rows':rows,'max_memory_gib':torch.cuda.max_memory_allocated()/2**30,
               'warning':'Fixed teacher states from checkpoint-selection validation, not independent test or rollout. Low/high split is block-average projection of prediction ERROR, not spectral quality or recovered image detail. No speed claim.'}
    temp = out/'metrics.json.tmp'
    temp.write_text(json.dumps(payload,indent=2,allow_nan=False))
    temp.replace(out/'metrics.json')
    print(json.dumps({'arm':args.arm,'step':ck['step'],'summary':summary,'peak_gib':payload['max_memory_gib']}),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--arm',choices=list(ARMS),required=True)
    p.add_argument('--data-root',default='eval_results/csra_rollout_data4k_20260912')
    p.add_argument('--output-dir',required=True)
    main(p.parse_args())
