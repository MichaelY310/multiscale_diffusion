"""Audit completed paired data/quality; report all variants, no winner picking."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from csra_state_replay_data import ROOT, PROTOCOL, PARENT_SHA, QUALITY_SEED
from probe_csra_front_keep import write_json, file_sha


def paired(values):
    x=np.asarray(values,dtype=float)
    rng=np.random.default_rng(321)
    means=x[rng.integers(0,len(x),(10000,len(x)))].mean(1)
    return dict(mean=float(x.mean()),paired_bootstrap95=np.quantile(means,[.025,.975]).tolist(),n=len(x))


def data_summary():
    rows=[];files=[]
    for shard in range(4):
        path=ROOT/'data'/f'shard_{shard}'/'manifest.json'
        manifest=json.loads(path.read_text())
        if not manifest.get('complete') or manifest.get('smoke') or manifest['parent_sha256']!=PARENT_SHA:
            raise ValueError('Missing completed data')
        for item in manifest['files']:
            p=path.parent/item['name']
            if file_sha(p)!=item['sha256']:raise ValueError('Filehash mismatch')
            d=torch.load(p,map_location='cpu',mmap=True,weights_only=True)
            if d['collector_parity_max_abs']!=0 or d['coarse_endpoint_max_abs']!=0:raise ValueError('Parity failure')
            rows.extend(dict(**r,seed=d['seed'],split=d['split']) for r in d['rows'])
            files.append(dict(shard=shard,seed=d['seed'],split=d['split'],images=len(d['labels']),
                              final_relative_latent_l2=d['final_relative_latent_l2']))
    means={split:{role:float(np.mean([r['main_mse'] for r in rows if r['split']==split and r['role']==role]))
                  for role in ('teacher','student')} for split in ('train','validation')}
    bytime={split:{str(t):{role:float(np.mean([r['main_mse'] for r in rows
                if r['split']==split and r['role']==role and r['t']==t])) for role in ('teacher','student')}
                for t in (299,259,215,171,128,84,40,0)} for split in means}
    write_json(ROOT/'data_summary.json',dict(complete=True,protocol=PROTOCOL,files=files,
        mean_mse=means,by_time=bytime,states_compared=len(rows),
        warning='Different trajectory-state distributions; larger/smaller MSE alone does not prove quality or causal instability.'))


def quality_summary():
    variants=('original','parent','teacher_only','mixed50');rows=[];metrics=[];verified=0
    for shard in range(4):
        path=ROOT/'quality'/f'shard_{shard}'/'metrics.json';d=json.loads(path.read_text())
        if (not d['complete'] or d['protocol']!=PROTOCOL or d['seed']!=QUALITY_SEED+shard*10000
            or d['updates']!=400 or d['parent_sha256']!=PARENT_SHA
            or d['coarse_endpoint_max_abs']!={k:0. for k in variants[1:]} or d['steps']!=[80,80,90]):
            raise ValueError('Quality contract mismatch')
        seen=set()
        for row in d['rows']:
            key=(row['image'],row['variant'])
            if key in seen or row['variant'] not in variants:raise ValueError('Duplicate/bad quality row')
            seen.add(key)
            if row['target']!=d['labels'][row['local_image']]:raise ValueError('Labels mismatch')
            p=path.parent/row['variant']/f"{row['local_image']:03d}.png"
            if file_sha(p)!=row['image_sha256']:raise ValueError('Image checksum mismatch')
            rows.append(row);verified+=1
        if len(seen)!=256:raise ValueError('Missing paired images')
        metrics.append(d)
    agg={v:{k:float(np.mean([r[k] for r in rows if r['variant']==v]))
                 for k in ('top1_correct','top5_correct','target_probability')} for v in variants}
    for v in variants:agg[v]['relative_latent_l2']=float(np.mean([d['relative_latent_l2'][v] for d in metrics]))
    lookup={(r['image'],r['variant']):r for r in rows};contrasts={}
    for a,b in (('mixed50','teacher_only'),('teacher_only','parent'),('mixed50','parent'),('mixed50','original')):
        contrasts[f'{a}_minus_{b}']={k:paired([float(lookup[i,a][k])-float(lookup[i,b][k]) for i in range(256)])
                                    for k in ('top1_correct','top5_correct','target_probability')}
    write_json(ROOT/'quality_summary.json',dict(complete=True,protocol=PROTOCOL,images_sha_verified=verified,
        aggregates=agg,contrasts=contrasts,warning='Paired-image uncertainty conditional on one training seed; '
            'classification and latent agreement are NOT FID/realism. Inspect all4fixed grids and failures.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-only',action='store_true')
    args=p.parse_args();data_summary()
    if not args.data_only:quality_summary()
