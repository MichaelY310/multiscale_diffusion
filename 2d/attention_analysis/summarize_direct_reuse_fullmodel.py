"""Verify every new image/video result and build a compact paired audit."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_ci(values):
    rng=random.Random(91560915)
    draws=sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(10000))
    return dict(mean=statistics.mean(values),ci95=[draws[250],draws[9749]],n=len(values))


def main(a):
    root=Path(a.root);report={'complete':False,'video':{},'image':{},'files_sha256':{}}
    for name in ('video_832','video_1280','video_temporal221'):
        folder=root/name;p=folder/'metrics.json'
        m=json.loads(p.read_text())
        if not m['complete'] or m['baseline_noop_max_abs']!=0:raise ValueError('Video not complete/parity failed')
        for variant in ('baseline','reuse6','reuse8','reuse10'):
            for ext in ('mp4','png'):
                f=folder/f'{variant}.{ext}'
                if not f.is_file() or f.stat().st_size<100:raise ValueError(f'Missing {f}')
                report['files_sha256'][str(f.relative_to(root))]=sha(f)
        rows=[]
        for e in m['entries']:
            for group in e['groups']:
                for row in group:
                    expected=len(m['layers_zero_based'][row['variant']])
                    if row['audit']['cross_stage_block_reuse_hits']!=expected:raise ValueError('Skip count mismatch')
            rows.append(dict(variant=e['variant'],denoise_baseline=e['baseline']['denoise']['median'],
                denoise_candidate=e['candidate']['denoise']['median'],total_baseline=e['baseline']['total']['median'],
                total_candidate=e['candidate']['total']['median'],reductions=e['reductions']))
        report['video'][name]=dict(rows=rows,factors=m['factors'],shape=m['shape'],
            warning=m['warning'],latent_relative_l2=m['latent_relative_l2'])
        report['files_sha256'][str(p.relative_to(root))]=sha(p)
    groups={};bench=[];parents=None;image_count=0
    for rank in range(4):
        folder=root/'image_preview'/f'shard_{rank}';p=folder/'metrics.json';m=json.loads(p.read_text())
        if not m['complete'] or m['no_op_max_abs']!=0 or m['rank']!=rank or m['batch']!=16:raise ValueError('Image protocol')
        if parents is not None and m['parent_protocol']!=parents:raise ValueError('Different checkpoints/protocol')
        parents=m['parent_protocol'];bench+=m['benchmark'];image_count+=m['batch']
        for row in m['rows']:
            i=row['image'];v=row['variant'];f=folder/v/f'{i-rank*16:03d}.png'
            if sha(f)!=row['image_sha256']:raise ValueError(f'Corrupt image {f}')
            if i in groups.setdefault(v,{}):raise ValueError('Duplicated image')
            groups[v][i]=row;report['files_sha256'][str(f.relative_to(root))]=row['image_sha256']
        report['files_sha256'][str(p.relative_to(root))]=sha(p)
    original=groups['original'];summary={}
    for v,rows in groups.items():
        if set(rows)!=set(original):raise ValueError('Unpaired results')
        for i,r in rows.items():
            if r['target']!=original[i]['target']:raise ValueError('Label mismatch')
        summary[v]={k:100*statistics.mean(float(r[k]) for r in rows.values()) for k in ('top1_correct','top5_correct')}
        summary[v]['target_probability']=statistics.mean(r['target_probability'] for r in rows.values())
        summary[v]['top1_difference_pp']=paired_ci([100*(int(r['top1_correct'])-int(original[i]['top1_correct'])) for i,r in rows.items()])
    if len(bench)!=4:raise ValueError('Missing4 image ABBA groups')
    report['image']=dict(images=image_count,total_png=sum(len(v) for v in groups.values()),summary=summary,
        denoise_latency_reduction_percent=paired_ci([g['latency_reduction_percent'] for g in bench]),
        benchmark=bench,parents=parents,warning='64 paired samples; classifier is NOT FID; shared GPU latency')
    for name in ('reuse6','dense'):
        p=root/f'train_{name}/complete.json';m=json.loads(p.read_text())
        if not m['complete'] or m['step']!=200:raise ValueError('Training incomplete')
        report.setdefault('training',{})[name]=m
    report['complete']=True
    (root/'summary_and_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('files_sha256','video')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='/home/yuheng/rdd_experiments/direct_reuse_fullmodel_20260915');main(p.parse_args())
