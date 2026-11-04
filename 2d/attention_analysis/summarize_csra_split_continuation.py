"""Verify matched800/1600 PNGs and paired differences; no FID extrapolation."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics


def read(root,folder,count):
    root=root/folder;audit=json.loads((root/'summary_and_audit.json').read_text());arms={};checked=0
    if not audit['complete']:raise ValueError('Remote audit incomplete')
    for rank in range(4):
        p=root/f'shard_{rank}/metrics.json'
        if hashlib.sha256(p.read_bytes()).hexdigest()!=audit['shard_sha256'][str(rank)]:raise ValueError('Metric hash')
        data=json.loads(p.read_text())
        for row in data['rows']:
            png=root/f'shard_{rank}'/row['variant']/f"{row['local_image']:03d}.png"
            if hashlib.sha256(png.read_bytes()).hexdigest()!=row['image_sha256']:raise ValueError('PNG hash')
            arm=arms.setdefault(row['variant'],{})
            if row['image'] in arm:raise ValueError('Duplicate image')
            arm[row['image']]=row;checked+=1
    if checked!=count or any(len(rows)!=256 for rows in arms.values()):raise ValueError('Wrong count')
    return audit,arms


def paired(a,b):
    rng=random.Random(51560916);result={}
    for metric in ('top1_correct','top5_correct','target_probability'):
        delta=[]
        for i in range(256):
            if a[i]['target']!=b[i]['target']:raise ValueError('Changed pairing')
            delta.append(float(a[i][metric])-float(b[i][metric]))
        samples=sorted(statistics.mean(rng.choices(delta,k=256)) for _ in range(4000))
        result[metric]=dict(mean=statistics.mean(delta),ci95=[samples[100],samples[3899]])
    return result


def main(root):
    root=Path(root);old,oa=read(root,'split_quality256',1280);new,na=read(root,'split1600_quality256',1024)
    if any(oa['original_dense'][i]['image_sha256']!=na['original_dense'][i]['image_sha256'] for i in range(256)):
        raise ValueError('Baseline PNGs do not exactly reproduce')
    comparisons={}
    for arm in ('front0_6_aligned','split3_3_aligned','split3_3_zero'):
        comparisons[arm+'_1600_minus_800']=paired(na[arm],oa[arm])
    comparisons['aligned_minus_zero_at1600']=paired(na['split3_3_aligned'],na['split3_3_zero'])
    comparisons['split_minus_front_at1600']=paired(na['split3_3_aligned'],na['front0_6_aligned'])
    validations={}
    for arm in ('front0_6_aligned','split3_3_aligned','split3_3_zero'):
        validations[arm]={str(step):json.loads((root/folder/arm/'complete.json').read_text())['validation']
            for step,folder in ((800,'training'),(1600,'continuation1600'))}
    output=dict(complete=True,verified_new_pngs=1024,verified_old_pngs=1280,baseline_png_parity=True,
        means800=old['means'],means1600=new['means'],validation=validations,paired=comparisons,
        warning='Same development256 images, one training seed, no multiplicity correction. Classifier/latent errors are not FID or video quality.')
    (root/'split1600_quality256/paired_continuation_summary.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');main(p.parse_args().root)
