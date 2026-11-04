"""Verify small readout-gate ablation; no speed or equivalence conclusions."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main(root):
    root=Path(root);groups={};checked=0
    for rank in range(4):
        folder=root/f'shard_{rank}';m=json.loads((folder/'metrics.json').read_text())
        if not m['complete'] or m['coarse_endpoint_max_abs']!=0 or m['protocol']['batch']!=16:raise ValueError('Incomplete')
        step=m['protocol']['step'];arm=groups.setdefault(step,{})
        for r in m['rows']:
            p=folder/r['variant']/f"{r['local_image']:03d}.png"
            if hashlib.sha256(p.read_bytes()).hexdigest()!=r['image_sha256']:raise ValueError('PNG changed')
            rr=arm.setdefault(r['variant'],{})
            if r['image'] in rr:raise ValueError('Duplicate image')
            rr[r['image']]=r;checked+=1
    if checked!=256:raise ValueError('Count')
    results={}
    for step,arms in groups.items():
        means={}
        for name,rows in arms.items():
            if len(rows)!=32:raise ValueError('Missing samples')
            means[name]={k:statistics.mean(float(r[k]) for r in rows.values())
                for k in ('top1_correct','top5_correct','target_probability')}
        results[step]=means
    output=dict(complete=True,verified_pngs=checked,means=results,
        warning='Only32 paired examples/parent; no quality-equivalence or speed claims; gate0 still executes attention.')
    (root/'summary_and_audit.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');main(p.parse_args().root)
