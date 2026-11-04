"""Stdlib-only local PNG verification and paired-image bootstrap; not FID."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics


def main(root):
    root=Path(root)
    audit=json.loads((root/'summary_and_audit.json').read_text())
    if not audit['complete']:raise ValueError('Remote audit incomplete')
    arms={};checked=0
    for rank in range(4):
        folder=root/f'shard_{rank}';path=folder/'metrics.json'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=audit['shard_sha256'][str(rank)]:
            raise ValueError('Changed metrics')
        data=json.loads(path.read_text())
        for row in data['rows']:
            png=folder/row['variant']/f"{row['local_image']:03d}.png"
            if hashlib.sha256(png.read_bytes()).hexdigest()!=row['image_sha256']:raise ValueError('PNG mismatch')
            arm=arms.setdefault(row['variant'],{})
            if row['image'] in arm:raise ValueError('Duplicate paired image')
            arm[row['image']]=row;checked+=1
    if checked!=1280 or any(len(v)!=256 for v in arms.values()):raise ValueError('Incomplete images')
    results={}
    for reference in ('split3_3_zero','front0_6_aligned','previous_prefix6_1600','original_dense'):
        out={};rng=random.Random(61560915)
        for metric in ('top1_correct','top5_correct','target_probability'):
            differences=[]
            for i in range(256):
                a,b=arms['split3_3_aligned'][i],arms[reference][i]
                if a['target']!=b['target']:raise ValueError('Label mismatch')
                differences.append(float(a[metric])-float(b[metric]))
            draws=sorted(statistics.mean(rng.choices(differences,k=256)) for _ in range(4000))
            out[metric]=dict(mean=statistics.mean(differences),ci95=[draws[100],draws[3899]])
        results['split3_3_aligned-minus-'+reference]=out
    output=dict(complete=True,locally_verified_pngs=checked,means=audit['means'],paired=results,
        warning='Fresh256 development pairs; image bootstrap conditional on one trained seed. Not FID, no multiple-testing correction, not equivalence evidence.')
    (root/'paired_summary_local.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');main(p.parse_args().root)
