"""Audit fixed2400 2x2 quality, pairing each image BEFORE source interaction."""
import hashlib
import json
from pathlib import Path
import statistics

from csra_neighbor_protocol import PROTOCOL, FORMAT, PARENT_SHA, OPTIMIZER_POLICY, FIXED_ARGS
from summarize_csra_prefix_quality_pilot import load_verified, paired_summary

ROOT=Path('eval_results/csra_neighbor_2x2_20260914')
FIELDS=('top1_correct','top5_correct','target_probability')


def contrast(a,b):
    if [r['image'] for r in a] != [r['image'] for r in b]:raise ValueError('Image pairing changed')
    return {f:paired_summary([r[f] for r in a],[r[f] for r in b]) for f in FIELDS}


def summarize():
    groups,shards,configs={}, {}, {}
    classifiers=set();pngs=0
    for mode in ('parent','off','on'):
        path=Path('eval_results/csra_staged_front_direct_control_quality256_20260914' if mode=='parent'
                  else f'eval_results/csra_neighbor_{mode}_quality256_20260914')
        groups[mode],shards[mode]=load_verified(path,prefix_depth=6,
            checkpoint_step=1600 if mode=='parent' else 2400,seed_base=20460913)
        for mp in sorted(path.glob('shard_*/metrics.json')):
            m=json.loads(mp.read_text());classifiers.add(m['classifier_sha256'])
            if mode=='parent':continue
            for arm in ('aligned','zero'):
                c=m['configs'][arm];a=c['args']
                if (c['format']!=FORMAT or c['training_protocol']!=PROTOCOL or c['step']!=2400
                        or c['parent']['sha256']!=PARENT_SHA[arm] or c['optimizer_policy']!=OPTIMIZER_POLICY
                        or c['front_keep']!=1 or c['skipped_layers']!=list(range(1,7)) or c['neighbor_rank']!=128
                        or a['neighbor']!=mode or a['coarse_source']!=arm
                        or any(a[k]!=v for k,v in FIXED_ARGS.items())):
                    raise ValueError('Changed checkpoint/protocol/data/source')
                if configs.setdefault(f'{mode}_{arm}',c)!=c:raise ValueError('Different weights across shards')
            for row in m['rows']:
                p=mp.parent/row['variant']/f"{row['local_image']:03d}.png"
                if hashlib.sha256(p.read_bytes()).hexdigest()!=row['image_sha256']:
                    raise ValueError('Missing or changed PNG')
                pngs+=1
    if pngs!=1536 or len(classifiers)!=1:raise ValueError('Missing images or changed classifier')
    original=groups['parent']['original_dense']
    if any(g['original_dense']!=original for g in groups.values()):
        raise ValueError('Original PNG/labels/classification changed across parent/off/on')
    aggregates={}
    for mode,g in groups.items():
        for arm,rows in g.items():
            if arm=='original_dense' and mode!='parent':continue
            key='original_dense' if arm=='original_dense' else f'{mode}_{arm}'
            aggregates[key]=dict(n=256,**{f:statistics.mean(float(r[f]) for r in rows) for f in FIELDS},
                relative_latent_l2=statistics.mean(s['relative_latent_l2'][arm] for s in shards[mode]))
    pairs={}
    for arm in ('aligned','zero'):
        pairs[f'on_minus_off_{arm}']=contrast(groups['on'][arm],groups['off'][arm])
        for mode in ('off','on'):
            pairs[f'{mode}_minus_parent_{arm}']=contrast(groups[mode][arm],groups['parent'][arm])
    for mode in ('off','on'):
        pairs[f'{mode}_source_minus_zero']=contrast(groups[mode]['aligned'],groups[mode]['zero'])
        pairs[f'{mode}_source_minus_original']=contrast(groups[mode]['aligned'],original)
    interaction={f:paired_summary(
        [float(a[f])-float(z[f]) for a,z in zip(groups['on']['aligned'],groups['on']['zero'])],
        [float(a[f])-float(z[f]) for a,z in zip(groups['off']['aligned'],groups['off']['zero'])]) for f in FIELDS}
    return dict(complete=True,protocol=PROTOCOL,images_sha_verified=pngs,aggregates=aggregates,
        paired_contrasts=pairs,source_by_neighbor_interaction=interaction,configurations=configs,shards=shards,
        warning='Sameadditional800, eachsource starts its corresponding fixed1600parent; all4resetAdam. Repeated256development images and one trainingseed, no multiplecomparison adjustment. Conditional image-bootstrap is not across trainingseeds. Classification is not FID/realism. Lowerridge/latentMSE is not qualitypercentage. Video test is separate untrained feasibility, not these trained weights.')


def grids():
    from PIL import Image,ImageDraw,ImageFont
    columns=[('off','original_dense','Original RDD'),('off','aligned','Reader / source'),
             ('on','aligned','Neighbor / source'),('off','zero','Reader / zero'),('on','zero','Neighbor / zero')]
    try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    except OSError:font=ImageFont.load_default()
    for rank in range(4):
        canvas=Image.new('RGB',(1280,2100),'white');draw=ImageDraw.Draw(canvas)
        for col,(mode,arm,label) in enumerate(columns):
            draw.text((col*256+5,16),label,font=font,fill='black')
            for i in range(8):
                path=Path(f'eval_results/csra_neighbor_{mode}_quality256_20260914')/f'shard_{rank}'/arm/f'{i:03d}.png'
                with Image.open(path) as picture:
                    if picture.size!=(256,256):raise ValueError('No rescaling or cherry-picking')
                    canvas.paste(picture.convert('RGB'),(col*256,52+i*256))
        canvas.save(ROOT/f'comparison_shard{rank}.png')


if __name__=='__main__':
    report=summarize();grids()
    (ROOT/'factorial_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('aggregates','paired_contrasts','source_by_neighbor_interaction')},indent=2))
