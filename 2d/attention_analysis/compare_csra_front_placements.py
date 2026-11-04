"""Matched fixed800 front0/front1 quality comparison, not a speed/FID claim."""
import json
from pathlib import Path
import statistics
from summarize_csra_prefix_quality_pilot import load_verified,paired_summary,VARIANTS


def compare(old_root,new_root):
    old,os=load_verified(old_root,prefix_depth=6,checkpoint_step=800,seed_base=20460913)
    new,ns=load_verified(new_root,prefix_depth=6,checkpoint_step=800,seed_base=20460913)
    if old['original_dense']!=new['original_dense']:raise ValueError('Reference/labels/classifier changed')
    classifiers=set()
    for root in (old_root,new_root):
        for p in root.glob('shard_*/metrics.json'):
            d=json.loads(p.read_text());classifiers.add(d['classifier_sha256'])
            for arm in ('aligned','zero'):
                a=d['configs'][arm]['args']
                assert a['max_steps']==800 and a['seed']==20279000 and a['coarse_source']==arm
                assert a['base_ckpt']=='checkpoints/RDD-Unified-L/0400000.pt'
                assert a['data_root']=='eval_results/csra_rollout_data4k_20260912'
    if len(classifiers)!=1:raise ValueError('Classifier mismatch')
    fields=('top1_correct','top5_correct','target_probability')
    pairs={arm:{f:paired_summary([r[f] for r in new[arm]],[r[f] for r in old[arm]])
                for f in fields} for arm in VARIANTS}
    interactions={f:paired_summary([float(a[f])-float(z[f]) for a,z in zip(new['aligned'],new['zero'])],
                                  [float(a[f])-float(z[f]) for a,z in zip(old['aligned'],old['zero'])]) for f in fields}
    l2={}
    for arm in VARIANTS:
        a=statistics.mean(s['relative_latent_l2'][arm] for s in os)
        b=statistics.mean(s['relative_latent_l2'][arm] for s in ns)
        l2[arm]={'front0':a,'front1':b,'relative_change_percent':100*(b/a-1) if a else None}
    return dict(complete=True,checkpoint_steps=[800,800],paired_front1_minus_front0=pairs,
                source_by_placement_interaction=interactions,relative_latent_l2=l2,
                warning='Same256 repeated development images; paired image-bootstrap intervals, not training-seed intervals or newholdout. No multiplecomparison correction. Equal18 executedblocks/6skips, samefull-width bridge andtrainingbudget; differentplacement anditscalibrationinitializer. No FID/speed evidence; Top1 doesnotmeasure realism.')


if __name__=='__main__':
    old=Path('eval_results/csra_prefix6_quality256_20260913')
    new=Path('eval_results/csra_front1_skip6_quality256_20260914')
    result=compare(old,new)
    (new/'placement_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
