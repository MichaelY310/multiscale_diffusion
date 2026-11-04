"""Matched training-schedule x coarse-source factorial quality pilot (not FID)."""
import json
from pathlib import Path
import statistics
from csra_staged_protocol import PROTOCOL
from summarize_csra_prefix_quality_pilot import load_verified, paired_summary

FIELDS = ('top1_correct','top5_correct','target_probability')
MODES = ('staged','direct_control')
ROOT = Path('eval_results/csra_staged_front_review_20260914')


def contrast(left, right):
    if [r['image'] for r in left] != [r['image'] for r in right]: raise ValueError('Unmatched images')
    return {f:paired_summary([r[f] for r in left],[r[f] for r in right]) for f in FIELDS}


def summarize():
    groups, shards, configs, classifiers = {}, {}, {}, set()
    audit = json.loads((ROOT/'completion_audit.json').read_text())
    if not audit['complete'] or audit['images_sha_verified'] != 1536: raise ValueError('Require full completion audit')
    for mode in MODES:
        path = Path(f'eval_results/csra_staged_front_{mode}_quality256_20260914')
        groups[mode],shards[mode] = load_verified(path,prefix_depth=6,checkpoint_step=1600,seed_base=20460913)
        for p in sorted(path.glob('shard_*/metrics.json')):
            m = json.loads(p.read_text())
            classifiers.add(m['classifier_sha256'])
            for arm in ('aligned','zero'):
                c = m['configs'][arm]
                a = c['args']
                if (c['training_protocol'] != PROTOCOL or c['training_mode'] != mode or c['front_keep'] != 1
                        or c['skipped_layers'] != list(range(1,7)) or c['step'] != 1600
                        or a['mode'] != mode or a['coarse_source'] != arm or a['seed'] != 20279000
                        or a['data_root'] != 'eval_results/csra_rollout_data4k_20260912'
                        or c['checkpoint_sha256'] != audit['checkpoints'][f'{mode}_{arm}']['sha256']):
                    raise ValueError('Wrong fixed1600 model/data/protocol')
                key = f'{mode}_{arm}'
                if configs.setdefault(key,c) != c: raise ValueError('Cross-shard checkpoint mismatch')
    groups['parent800'],shards['parent800'] = load_verified(
        Path('eval_results/csra_front1_skip6_quality256_20260914'),prefix_depth=6,checkpoint_step=800,seed_base=20460913)
    original = groups['parent800']['original_dense']
    if len(classifiers) != 1 or any(g['original_dense'] != original for g in groups.values()):
        raise ValueError('Changed baseline images/labels/classification')
    aggregates = {}
    for mode,g in groups.items():
        for arm,rows in g.items():
            if arm == 'original_dense' and mode != 'staged': continue
            key = 'original_dense' if arm == 'original_dense' else f'{mode}_{arm}'
            aggregates[key] = dict(n=256,**{f:statistics.mean(float(r[f]) for r in rows) for f in FIELDS},
                latent_relative_l2=statistics.mean(s['relative_latent_l2'][arm] for s in shards[mode]))
    pairs = {}
    for arm in ('aligned','zero'):
        pairs[f'staged_minus_direct_{arm}'] = contrast(groups['staged'][arm],groups['direct_control'][arm])
        for mode in MODES:
            pairs[f'{mode}1600_minus_parent800_{arm}'] = contrast(groups[mode][arm],groups['parent800'][arm])
    for mode in MODES:
        pairs[f'{mode}_aligned_minus_zero'] = contrast(groups[mode]['aligned'],groups[mode]['zero'])
        pairs[f'{mode}_aligned_minus_original'] = contrast(groups[mode]['aligned'],original)
    interaction = {f:paired_summary(
        [float(a[f])-float(z[f]) for a,z in zip(groups['staged']['aligned'],groups['staged']['zero'])],
        [float(a[f])-float(z[f]) for a,z in zip(groups['direct_control']['aligned'],groups['direct_control']['zero'])]) for f in FIELDS}
    return dict(complete=True,aggregates=aggregates,paired_contrasts=pairs,
        source_by_training_schedule_interaction=interaction,configurations=configs,shards=shards,
        warning='Same1600 totalupdates; staged deliberately givesfine800 updates versusdirect1600. Samefixed inferencegraph,source/data/loss/precision. Parent800 is learningcurve context, not equalbudget. Repeated256 development images, bootstrap conditional on fixedsingle-training-seed checkpoints, not independent trainingseeds; unadjusted comparisons. Classification not FID/realism, latentdeviation not qualitypercentage. No video or speed claim.')


if __name__ == '__main__':
    report = summarize()
    (ROOT/'factorial_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('aggregates','paired_contrasts','source_by_training_schedule_interaction')},indent=2))
