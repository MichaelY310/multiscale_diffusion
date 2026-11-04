"""Audit the fixed2400 loss/source factorial experiment against fixed1600.

Stdlib only. All inputs are read-only. No checkpoint selection, image selection,
sampling, training or FID computation occurs here. The repeated development
image bootstrap is conditional on these particular trained checkpoints.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from summarize_csra_prefix_quality_pilot import load_verified, paired_summary

PARENT_SHA = {
    'aligned':'bed405396a43aa01e987c9e2ae5cbb3cdcca14f158846d40a31f56a1e6dad689',
    'zero':'9dcd2b80501d5bb3593fd73f5e55f7dcb33d1ddc85839ba41782debfb22b2156',
}
FIELDS = ('top1_correct','top5_correct','target_probability')


def check_checkpoint(config, loss, source, calibration_sha):
    e = config['jump_supervision']
    a = config['args']
    expected_name = f'RDD-CSRA-prefix6-jump-loss-{loss}-{source}-L400K-20260914'
    p = Path(config['checkpoint'])
    if (p.name != 'latest.pt' or p.parent.name != expected_name or config['step'] != 2400
            or e['name'] != 'prefix6_jump_loss_2x2_v1' or e['loss_mode'] != loss
            or e['parent_step'] != 1600 or e['fixed_endpoint'] != 2400 or e['extra_steps'] != 800
            or e['parent_sha256'] != PARENT_SHA[source] or e['calibration_sha256'] != calibration_sha
            or e['per_gpu_batch'] != 64 or e['world_size'] != 4 or e['accumulation'] != 1
            or a['coarse_source'] != source or a['jump_loss_mode'] != loss
            or a['bridge_kind'] != 'conditional_attention' or a['prefix_depth'] != 6
            or a['seed'] != 20279000):
        raise ValueError('Wrong fixed endpoint, objective, source, parent or training configuration')


def same_reference(groups):
    originals = [v['original_dense'] for v in groups.values()]
    if any(rows != originals[0] for rows in originals[1:]):
        raise ValueError('Dense PNG hashes, labels or classification outputs changed across comparisons')


def paired_fields(left, right, repetitions=10000):
    if [r['image'] for r in left] != [r['image'] for r in right]:
        raise ValueError('Mismatched image order')
    return {f:paired_summary([r[f] for r in left],[r[f] for r in right],repetitions=repetitions)
            for f in FIELDS}


def interaction(groups, repetitions=10000):
    # Per-image difference-in-differences, paired BEFORE bootstrapping.
    return {f:paired_summary(
        [float(a[f])-float(z[f]) for a,z in zip(groups['jump']['aligned'],groups['jump']['zero'])],
        [float(a[f])-float(z[f]) for a,z in zip(groups['old']['aligned'],groups['old']['zero'])],
        repetitions=repetitions) for f in FIELDS}


def summarize(root, parent_root):
    calibration_sha = hashlib.sha256((root/'calibration.json').read_bytes()).hexdigest()
    groups, shards, config_by_arm = {}, {}, {}
    classifier_shas, trainer_shas = set(), set()
    paths = dict(parent=parent_root,old=root/'quality_old',jump=root/'quality_jump')
    for name, path in paths.items():
        groups[name], shards[name] = load_verified(path,prefix_depth=6,
            checkpoint_step=1600 if name=='parent' else 2400,seed_base=20460913)
        for mp in sorted(path.glob('shard_*/metrics.json')):
            m = json.loads(mp.read_text(encoding='utf-8'))
            classifier_shas.add(m['classifier_sha256'])
            if name != 'parent':
                for source in ('aligned','zero'):
                    config = m['configs'][source]
                    check_checkpoint(config,name,source,calibration_sha)
                    trainer_shas.add(config['jump_supervision']['trainer_sha256'])
                    key = name+'_'+source
                    if key in config_by_arm and config_by_arm[key] != config:
                        raise ValueError('Ranks used inconsistent checkpoint configurations')
                    config_by_arm[key] = config
    if len(classifier_shas) != 1 or len(trainer_shas) != 1:
        raise ValueError('Classifier or trainer source changed across arms')
    same_reference(groups)
    aggregates = {}
    for name, group in groups.items():
        for arm, rows in group.items():
            if arm=='original_dense' and name!='parent':
                continue
            key = 'original_dense' if arm=='original_dense' else name+'_'+arm
            aggregates[key] = dict(n=len(rows),top1_count=sum(r['top1_correct'] for r in rows),
                top5_count=sum(r['top5_correct'] for r in rows),
                **{f:statistics.mean(float(r[f]) for r in rows) for f in FIELDS},
                relative_latent_l2=statistics.mean(s['relative_latent_l2'][arm] for s in shards[name]))
    contrasts = {}
    for arm in ('aligned','zero'):
        contrasts[f'jump_minus_old_{arm}'] = paired_fields(groups['jump'][arm],groups['old'][arm])
        for mode in ('old','jump'):
            contrasts[f'{mode}2400_minus_parent1600_{arm}'] = paired_fields(groups[mode][arm],groups['parent'][arm])
    for mode in ('parent','old','jump'):
        contrasts[f'{mode}_aligned_minus_zero'] = paired_fields(groups[mode]['aligned'],groups[mode]['zero'])
        contrasts[f'{mode}_aligned_minus_original'] = paired_fields(groups[mode]['aligned'],groups[mode]['original_dense'])
    return dict(complete=True,aggregates=aggregates,paired_contrasts=contrasts,
        source_loss_interaction=interaction(groups),unchanged_dense_images=256,
        calibration_sha256=calibration_sha,classifier_sha256=next(iter(classifier_shas)),
        trainer_sha256=next(iter(trainer_shas)),shards=shards,configurations=config_by_arm,
        warning='Matched fixed2400 endpoints; parent fixed1600. Repeated256-image development set, not independent held-out evidence. '
                'Paired bootstrap over images conditional on checkpoints, not training seeds; unadjusted multiple comparisons. '
                'Classification and latent deviation do not measure FID/realism. Latent value is per-image normratio averaged; '
                'equal64-image shard averages preserve the256-image mean. Training-only loss change adds no inference compute.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('eval_results/csra_prefix6_jump_loss_20260914'))
    p.add_argument('--parent-root',type=Path,default=Path('eval_results/csra_prefix6_1600_quality256_20260914'))
    args=p.parse_args();report=summarize(args.root,args.parent_root)
    (args.root/'factorial_summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('complete','aggregates','paired_contrasts','source_loss_interaction','warning')},indent=2))
