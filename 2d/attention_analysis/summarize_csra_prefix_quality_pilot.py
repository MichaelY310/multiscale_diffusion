"""Read-only-input summary of the matched 256-image pilot (stdlib only).

Classification is a semantic diagnostic, not FID, realism, or a timing result.
Bootstrap intervals resample matched image pairs, not independently each arm.
They are conditional pilot intervals; four RNG shards are not four model seeds.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics


VARIANTS = ('original_dense', 'aligned', 'zero')


def paired_summary(left, right, repetitions=10000, seed=20369999):
    if len(left) != len(right) or not left:
        raise ValueError('Need nonempty matched arrays')
    differences = [float(a) - float(b) for a, b in zip(left, right)]
    if not all(math.isfinite(x) for x in differences):
        raise ValueError('Nonfinite metric')
    rng = random.Random(seed)
    n = len(differences)
    means = sorted(sum(rng.choices(differences, k=n)) / n for _ in range(repetitions))
    return dict(mean_difference=statistics.mean(differences),
                paired_bootstrap_95=[means[int(.025 * repetitions)], means[int(.975 * repetitions)]],
                left_greater=sum(d > 0 for d in differences),
                right_greater=sum(d < 0 for d in differences),
                equal=sum(d == 0 for d in differences), n=n)


def load_verified(root, *, prefix_depth=8, checkpoint_step=3200, seed_base=20360913):
    metrics = sorted(root.glob('shard_*/metrics.json'))
    if len(metrics) != 4:
        raise ValueError('Need exactly four completed shards')
    grouped = {v: {} for v in VARIANTS}
    shards = []
    for rank, path in enumerate(metrics):
        m = json.loads(path.read_text(encoding='utf-8'))
        if (not m['complete'] or m['rank'] != rank or m['seed'] != seed_base + rank * 10000
                or m['images_per_variant'] != 64 or len(m['rows']) != 192
                or m['coarse_endpoint_max_abs'] != {'aligned': 0., 'zero': 0.}
                or m['steps'] != [80, 80, 90] or m['cfg'] != 1. or m['upsample'] != 'noise_filling'
                or m['prefix_depth'] != prefix_depth
                or any(m['configs'][v]['step'] != checkpoint_step for v in ('aligned', 'zero'))):
            raise ValueError(f'Wrong/incomplete configuration: {path}')
        for row in m['rows']:
            v, i = row['variant'], row['image']
            if v not in grouped or i in grouped[v] or i != rank * 64 + row['local_image']:
                raise ValueError('Duplicate/mismatched image')
            if row['target'] != m['labels'][row['local_image']]:
                raise ValueError('Wrong actual class label')
            grouped[v][i] = row
        shards.append(dict(rank=rank, seed=m['seed'], peak_memory_gib=m['peak_memory_gib'],
                           relative_latent_l2=m['relative_latent_l2'],
                           metrics_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    for v in VARIANTS:
        if set(grouped[v]) != set(range(256)):
            raise ValueError('Incomplete variant')
        for i in range(256):
            if grouped[v][i]['target'] != grouped['original_dense'][i]['target']:
                raise ValueError('Unpaired labels')
    return {v: [grouped[v][i] for i in range(256)] for v in VARIANTS}, shards


def main(root, *, prefix_depth=8, checkpoint_step=3200, seed_base=20360913):
    groups, shards = load_verified(root,prefix_depth=prefix_depth,checkpoint_step=checkpoint_step,seed_base=seed_base)
    aggregates, pairs = {}, {}
    fields = ('top1_correct', 'top5_correct', 'target_probability')
    for v, rows in groups.items():
        aggregates[v] = dict(n=256, top1_count=sum(r['top1_correct'] for r in rows),
                             top5_count=sum(r['top5_correct'] for r in rows),
                             **{f: statistics.mean(float(r[f]) for r in rows) for f in fields},
                             mean_shard_relative_latent_l2=statistics.mean(
                                 s['relative_latent_l2'][v] for s in shards))
    for a, b in (('aligned', 'zero'), ('aligned', 'original_dense'), ('zero', 'original_dense')):
        pairs[f'{a}_minus_{b}'] = {f: paired_summary([r[f] for r in groups[a]],
                                                   [r[f] for r in groups[b]]) for f in fields}
    report = dict(complete=True, prefix_depth=prefix_depth,checkpoint_step=checkpoint_step,seed_base=seed_base,
                  aggregates=aggregates, paired_comparisons=pairs, shards=shards,
                  unique_target_classes=len({r['target'] for r in groups['original_dense']}),
                  displayed_rows={str(rank): [dict(image=i, target=groups['original_dense'][i]['target'],
                       name=groups['original_dense'][i]['target_name']) for i in range(rank*64, rank*64+8)]
                       for rank in range(4)},
                  warning='Development pilot, not FID or speed; reused seeds are not a fresh held-out test. Paired bootstrap over256 images,10000 resamples,seed20369999; conditional on the fixed checkpoints and four RNG shards. No multiple-comparison adjustment, no between-training-seed inference. Classifier scores do not prove realism. relative_l2 averages per-image norm(error)/norm(reference); averaging equal64-image shard means gives the same256-image mean.')
    (root/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('eval_results/csra_prefix8_quality256_20260913'))
    parser.add_argument('--prefix-depth',type=int,default=8)
    parser.add_argument('--checkpoint-step',type=int,default=3200)
    parser.add_argument('--seed-base',type=int,default=20360913)
    args=parser.parse_args()
    main(args.root,prefix_depth=args.prefix_depth,checkpoint_step=args.checkpoint_step,seed_base=args.seed_base)
