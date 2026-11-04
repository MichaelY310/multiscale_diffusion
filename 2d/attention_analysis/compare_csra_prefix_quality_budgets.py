"""Audit and compare two evaluations of the SAME development images (stdlib)."""
import argparse
import json
from pathlib import Path
from summarize_csra_prefix_quality_pilot import load_verified, paired_summary, VARIANTS


def compare(old_root, new_root):
    old, _ = load_verified(old_root, prefix_depth=6, checkpoint_step=800, seed_base=20460913)
    new, _ = load_verified(new_root, prefix_depth=6, checkpoint_step=1600, seed_base=20460913)
    if old['original_dense'] != new['original_dense']:
        raise ValueError('Baseline images, labels or classifier predictions changed')
    classifiers = set()
    for root in (old_root, new_root):
        for path in root.glob('shard_*/metrics.json'):
            classifiers.add(json.loads(path.read_text())['classifier_sha256'])
    if len(classifiers) != 1:
        raise ValueError('Classifier mismatch')
    fields = ('top1_correct', 'top5_correct', 'target_probability')
    shifts = {}
    for arm in VARIANTS:
        if [r['target'] for r in old[arm]] != [r['target'] for r in new[arm]]:
            raise ValueError('Labels changed')
        shifts[arm] = {f: paired_summary([r[f] for r in new[arm]], [r[f] for r in old[arm]]) for f in fields}
    return dict(complete=True, unchanged_original_images=256, shifts_1600_minus_800=shifts,
                warning='Repeated fixed development set, not independent held-out evidence. Paired bootstrap over images; not over training seeds. Classification is not FID or realism.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--old-root', type=Path, required=True)
    p.add_argument('--new-root', type=Path, required=True)
    a = p.parse_args()
    report = compare(a.old_root, a.new_root)
    (a.new_root/'budget_comparison.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
