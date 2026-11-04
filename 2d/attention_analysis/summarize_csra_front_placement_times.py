"""Matched front0/front1 trained800 diagnosis; aggregate images, not 2880 IID times."""
import hashlib
import json
import sys
from pathlib import Path
import statistics
from summarize_csra_prefix_quality_pilot import paired_summary

ANCHORS = {299, 259, 215, 171, 128, 84, 40, 0}
SEEDS = (20860914, 20870914)
ARMS = ('aligned', 'zero')
ROOT = Path('eval_results/csra_front_placement_times_20260914')


def summarize(root=ROOT):
    paths = sorted(root.glob('front*/seed_*.json'))
    expected = {root/f'front{front}'/f'seed_{seed}_{arm}.json'
                for front in (0, 1) for seed in SEEDS for arm in ARMS}
    if set(paths) != expected:
        raise ValueError('Require exactly eight prespecified probes')
    groups, fingerprints, reference, final, peaks = {}, {}, {}, {}, []
    for path in paths:
        data = json.loads(path.read_text())
        front = int(path.parent.name[-1])
        if (not data['complete'] or data['config']['step'] != 800 or data['batch'] != 16
                or data['steps'] != [80, 80, 90] or data['cfg'] != 1.
                or data['upsample'] != 'noise_filling' or data['coarse_parity_max_abs'] != 0.
                or data['instrumentation_final_max_abs'] != {'teacher': 0., 'student': 0.}):
            raise ValueError('Checkpoint/sampler/parity contract changed')
        key = f"front{front}_{data['arm']}"
        group = groups.setdefault(key, [])
        seen = set()
        for row in data['rows']:
            identifier = (row['state_role'], row['image'], row['t'])
            if identifier in seen:
                raise ValueError('Duplicate time/image')
            seen.add(identifier)
            if row['state_role'] == 'teacher':
                # Every arm must see the exact same teacher trajectory and labels.
                rk = (data['seed'], row['image'], row['t'])
                rv = (row['label'], row['target_power'], row['low_target_power'], row['high_target_power'])
                if reference.setdefault(rk, rv) != rv:
                    raise ValueError('Teacher-state baseline differs across arms')
            group.append(dict(seed=data['seed'], **row))
        times = {row['t'] for row in data['rows']}
        expected_rows = {(role, i, t) for role in ('teacher', 'student') for i in range(16) for t in times}
        if len(times) != 90 or not ANCHORS <= times or seen != expected_rows:
            raise ValueError('Missing observations')
        final.setdefault(key, []).append(data['final_relative_latent_l2'])
        peaks.append(data['peak_allocated_gib'])
        fingerprints[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    subsets = {'all90': lambda r: True, 'anchors8': lambda r: r['t'] in ANCHORS,
               'between82': lambda r: r['t'] not in ANCHORS,
               'early299_202': lambda r: r['t'] >= 202,
               'middle201_101': lambda r: 101 <= r['t'] < 202,
               'late100_0': lambda r: r['t'] <= 100}
    totals, curves, per_image = {}, {}, {}
    for key, rows in groups.items():
        totals[key], curves[key], per_image[key] = {}, {}, {}
        for role in ('teacher', 'student'):
            relevant = [r for r in rows if r['state_role'] == role]
            curves[key][role] = [dict(t=t, **{f:statistics.mean(r[f] for r in relevant if r['t'] == t)
                for f in ('main_mse', 'target_power', 'low_mse', 'high_mse', 'local_jump_mse')})
                for t in sorted({r['t'] for r in relevant}, reverse=True)]
            totals[key][role], per_image[key][role] = {}, {}
            for subset, take in subsets.items():
                selected = [r for r in relevant if take(r)]
                totals[key][role][subset] = {f:statistics.mean(r[f] for r in selected)
                    for f in ('main_mse', 'target_power', 'low_mse', 'high_mse', 'local_jump_mse')}
                per_image[key][role][subset] = [statistics.mean(r['main_mse'] for r in selected
                    if r['seed'] == seed and r['image'] == i) for seed in SEEDS for i in range(16)]
        totals[key]['final_relative_latent_l2_mean_of_two_batches'] = statistics.mean(final[key])
    pairs = {arm:{role:{part:paired_summary(per_image[f'front1_{arm}'][role][part],
                                           per_image[f'front0_{arm}'][role][part])
                        for part in subsets} for role in ('teacher', 'student')} for arm in ARMS}
    return dict(complete=True,files_sha256=fingerprints,images_per_arm=32,rows=23040,
                peak_allocated_gib=max(peaks),means=totals,curves=curves,
                paired_front1_minus_front0=pairs,
                warning='Same32 development images, one trained checkpoint per arm. Bootstrap images after averaging times; not independent training seeds. Teacher targets on matched teacher states are exact across arms. On student states each teacher is recomputed on that student state, so cross-arm state difficulty changes. Anchor/between averages have different time distributions, not a causal interpolation test. Local jump error is not final FID attribution; no speed or quality claims.')


def plot(report, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    colors = {'front0_aligned':'#2166ac', 'front1_aligned':'#b2182b',
              'front0_zero':'#67a9cf', 'front1_zero':'#ef8a62'}
    for ax, role in zip(axes, ('teacher', 'student')):
        for key, data in report['curves'].items():
            points = data[role]
            ax.plot([r['t'] for r in points], [r['main_mse'] for r in points],
                    '--' if key.endswith('zero') else '-', color=colors[key], label=key)
            anchors = [r for r in points if r['t'] in ANCHORS]
            ax.scatter([r['t'] for r in anchors], [r['main_mse'] for r in anchors],
                       color=colors[key], s=14)
        ax.set_title('Matched original teacher states' if role == 'teacher' else 'Each student\'s own states')
        ax.set_xlim(299, 0)
        ax.set_xlabel('Fine timestep (denoising left to right)')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Main-head MSE to teacher evaluated at current state')
    axes[1].legend(fontsize=8)
    fig.suptitle('Fixed800, same32 development images | dots: 8 trained times; lines: all90')
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    report = summarize()
    (ROOT/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    if '--plot' in sys.argv:
        plot(report, ROOT/'all90_front_comparison.png')
    print(json.dumps(report['means'], indent=2))
