"""Image-clustered summaries of an oracle diagnostic, never a speed claim."""
import hashlib
import json
from pathlib import Path
import statistics
from summarize_csra_prefix_quality_pilot import paired_summary

ROOT = Path('eval_results/csra_bridge_bands_20260914')


def main():
    rows, fingerprints, configurations, peaks = [], {}, {}, []
    for shard in range(4):
        p = ROOT/f'shard_{shard}.json'
        m = json.loads(p.read_text())
        if (not m['complete'] or m['version'] != 'csra_bridge_cell_bands_v1'
                or m['shard'] != shard or len(m['rows']) != 512
                or m['forward_parity_max_abs'] != 0 or m['teacher_suffix_parity_max_abs'] != 0
                or not m['rng_unchanged']):
            raise ValueError('Incomplete/unverified diagnostic')
        fingerprints[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        peaks.append(m['peak_allocated_gib'])
        for arm, c in m['configurations'].items():
            if c['step'] != 1600 or c['training_mode'] != 'direct_control':
                raise ValueError('Wrong checkpoint')
            if configurations.setdefault(arm, c) != c:
                raise ValueError('Inconsistent checkpoint')
        rows.extend(m['rows'])
    means, contrasts = {}, {}
    for arm in ('aligned', 'zero'):
        ar = [r for r in rows if r['arm'] == arm]
        grouped = {}
        for r in ar:
            grouped.setdefault((r['seed'], r['image']), []).append(r)
        if len(grouped) != 128 or any(len(g) != 8 or len({r['t'] for r in g}) != 8 for g in grouped.values()):
            raise ValueError('Require128 distinct matched images with8 times each')
        modes = ('actual','oracle_cell_mean','oracle_within_cell','oracle_full')
        output = {k:statistics.mean(r['errors'][k]['main_mse'] for r in ar) for k in modes}
        values = {k:[statistics.mean(r['errors'][k]['main_mse'] for r in g) for g in grouped.values()]
                  for k in modes}
        contrasts[arm] = {k:paired_summary(values[k], values['actual']) for k in modes if k != 'actual'}
        means[arm] = dict(main_mse=output,
            main_mse_change_pct={k:100*(v/output['actual']-1) for k,v in output.items()},
            **{k:statistics.mean(r[k] for r in ar) for k in ('hidden_mse','cell_mean_mse','within_cell_mse',
               'hidden_nmse','cell_mean_energy_per_dof','within_cell_energy_per_dof')})
        means[arm]['cell_mean_fraction'] = means[arm]['cell_mean_mse']/means[arm]['hidden_mse']
    result = dict(complete=True, means=means, paired_image_average_contrasts=contrasts,
        files_sha256=fingerprints, configurations=configurations, peak_allocated_gib=peaks,
        warning='128 reused development images,8 anchors averaged per image before bootstrap. '
                'Fixed two training seeds/arms do not estimate retraining variation; unadjusted comparisons. '
                'Oracle cut uses original teacher hidden in a trained student suffix: representation mismatch '
                'and nonlinear interactions prevent interpreting isolated replacements as causal frequency importance. '
                'No new FID, generated images, training or speed measurement.')
    # Both arms share one training seed; source mode differs, not random seed.
    result['warning'] = result['warning'].replace('Fixed two training seeds/arms', 'Fixed two checkpoints from one training seed')
    (ROOT/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
