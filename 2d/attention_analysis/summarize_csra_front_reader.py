"""Summarize fixed800 read-only interventions. Never rank them as trained methods."""
import hashlib
import json
from pathlib import Path
import statistics as stats
from summarize_csra_prefix_quality_pilot import paired_summary

ROOT = Path('eval_results/csra_front_reader_audit_20260914')


def summarize(root=ROOT):
    paths = sorted(root.glob('front*/shard_*.json'))
    expected = {root/f'front{f}'/f'shard_{s}.json' for f in (0,1) for s in range(4)}
    if set(paths) != expected: raise ValueError('Missing/extra audit files')
    groups, sha, peaks, checkpoint_sha = {}, {}, [], {}
    for path in paths:
        d = json.loads(path.read_text())
        if (not d['complete'] or d['config']['step'] != 800 or len(d['rows']) != 256
                or d['forward_parity_max_abs'] or d['teacher_suffix_parity_max_abs'] or not d['rng_unchanged']):
            raise ValueError('Incomplete or failed audit')
        key = f"front{d['front']}"
        if checkpoint_sha.setdefault(key,d['config']['checkpoint_sha256']) != d['config']['checkpoint_sha256']:
            raise ValueError('Checkpoint changed across shards')
        groups.setdefault(key, []).extend(d['rows'])
        sha[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        peaks.append(d['peak_allocated_gib'])
    totals, paired = {}, {}
    for key, rows in groups.items():
        ids = sorted({(r['seed'],r['image']) for r in rows})
        if len(ids) != 128 or len({(r['seed'],r['image'],r['t']) for r in rows}) != 1024:
            raise ValueError('Repeated/missing development images')
        modes = set(rows[0]['errors'])
        totals[key] = {mode:stats.mean(r['errors'][mode]['main_mse'] for r in rows) for mode in modes}
        base = [stats.mean(r['errors']['trained_aligned']['main_mse'] for r in rows
                          if (r['seed'],r['image']) == i) for i in ids]
        paired[key] = {mode:paired_summary([stats.mean(r['errors'][mode]['main_mse'] for r in rows
                                 if (r['seed'],r['image']) == i) for i in ids],base) for mode in modes}
        totals[key]['frontend_shift_nmse'] = stats.mean(r['frontend_shift_nmse'] for r in rows)
        totals[key]['bridge_hidden_nmse'] = stats.mean(r['bridge_hidden_nmse'] for r in rows)
    return dict(complete=True,files_sha256=sha,checkpoint_sha256=checkpoint_sha,
                peak_allocated_gib=max(peaks),means=totals,paired_intervention_minus_native=paired,
                warning='Each image averages8times before bootstrap. Same128 repeated development images and single trained seed; intervals conditional on checkpoints. Counterfactual zeros/swaps may be OOD. Oracle teacher hidden requires the skipped blocks and is NOT a speedup or deployable model. Training schedule hypotheses are not proven by these interventions.')


if __name__ == '__main__':
    result = summarize()
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['means'],indent=2))
