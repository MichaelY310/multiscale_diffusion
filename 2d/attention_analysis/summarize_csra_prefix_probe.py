#!/usr/bin/env python3
"""Developmental paired batch-cluster bootstrap; never an independent test claim."""
import argparse
import json
import math
from pathlib import Path
import random
import statistics


def quantile(values, p):
    ordered = sorted(values)
    z = (len(ordered)-1)*p
    a = int(z)
    b = min(a+1,len(ordered)-1)
    return ordered[a]*(1-z+a)+ordered[b]*(z-a)


def bootstrap_gain(rows, reference, candidate, time=None):
    groups = {}
    for row in rows:
        if row['mode'] not in (reference,candidate) or (time is not None and row['t'] != time):
            continue
        groups.setdefault(row['seed'],{}).setdefault(row['mode'],[]).append(row['mse'])
    pairs = [(statistics.mean(g[reference]),statistics.mean(g[candidate])) for g in groups.values()]
    if len(pairs) != 16: raise ValueError('Expected16 original validation batch seeds')
    if any(len(g[reference]) != len(g[candidate]) for g in groups.values()): raise ValueError('Unpaired')
    rng = random.Random(20260913)
    values = []
    for _ in range(2000):
        draw = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        values.append(100*(1-sum(p[1] for p in draw)/sum(p[0] for p in draw)))
    return {'reference':reference,'candidate':candidate,'time':time,
            'gain_percent':100*(1-sum(p[1] for p in pairs)/sum(p[0] for p in pairs)),
            'batch_cluster_bootstrap_95':[quantile(values,.025),quantile(values,.975)],
            'batches':len(pairs)}


def main(root):
    data = json.loads((root/'metrics.json').read_text())
    if not data['complete'] or data['parity_max_abs'] != 0: raise ValueError('Incomplete/nonparity')
    rows = data['rows']
    modes = list(data['validation_mse'])
    if len(rows) != 128*8*5: raise ValueError('Incomplete prediction rows')
    summaries = {}
    for mode in modes:
        selected = [r for r in rows if r['mode']==mode]
        if len({(r['seed'],r['image'],r['t']) for r in selected}) != 128*8:
            raise ValueError('Duplicate/missing state')
        result = {k:statistics.mean(r[k] for r in selected) for k in ('mse','low_mse','high_mse','target_power')}
        result['relative_l2_rms'] = math.sqrt(result['mse']/result['target_power'])
        if abs(result['mse']-data['validation_mse'][mode]) > 1e-6: raise ValueError('Aggregation mismatch')
        if abs(result['mse']-result['low_mse']-result['high_mse']) > 1e-5: raise ValueError('Projection energy mismatch')
        summaries[mode] = result
    gains = [bootstrap_gain(rows,ref,'coarse_aligned') for ref in modes if ref != 'coarse_aligned']
    times = [bootstrap_gain(rows,'fine_only','coarse_aligned',t) for t in (299,259,215,171,128,84,40,0)]
    result = {'complete':True,'summary':summaries,'paired_gains':gains,'per_time_gains':times,
              'warning':'192fit/64calibration/128validation; hyperparameter chosen only on calibration. Descriptive paired bootstrap of16 batch seeds (8images per seed, preserve wrong-source within-batch dependence),2000resamples. One fixed teacher/data subset; validation previously used in other studies; no training-seed repetition, multiplicity correction or independenttest. Hidden prediction, not FID/quality/speed.'}
    (root/'paired_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('eval_results/csra_prefix_information_20260913'))
    main(p.parse_args().root)
