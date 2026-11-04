#!/usr/bin/env python3
"""Paired descriptive bootstrap, sampling entire trajectories not time rows."""
import json
import random
from pathlib import Path


def summarize(root):
    output = {}
    for version in ('v2','v5'):
        pair = [json.loads((root/f'{version}_{mode}'/'metrics.json').read_text()) for mode in ('aligned','zero')]
        assert all(p['complete'] and len(p['rows']) == 1024 for p in pair)
        arms = [{(r['seed'],r['image'],r['t']):r for r in p['rows']} for p in pair]
        assert arms[0].keys() == arms[1].keys() and len(arms[0]) == 1024
        trajectories = sorted({k[:2] for k in arms[0]})
        assert len(trajectories) == 128
        result = {}
        times = sorted({k[2] for k in arms[0]},reverse=True)
        for t in times:
            result[str(t)] = {}
            for metric in ('main_mse','low_error_mse','high_error_mse'):
                a,z = [[arm[(*traj,t)][metric] for traj in trajectories] for arm in arms]
                rng = random.Random(20260913)
                estimates=[]
                for _ in range(2000):
                    ids = [rng.randrange(len(a)) for _ in a]
                    estimates.append(100*(1-sum(a[i] for i in ids)/sum(z[i] for i in ids)))
                estimates.sort()
                result[str(t)][metric] = {'aligned':sum(a)/len(a),'zero':sum(z)/len(z),
                    'gain_percent':100*(1-sum(a)/sum(z)),
                    'pointwise_95_bootstrap':[estimates[49],estimates[1949]]}
        output[version]=result
    payload={'results':output,'warning':'Descriptive paired trajectory bootstrap on checkpoint-selection validation. Pointwise intervals not corrected for multiple times/bands. No repeated training seeds, independent test, image-quality or speed inference.'}
    (root/'paired_summary.json').write_text(json.dumps(payload,indent=2,allow_nan=False))
    for version,results in output.items():
        print(version)
        for t,metrics in results.items():
            print(t, ' / '.join(f'{k}: {v["gain_percent"]:+.3f}% [{v["pointwise_95_bootstrap"][0]:+.3f}, {v["pointwise_95_bootstrap"][1]:+.3f}]'
                               for k,v in metrics.items()))


if __name__=='__main__':
    summarize(Path('eval_results/csra_time_source_20260913'))
