#!/usr/bin/env python3
"""Audit matched auxiliary-supervision results, never infer FID or speed."""
import json
import hashlib
from pathlib import Path
import statistics
import argparse


def summarize(root):
    arms={}; original_hashes={}; grids=[]
    for aux in ('on','off'):
        for source in ('aligned','zero'):
            name=f'{aux}_{source}'
            complete=root/f'checkpoints/RDD-CSRA-prefix8-hidden-{aux}-{source}-L400K-20260913/complete.json'
            done=json.loads(complete.read_text())
            assert done['complete'] and not done['smoke'] and done['step']==4000
            events=[]
            for line in (root/f'logs/csra_prefix8_hidden_{name}_20260913.log').read_text().splitlines():
                if line.startswith('{'):
                    try:events.append(json.loads(line))
                    except json.JSONDecodeError:pass
            val=[e for e in events if e.get('event')=='validation' and e['step']==4000][-1]
            assert abs(val['objective']-done['best_objective'])<1e-12
            arm={'validation':val,'samples':[],'experiment':done['experiment']}
            for seed in (20260921,20260923,20275004):
                d=root/f'eval_results/csra_prefix8_hidden_{aux}_pair_20260913/seed_{seed}'
                m=json.loads((d/'metrics.json').read_text()); key=f'prefix_{source}'
                assert m['complete'] and m['coarse_endpoint_max_abs_vs_original'][key]==0
                c=m['configurations'][key]
                assert c['step']==4000 and c['hidden_supervision']['aux']==aux
                assert c['fine_calls']==c['reused_fine_calls']==90
                assert len(list(d.glob('*/*.png')))==12 and (d/'comparison_grid.png').is_file()
                for p in sorted((d/'original_dense').glob('*.png')):
                    digest=hashlib.sha256(p.read_bytes()).hexdigest(); hk=(seed,p.name)
                    assert original_hashes.setdefault(hk,digest)==digest
                arm['samples'].append({'seed':seed,'batch_relative_latent_l2':m['relative_latent_l2'][key]})
                grids.append(str((d/'comparison_grid.png').relative_to(root)))
            arm['mean_batch_relative_latent_l2']=statistics.mean(s['batch_relative_latent_l2'] for s in arm['samples'])
            arms[name]=arm
    def decrease(a,b):return 100*(1-a/b)
    comparisons={source:{
        'main_mse_reduction_pct':decrease(arms[f'on_{source}']['validation']['main_mse'],arms[f'off_{source}']['validation']['main_mse']),
        'hidden_nmse_reduction_pct':decrease(arms[f'on_{source}']['validation']['hidden_nmse'],arms[f'off_{source}']['validation']['hidden_nmse']),
        'mean_latent_l2_reduction_pct':decrease(arms[f'on_{source}']['mean_batch_relative_latent_l2'],arms[f'off_{source}']['mean_batch_relative_latent_l2'])
    } for source in ('aligned','zero')}
    return dict(complete=True,arms=arms,aux_vs_equal_budget_off=comparisons,grids=sorted(set(grids)),
                warning='3 reused development seeds,4 images each; means of batch relative L2, not per-image mean/FID/semantic quality/speed. All selected best4000; all original8/16 unchanged. Hidden loss explicitly optimized; its decrease alone is not evidence of improved images.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--output',default='eval_results/csra_prefix8_hidden_2x2_summary_20260913.json')
    a=p.parse_args();result=summarize(Path(a.root));Path(a.output).write_text(json.dumps(result,indent=2))
    print(json.dumps({'aux_vs_equal_budget_off':result['aux_vs_equal_budget_off'],
                      'means':{k:v['mean_batch_relative_latent_l2'] for k,v in result['arms'].items()}}))
