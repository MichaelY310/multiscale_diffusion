"""Summarize fixed-checkpoint trajectory diagnosis; stdlib only."""
import argparse
import json
import math
from pathlib import Path
import statistics


def summarize(root):
    docs=[json.loads(p.read_text()) for p in sorted(root.glob('seed_*.json'))]
    if len(docs)!=4 or {(d['seed'],d['arm']) for d in docs}!={(s,a) for s in (20660914,20670914) for a in ('aligned','zero')}:
        raise ValueError('Expected exactly four completed probes')
    rows=[]
    for d in docs:
        if (not d['complete'] or d['batch']!=16 or d['prefix_depth']!=6 or d['config']['step']!=1600
            or d['instrumentation_final_max_abs']!={'teacher':0.,'student':0.} or d['coarse_parity_max_abs']!=0.
            or len(d['rows'])!=2880):raise ValueError('Invalid or unmatched diagnostic')
        gains={r['t']:r for r in d['gains']}
        if len(gains)!=90:raise ValueError('Missing times')
        keys={(r['state_role'],r['image'],r['t']) for r in d['rows']}
        if len(keys)!=2880:raise ValueError('Duplicate observations')
        for r in d['rows']:
            for k in ('main_mse','low_mse','high_mse','local_jump_mse'):
                if not math.isfinite(r[k]) or r[k]<0:raise ValueError('Invalid metric')
            g=gains[r['t']]
            rows.append(dict(seed=d['seed'],arm=d['arm'],**r,
                             jump_high_mse=r['high_mse']*g['high_gain']**2))
    table={};bins={'early_299_200':(200,299),'middle_199_100':(100,199),'late_99_1':(1,99),'final_identity_0':(0,0)}
    for arm in ('aligned','zero'):
        table[arm]={}
        for role in ('teacher','student'):
            group=[r for r in rows if r['arm']==arm and r['state_role']==role]
            total=sum(r['local_jump_mse'] for r in group)
            table[arm][role]=dict(main_mse=statistics.mean(r['main_mse'] for r in group),
                high_error_fraction=sum(r['high_mse'] for r in group)/sum(r['main_mse'] for r in group),
                local_jump_mse=statistics.mean(r['local_jump_mse'] for r in group),
                local_jump_high_fraction=sum(r['jump_high_mse'] for r in group)/total,
                by_window={name:dict(points=len({r['t'] for r in group if lo<=r['t']<=hi}),
                    local_error_sum_fraction=sum(r['local_jump_mse'] for r in group if lo<=r['t']<=hi)/total)
                    for name,(lo,hi) in bins.items()},
                by_time={str(t):dict(main_mse=statistics.mean(r['main_mse'] for r in group if r['t']==t),
                    local_jump_mse=statistics.mean(r['local_jump_mse'] for r in group if r['t']==t)) for t in sorted({r['t'] for r in group},reverse=True)})
    return dict(complete=True,table=table,rows_checked=len(rows),
        mean_final_relative_l2={a:statistics.mean(d['final_relative_latent_l2'] for d in docs if d['arm']==a) for a in ('aligned','zero')},
        warning='32 development images per arm; all90 times are correlated along trajectories. Local jump error sums/shares are descriptive, not final error attribution, FID sensitivity, or proof of on-policy improvement. Local0 identity contributes zero despite nonzero head error.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    result=summarize(a.root)
    (a.root/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    # All90 raw means stay in JSON, not a huge terminal dump.
    concise={arm:{role:{k:v for k,v in values.items() if k!='by_time'} for role,values in roles.items()} for arm,roles in result['table'].items()}
    print(json.dumps(dict(rows=result['rows_checked'],table=concise,final_l2=result['mean_final_relative_l2']),indent=2))
