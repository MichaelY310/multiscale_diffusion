#!/usr/bin/env python3
"""Aggregate the completed all-time probe, with no new GPU work."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diagnose_csra_prefix_all_times import bracket_reference
from probe_csra_prefix_information import TIMES, write_json


def main():
    root = Path('eval_results/csra_prefix8_all_times_20260913')
    paths = sorted(root.glob('budget_*/seed_*.json'))
    if len(paths)!=8: raise ValueError('Expected exactly eight completed probes')
    groups={}
    for path in paths:
        m=json.loads(path.read_text())
        if not m['complete'] or m['teacher_final_parity_max_abs'] or m['coarse_endpoint_parity_max_abs']:
            raise ValueError('Incomplete or failed parity')
        key=f"{m['budget']}_{m['arm']}"
        groups.setdefault(key,[]).extend(m['rows'])
    if set(groups)!={'1600_aligned','1600_zero','3200_aligned','3200_zero'}:
        raise ValueError('Wrong budget/source groups')
    summary={}
    fig,axes=plt.subplots(3,1,figsize=(12,12),sharex=True)
    for key,rows in groups.items():
        times=sorted({r['t'] for r in rows},reverse=True)
        if len(rows)!=720 or len(times)!=90: raise ValueError('Missing records')
        points=[]
        for t in times:
            sample=[r for r in rows if r['t']==t]
            avg=lambda k:sum(r[k] for r in sample)/len(sample)
            points.append(dict(t=t,main_mse=avg('main_mse'),target_power=avg('target_power'),
                               low_mse=avg('low_mse'),high_mse=avg('high_mse')))
        anchors={r['t']:r['main_mse'] for r in points if r['t'] in TIMES}
        for r in points:
            r['anchor_reference']=bracket_reference(r['t'],anchors)
            r['ratio_to_anchor_interpolation']=r['main_mse']/max(r['anchor_reference'],1e-20)
            r['normalized_mse']=r['main_mse']/max(r['target_power'],1e-20)
        between=[r for r in points if r['t'] not in TIMES]
        interior=[r for r in between if r['t']>40]
        aggregate=lambda rs:sum(r['main_mse'] for r in rs)/sum(r['anchor_reference'] for r in rs)
        intervals=[]
        for lo,hi in zip(sorted(TIMES)[:-1],sorted(TIMES)[1:]):
            select=[r for r in between if lo<r['t']<hi]
            intervals.append(dict(lo=lo,hi=hi,points=len(select),
                aggregate_mse_over_anchor_reference=aggregate(select),
                maximum_point_ratio=max(r['ratio_to_anchor_interpolation'] for r in select)))
        summary[key]=dict(points=points,intervals=intervals,
            all_untrained_aggregate_ratio=aggregate(between),
            untrained_above40_aggregate_ratio=aggregate(interior),
            maximum_point_ratio=max(r['ratio_to_anchor_interpolation'] for r in between))
        color={'1600_aligned':'#2166ac','1600_zero':'#ef8a62','3200_aligned':'#053061','3200_zero':'#b2182b'}[key]
        label=key.replace('_',' / ')
        style='--' if key.startswith('1600') else '-'
        for ax,field in zip(axes,('main_mse','normalized_mse','ratio_to_anchor_interpolation')):
            ax.plot(times,[r[field] for r in points],style,color=color,label=label,linewidth=1.5)
            selected=[r for r in points if r['t'] in TIMES]
            ax.scatter([r['t'] for r in selected],[r[field] for r in selected],color=color,s=17,zorder=4)
    for ax,label in zip(axes,('Main-head MSE (same teacher state)','MSE / teacher target power','MSE / linearly interpolated trained-time MSE')):
        ax.set_ylabel(label);ax.grid(alpha=.25);ax.set_yscale('log')
        for t in TIMES: ax.axvline(t,color='gray',alpha=.2,linewidth=.8)
    axes[-1].axhline(1,color='black',linewidth=1,alpha=.4)
    axes[-1].set_xlabel('Global fine timestep (denoising: left to right)')
    axes[-1].set_xlim(299,0)
    axes[0].legend(ncol=2)
    fig.suptitle('8 trained times vs all90 sampling times | 2 development seeds x4 images\nDots: trained times; all predictions evaluated on identical ORIGINAL teacher trajectories')
    fig.tight_layout(rect=[0,0,1,.95])
    fig.savefig(root/'all_time_error_curves.png',dpi=160)
    plt.close(fig)
    write_json(root/'summary.json',dict(complete=True,groups=summary,files=[str(p) for p in paths],
        warning='Interpolation is a descriptive error reference, not a causal model. Time/noise/target difficulty co-vary. Report interval0–40 separately because final t0 output is special. Ratios use aggregated MSE over eight images, not averaged per-image relative errors. Development diagnostic only; no FID or speed claim.'))
    print(json.dumps({k:{f:v for f,v in d.items() if f not in ('points','intervals')} for k,d in summary.items()}))


if __name__=='__main__': main()
