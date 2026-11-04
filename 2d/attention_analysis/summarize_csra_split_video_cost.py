"""Verify complete paired video workloads. No learned-video quality inference."""
import argparse
import hashlib
import json
from pathlib import Path
import math


def main(root):
    root=Path(root);rows=[];count=0;hashes={}
    for case in ('832_spatial','832_st','1280_spatial','1280_st'):
        path=root/case/'metrics.json';m=json.loads(path.read_text())
        hashes[case]=hashlib.sha256(path.read_bytes()).hexdigest()
        if not m['complete'] or m['groups']!=8 or m['NFE']!=3 or m['frames']!=29:
            raise ValueError('Incomplete/mismatched case')
        if m['duplicate_max_abs']!=0 or m['baseline_wrapper_max_abs']!=0:raise ValueError('Parity')
        if not m['two_models_two_readers_and_vae_resident_in_all_arms']:raise ValueError('Residency')
        if m['transition']!='noise_filling' or m['vsa_sparsity']!=.8:raise ValueError('Sampler')
        if {e['variant'] for e in m['entries']}!={'prefix6','split3_3'}:raise ValueError('Missing arm')
        for e in m['entries']:
            if len(e['groups'])!=8:raise ValueError('Missing groups')
            for i,g in enumerate(e['groups']):
                arm=e['variant'];expected=['baseline',arm,arm,'baseline'] if i%2==0 else [arm,'baseline','baseline',arm]
                if g['order']!=expected or [r['variant'] for r in g['rows']]!=expected:raise ValueError('Order')
                for r in g['rows']:
                    candidate=r['variant']!='baseline';audit=r['audit']
                    skips=list(range(6)) if arm=='prefix6' else [0,1,2,6,7,8]
                    if audit['executed']!=[m['layers'],m['layers'],m['layers']-6*int(candidate)]:raise ValueError('Executed blocks')
                    if audit['skipped']!=(skips if candidate else []):raise ValueError('Skipped blocks')
                    if audit['bridge_calls']!=(1 if arm=='prefix6' else 2)*int(candidate):raise ValueError('Reader calls')
                    if len(audit['reader_seconds'])!=audit['bridge_calls']:raise ValueError('Reader accounting')
                    if not all(math.isfinite(r[k]) and r[k]>0 for k in ('denoise_seconds','vae_seconds','total_seconds')):raise ValueError('Timing')
                    if abs(r['total_seconds']-r['denoise_seconds']-r['vae_seconds'])>1e-8:raise ValueError('Time sum')
                    count+=1
            rows.append(dict(case=case,variant=e['variant'],baseline=e['baseline'],candidate=e['candidate'],
                paired_reduction_ci95=e['paired_reduction_ci95'],bridge_seconds=e['bridge_seconds']))
        for name in ('baseline','prefix6','split3_3'):
            video=root/case/f'{name}_UNTRAINED_COST_ONLY.mp4'
            if not video.is_file() or video.stat().st_size<1000:raise ValueError('Missing video')
    if count!=256:raise ValueError('Wrong official timing count')
    result=dict(complete=True,verified_timed_trajectories=count,metrics_sha256=hashes,rows=rows,
        warning='Both video readers untrained; FastWan not RDD-trained. FP32 VAE with TF32 disabled. Not trained-video quality or optimized production pipeline performance.')
    (root/'summary_and_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');main(p.parse_args().root)
