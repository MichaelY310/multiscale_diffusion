"""Paired image-level (all8 times averaged) front placement diagnostic summary."""
import argparse
import json
from pathlib import Path
import statistics
from summarize_csra_prefix_quality_pilot import paired_summary


def summarize(root):
    metrics=[json.loads((root/f'front{i}'/'metrics.json').read_text()) for i in (0,1)]
    for front,m in enumerate(metrics):
        assert m['complete'] and m['front_keep']==front and m['skip_count']==6
        assert len(m['executed_layers'])==18 and m['suffix_teacher_parity_max_abs']==0
        assert m['images']=={'fit':192,'calibration':64,'validation':128}
    assert metrics[0]['fingerprints']==metrics[1]['fingerprints']
    fields=('main_mse','hidden_mse','target_nmse','delta_nmse')
    values={}
    for front,m in enumerate(metrics):
        for mode in m['means']:
            rows=[r for r in m['rows'] if r['mode']==mode]
            keys=sorted({(r['seed'],r['image']) for r in rows})
            assert len(keys)==128 and len(rows)==1024
            values[front,mode]={f:[statistics.mean(r[f] for r in rows if (r['seed'],r['image'])==k)
                                  for k in keys] for f in fields}
    comparisons={}
    for mode in metrics[0]['means']:
        comparisons[f'front1_minus_front0/{mode}']={f:paired_summary(values[1,mode][f],values[0,mode][f]) for f in fields}
    for front in (0,1):
        for control in ('fine_only','coarse_roll','coarse_wrong','coarse_mean'):
            comparisons[f'front{front}/aligned_minus_{control}']={
                'main_mse':paired_summary(values[front,'coarse_aligned']['main_mse'],values[front,control]['main_mse'])}
    result={'complete':True,'means':{f'front{f}':m['means'] for f,m in enumerate(metrics)},
            'paired_comparisons':comparisons,'fingerprints':metrics[0]['fingerprints'],
            'warning':'128 repeated-development images; average8 times withinimage before bootstrap. No independent trainingseed inference or multiplecomparison correction. Linear diagnostic only, not trained nonlinear bridge or rollout/FID/speed. Hidden targets differ, final output teacher is same. Delta-normalized error can worsen even if absolute/output errors improve.'}
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('eval_results/csra_front_keep6_information_20260914'))
    summarize(p.parse_args().root)
