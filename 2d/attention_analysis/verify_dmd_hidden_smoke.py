"""Audit matched hidden-on/off smoke telemetry. No model load or GPU use."""
import argparse
import json
import math
from pathlib import Path


def audit(root):
    runs={}
    for name in ('baseline','hidden'):
        folder=root/f'smoke_{name}'
        protocol=json.loads((folder/'protocol.json').read_text())
        progress=json.loads((folder/'progress.json').read_text())
        rows=[json.loads(line) for line in (folder/'train.jsonl').read_text().splitlines() if line.strip()]
        train=[row for row in rows if row.get('state')=='training']
        assert progress['complete'] and progress['state']=='smoke_passed'
        assert progress['step']==3 and protocol['world_size']==4 and protocol['accumulation']==1
        assert protocol['args']['global_batch']==256 and protocol['args']['batch_per_gpu']==64
        assert [r['score_stage'] for r in train]==[2,1,0]
        assert all(math.isfinite(r['student_grad']) and r['student_grad']>0 for r in train)
        runs[name]=(protocol,train)
    b,h=runs['baseline'],runs['hidden']
    assert b[0]['initial_signature']==h[0]['initial_signature']
    for key in ('initial','dmd_teacher','score_teacher','lr','critic_lr','critic_warmup','ramp_steps'):
        assert b[0]['args'][key]==h[0]['args'][key],key
    assert not b[0]['hidden_distillation']['enabled'] and h[0]['hidden_distillation']['enabled']
    first_b,first_h=b[1][0],h[1][0]
    checks=('pair_mse_8_16_32','onpolicy_mse','variance_mse','dmd_surrogate','lpips','dmd_gradient_rms')
    assert all(first_b[k]==first_h[k] for k in checks),'Initial non-hidden objective changed'
    added=first_h['hidden_weight']*first_h['hidden_loss']
    difference=first_h['loss']-first_b['loss']
    assert abs(difference-added)<1e-6
    return dict(passed=True,matched_initial_signature=b[0]['initial_signature'],
                all_initial_nonhidden_terms_exactly_equal=True,
                loss_difference=difference,expected_weighted_hidden_loss=added,
                peak_allocated_gib={k:max(r['max_rank_peak_gib'] for r in v[1]) for k,v in runs.items()},
                global_batch=256,batch_per_gpu=64,accumulation=1,
                tested_score_stages=[2,1,0],quality_improvement_proven=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args(); report=audit(a.root)
    (a.root/'smoke_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
