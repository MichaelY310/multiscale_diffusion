"""Audit only this task's completed outputs; never signals any process."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import torch


def main(root):
    steps=(0,1,200,800,1000,2000,3000,4000,5000)
    audit={'time':datetime.now(timezone.utc).isoformat(),'jobs':{},'all_complete':True,
           'quality_passed':False,'quality_note':'Execution completion is NOT quality acceptance; fixed64 is not FID.'}
    for family in ('high_heavy','balanced_plus'):
        run=root/family
        progress=json.loads((run/'progress.json').read_text())
        checkpoint=torch.load(run/'checkpoints/latest.pt',map_location='cpu',mmap=True,weights_only=True)
        row={'training_step':progress['step'],'state':progress['state'],
             'checkpoint_step':checkpoint['step'],'checkpoint_family':checkpoint['family'],
             'model_values':sum(t.numel() for t in checkpoint['model'].values()),
             'optimizer_saved':'optimizer' in checkpoint,'rng_ranks':len(checkpoint['generator_states']),
             'checkpoint_bytes':(run/'checkpoints/latest.pt').stat().st_size,'previews':{}}
        del checkpoint
        for step in steps:
            meta=json.loads((run/f'preview_{step:07d}/complete.json').read_text())
            validation=json.loads((run/f'validation_{step:07d}.json').read_text())
            preview=run/f'preview_{step:07d}'
            item={'complete':meta['complete'],'samples':meta['samples'],
                  'teacher_pngs':len(list(preview.glob('teacher_*.png'))),
                  'student_pngs':len(list(preview.glob('student_*.png'))),
                  'grid_exists':(preview/'comparison_first16.jpg').is_file(),
                  'finite_fraction':validation['finite_batch_fraction']}
            item['passed']=bool(item['complete'] and item['samples']==64 and
                                item['teacher_pngs']==item['student_pngs']==64 and
                                item['grid_exists'] and item['finite_fraction']==1)
            row['previews'][str(step)]=item
        row['complete']=bool(progress['complete'] and row['training_step']==row['checkpoint_step']==5000
                             and row['checkpoint_family']==family and row['optimizer_saved'] and
                             row['rng_ranks']==4 and all(v['passed'] for v in row['previews'].values()))
        audit['all_complete'] &= row['complete']
        audit['jobs'][family]=row
    audit['tmux_sessions_present']={}
    for name in ('rdd_dmd_reuse_a_20260915','rdd_dmd_reuse_b_20260915','rdd_dmd_reuse_monitor_20260915'):
        result=subprocess.run(['tmux','has-session','-t','='+name],capture_output=True)
        audit['tmux_sessions_present'][name]=(result.returncode==0)
    (root/'completion_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))
    if not audit['all_complete']:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    main(Path(p.parse_args().root))
