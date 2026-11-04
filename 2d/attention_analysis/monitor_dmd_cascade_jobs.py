"""Read-only job monitor plus bounded preview decoding; NEVER sends signals.

Runs only on this task's run directories. A stale/nonfinite job is recorded,
not automatically killed or silently restarted. Human/agent can inspect logs.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--families',default='high_heavy,balanced');p.add_argument('--interval',type=int,default=60)
    a=p.parse_args();root=Path(a.root);repo=Path(__file__).resolve().parents[2]
    families=a.families.split(',');observed={};idle={};errors={}
    while True:
        status={'time':datetime.now(timezone.utc).isoformat(),'jobs':{},'all_complete':True}
        for family in families:
            run=root/family;path=run/'progress.json'
            if not path.exists():
                status['jobs'][family]={'state':'not_started'};status['all_complete']=False;continue
            try:data=json.loads(path.read_text())
            except (json.JSONDecodeError,OSError):status['all_complete']=False;continue
            if observed.get(family)!=data.get('step'):observed[family]=data.get('step');idle[family]=time.time()
            data['seconds_since_step_change']=round(time.time()-idle[family])
            data['stale_over_20min']=data['seconds_since_step_change']>1200 and not data.get('complete',False)
            status['all_complete'] &= data.get('complete',False)
            # One tiny VAE process, only if >=7GiB remains on a task-authorized GPU.
            pending=[f for f in sorted(run.glob('validation_*.json')) if f.stem.split('_')[-1].isdigit()]
            gate=root/'speed_gate.json'
            if not gate.exists() or not json.loads(gate.read_text()).get('complete',False):pending=[]
            for meta in pending:
                step=int(meta.stem.split('_')[-1]);preview=run/f'preview_{step:07d}'
                if (preview/'complete.json').exists() or (family,step) in errors:continue
                if len(list(run.glob(f'validation_{step:07d}_rank*.pt')))!=4:continue
                memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
                choices=[(int(line.split(',')[1]),int(line.split(',')[0])) for line in memory.strip().splitlines()]
                free,gpu=max(choices)
                if free<7000:break
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
                with (run/f'preview_{step:07d}.log').open('a') as log:
                    result=subprocess.run([str(repo/'.venv-rdd-eval/bin/python'),'-u',str(repo/'2d/attention_analysis/decode_dmd_cascade_previews.py'),
                         '--run',str(run),'--step',str(step)],env=env,stdout=log,stderr=subprocess.STDOUT)
                if result.returncode:errors[(family,step)]=result.returncode
            data['preview_errors']={str(step):rc for (fam,step),rc in errors.items() if fam==family}
            status['jobs'][family]=data
        tmp=root/'monitor_status.tmp';tmp.write_text(json.dumps(status,indent=2));os.replace(tmp,root/'monitor_status.json')
        print(json.dumps(status),flush=True)
        if status['all_complete']:break
        time.sleep(a.interval)
