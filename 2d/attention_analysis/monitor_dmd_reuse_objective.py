"""Bounded task-specific monitor/preview queue. NO process signalling at all."""
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime,timezone

ROOT=Path('/home/yuheng/rdd_experiments/dmd_reuse_objective_20260915')
REPO=Path(__file__).resolve().parents[2]
FAMILIES=('control','dmd_lpips')
if __name__=='__main__':
    observed={};changed={};errors={}
    while True:
        status={'time':datetime.now(timezone.utc).isoformat(),'jobs':{},'all_complete':True,'preview_errors':{}}
        pending=[]
        for family in FAMILIES:
            run=ROOT/family
            try: data=json.loads((run/'progress.json').read_text())
            except (FileNotFoundError,json.JSONDecodeError): data={'state':'not_started','complete':False}
            state=(data.get('step'),data.get('warm_step'))
            if observed.get(family)!=state: changed[family]=time.time();observed[family]=state
            data['seconds_since_progress']=round(time.time()-changed.get(family,time.time()))
            data['stale_over_30min']=data['seconds_since_progress']>1800 and not data.get('complete')
            status['jobs'][family]=data;status['all_complete'] &= bool(data.get('complete'))
            for folder in (run,run/'holdout256'):
                for path in sorted(folder.glob('validation_*.json')):
                    step=int(path.stem.split('_')[-1]);destination=folder/f'preview_{step:07d}'
                    key=f'{family}/{folder.name}/{step}'
                    if (destination/'complete.json').exists():continue
                    status['all_complete']=False
                    if key in errors:continue
                    if len(list(folder.glob(f'validation_{step:07d}_rank*.pt')))==4:
                        pending.append((folder,step,key))
        if pending:
            memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
            choices=[(int(line.split(',')[1]),int(line.split(',')[0])) for line in memory.strip().splitlines()]
            free,gpu=max(choices)
            if free>=10000:
                folder,step,key=pending[0]
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
                with (folder/f'preview_{step:07d}.log').open('a') as log:
                    result=subprocess.run([str(REPO/'.venv-rdd-eval/bin/python'),'-u',str(REPO/'2d/attention_analysis/decode_dmd_cascade_previews.py'),
                                           '--run',str(folder),'--step',str(step)],env=env,stdout=log,stderr=subprocess.STDOUT)
                if result.returncode: errors[key]=result.returncode
        status['preview_errors']=errors
        temp=ROOT/'monitor_status.tmp';temp.write_text(json.dumps(status,indent=2));os.replace(temp,ROOT/'monitor_status.json')
        print(json.dumps(status),flush=True)
        if status['all_complete']:break
        time.sleep(60)
