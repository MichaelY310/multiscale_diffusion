"""Preview the authorized hidden pilot and prune ONLY the two named old runs.

Does not signal any process. New previews use only physical GPUs 0..3 and wait
for memory. Training budgets/objectives are never modified by this monitor.
"""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time

REPO=Path(__file__).resolve().parents[2]
ROOT=Path('/home/yuheng/rdd_experiments/dmd_reuse_hidden_20260916')
RUN=ROOT/'pilot_hidden'
OLD=Path('/home/yuheng/rdd_experiments/dmd_reuse_objective_20260915')
PY=REPO/'.venv-rdd-eval/bin/python'


def read(path,default=None):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default


def main():
    ROOT.mkdir(exist_ok=True);last_cleanup=0;last_progress=None;changed=time.time()
    attempts={};errors={};cleanup_status=None
    while True:
        now=time.time();data=read(RUN/'progress.json',{'complete':False,'state':'not_started'})
        token=(data.get('step'),data.get('warm_step'))
        if token!=last_progress:last_progress=token;changed=now
        status={'time':datetime.now(timezone.utc).isoformat(),'job':data,
                'seconds_since_progress':round(now-changed),'preview_errors':errors}
        if now-last_cleanup>=300:
            # Save an exact audit in the old root; a concurrent latest replacement
            # makes the pruner refuse/skip, not delete a newer unverified file.
            with (ROOT/'retention.log').open('a') as log:
                proc=subprocess.run([str(PY),str(REPO/'2d/attention_analysis/prune_old_dmd_objective_checkpoints.py'),'--apply'],
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT)
            cleanup_status={'time':datetime.now(timezone.utc).isoformat(),'returncode':proc.returncode}
            last_cleanup=now
        pending=[]
        for folder in (RUN,RUN/'holdout256'):
            for path in sorted(folder.glob('validation_*.json')):
                step=int(path.stem.split('_')[-1]);key=f'{folder.name}/{step}'
                if read(folder/f'preview_{step:07d}/complete.json',{}).get('complete'):continue
                if len(list(folder.glob(f'validation_{step:07d}_rank*.pt')))==4:
                    pending.append((folder,step,key))
        eligible=[job for job in pending if attempts.get(job[2],0)<3]
        if eligible:
            raw=subprocess.check_output(['nvidia-smi','-i','0,1,2,3','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
            free,gpu=max((int(line.split(',')[1]),int(line.split(',')[0])) for line in raw.strip().splitlines())
            status['preview_free_mib']=free
            if free>=8000:
                folder,step,key=eligible[0]
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',
                         HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
                with (folder/f'preview_{step:07d}.log').open('a') as log:
                    proc=subprocess.run([str(PY),'-u',str(REPO/'2d/attention_analysis/decode_dmd_cascade_previews.py'),
                                         '--run',str(folder),'--step',str(step)],env=env,stdout=log,stderr=subprocess.STDOUT)
                attempts[key]=attempts.get(key,0)+1
                if proc.returncode:errors[key]=proc.returncode
                else:errors.pop(key,None)
        status['retention']=cleanup_status
        status['pending_previews']=[job[2] for job in pending]
        status['new_pilot_complete']=bool(data.get('complete')) and not pending
        # Continue lightweight retention after the pilot while the old authorized
        # DMD run is still saving snapshots. No recovery or retuning of that run.
        old_complete=all(read(OLD/name/'progress.json',{}).get('complete',False) for name in ('control','dmd_lpips'))
        status['old_runs_finished']=old_complete
        tmp=ROOT/'monitor_status.tmp';tmp.write_text(json.dumps(status,indent=2));os.replace(tmp,ROOT/'monitor_status.json')
        print(json.dumps(status),flush=True)
        if status['new_pilot_complete'] and old_complete and cleanup_status and cleanup_status['returncode']==0:break
        time.sleep(60)


if __name__=='__main__':main()
