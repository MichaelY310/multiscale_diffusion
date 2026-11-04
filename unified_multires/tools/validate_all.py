#!/usr/bin/env python3
"""Serial isolated author/port conformance; CPU only, never fetches assets."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--training-steps',type=int,default=128);parser.add_argument('--filter',default='');args=parser.parse_args()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONPATH=os.pathsep.join([str(ROOT),str(ROOT/'_deps')]))
    methods=['relay','ssd','speed','pixelflow','dfm','lapflow','ppflow','nami','mrflow','dctdiff','diffusability','mdm','dimr','rdd']
    jobs=[('component/'+m,['tools/conformance.py',m]) for m in methods]
    jobs += [('training/'+m,['tools/validate_training.py',m,'--steps',str(args.training_steps)]) for m in methods if m not in ('speed','nami','mrflow')]
    jobs += [('inference/'+m,['tools/validate_inference.py',m]) for m in ('speed','nami','mrflow')]
    jobs += [('training/rmd',['tools/validate_rmd.py','--steps',str(args.training_steps)])]
    jobs += [('native_entrypoint/rmd',['tools/validate_rmd_native_entrypoint.py'])]
    jobs += [('continuous_rng/dfm',['tools/validate_dfm_rng.py'])]
    jobs += [('contracts',['-m','pytest','-q','tests'])]
    directory=ROOT/'runs'/'suite';directory.mkdir(parents=True,exist_ok=True);records=[]
    report_path=ROOT/'evidence'/'suite.json'
    if args.filter and report_path.exists():
        old=json.loads(report_path.read_text())
        if old['training_steps']!=args.training_steps:raise ValueError('Filtered runs must retain the existing training_steps; use a full run to change the budget')
        records=old['cases']
    jobs=[job for job in jobs if args.filter in job[0]]
    if not jobs:raise ValueError('No validation case matches --filter')
    for name,arguments in jobs:
        started=time.time();cmd=[sys.executable,*arguments]
        result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        log=directory/(name.replace('/','_')+'.log');log.write_text(result.stdout)
        row=dict(case=name,status='passed' if result.returncode==0 else 'failed',exit_code=result.returncode,seconds=time.time()-started,argv=cmd,log=str(log.relative_to(ROOT)))
        records=[r for r in records if r['case']!=name];records.append(row);report_path.write_text(json.dumps(dict(device='cpu',gpu_used=False,training_steps=args.training_steps,cases=records),indent=2))
        print(json.dumps(row),flush=True)
    return any(r['exit_code'] for r in records)
if __name__=='__main__':raise SystemExit(main())
