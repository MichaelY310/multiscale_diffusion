#!/usr/bin/env python3
"""Run every preset through real CLI processes, including exact checkpoint resume."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'_deps')]
import torch

def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    if hasattr(a,'shape'):return bool((a==b).all())
    return a==b

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--filter',default='');parser.add_argument('--steps',type=int,default=4);parser.add_argument('--directory',default='');args=parser.parse_args()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONPATH=os.pathsep.join([str(ROOT),str(ROOT/'_deps')]))
    reports=[];directory=ROOT/'runs'/'cli_validation'/args.directory;directory.mkdir(parents=True,exist_ok=True)
    for path in sorted((ROOT/'configs').rglob('*.json')):
        if args.filter not in path.stem:continue
        if 'inference_only' in path.parts or 'paper_recipes' in path.parts:continue
        name=path.stem;started=time.time();failure=None;commands=[]
        def command(arguments):
            cmd=[sys.executable,'-m','umdiff.cli',*arguments];result=subprocess.run(cmd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            commands.append({'argv':cmd,'exit_code':result.returncode,'log':result.stdout[-5000:]})
            if result.returncode:raise RuntimeError(result.stdout[-5000:])
        try:
            full=directory/name/'full';split=directory/name/'resumed'
            common=['--config',str(path),'--synthetic','--batch-size','8','--seed','42']
            command(['train',*common,'--steps',str(args.steps),'--output',str(full)])
            first=args.steps//2
            command(['train',*common,'--steps',str(first),'--output',str(split)])
            command(['train',*common,'--steps',str(args.steps-first),'--output',str(split),'--resume',str(split/'checkpoint.pt')])
            a=torch.load(full/'checkpoint.pt',weights_only=False,map_location='cpu');b=torch.load(split/'checkpoint.pt',weights_only=False,map_location='cpu')
            keys=('model','fake','real','optimizer','fake_optimizer','step','torch_rng','numpy_rng','python_rng') if a.get('format')=='rmd1' else ('model','ema','optimizer','stream','step','torch_rng','numpy_rng','python_rng')
            for key in keys:
                if not equal(a[key],b[key]):raise AssertionError('Resume differs: '+key)
            command(['sample','--checkpoint',str(full/'checkpoint.pt'),'--output',str(full/'sample.pt'),'--count','2'])
            command(['sample','--checkpoint',str(split/'checkpoint.pt'),'--output',str(split/'sample.pt'),'--count','2'])
            if not equal(torch.load(full/'sample.pt',weights_only=True),torch.load(split/'sample.pt',weights_only=True)):raise AssertionError('Resumed sampling differs')
        except Exception as exc:failure=str(exc)
        reports.append({'name':name,'status':'failed' if failure else 'passed','steps':args.steps,'checkpoint_resume_bitwise':failure is None,'sample_resume_bitwise':failure is None,'seconds':time.time()-started,'failure':failure,'commands':commands})
        report_path=ROOT/'evidence'/args.directory/'cli_validation.json';report_path.parent.mkdir(parents=True,exist_ok=True)
        old=json.loads(report_path.read_text()).get('cases',[]) if args.filter and report_path.exists() else []
        merged={row['name']:row for row in old}
        merged.update({row['name']:row for row in reports})
        report_path.write_text(json.dumps({'device':'cpu','gpu_used':False,'cases':list(merged.values())},indent=2))
        print(json.dumps({k:reports[-1][k] for k in ('name','status','seconds','failure')}),flush=True)
    if not reports:raise ValueError('No preset matched the requested filter')
    return any(r['status']!='passed' for r in reports)
if __name__=='__main__':raise SystemExit(main())
