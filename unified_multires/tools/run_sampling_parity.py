"""CPU-only reproducible sampling suite. No model/data downloads or training.

The common suite covers all18 comparison files, two random seeds/batch sizes,
and an extra original PixelFlow Euler path. Each job retains a JSON report.
"""
import argparse,concurrent.futures,hashlib,json,os,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def snapshot():
    paths=list((ROOT/'umdiff').rglob('*.py'))+list((ROOT/'configs').rglob('*.json'))
    paths+=list((ROOT/'tools').glob('*.py'))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def plan(out,suite):
    jobs=[]
    def add(name,script,*args):jobs.append(dict(name=name,argv=[sys.executable,str(ROOT/'tools'/script),*map(str,args)]))
    if suite in ('common','all'):
        for seed,batch in ((641,1),(972,2)):
            case=out/f'seed{seed}_batch{batch}'
            for path in sorted((ROOT/'configs/comparison_imagenet256').glob('*/*.json')):
                c=json.loads(path.read_text());solver=c['method'].get('process',{}).get('args',{}).get('solver','euler')
                name=f'{seed}_{c["suite"]}_{c["id"]}'
                add(name,'validate_comparison_sampling.py',path,'--solver',solver,'--seed',seed,'--batch',batch,'--output',case)
                if c['id']=='pixelflow_xl_4':add(name+'_euler','validate_comparison_sampling.py',path,'--solver','euler','--seed',seed,'--batch',batch,'--output',case)
    if suite in ('native','all'):
        from run_scoped_validation import jobs as native_jobs
        jobs.extend(j for j in native_jobs(out/'native',False) if j['name'].startswith(('profile_sampling_','sampling_','inference_')))
        add('native_pixelflow_dopri5','audit_dopri5_sampling.py','--output',out/'native/dopri5')
        add('native_rmd_sampling','validate_rmd_sampling.py','--output',out/'native/rmd_sampling.json')
    return jobs


def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',choices=['common','native','all'],default='all')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=2);args=p.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'PLAN.json').exists():raise ValueError('Use a new output directory to retain previous attempts')
    frozen=snapshot();jobs=plan(out,args.suite)
    (out/'SOURCE_BEFORE.json').write_text(json.dumps(frozen,indent=2)+'\n')
    (out/'PLAN.json').write_text(json.dumps(jobs,indent=2)+'\n')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
             OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    def run(job):
        start=time.monotonic();log=out/(job['name']+'.log')
        with log.open('w') as f:r=subprocess.run(job['argv'],env=env,cwd=ROOT.parent,stdout=f,stderr=subprocess.STDOUT)
        row=dict(job,status='passed' if r.returncode==0 else 'failed',returncode=r.returncode,
                 seconds=time.monotonic()-start,source_unchanged=snapshot()==frozen,log=str(log.relative_to(out)))
        print(json.dumps({k:row[k] for k in ('name','status','seconds','source_unchanged')}),flush=True)
        return row
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run,j) for j in jobs]
        for f in concurrent.futures.as_completed(futures):
            rows.append(f.result());(out/'PROGRESS.json').write_text(json.dumps(rows,indent=2)+'\n')
    stable=snapshot()==frozen
    report=dict(status='passed' if stable and all(r['status']=='passed' and r['source_unchanged'] for r in rows) else 'failed',
                jobs=rows,source_unchanged=stable,gpu_used=False,model_or_data_downloads=False)
    (out/'SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n')
    return report['status']!='passed'


if __name__=='__main__':raise SystemExit(main())
