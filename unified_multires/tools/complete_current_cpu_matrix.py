#!/usr/bin/env python3
"""Finish/revalidate every released training profile against current file hashes.

Reuses only passed reports with identical config, implementation and tracer
hashes. Historical attempts are preserved. No GPU, assets or training downloads.
"""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
HARN=ROOT/'tools/validate_full_forward_backward.py'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=2);p.add_argument('--threads',type=int,default=8);p.add_argument('--reduced',action='store_true');p.add_argument('--wait-for-suite',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);summary=a.output/'SUMMARY.json'
    if a.wait_for_suite:
        while not summary.exists() or json.loads(summary.read_text()).get('remaining'):
            time.sleep(5)
    manifest=json.loads((ROOT/'configs/paired_experiments/manifest.json').read_text())
    names=[r['id'] for r in manifest['profiles'] if r['paper'] not in ('rmd','diffusability')]
    expected=dict(harness_sha256=sha(HARN),implementation_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'umdiff').glob('*.py'))},verification_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'tools/activation_trace.py',ROOT/'tools/validate_training.py')})
    results=[];attempts=a.output/'attempts';attempts.mkdir(exist_ok=True)
    for name in names:
        path=a.output/(name+'.json');config=ROOT/'configs/paired_experiments'/(name+'.json')
        existing=json.loads(path.read_text()) if path.exists() else {}
        reuse=existing.get('status')=='passed' and len(existing.get('steps',[]))==a.steps and existing.get('full_released_architecture')==(not a.reduced) and existing.get('config_sha256')==sha(config) and all(existing.get(k)==v for k,v in expected.items())
        if not reuse:
            if path.exists():
                token=str(time.time_ns());(attempts/(name+'_'+token+'.json')).write_bytes(path.read_bytes())
                log=a.output/(name+'.log')
                if log.exists():(attempts/(name+'_'+token+'.log')).write_bytes(log.read_bytes())
            cmd=[sys.executable,str(HARN),'--profile',name,'--output',str(a.output),'--steps',str(a.steps),'--threads',str(a.threads)]
            if a.reduced:cmd.append('--reduced')
            print(json.dumps(dict(profile=name,event='execute_current')),flush=True)
            with (a.output/(name+'.log')).open('w') as log:code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''}).returncode
            existing=json.loads(path.read_text());assert code==0 and existing['status']=='passed',name
        assert all(existing.get(k)==v for k,v in expected.items()),'Code changed while matrix was running'
        results.append(dict(profile=name,status=existing['status'],reused=reuse,seconds=existing['seconds'],config_sha256=existing['config_sha256']))
        summary.write_text(json.dumps(dict(results=results,remaining=names[len(results):],current_provenance=expected),indent=2)+'\n')
        print(json.dumps(results[-1]),flush=True)
    return 0
if __name__=='__main__':raise SystemExit(main())
