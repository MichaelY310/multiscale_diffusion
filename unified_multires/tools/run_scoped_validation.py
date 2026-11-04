#!/usr/bin/env python3
"""Fixed-code CPU acceptance suite for explicitly supported public algorithms.

No downloads, CUDA, formal runs or old monitors. Local process IPC is needed
for worker/Gloo tests. --only reruns selected jobs while retaining old attempts.
"""
import argparse,concurrent.futures,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def snapshot():
    files=list((ROOT/'umdiff').rglob('*.py'))+list((ROOT/'configs').rglob('*.json'))
    files+=list((ROOT/'tools').glob('validate*.py'))
    files += [ROOT/'tools'/name for name in ('activation_trace.py','conformance.py','compare_released_profiles.py','complete_current_cpu_matrix.py','run_scoped_validation.py')]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}


def jobs(base,full):
    rows=[]
    def add(name,script,*args):rows.append(dict(name=name,argv=[sys.executable,str(ROOT/'tools'/script),*map(str,args)]))
    add('environment','record_cpu_parity_environment.py','--output',base)
    if full:add('full_current','complete_current_cpu_matrix.py','--output',base/'full_current','--steps',2,'--threads',4)
    add('reduced_current','complete_current_cpu_matrix.py','--output',base/'reduced_current','--steps',2,'--threads',2,'--reduced')
    for m in ('relay','ssd','pixelflow','dfm','lapflow','ppflow','dctdiff','mdm','dimr','diffusability','rmd','rdd'):
        add('long_'+m,'validate_activations.py','--method',m,'--steps',128,'--directory',base/'long_release')
    for m in ('dctdiff','mdm','mdm_vdm','ssd','ppflow','dimr','dfm','lapflow','lapflow3','relay','rdd','diffusability'):
        add('extension_'+m,'validate_recipe_extensions.py',m,'--steps',128,'--threads',2,'--directory',base/'extensions_current'/m)
    add('codec_full','validate_recipe_extensions.py','diffusability','--full-codec','--steps',2,'--threads',4,'--directory',base/'codec_current')
    add('rmd_full','validate_rmd_current_profile.py','--full-backbone','--steps',17,'--threads',4,'--output',base/'rmd')
    for name,script in [('lifecycle','validate_fresh_process_lifecycle.py'),('rmd_lifecycle','validate_rmd_fresh_process.py'),('codec_lifecycle','validate_codec_fresh_process.py'),('codec_boundaries','validate_codec_boundaries.py'),('data_workers','validate_stateful_data.py'),('data_cli','validate_data_cli.py'),('cpu_ddp','validate_cpu_ddp.py'),('cpu_scaler','validate_cpu_scaler.py'),('rdd_optional','validate_rdd_optional_objectives.py'),('native_ssd','validate_ssd_native_loop.py')]:
        add(name,script,'--output',base/name)
    for levels in (2,3):add('native_ppflow'+str(levels),'validate_ppflow_native_main.py','--levels',levels,'--output',base/'native_ppflow')
    for m in ('lapflow','ppflow'):add('fix_'+m,'validate_source_fixes.py',m,'--output',base/'source_fixes')
    for m in ('speed','nami','mrflow'):add('inference_'+m,'validate_inference.py',m,'--output-dir',base/'inference_layers')
    for m in ('qwen','pixelgen','wan'):add('inference_'+m,'validate_remaining_inference.py',m,'--output',base/'inference_new')
    for m in ('mrflow_qwen','speed_pixelgen','speed_wan'):add('workflow_'+m,'validate_new_workflows_cli.py',m,'--output',base/'workflow_cli')
    for m in ('ppflow','mdm'):add('sampling_'+m,'validate_current_sampling.py',m,'--output',base/'sampling')
    for m in ('dctdiff','dctdiff_imagenet','dimr','pixelflow','dfm','dfm_yaml','lapflow_paper','ssd','relay_low','relay_high','rdd'):
        add('profile_sampling_'+m,'validate_profile_inference.py',m,'--output',base/'sampling_profiles')
    for m in ('relay','ppflow','rdd','lapflow'):add('component_'+m,'validate_generic_components_current.py',m,'--output',base/'components'/(m+'.json'))
    add('ssd_branches','validate_ssd_resolution_branches.py','--threads',2,'--output',base/'ssd_resolution_branches.json')
    add('cat','validate_cat_published_codec.py','--steps',2,'--threads',2,'--output',base/'cat_codec')
    add('cat_full','validate_cat_published_codec.py','--full-backbone','--steps',1,'--threads',4,'--output',base/'cat_codec_full')
    add('cat_api','validate_cat_api_contract.py','--output',base/'cat_api')
    add('gap_regressions','validate_cpu_gap_fixes.py','--output',base/'gap_regressions.json')
    add('tracer_negative_controls','validate_activations.py','--selftest','--directory',base/'tracer_selftest')
    rows.append(dict(name='pytest',argv=[sys.executable,'-m','pytest','-q',str(ROOT/'tests/test_component_contracts.py')]))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'evidence/unified_scope_20261012');p.add_argument('--workers',type=int,default=2);p.add_argument('--full',action='store_true');p.add_argument('--only',nargs='*');p.add_argument('--plan',action='store_true');a=p.parse_args()
    base=a.output.resolve();base.mkdir(parents=True,exist_ok=True);all_jobs=jobs(base,a.full)
    if a.only:
        unknown=set(a.only)-{j['name'] for j in all_jobs}
        if unknown:p.error('Unknown jobs '+repr(unknown))
        selected=[j for j in all_jobs if j['name'] in a.only]
    else:selected=all_jobs
    frozen=snapshot();tag=str(time.time_ns());attempt=base/'suite_attempts'/tag;attempt.mkdir(parents=True)
    (attempt/'source_snapshot.json').write_text(json.dumps(frozen,indent=2)+'\n')
    (attempt/'plan.json').write_text(json.dumps(selected,indent=2)+'\n')
    if a.plan:print(json.dumps(dict(jobs=len(selected),path=str(attempt/'plan.json'))));return 0
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1'}
    def run(job):
        before=snapshot()
        if before!=frozen:return dict(**job,status='failed',failure='Code changed before job start')
        started=time.time();log=attempt/(job['name']+'.log')
        with log.open('w') as f:code=subprocess.run(job['argv'],cwd=ROOT.parent,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
        stable=snapshot()==frozen
        return dict(**job,status='passed' if code==0 and stable else 'failed',returncode=code,code_unchanged=stable,seconds=time.time()-started,log=str(log.relative_to(base)))
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(run,j):j for j in selected}
        for future in concurrent.futures.as_completed(futures):
            row=future.result();results.append(row);print(json.dumps({k:row.get(k) for k in ('name','status','seconds','failure')}),flush=True)
            (attempt/'SUMMARY.json').write_text(json.dumps(dict(status='running',jobs=results,remaining=[j['name'] for j in selected if j['name'] not in {r['name'] for r in results}]),indent=2)+'\n')
    status='passed' if all(r['status']=='passed' for r in results) else 'failed'
    report=dict(status=status,jobs=results,gpu_used=False,formal_training=False,source_snapshot='source_snapshot.json',all_code_unchanged=snapshot()==frozen)
    (attempt/'SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(status=status,attempt=str(attempt),jobs=len(results))),flush=True);return status!='passed'
if __name__=='__main__':raise SystemExit(main())
