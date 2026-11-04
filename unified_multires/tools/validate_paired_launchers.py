#!/usr/bin/env python3
"""Two fresh processes exercise the actual new reader/train/save path, CPU only."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'_deps'),str(ROOT/'tools')]
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch
from paired_experiments import compile_plan,load_manifest,execute
from validate_paired_experiments import reduced
from compare_released_profiles import compare_state

def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=['original','unified']);p.add_argument('--temp',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'evidence/paired_experiments_20261010/launcher_smoke.json');a=p.parse_args()
    torch.set_num_threads(2)
    if a.worker:
        m=load_manifest();entry=next(e for e in m['profiles'] if e['id']=='rdd_unified_xl_300_300_400')
        plan=compile_plan(m,entry,'fair-rdd');c,_=reduced(plan['contract']['method_config'])
        c['training'].pop('input_transform',None);c['training'].pop('initialization',None)
        plan['contract'].update(method_config=c,data=str(a.temp/'fixture.pt'))
        plan['contract']['controls'].update(budget=2,global_batch=2,micro_batch=1,accumulation=2)
        plan.update(missing_assets=[],blocked_reasons=[])
        plan['validation_overrides']='Synthetic local input, reduced backbone, FP32CPU,2updates,batch2; no paper-quality run.'
        execute(plan,a.worker,a.temp/'runs');return 0
    temp=Path(tempfile.mkdtemp(prefix='umdiff_launcher_',dir='/tmp'));report=None
    try:
        g=torch.Generator().manual_seed(231);torch.save({'latents':torch.randn(8,4,32,32,generator=g),'labels':torch.arange(8)},temp/'fixture.pt')
        for backend in ('original','unified'):
            result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',backend,'--temp',str(temp)],
                capture_output=True,text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
            if result.returncode:raise RuntimeError(backend+': '+result.stdout+result.stderr)
        folders=[temp/'runs'/backend/'rdd_unified_xl_300_300_400' for backend in ('original','unified')]
        states=[torch.load(d/'checkpoint.pt',map_location='cpu',weights_only=False) for d in folders]
        checks={key:compare_state(states[0][key],states[1][key]) for key in ('model','optimizer','ema','torch_rng','scaler')}
        for key in ('step','recipe','images_seen','data_binding'):
            if states[0][key]!=states[1][key]:raise AssertionError(key+' differs')
        trace_exact=(folders[0]/'trace.jsonl').read_bytes()==(folders[1]/'trace.jsonl').read_bytes()
        same=trace_exact and all(v['exact']==v['tensors'] for v in checks.values())
        report=dict(status='byte_exact' if same else 'different',steps=2,gpu_used=False,formal_training=False,
          checks=checks,trace_exact=trace_exact,checkpoint_sizes=[(d/'checkpoint.pt').stat().st_size for d in folders],
          checkpoint_sha256=[hashlib.sha256((d/'checkpoint.pt').read_bytes()).hexdigest() for d in folders],
          scope='Actual new reader/loop/save, two fresh CPU processes, reduced RDD and synthetic local fixture; NOT native author CLI.',
          temporary_payloads_deleted=True,full_paper_result_certified=False)
    finally:shutil.rmtree(temp)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report));return report['status']!='byte_exact'

if __name__=='__main__':raise SystemExit(main())
