#!/usr/bin/env python3
"""Current generic numerical checks, with artifacts confined to this run."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'_deps')]
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch
import conformance
def main():
    p=argparse.ArgumentParser();p.add_argument('method',choices=['relay','ppflow','rdd','lapflow']);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output=a.output.resolve();torch.set_num_threads(2);started=time.time()
    conformance.ARTIFACT_ROOT=a.output.parent/'artifacts';conformance.RUNNERS[a.method]()
    nonbyte=[c for c in conformance.CHECKS if not c['bitwise']]
    if a.method=='lapflow':assert len(nonbyte)==2 and all(c['name']=='lapflow/reconstruction' and c['max_abs']==2.384185791015625e-7 for c in nonbyte)
    else:assert not nonbyte
    r=dict(method=a.method,status='passed',device='cpu',gpu_used=False,seconds=time.time()-started,checks=conformance.CHECKS,source_executions=conformance.native.AUDIT,
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),verification_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [ROOT/'tools/conformance.py',ROOT/'tools/conformance_more.py',ROOT/'tools/conformance_additional.py']},
        scope='Generic component math and short controlled updates; LapFlow two reconstruction identities retain explicitly classified FP32 roundoff, not a native pair difference.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(method=a.method,status='passed',checks=len(conformance.CHECKS),nonbyte=len(nonbyte))))
if __name__=='__main__':main()
