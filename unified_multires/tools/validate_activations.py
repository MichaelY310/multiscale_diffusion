#!/usr/bin/env python3
"""Supplement existing native CPU training with module activation parity."""
import argparse
from collections import Counter
from copy import deepcopy
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
sys.path[:0]=[str(ROOT),str(ROOT/'_deps')]
import torch
from activation_trace import ActivationAudit

METHODS=['relay','ssd','pixelflow','dfm','lapflow','ppflow','dctdiff','mdm','dimr','diffusability','rmd','rdd']


def selftest(directory):
    class Toy(torch.nn.Module):
        def __init__(self,scale=1.):
            super().__init__();self.linear=torch.nn.Linear(3,3);self.bridge=torch.nn.Identity();self.tail=torch.nn.Linear(3,1);self.scale=scale
        def forward(self,x):
            h=self.bridge(self.linear(x)*self.scale)
            # Repeated call and structured output; functionally cancelled scale.
            return {'value':self.tail(h/self.scale),'again':(self.bridge(h),None),
                    'unused':h.detach().clone().requires_grad_()}
    torch.manual_seed(3);a=Toy();b=deepcopy(a);b.scale=2.
    audit=ActivationAudit('selftest_difference',[('network',a,b)],directory)
    x=torch.randn(2,3)
    audit.begin(0);audit.side('native');ya=a(x)['value'];audit.side('unified');yb=b(x)['value']
    ya.sum().backward();yb.sum().backward();audit.parameters(a,b);audit.finish_segment();result=audit.close()
    assert torch.equal(ya,yb)
    assert result['totals']['parameter_gradient']['exact']==result['totals']['parameter_gradient']['pairs']
    assert result['totals']['forward_output']['exact']<result['totals']['forward_output']['pairs']
    assert result['totals']['activation_gradient']['exact']<result['totals']['activation_gradient']['pairs']
    assert not result['structural_mismatches']
    assert result['gradient_coverage']['requires_grad_but_unused_output_pairs']>=1
    mask=ActivationAudit('selftest_masks',[],directory);mask.begin(0)
    infinite_mask=torch.tensor([0.,float('-inf'),float('inf')])
    mask.compare_tensor('forward_input','equal_mask',infinite_mask,infinite_mask.clone())
    assert mask.totals['forward_input']['within_tolerance']==1
    mask.compare_tensor('forward_input','opposite_mask',torch.tensor([float('-inf')]),torch.tensor([float('inf')]))
    assert mask.totals['forward_input']['within_tolerance']==1
    mask.close()
    # Input snapshots survive in-place operations; unused outputs are not zero gradients.
    a=torch.nn.Sequential(torch.nn.Linear(3,3),torch.nn.ReLU(inplace=True));b=deepcopy(a)
    audit=ActivationAudit('selftest_equal_inplace',[('network',a,b)],directory)
    audit.begin(0);audit.side('native');ya=a(x);audit.side('unified');yb=b(x)
    ya.sum().backward();yb.sum().backward();audit.parameters(a,b);audit.finish_segment();equal=audit.close()
    assert not equal['structural_mismatches']
    assert all(s['pairs']==s['exact'] for s in equal['totals'].values())
    result=dict(status='passed',checks=['detect hidden activation mismatch despite equal final output and parameter gradients',
                                       'detect hidden activation gradient mismatch','repeated module calls and nested outputs',
                                       'unused output gradient presence','in-place ReLU without autograd view hooks',
                                       'identical infinite masks accepted, opposite infinite masks rejected'],
                divergent_fixture=result,identical_fixture=equal)
    (directory/'selftest.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(selftest='passed')),flush=True)


def run_method(method,steps,directory):
    import validate_training as training
    import conformance
    from umdiff import native
    torch.set_num_threads(2);training.seed(20261008)
    audits=[];started=time.time();failure=None;details={}
    def factory(name,pairs):
        # Author BTCHW facade vs unified direct BCHW encode/decode are different
        # interfaces. Audit all shared VAE submodules, not this facade root.
        audit=ActivationAudit(name,pairs,directory,excluded_modules=('',) if method=='diffusability' else ())
        audits.append(audit)
        original_finish=audit.finish_segment
        def finish():
            original_finish()
            if (audit.step+1)%32==0 and audit.segment in ('training','phase2'):
                print(json.dumps(dict(method=method,variant=name,completed_steps=audit.step+1,
                                      totals=audit.totals,structural_mismatches=len(audit.structural_mismatches))),flush=True)
        audit.finish_segment=finish
        return audit
    checkpoints=ROOT/'runs'/directory.resolve().relative_to(ROOT/'evidence')/'checkpoints'
    conformance.ARTIFACT_ROOT=directory/'artifacts'
    training.AUDIT_FACTORY=factory;training.CHECKPOINT_ROOT=checkpoints
    try:
        if method=='rmd':
            import validate_rmd
            validate_rmd.AUDIT_FACTORY=factory;validate_rmd.CHECKPOINT_ROOT=checkpoints
            # Preserve the original main() pre-training RNG consumption, so the
            # traced checkpoint can also be checked against the earlier run.
            with torch.no_grad():validate_rmd.components()
            conformance.CHECKS.clear()
            details=validate_rmd.train(steps)
        else:
            details=training.RUNNERS[method](steps)
    except Exception:
        failure=traceback.format_exc();print(failure,file=sys.stderr,flush=True)
    variants=[audit.close() for audit in audits]
    totals={k:dict(pairs=0,exact=0,byte_exact=0,within_tolerance=0,max_abs=0.) for k in ('forward_input','forward_output','activation_gradient','parameter_gradient')}
    for variant in variants:
        for kind,s in variant['totals'].items():
            for key in ('pairs','exact','byte_exact','within_tolerance'):totals[kind][key]+=s[key]
            totals[kind]['max_abs']=max(totals[kind]['max_abs'],s['max_abs'])
    structural=sum(len(v['structural_mismatches']) for v in variants)
    passed=not failure and not structural and all(s['pairs'] and s['byte_exact']==s['pairs'] for s in totals.values())
    existing=Counter();original_max={};original_nonexact=Counter()
    for check in conformance.CHECKS:
        scope=check.get('scope','unspecified');existing[scope]+=1
        original_max[scope]=max(original_max.get(scope,0.),check.get('max_abs',0.))
        if not check.get('bitwise',False):original_nonexact[scope]+=1
    passed=passed and not original_nonexact
    report=dict(method=method,status='passed' if passed else 'failed',device='cpu',gpu_used=False,
                steps=steps,seconds=time.time()-started,totals=totals,variants=variants,
                original_checks=dict(counts=existing,nonexact=original_nonexact,max_abs=original_max),
                source_executions=native.AUDIT,training_details=details,failure=failure,
                harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                implementation_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'umdiff').glob('*.py'))},
                verification_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'tools/activation_trace.py',ROOT/'tools/validate_training.py')},
                exclusions=['Diffusability: only the author BTCHW wrapper root is excluded; every executed shared VAE submodule is traced. The extra no-loss diagnostic encode is suspended in both alignment and coverage; both actual loss graphs remain unchanged.'] if method=='diffusability' else [],
                scope='Same existing CPU reduced author-backbone synthetic native-specific fixture. Every executed module boundary is observed; not every functional operator. Shared author backbones are reused, not independently rewritten. No pretrained assets or datasets.')
    directory.mkdir(parents=True,exist_ok=True)
    (directory/(method+'.json')).write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(method=method,status=report['status'],seconds=report['seconds'],totals=totals,structural_mismatches=structural)),flush=True)
    return 0 if passed else 1


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--method',choices=METHODS);parser.add_argument('--all',action='store_true')
    parser.add_argument('--steps',type=int,default=128);parser.add_argument('--directory',type=Path,default=ROOT/'evidence/activation_parity')
    parser.add_argument('--selftest',action='store_true');parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();args.directory.mkdir(parents=True,exist_ok=True)
    if args.selftest:selftest(args.directory);return 0
    if args.method:return run_method(args.method,args.steps,args.directory)
    if not args.all:parser.error('choose --method, --all or --selftest')
    # Separate processes avoid author packages with colliding top-level names.
    queue=list(METHODS);running=[];results={}
    while queue or running:
        while queue and len(running)<args.workers:
            method=queue.pop(0);log=(args.directory/(method+'.log')).open('w')
            command=[sys.executable,__file__,'--method',method,'--steps',str(args.steps),'--directory',str(args.directory)]
            running.append((method,subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT),log))
            print(json.dumps(dict(started=method)),flush=True)
        for method,process,log in list(running):
            code=process.poll()
            if code is not None:
                log.close();running.remove((method,process,log));results[method]=code
                print(json.dumps(dict(completed=method,exit_code=code)),flush=True)
        if running:time.sleep(.2)
    (args.directory/'run_status.json').write_text(json.dumps(results,indent=2))
    return int(any(results.values()))


if __name__=='__main__':raise SystemExit(main())
