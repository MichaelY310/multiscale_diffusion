#!/usr/bin/env python3
"""Strict per-layer author-objective vs unified-objective CPU comparisons.

No shared Trainer and no post-clip gradients. Full released image architecture
and grid are retained. Models use common nonzero random weights, explicitly
not downloaded paper checkpoints. Every profile runs in a fresh process.
"""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'_deps')]
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np
import torch
from umdiff import build,native
from umdiff.source_objectives import setup,unpack_model
from activation_trace import ActivationAudit,byte_view
from validate_training import nonzero_output
from types import SimpleNamespace

def author_backward(name,loss,images,model):
    """Execute the pinned trainer's backward statement, with CPU dependencies."""
    if name.startswith('relay'):
        scale=25 if name=='relay_low' else 1
        native.statements('relay/training/training_loop.py',153,154,
            dict(images=images,loss=loss,loss_scaling=scale))
        return loss.sum().mul(scale/(32 if images.shape[-1]==64 else 256))
    if name=='ssd':
        fn=native.method('ssd/guided_diffusion/fp16_util.py','backward',dict(th=torch))
        fn(SimpleNamespace(use_fp16=False),loss)
        return loss
    source_line={'dctdiff':('dctdiff/train.py',131),'dimr':('dimr/train.py',198),
        'dfm':('dfm/train.py',465),'dfm_yaml':('dfm/train.py',465),'dfm_repa':('dfm/train.py',465),
        'pixelflow':('pixelflow/train.py',127),'ppflow':('ppflow/train.py',327),
        'ppflow3':('ppflow/train.py',327),'lapflow':('lapflow/feb12_train_ours.py',429),
        'lapflow_paper':('lapflow/feb12_train_ours.py',429),
        'mdm':('mdm/ml-mdm-matryoshka/ml_mdm/trainer.py',75),
        'mdm1024':('mdm/ml-mdm-matryoshka/ml_mdm/trainer.py',75)}
    if name=='rdd':
        path=ROOT.parent/'2d/train_RDD_imagenet256_unified.py'
        node=next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.Expr) and n.lineno==592)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),dict(total_loss=loss,accum_steps=1))
        native.AUDIT.append(dict(source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),line_span=[592,592],execution='unchanged author backward statement; accumulation=1'))
        return loss
    relative,line=source_line[name]
    namespace=dict(loss=loss)
    if name.startswith('dfm') or name in ('dctdiff','dimr'):
        from accelerate import Accelerator
        namespace['accelerator']=Accelerator(cpu=True,mixed_precision='no',gradient_accumulation_steps=1)
    native.statements(relative,line,line,namespace)
    return loss

def snapshot_rng():
    return torch.get_rng_state().clone(),copy.deepcopy(np.random.get_state()),random.getstate()

def restore_rng(state):
    torch.set_rng_state(state[0]);np.random.set_state(state[1]);random.setstate(state[2])

def fingerprint(state):
    a,b,c=state
    return [hashlib.sha256(a.numpy().tobytes()).hexdigest(),
            hashlib.sha256(b[1].tobytes()+repr((b[0],*b[2:])).encode()).hexdigest(),
            hashlib.sha256(repr(c).encode()).hexdigest()]

def profile_name(name):
    if name.startswith('relay_'):return 'relay_low' if name.endswith('_low') else 'relay_high'
    if name.startswith('dctdiff_'):return 'dctdiff'
    if name.startswith('dimr_'):return 'dimr'
    if name.startswith('ppflow_'):return 'ppflow3' if name.endswith('_3') else 'ppflow'
    if name.startswith('dfm_'):return 'dfm_repa' if 'repa' in name else 'dfm_yaml' if '_b_' in name else 'dfm'
    if name.startswith('lapflow_'):return 'lapflow_paper' if name.endswith('_paper') else 'lapflow'
    if name.startswith('mdm_'):return 'mdm1024' if name.endswith('1024') else 'mdm'
    return name.split('_')[0]

def run(name,out,steps,threads,reduced=False):
    started=time.time();torch.set_num_threads(threads)
    torch.manual_seed(20261012);np.random.seed(20261012);random.seed(20261012)
    torch.use_deterministic_algorithms(True)
    path=ROOT/'configs/paired_experiments'/(name+'.json')
    config=json.loads(path.read_text());method_name=profile_name(name)
    report=dict(profile=name,status='running',device='cpu',gpu_used=False,
        config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),config=copy.deepcopy(config),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        implementation_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT/'umdiff').glob('*.py'))},
        verification_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT/'tools/activation_trace.py',ROOT/'tools/validate_training.py')},
        full_released_architecture=not reduced,full_input_grid=not reduced,
        author_cli_executed=False,pretrained_weights=False,real_data=False,
        scope='Author forward/loss bodies vs unified Method; independent backward before clipping/repair. Shared released backbone implementation; CPU FP32, synthetic model-domain inputs and common nonzero random checkpoint. Not a CLI/data/quality/CUDA certification.')
    audit=None
    try:
        if reduced:
            from validate_paired_experiments import reduced as reduce_config
            config,_=reduce_config(config)
            if method_name.startswith('relay'):
                config['training']['optimizer']['loss_multiplier']=(25 if method_name=='relay_low' else 1)/(32 if config['image_shape'][-1]==64 else 256)
        if method_name=='mdm1024':
            config['model']['args']['config']['initialize_inner_with_pretrained']=None
        # A test checkpoint must exercise hidden gradients instead of passing
        # trivially with zero-initialized final projection heads.
        a=nonzero_output(build('model',config['model']));b=copy.deepcopy(a)
        report['parameters']=sum(p.numel() for p in a.parameters())
        batch=8 if method_name.startswith('ppflow') else 4 if method_name=='pixelflow' else 3 if method_name=='mdm' else 2 if method_name.startswith('lapflow') or method_name=='rdd' or (method_name=='mdm1024' and reduced) else 1
        images=torch.randn(batch,*config['image_shape'])
        labels=torch.arange(batch,dtype=torch.long)%1000
        conditions={}
        if method_name=='dfm_repa':
            args=unpack_model(config['model'])['args'];tokens=(args['input_size']//2)**2
            conditions['teacher_features']=[torch.randn(batch,tokens,dim) for dim in args['z_dims']]
        method,original,unified,_,notes=setup(method_name,config,images,labels,conditions)
        report.update(effective_config=config,input_shape=list(images.shape),deviations=notes,
                      micro_batch=batch,initialization='same nonzero random state',precision='float32')
        audit=ActivationAudit(name,[('network',a,b)],out/'layers')
        report['steps']=[]
        for step in range(steps):
            # Exercise both released stage branches. These are explicitly
            # selected experiment inputs, not a claim about a long-run RNG.
            if method_name.startswith('dfm'):np.random.seed(0 if step%2==0 else 4)
            if method_name.startswith('lapflow'):random.seed(1 if step%2==0 else 0)
            a.train();b.train();a.zero_grad(set_to_none=True);b.zero_grad(set_to_none=True)
            before=snapshot_rng();audit.begin(step);audit.side('native')
            raw_a=original(a,step)
            audit.side('native','backward_recompute')
            la=author_backward(method_name,raw_a if method_name.startswith('relay') else raw_a.mean(),images,a)
            after=snapshot_rng();restore_rng(before)
            audit.side('unified');raw_b=unified(b,step)
            lb=raw_b.sum()*config['training']['optimizer']['loss_multiplier'] if method_name.startswith('relay') else raw_b.mean()
            audit.side('unified','backward_recompute');lb.backward()
            audit.parameters(a,b);audit.finish_segment()
            row=dict(step=step,native_loss=la.item(),unified_loss=lb.item(),
                loss_bytes_equal=la.dtype==lb.dtype and torch.equal(byte_view(la),byte_view(lb)),
                rng_equal=fingerprint(after)==fingerprint(snapshot_rng()),
                raw_loss_bytes_equal=raw_a.shape==raw_b.shape and raw_a.dtype==raw_b.dtype and torch.equal(byte_view(raw_a),byte_view(raw_b)),
                branch_control='both stages by fixed NumPy states' if method_name.startswith('dfm') else 'both stages by fixed Python states' if method_name.startswith('lapflow') else None)
            report['steps'].append(row)
            if not row['loss_bytes_equal'] or not row['raw_loss_bytes_equal']:
                raise AssertionError('Author/unified loss bytes differ')
            if not row['rng_equal']:raise AssertionError('Author/unified post-backward RNG diverged')
            if not all(v['pairs']==v['byte_exact'] for v in audit.totals.values()) or audit.structural_mismatches:
                raise AssertionError('Layer values or raw gradients differ; inspect first_differences')
            # Independent, identical update to make the next graph nontrivial.
            # Optimizer lifecycle is deliberately outside this fwd/bwd test.
            with torch.no_grad():
                for net in (a,b):
                    for p in net.parameters():
                        if p.grad is not None:
                            if not torch.isfinite(p.grad).all():raise AssertionError('Nonfinite raw gradient')
                            p.add_(p.grad,alpha=-1e-6)
            print(json.dumps(dict(event='step',profile=name,**row)),flush=True)
        report['activation']=audit.close();audit=None
        totals=report['activation']['totals']
        if not all(v['pairs']>0 and v['pairs']==v['byte_exact'] for v in totals.values()):
            raise AssertionError('Empty or nonexact forward/backward coverage')
        report['status']='passed'
    except Exception:
        report['status']='failed';report['failure']=traceback.format_exc()
    finally:
        if audit is not None:report['activation']=audit.close()
        report['source_executions']=copy.deepcopy(native.AUDIT)
        report['seconds']=time.time()-started
        report['cuda_initialized']=torch.cuda.is_initialized()
        out.mkdir(parents=True,exist_ok=True)
        (out/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:report[k] for k in ('profile','status','seconds')}),flush=True)
        if report.get('failure'):print(report['failure'],flush=True)
    return report['status']!='passed'

def main():
    p=argparse.ArgumentParser();p.add_argument('--profile');p.add_argument('--all',action='store_true')
    p.add_argument('--steps',type=int,default=2);p.add_argument('--threads',type=int,default=8)
    p.add_argument('--reduced',action='store_true');p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.all:
        rows=json.loads((ROOT/'configs/paired_experiments/manifest.json').read_text())['profiles']
        names=[r['id'] for r in rows if r['paper'] not in ('rmd','diffusability')]
        results=[];args.output.mkdir(parents=True,exist_ok=True)
        for name in names:
            cmd=[sys.executable,__file__,'--profile',name,'--output',str(args.output),'--steps',str(args.steps),'--threads',str(args.threads)]
            if args.reduced:cmd.append('--reduced')
            with (args.output/(name+'.log')).open('w') as log:
                code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
            result=json.loads((args.output/(name+'.json')).read_text())
            results.append(dict(profile=name,status=result['status'],returncode=code,seconds=result['seconds']))
            (args.output/'SUMMARY.json').write_text(json.dumps(dict(results=results,remaining=names[len(results):]),indent=2)+'\n')
            print(json.dumps(results[-1]),flush=True)
        return any(r['returncode'] for r in results)
    if not args.profile:p.error('Use --profile or --all')
    return run(args.profile,args.output,args.steps,args.threads,args.reduced)

if __name__=='__main__':raise SystemExit(main())
