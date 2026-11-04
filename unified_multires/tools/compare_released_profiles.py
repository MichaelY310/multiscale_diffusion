#!/usr/bin/env python3
"""Paired short runs of full released networks, with every deviation recorded.

These are conformance runs, not pretrained/FID replications. Original loss
bodies are executed separately from Method; the native optimizer and EMA do
not use Trainer. Module classes are intentionally reused from the pinned repo.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / '_deps'), str(ROOT / 'tools')]
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import numpy as np
import torch
import torch.nn.functional as F
from umdiff import build, native
from umdiff.engine import Method
from umdiff.native_models import activate
from umdiff.randomness import GlobalRandomStream
from umdiff.training import Trainer
from activation_trace import ActivationAudit, byte_view

OUT = ROOT / 'evidence/current_recipe_parity_20261009'
PROFILES = {
    'dctdiff': 'dctdiff_cifar10_uvit_mid_deep_2by2',
    'dctdiff_imagenet': 'dctdiff_imgnet64_uvit_mid_deep_2by2',
    'dimr': 'dimr_DiMR-G-2R_imagenet256',
    'pixelflow': 'pixelflow_xl_c2i',
    'ppflow': 'ppflow_xl_2',
    'ppflow3': 'ppflow_xl_3',
    'dfm': 'dfm_sit_xl',
    'dfm_yaml': 'dfm_sit_b_released_yaml',
    'lapflow': 'lapflow_mot_xl_2',
    'lapflow_paper': 'lapflow_imagenet_xl_2_paper',
    'ssd': 'ssd_celeba64',
    'mdm': 'mdm_cc12m256',
    'mdm1024': 'mdm_cc12m1024',
    'rdd': 'rdd_unified_xl_300_300_400',
    'relay_low': 'relay_imagenet_low',
    'relay_high': 'relay_imagenet_high',
}
STREAM = GlobalRandomStream()

def seed(value):
    torch.manual_seed(value)
    np.random.seed(value)
    random.seed(value)

def rng_fingerprint():
    state=np.random.get_state()
    return {'torch':hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
            'numpy':hashlib.sha256(state[1].tobytes()+repr((state[0],*state[2:])).encode()).hexdigest(),
            'python':hashlib.sha256(repr(random.getstate()).encode()).hexdigest()}

from umdiff.source_objectives import setup, unpack_model

def tensor_stats(expected, actual):
    same = expected.shape == actual.shape and expected.dtype == actual.dtype
    exact = same and torch.equal(byte_view(expected), byte_view(actual))
    difference = float((expected.float() - actual.float()).abs().max()) if same and expected.numel() else 0.
    return dict(exact=bool(exact), max_abs=difference, shape=list(expected.shape), dtype=str(expected.dtype))

def compare_state(a, b):
    # Stream per tensor, do not duplicate optimizer states or large weights.
    result = dict(tensors=0, exact=0, max_abs=0.)
    def visit(x, y):
        if isinstance(x, torch.Tensor):
            s = tensor_stats(x, y)
            result['tensors'] += 1; result['exact'] += s['exact']
            result['max_abs'] = max(result['max_abs'], s['max_abs'])
        elif isinstance(x, dict):
            if x.keys() != y.keys(): raise AssertionError(f'state dictionary keys differ: {list(x.keys()-y.keys())[:10]}, {list(y.keys()-x.keys())[:10]}')
            for k in x: visit(x[k], y[k])
        elif isinstance(x, (list, tuple)):
            if len(x) != len(y): raise AssertionError('state sequence lengths differ')
            for u, v in zip(x, y): visit(u, v)
        elif x != y:
            raise AssertionError(f'non-tensor state differs: {x!r}, {y!r}')
    visit(a, b)
    return result

def named_optimizer_state(optimizer, model):
    """Match states by parameter name, not IDs shifted by frozen parameters.

    The author may include frozen position embeddings in parameter groups;
    Adam never creates a state/updates a parameter with gradient=None.
    Trainer omits these from its groups. Preserve every active group field.
    """
    names = {id(p): name for name, p in model.named_parameters()}
    state = {names[id(p)]: v for p, v in optimizer.state.items()}
    groups = [{**{k:v for k,v in group.items() if k!='params'},
               'params':[names[id(p)] for p in group['params'] if p.requires_grad]}
              for group in optimizer.param_groups]
    return dict(state=state,param_groups=groups)

def run(name, steps, threads, precision):
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time(); report = dict(profile=name, status='running', device='cpu', gpu_used=False,
                                        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path = ROOT / 'configs/paper_recipes' / (PROFILES[name] + '.json')
    config = json.loads(path.read_text())
    report.update(template=str(path.relative_to(ROOT)), template_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  config=deepcopy(config), full_paper_quality_verified=False, original_best_config_fully_executed=False)
    audit = None
    try:
        torch.set_num_threads(threads); seed(20261009)
        method, original, port, images, notes = setup(name, config)
        recipe = deepcopy(config['training']['optimizer'])
        recipe['lr'] = float(recipe['lr'])
        recipe['precision'] = precision
        if name.startswith('relay'):
            # Native loop loss.sum()*ls/t compared against actual logical batch.
            recipe['loss_reduction']='sum'
            recipe['loss_multiplier'] = (25 if name == 'relay_low' else 1) / (32 if images.shape[-1]==64 else 256)
            recipe['accumulation_steps']=1
        effective_model=deepcopy(config['model'])
        if name=='mdm1024':
            effective_model['args']['config']['initialize_inner_with_pretrained']=None
            notes.append('Required pretrained256 inner model unavailable; explicitly disabled only for this random-weight architecture conformance run.')
        report.update(effective_recipe=recipe, deviations=notes,
                      precision_deviation=precision != config['training']['optimizer'].get('precision','float32'),
                      full_released_architecture=True, actual_input_shape=list(images.shape))
        print(json.dumps(dict(event='build', profile=name, shape=list(images.shape))), flush=True)
        report['effective_model']=effective_model
        a = build('model', effective_model)
        b = deepcopy(a)
        model_parameters = sum(p.numel() for p in a.parameters())
        report['parameters'] = model_parameters
        # Method scheduling is executed in the port call, not Trainer's sampling.
        class ForwardMethod:
            config = method.config
            def training(self, net, x, t, stream, noise=None, model_kwargs=None, active=None, context=None):
                from umdiff.engine import TrainingResult
                value = port(net, context['step'])
                return TrainingResult(value, None, None, None, {})
        trainer = Trainer(ForwardMethod(), b, **recipe)
        ema = deepcopy(a).eval().requires_grad_(False)
        factory = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}[recipe.get('optimizer_name','adamw')]
        optimizer = factory(a.parameters(), lr=recipe['lr'], betas=recipe.get('betas',(.9,.999)),
                            weight_decay=recipe.get('weight_decay',0.), eps=recipe.get('eps',1e-8))
        scaler = torch.amp.GradScaler('cpu', enabled=precision=='float16')
        audit = ActivationAudit(name, [('network', a, b)], OUT / 'activation')
        trajectories = []
        for step in range(steps):
            print(json.dumps(dict(event='step', profile=name, step=step, parameters=model_parameters)), flush=True)
            audit.begin(step); audit.side('native'); a.train(); optimizer.zero_grad(set_to_none=True)
            from umdiff.recipes import lr_factor
            if recipe.get('lr_schedule'):
                for group in optimizer.param_groups:
                    group['lr'] = recipe['lr'] * lr_factor(step, recipe['lr_schedule'], images_seen=step*len(images)*recipe.get('image_count_multiplier',1))
            seed(1100+step)
            with torch.autocast('cpu', dtype={'float16':torch.float16,'bfloat16':torch.bfloat16}.get(precision,torch.bfloat16), enabled=precision!='float32'):
                loss_a = original(a, step)
                objective = (loss_a.sum() if recipe.get('loss_reduction')=='sum' else loss_a.mean()) * recipe.get('loss_multiplier',1.)
            audit.side('native','backward_recompute'); scaler.scale(objective).backward(); scaler.unscale_(optimizer)
            if recipe.get('gradient_repair'):
                for param in a.parameters():
                    if param.grad is not None:torch.nan_to_num(param.grad,out=param.grad,**recipe['gradient_repair'])
            if recipe.get('grad_clip'):
                torch.nn.utils.clip_grad_norm_(a.parameters(), recipe['grad_clip'], error_if_nonfinite=True)
            scaler.step(optimizer); scaler.update()
            with torch.no_grad():
                decay = recipe.get('ema_decay',.999)
                if recipe.get('ema_schedule'):
                    half = recipe['ema_schedule']['half_life_images']; ramp=recipe['ema_schedule'].get('rampup_ratio')
                    if ramp is not None: half=min(half,step*len(images)*recipe.get('image_count_multiplier',1)*ramp)
                    decay=.5**(len(images)*recipe.get('image_count_multiplier',1)/max(half,1e-8))
                for q, p in zip(ema.parameters(), a.parameters()):
                    if recipe.get('ema_style')=='scaled_add': q.mul_(decay).add_((1-decay)*p)
                    elif recipe.get('ema_style')=='source_lerp': q.copy_(p.detach().lerp(q,decay))
                    else: q.mul_(decay).add_(p,alpha=1-decay)
                for q,p in zip(ema.buffers(),a.buffers()):q.copy_(p)
            native_rng=rng_fingerprint()
            audit.side('unified'); seed(1100+step)
            # Hooks for checkpoint recomputation remain active. Trainer invokes
            # backward internally; compare call ordinals, not spelling of phase.
            audit.execution='forward'
            result = trainer.step(images,None)
            # Native backward checkpoint calls are semantically the same calls;
            # Trainer hooks don't expose a backward callback, so normalize only
            # this metadata, preserving every captured value and gradient.
            for events in audit.events.values():
                for record in events.values():record['execution']='training'
            audit.parameters(a,b); audit.finish_segment()
            row=dict(step=step,native_loss=float(loss_a.detach().mean()),unified_loss=float(result.loss.detach().mean()),
                     model=compare_state(a.state_dict(),b.state_dict()),
                     optimizer=compare_state(named_optimizer_state(optimizer,a),named_optimizer_state(trainer.optimizer,b)),
                     ema=compare_state(ema.state_dict(),trainer.ema.state_dict()))
            row['loss_scalar_exact']=row['native_loss']==row['unified_loss']
            row['loss_scalar_abs']=abs(row['native_loss']-row['unified_loss'])
            row['post_step_global_rng_byte_exact']=native_rng==rng_fingerprint()
            row['post_step_native_rng']=native_rng;row['post_step_unified_rng']=rng_fingerprint()
            trajectories.append(row); report['trajectory']=trajectories
            print(json.dumps(dict(event='compared',profile=name,**row)),flush=True)
        report['activation']=audit.close(); audit=None
        totals=report['activation']['totals']
        exact = not report['activation']['structural_mismatches'] and all(s['pairs']==s['byte_exact'] for s in totals.values())
        exact = exact and all(row[k]['tensors']==row[k]['exact'] for row in trajectories for k in ('model','optimizer','ema'))
        exact=exact and all(row['post_step_global_rng_byte_exact'] for row in trajectories)
        report['status']='byte_exact' if exact else 'different'
        report['byte_exact_scope']='Executed module inputs/outputs, activation gradients, parameter gradients, active optimizer state, all model/EMA tensors. Scalar loss is reported separately.'
        report['optimizer_matching']='States matched by parameter name, frozen position embeddings excluded from group-ID comparison; all active states retained.'
        report['hook_metadata']='Native/unified checkpoint execution phase labels normalized to training; tensor values and call ordinals unchanged.'
        report['steps']=steps
        # Small state manifest rather than dozens of GB of duplicate checkpoints.
        report['final_state_manifest']={key:hashlib.sha256(byte_view(value.detach()).numpy().tobytes()).hexdigest() for key,value in b.state_dict().items()}
    except Exception:
        report['status']='failed';report['failure']=traceback.format_exc()
        print(report['failure'],flush=True)
        if audit is not None:report['activation']=audit.close()
    report['seconds']=time.time()-started
    (OUT/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report.get(k) for k in ('profile','status','parameters','seconds','failure')}),flush=True)
    return report['status']!='byte_exact'

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('profile',choices=PROFILES)
    parser.add_argument('--steps',type=int,default=2)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--precision',choices=['float32','bfloat16','float16'],default='float32')
    args=parser.parse_args()
    raise SystemExit(run(args.profile,args.steps,args.threads,args.precision))
