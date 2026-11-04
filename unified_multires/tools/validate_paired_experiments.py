#!/usr/bin/env python3
"""Small, continuous-RNG CPU checks of the two NEW source-core launch backends.

Source side manually applies Adam/EMA/accumulation, unified side uses Trainer.
No data/weights downloads; no formal training; no retained large checkpoints.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'_deps'),str(ROOT/'tools')]
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np
import torch
from umdiff import build
from umdiff.training import Trainer
from umdiff.recipes import lr_factor
from paired_experiments import load_manifest,compile_plan,selected,BoundCoreMethod
from compare_released_profiles import compare_state,named_optimizer_state,unpack_model,rng_fingerprint
from activation_trace import ActivationAudit,byte_view
from validate_training import nonzero_output

def reduced(config):
    c=copy.deepcopy(config);m=unpack_model(c['model']);a=m['args'];name=m['name']
    deviations=['Synthetic inputs; random common nonzero initial weights; no pretrained quality assertion.',
      'Reduced author backbone width/depth; FP32 CPU; logical batch reduced to two microbatches.',
      'Continuous RNG across updates: restore the paired pre-step state only; no per-step reseeding.']
    if name=='native_dctdiff':a.update(embed_dim=32,depth=2,num_heads=4)
    elif name=='native_dimr':
        for stage in a['config']['stage_configs']:stage.update(dim=32,hidden_dim=64,num_attention_heads=4,num_blocks=1)
    elif name=='native_pixelflow':a.update(depth=2,num_attention_heads=2)
    elif name=='native_ppflow':a.update(hidden_size=32,depth=2,num_heads=4)
    elif name=='native_dfm':a.update(hidden_size=32,decoder_hidden_size=32,depth=2,encoder_depth=1,num_heads=4)
    elif name=='native_lapflow':a.update(hidden_size=32,depth=2,num_heads=4)
    elif name=='native_ssd':a.update(model_channels=32,num_res_blocks=1)
    elif name=='native_relay':a.update(model_channels=32,num_blocks=1,channel_mult=[1,2])
    elif name=='native_rdd_unified':a.update(hidden_size=32,depth=2,num_heads=4)
    elif name=='native_mdm':
        def shrink(cfg):
            cfg['resolution_channels']=[32]*len(cfg['resolution_channels'])
            cfg['num_resnets_per_resolution']=[1]*len(cfg['resolution_channels'])
            cfg['num_attention_layers']=[int(v>0) for v in cfg['num_attention_layers']]
            cfg['conditioning_feature_dim']=32;cfg['conditioning_feature_proj_dim']=32
            cfg['resnet_config']['num_groups_norm']=8
            cfg['temporal_dim']=128
            if 'initialize_inner_with_pretrained' in cfg:cfg['initialize_inner_with_pretrained']=None
            if cfg.get('inner_config'):shrink(cfg['inner_config'])
        shrink(a['config']);c['image_shape']=[3,64,64]
        deviations.append('MDM grid reduced to64; original nested scale ratios/SNR/loss weights retained; required pretrained inner initialization bypassed only in this synthetic test.')
    else:raise ValueError('No reduced conformance adapter: '+name)
    c['training']['optimizer']['precision']='float32';c['training']['optimizer']['accumulation_steps']=2
    return c,deviations

def save_rng():return (torch.get_rng_state(),copy.deepcopy(np.random.get_state()),random.getstate())
def restore_rng(state):torch.set_rng_state(state[0]);np.random.set_state(state[1]);random.setstate(state[2])

def run(plan,out,steps=2,match_batch=False):
    started=time.time();name=plan['id'];report=dict(profile=name,mode=plan['mode'],contract_sha256=plan['contract_sha256'],
      gpu_used=False,formal_training=False,full_paper_result_certified=False,author_cli_lifecycle_certified=False)
    audit=None
    try:
        config,deviations=reduced(plan['contract']['method_config']);recipe=config['training']['optimizer']
        accumulation=plan['contract']['controls']['accumulation'] if match_batch else 2
        recipe['accumulation_steps']=accumulation
        torch.manual_seed(20261010);np.random.seed(20261010);random.seed(20261010)
        a=nonzero_output(build('model',config['model']));b=copy.deepcopy(a)
        original=BoundCoreMethod(config,plan['source_profile'],'original')
        trainer=Trainer(BoundCoreMethod(config,plan['source_profile'],'unified'),b,**recipe)
        factory={'adam':torch.optim.Adam,'adamw':torch.optim.AdamW}[recipe.get('optimizer_name','adamw')]
        opt=factory([p for p in a.parameters() if p.requires_grad],lr=float(recipe['lr']),
          betas=recipe.get('betas',(.9,.999)),weight_decay=recipe.get('weight_decay',0.),eps=recipe.get('eps',1e-8))
        ema=copy.deepcopy(a).eval().requires_grad_(False)
        audit=ActivationAudit(name,[('network',a,b)],out/'activation')
        micro=8 if plan['paper']=='ppflow' else 4 if plan['paper']=='pixelflow' else 3 if plan['paper']=='mdm' else 1
        if match_batch:
            micro=plan['contract']['controls']['micro_batch']
            deviations=[d for d in deviations if 'logical batch reduced' not in d]
            deviations.append('Full configured global batch/microbatch/accumulation retained; reduced backbone and FP32CPU still apply.')
        batch=accumulation*micro;rows=[]
        for step in range(steps):
            images=torch.randn(batch,*config['image_shape'])
            labels=torch.arange(batch)%1000
            conditions={'lm_outputs':torch.randn(batch,8,32),'lm_mask':torch.ones(batch,8)} if plan['paper']=='mdm' else {}
            prior=save_rng();audit.begin(step);audit.side('native');a.train();opt.zero_grad(set_to_none=True)
            if recipe.get('lr_schedule'):
                for group in opt.param_groups:group['lr']=float(recipe['lr'])*lr_factor(step,recipe['lr_schedule'],images_seen=step*batch*recipe.get('image_count_multiplier',1))
            losses=[]
            for i in range(accumulation):
                start=i*micro;end=start+micro
                result=original.training(a,images[start:end],None,None,
                  model_kwargs={'labels':labels[start:end],'conditions':{k:v[start:end] for k,v in conditions.items()}},context={'step':step})
                loss=result.loss.mean();losses.append(loss.detach());objective=loss/accumulation*recipe.get('loss_multiplier',1.)
                audit.execution='backward_recompute';objective.backward();audit.execution='forward'
            if recipe.get('gradient_repair'):
                for p in a.parameters():
                    if p.grad is not None:torch.nan_to_num(p.grad,out=p.grad,**recipe['gradient_repair'])
            if recipe.get('grad_clip'):torch.nn.utils.clip_grad_norm_(a.parameters(),recipe['grad_clip'],error_if_nonfinite=True)
            opt.step()
            with torch.no_grad():
                decay=recipe.get('ema_decay',.999)
                if recipe.get('ema_schedule'):
                    half=recipe['ema_schedule']['half_life_images'];ramp=recipe['ema_schedule'].get('rampup_ratio')
                    if ramp is not None:half=min(half,step*batch*recipe.get('image_count_multiplier',1)*ramp)
                    decay=.5**(batch*recipe.get('image_count_multiplier',1)/max(half,1e-8))
                for q,p in zip(ema.parameters(),a.parameters()):
                    if recipe.get('ema_style')=='scaled_add':q.mul_(decay).add_((1-decay)*p)
                    elif recipe.get('ema_style')=='source_lerp':q.copy_(p.detach().lerp(q,decay))
                    else:q.mul_(decay).add_(p,alpha=1-decay)
                for q,p in zip(ema.buffers(),a.buffers()):q.copy_(p)
            after=rng_fingerprint();restore_rng(prior);audit.side('unified')
            result=trainer.step(images,None,model_kwargs={'labels':labels,'conditions':conditions})
            for events in audit.events.values():
                for event in events.values():event['execution']='training'
            audit.parameters(a,b);audit.finish_segment()
            rows.append(dict(step=step,loss_exact=bool(torch.equal(torch.stack(losses).mean(),result.loss.mean())),
              native_loss=torch.stack(losses).mean().item(),unified_loss=result.loss.mean().item(),
              rng_exact=after==rng_fingerprint(),model=compare_state(a.state_dict(),b.state_dict()),
              optimizer=compare_state(named_optimizer_state(opt,a),named_optimizer_state(trainer.optimizer,b)),
              ema=compare_state(ema.state_dict(),trainer.ema.state_dict())))
        activation=audit.close();audit=None
        exact=not activation['structural_mismatches'] and all(v['pairs']==v['byte_exact'] for v in activation['totals'].values())
        exact=exact and all(row['loss_exact'] and row['rng_exact'] and all(row[k]['exact']==row[k]['tensors'] for k in ('model','optimizer','ema')) for row in rows)
        report.update(status='byte_exact' if exact else 'different',steps=steps,trajectory=rows,activation=activation,test_seed=20261010,
          effective_config=config,effective_logical_batch=batch,effective_micro_batch=micro,accumulation=accumulation,deviations=deviations,
          final_state_sha256=digest_state(b),parameters=sum(p.numel() for p in a.parameters()))
    except Exception:
        report.update(status='failed',failure=traceback.format_exc());print(report['failure'],flush=True)
        if audit is not None:report['activation']=audit.close()
    report['seconds']=time.time()-started;out.mkdir(parents=True,exist_ok=True)
    (out/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report.get(k) for k in ('profile','status','seconds','failure')}),flush=True)
    return report

def digest_state(model):
    return {k:hashlib.sha256(byte_view(v.detach()).numpy().tobytes()).hexdigest() for k,v in model.state_dict().items()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['paper','fair-rdd'],default='fair-rdd')
    p.add_argument('--papers',default='all');p.add_argument('--profile');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=2);p.add_argument('--match-batch',action='store_true');p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);a=p.parse_args()
    if a.steps<1 or a.steps>4:p.error('Small conformance checks allow1-4 updates only')
    torch.set_num_threads(2);manifest=load_manifest();entries,_=selected(manifest,a);reports=[];skipped=[]
    a.output.mkdir(parents=True,exist_ok=True)
    for entry in entries:
        plan=compile_plan(manifest,entry,a.mode)
        if entry['paper'] not in {'rdd','dimr','ppflow','dfm','lapflow','pixelflow','relay','ssd','mdm','dctdiff'} or entry['id']=='dfm_sit_xl_repa_localfeatures':
            skipped.append(dict(id=entry['id'],reason='Specialized codec/distillation/teacher pipeline; use separate component evidence.'));continue
        if a.mode=='fair-rdd' and not entry['fair_eligible']:continue
        if a.worker:
            reports.append(run(plan,a.output,a.steps,a.match_batch))
        else:
            # Author repos deliberately reuse top-level module names such as
            # models/libs/sampler. A fresh process per profile is mandatory.
            import subprocess
            command=[sys.executable,str(Path(__file__).resolve()),'--mode',a.mode,'--profile',entry['id'],
                '--output',str(a.output),'--steps',str(a.steps),'--worker']
            if a.match_batch:command+=['--match-batch']
            with (a.output/(entry['id']+'.log')).open('w') as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
            path=a.output/(entry['id']+'.json')
            report=json.loads(path.read_text()) if path.exists() else dict(profile=entry['id'],status='failed',seconds=0.,failure='Worker exited '+str(result.returncode))
            if result.returncode and report['status']=='byte_exact':report['status']='failed'
            reports.append(report);print(json.dumps({k:report.get(k) for k in ('profile','status','seconds','failure')}),flush=True)
    result=dict(status='passed' if reports and all(r['status']=='byte_exact' for r in reports) else 'failed',
      mode=a.mode,profiles=len(reports),summaries=[{k:r[k] for k in ('profile','status','seconds')} for r in reports],
      skipped=skipped,gpu_used=False,formal_training=False,full_paper_result_certified=False,
      scope='Reduced nonzero author backbones, synthetic inputs, CPUFP32;2-microbatch accumulation;continuous RNG;shared source-core frontend.')
    (a.output/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    return result['status']!='passed'

if __name__=='__main__':raise SystemExit(main())
