#!/usr/bin/env python3
"""Plan/check or explicitly run matched author-core/unified-core experiments.

Never describe these shared-lifecycle runs as complete author-CLI replications.
Unidentified/private/full-pipeline recipes fail closed, not guessed or skipped.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'_deps')]
MANIFEST=ROOT/'configs/paired_experiments/manifest.json'
DEFAULT_OUT=ROOT/'evidence/paired_experiments_20261010'
CORE_PROFILES={'dctdiff','dimr','pixelflow','ppflow','dfm','lapflow','ssd','mdm','rdd','relay'}

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def unpack_repa(config):
    return config.get('batch',{}).get('args',{}).get('repa_weight',0)>0

def source_profile(entry):
    p=entry['paper'];name=entry['id']
    if p=='rdd':return 'rdd'
    if p=='relay':return 'relay_low' if name.endswith('_low') else 'relay_high'
    if p=='ppflow':return 'ppflow3' if name.endswith('_3') else 'ppflow'
    if p=='dfm':return 'dfm_yaml' if name=='dfm_sit_b_released_yaml' else 'dfm'
    if p=='lapflow':return 'lapflow_paper' if name.endswith('_paper') else 'lapflow'
    if p=='mdm':return 'mdm1024' if name.endswith('1024') else 'mdm'
    return p

def load_manifest():
    return json.loads(MANIFEST.read_text())

def selected(manifest,args):
    names=set(args.papers.split(',')) if args.papers!='all' else set(manifest['papers'])
    unknown=names-set(manifest['papers'])
    if unknown:raise ValueError('Unknown papers: '+str(sorted(unknown)))
    rows=[p for p in manifest['profiles'] if p['paper'] in names]
    if args.profile:
        rows=[p for p in rows if p['id']==args.profile]
        if not rows:raise ValueError('Unknown profile or excluded paper: '+args.profile)
    return rows,names

def check_pins(config):
    for pin in config['evidence'].get('source_pins',[]):
        root=ROOT.parent if pin.get('root')=='workspace' else ROOT/'sources'
        if hashlib.sha256((root/(pin['file'].removeprefix('multiscale_diffusion/') if pin.get('root')=='workspace' else pin['file'])).read_bytes()).hexdigest()!=pin['sha256']:
            raise ValueError('Source hash changed: '+pin['file'])

def author_cli_reference(entry):
    """Unmodified entrypoint references; not falsely rendered as matched commands."""
    refs={'relay':['train.py','generate_imagenet.py'],
      'ssd':['scripts/image_train.py','scripts/image_sample.py'],
      'pixelflow':['train.py','sample_ddp.py'],
      'ppflow':['train.py','sample_ddp.py'],
      'dfm':['train.py','generate.py'],
      'lapflow':['feb12_train_ours.py','feb12_test_ours.py'],
      'dctdiff':['train.py'], 'dimr':['train.py','eval.py'],
      'mdm':['ml-mdm-matryoshka/ml_mdm/clis/train_parallel.py'],
      'diffusability':['src/train.py'],'rmd':['src/rmd/cli.py']}
    if entry['paper']=='rdd':return [str(ROOT.parent/'2d/train_RDD_imagenet256_unified.py')]
    return [str(ROOT/'sources'/entry['paper']/p) for p in refs.get(entry['paper'],[])]

def compile_plan(manifest,entry,mode,bindings=None):
    path=MANIFEST.parent/entry['config']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['config_sha256']:
        raise ValueError('Prepared profile hash changed; rebuild the manifest: '+entry['id'])
    config=json.loads(path.read_text());check_pins(config)
    c=copy.deepcopy(entry['paper_contract']);reasons=[]
    if mode=='fair-rdd':
        fair=copy.deepcopy(manifest['fair_rdd'])
        c=dict(dataset=fair['dataset'],resolution=256,task=fair['task'],
          global_batch=fair['global_batch'],world_size=fair['world_size'],micro_batch=fair['micro_batch'],
          accumulation=fair['accumulation'],budget=fair['updates'],seed=fair['seed'],optimizer=fair['optimizer'],
          implementation_scope=entry['paper_contract']['implementation_scope'])
        if not entry['fair_eligible']:reasons.append('Different benchmark/task, codec-only objective, unselected profile or teacher-feature branch; no silent RDD conversion.')
        # Only the requested experiment controls change. Native algorithm,
        # representation, target, initialization, decoder and solver remain.
        config['training']['optimizer']=copy.deepcopy(fair['optimizer'])
    else:
        batch=c['global_batch']
        if isinstance(batch,int):
            # Native CLI topology is not claimed. Record a controlled, single
            # process batch split; retain exact arithmetic style/schedules.
            micro=8 if entry['paper']=='ppflow' else 4 if entry['paper']=='pixelflow' else 1
            c.update(micro_batch=micro,accumulation=batch//micro)
        if not isinstance(c['global_batch'],int):reasons.append('Released/paper global batch not identified.')
        if not isinstance(c['budget'],int):reasons.append('Exact update budget not identified; epochs/image-count budgets require the native data/lifecycle.')
        if c['seed'] is None:reasons.append('Author best-run seed not identified; do not silently replace it with0.')
    if entry['paper'] not in CORE_PROFILES:reasons.append('Requires a separate codec/distillation or full pretrained inference pipeline; common scratch trainer is inapplicable.')
    if c.get('world_size')!=1:raise ValueError('This shared-core driver supports exactly one process; no false DDP plan.')
    if isinstance(c.get('global_batch'),int):
        if c['world_size']*c['micro_batch']*c['accumulation']!=c['global_batch']:raise ValueError('Effective global batch mismatch')
        if entry['paper']=='ppflow' and c['micro_batch']%8:raise ValueError('PPFlow source batch packing requires a microbatch divisible by8 in this adapter')
        if entry['paper']=='pixelflow' and c['micro_batch']%4:raise ValueError('PixelFlow source stage-balanced packing requires a microbatch divisible by4')
        config['training']['optimizer']['accumulation_steps']=c['accumulation']
        # Relay uses an image clock per global batch; its template describes
        # eight native ranks. Never carry that multiplier into one process.
        if entry['paper']=='relay':
            opt=config['training']['optimizer'];opt['image_count_multiplier']=1
            denominator=32 if entry['id'].endswith('_low') else 256
            opt['loss_reduction']='sum'
            opt['loss_multiplier']=(25 if entry['id'].endswith('_low') else 1)/denominator
    required=sorted(set(re.findall(r'@([a-z][a-z0-9_]*)@',json.dumps(config))))
    bindings=bindings or {};assets=bindings.get('assets',{})
    # A profile may override a same-paper data binding (e.g.CIFAR vsFFHQ).
    data=bindings.get('data',{}).get(entry['id'],bindings.get('data',{}).get(entry['paper']))
    local={k:str(Path(assets[k]).resolve()) for k in required if k in assets}
    def replace(value):
        if isinstance(value,str):return re.sub(r'@([a-z][a-z0-9_]*)@',lambda m:local.get(m[1],m[0]),value)
        if isinstance(value,list):return [replace(v) for v in value]
        if isinstance(value,dict):return {k:replace(v) for k,v in value.items()}
        return value
    config=replace(config)
    contract=dict(profile=entry['id'],mode=mode,controls=c,method_config=config,data=data,assets=local,
      full_paper_result_certified=False,author_cli_lifecycle_certified=False)
    from author_cli_plans import commands
    return dict(id=entry['id'],paper=entry['paper'],mode=mode,contract=contract,contract_sha256=digest(contract),
      required_assets=required,missing_assets=[k for k in required if k not in local],
      blocked_reasons=reasons,source_profile=source_profile(entry),
      original_author_cli_reference=author_cli_reference(entry),production_entrypoints=commands(entry,contract),
      scope='Pinned author source core / unified components with identical shared training lifecycle; not unmodified native CLI')

class BoundCoreMethod:
    """Both launchers enter the same reader/optimizer; only loss construction differs."""
    def __init__(self,config,profile,backend):
        self.config=config;self.profile=profile;self.backend=backend
    def training(self,net,images,t,stream,noise=None,model_kwargs=None,active=None,context=None):
        from umdiff.source_objectives import setup
        from umdiff.engine import TrainingResult
        kwargs=model_kwargs or {};ctx=context or {}
        profile='dfm_repa' if unpack_repa(self.config) else self.profile
        _,source,unified,_,_=setup(profile,self.config,images=images,
            labels=kwargs.get('labels'),input_conditions=kwargs.get('conditions',{}))
        loss=(source if self.backend=='original' else unified)(net,ctx.get('step',0))
        # Native CLIs log/reduce the microbatch scalar. Returning per-example
        # vectors on only one backend made the accumulated log mean round in a
        # different order, even with identical gradients and optimizer states.
        return TrainingResult(loss if self.config['training']['optimizer'].get('loss_reduction')=='sum' else loss.mean(),None,None,None,{})

def execute(plan,backend,output):
    if plan['blocked_reasons']:raise ValueError('Recipe is blocked: '+'; '.join(plan['blocked_reasons']))
    if plan['missing_assets']:raise ValueError('Missing exact local assets: '+str(plan['missing_assets']))
    contract=plan['contract'];data=contract['data']
    if not data or not Path(data).exists():raise ValueError('Supply a real local data path in --bindings; synthetic formal training is forbidden')
    for name,path in contract['assets'].items():
        if not Path(path).exists():raise FileNotFoundError(name+': '+path)
    if int(os.environ.get('WORLD_SIZE','1'))!=1:raise ValueError('The shared source-core driver is single process')
    # No current GPU authorization. Do not renew the expired window implicitly.
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import random
    import numpy as np
    import torch
    from umdiff import build
    from umdiff.cli import OfflineTensors,TensorShards,ImageFolder
    from umdiff.input_transforms import LocalMomentDataset
    from umdiff.training import Trainer,data_binding
    from umdiff.recipes import initialize,freeze_parameters
    c=contract['method_config'];controls=contract['controls'];seed=controls['seed']
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    output=output/backend/plan['id']
    if output.exists():raise FileExistsError('Refuse to overwrite an experiment directory: '+str(output))
    method=BoundCoreMethod(c,plan['source_profile'],backend)
    model=build('model',c['model'])
    initializations=c['training'].get('initialization',[])
    if isinstance(initializations,dict):initializations=[initializations]
    for item in initializations:initialize(model,item)
    if c['training'].get('freeze'):freeze_parameters(model,c['training']['freeze'])
    trainer=Trainer(method,model,seed=seed,**c['training']['optimizer'])
    transform=build('input_transform',c['training']['input_transform']) if c['training'].get('input_transform') else None
    path=Path(data);recipe=c['training'].get('data',{})
    if path.is_file():dataset=OfflineTensors(path)
    elif (path/'manifest.json').exists():dataset=TensorShards(path)
    elif recipe.get('kind')=='moments_npy':dataset=LocalMomentDataset(path)
    else:dataset=ImageFolder(path,recipe.get('image_size',c['image_shape'][-1]),recipe)
    batch=controls['global_batch'];steps=controls['budget']
    if len(dataset)<batch:raise ValueError('Dataset smaller than one full global batch')
    trainer.data_binding=data_binding(path,batch,seed,files=getattr(dataset,'files',getattr(getattr(dataset,'images',None),'paths',None)))
    output.mkdir(parents=True);(output/'contract.json').write_text(json.dumps(plan,indent=2)+'\n')
    cache={}
    with (output/'trace.jsonl').open('w') as trace:
        for step in range(steps):
            epoch_size=len(dataset)//batch*batch if recipe.get('drop_last',False) else len(dataset)
            absolute=[step*batch+i for i in range(batch)];epochs={a//epoch_size for a in absolute}
            cache={epoch:cache.get(epoch) for epoch in epochs}
            for epoch in epochs:
                if cache[epoch] is None:cache[epoch]=torch.randperm(len(dataset),generator=torch.Generator().manual_seed(seed+epoch))
            indices=[int(cache[a//epoch_size][a%epoch_size]) if recipe.get('shuffle') else a%epoch_size for a in absolute]
            value=torch.utils.data.default_collate([dataset[i] for i in indices]);images=value['images']
            if 'labels' not in value and plan['paper'] in {'rdd','dimr','ppflow','dfm','lapflow','pixelflow'}:raise ValueError('Class-conditional training requires real dataset labels')
            features={k:v for k,v in value.items() if k not in ('images','labels')}
            if plan['paper']=='mdm' and not {'lm_outputs','lm_mask'}<=set(features):raise ValueError('MDM requires real local T5 features and masks')
            if transform is not None:images=transform(images,trainer.stream)
            if unpack_repa(c):
                if 'teacher_features' not in features:raise ValueError('REPA requires local teacher_features from the specified teacher')
                features['teacher_features']=[features['teacher_features']]
            if list(images.shape[1:])!=c['image_shape']:raise ValueError('Data/codec output shape differs from the recipe')
            result=trainer.step(images,None,model_kwargs={'labels':value.get('labels'),'conditions':features})
            row=dict(step=trainer.step_index,loss=result.loss.mean().item());trace.write(json.dumps(row)+'\n')
            if trainer.step_index%100==0:trace.flush();print(json.dumps(row),flush=True)
            if trainer.step_index%50000==0:trainer.save(output/'checkpoint.pt')
    trainer.save(output/'checkpoint.pt')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',required=True,choices=['original','unified'])
    p.add_argument('--mode',choices=['paper','fair-rdd'],default='paper')
    p.add_argument('--action',choices=['plan','check','validate','run'],default='plan')
    p.add_argument('--entrypoint',choices=['native-cli','matched-core'],default='native-cli',
        help='native-cli plans the genuine original/unified CLIs; matched-core is the executable controlled frontend')
    p.add_argument('--papers',default='all');p.add_argument('--profile')
    p.add_argument('--bindings',type=Path);p.add_argument('--output',type=Path,default=DEFAULT_OUT)
    a=p.parse_args();manifest=load_manifest();entries,names=selected(manifest,a)
    bindings=json.loads(a.bindings.read_text()) if a.bindings else None
    plans=[compile_plan(manifest,row,a.mode,bindings) for row in entries]
    pipeline_only=[name for name in names if name not in {p['paper'] for p in entries}]
    result=dict(backend=a.backend,mode=a.mode,entrypoint=a.entrypoint,plans=plans,
      pipeline_only=[dict(paper=name,reason='Inference-only release; no scratch training experiment',recipe=manifest['papers'][name]) for name in sorted(pipeline_only)],
      unreleased=manifest['unreleased'],full_paper_result_certified=False,gpu_used=False,
      formal_training_requested=a.action=='run',formal_training_started=False,configuration_source=str(MANIFEST))
    a.output.mkdir(parents=True,exist_ok=True)
    for plan in plans:
        path=(a.output/'resolved'/a.mode/(plan['id']+'.json')).resolve()
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(plan['contract']['method_config'],indent=2)+'\n')
        plan['contract']['resolved_config_path']=str(path)
        plan['resolved_config_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        plan['contract_sha256']=digest(plan['contract'])
        from author_cli_plans import commands
        entry=next(e for e in entries if e['id']==plan['id'])
        plan['production_entrypoints']=commands(entry,plan['contract'])
    target=a.output/(a.mode+'_'+a.backend+'_plan.json');target.write_text(json.dumps(result,indent=2)+'\n')
    if a.action=='validate':
        import subprocess
        cmd=[sys.executable,str(ROOT/'tools/validate_paired_experiments.py'),'--mode',a.mode,'--output',str(a.output/'validation'/a.mode)]
        if a.profile:cmd+=['--profile',a.profile]
        elif a.papers!='all':cmd+=['--papers',a.papers]
        return subprocess.call(cmd,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
    if a.action=='run':
        if not a.profile:p.error('Formal training requires exactly one explicit --profile; there is no automatic full suite')
        if a.entrypoint=='native-cli':raise ValueError('Native/unified production CLI pair is not certified: '+'; '.join(plans[0]['production_entrypoints']['blocked_reasons']))
        execute(plans[0],a.backend,a.output/'runs')
    print(json.dumps(dict(plan=str(target),profiles=len(plans),eligible_source_core=sum(not x['blocked_reasons'] for x in plans),eligible_native_cli=0,
      pipeline_only=sorted(pipeline_only),action=a.action,formal_training_started=a.action=='run',gpu_used=False,
      scope='source-core controlled experiments; full paper FID/native CLI not certified'),ensure_ascii=False))
    return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,FileNotFoundError,FileExistsError) as error:
        print(str(error),file=sys.stderr);raise SystemExit(2)
