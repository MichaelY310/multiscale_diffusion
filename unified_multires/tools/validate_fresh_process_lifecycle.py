#!/usr/bin/env python3
"""Fresh-process author-core/unified lifecycle for every published image profile.

Original uses an independent optimizer/EMA loop, not Trainer. This completes
source-core resume checks; it deliberately does NOT certify every author's CLI.
"""
import argparse,copy,hashlib,json,os,random,subprocess,sys,tempfile,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'_deps')]
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np
import torch
from umdiff import build,native
from umdiff.training import Trainer
from umdiff.engine import TrainingResult
from umdiff.source_objectives import setup,unpack_model
from umdiff.recipes import lr_factor
from validate_paired_experiments import reduced
from validate_full_forward_backward import profile_name,snapshot_rng,restore_rng,fingerprint
from validate_training import nonzero_output
from activation_trace import ActivationAudit,byte_view
from compare_released_profiles import compare_state


def manifest(value):
    rows=[]
    def visit(x,path):
        if torch.is_tensor(x):rows.append(dict(path=path,shape=list(x.shape),dtype=str(x.dtype),sha256=hashlib.sha256(byte_view(x.detach()).numpy().tobytes()).hexdigest()))
        elif isinstance(x,np.ndarray):rows.append(dict(path=path,shape=list(x.shape),dtype=str(x.dtype),sha256=hashlib.sha256(x.tobytes()).hexdigest()))
        elif isinstance(x,dict):
            for k,v in x.items():visit(v,path+'/'+str(k))
        elif isinstance(x,(list,tuple)):
            rows.append(dict(path=path,sequence_type=type(x).__name__,length=len(x)))
            for i,v in enumerate(x):visit(v,path+'/'+str(i))
        else:rows.append(dict(path=path,type=type(x).__name__,value=x))
    visit(value,'root');return rows


def worker(a):
    torch.set_num_threads(2);torch.manual_seed(917);np.random.seed(917);random.seed(917)
    path=ROOT/'configs/paired_experiments'/(a.profile+'.json');c,notes=reduced(json.loads(path.read_text()))
    name=profile_name(a.profile);recipe=c['training']['optimizer'];recipe['optimizer_parameters']='all'
    if name=='relay_high':
        unpack_model(c['model'])['args']['img_resolution']=64;c['image_shape']=[3,64,64]
        notes.append('Relay high grid reduced from256 to64 for four-update resume lifecycle; original256 full-forward/backward certificate remains separate.')
    if name.startswith('relay'):
        recipe['loss_reduction']='sum';recipe['loss_multiplier']=(25/32 if name=='relay_low' else 1/256)
    if name=='mdm1024':c['model']['args']['config']['initialize_inner_with_pretrained']=None
    # Two microbatches for each update, preserving native packing divisibility.
    micro=8 if name.startswith('ppflow') else 4 if name=='pixelflow' else 3 if name.startswith('mdm') else 2
    count=2*micro;data_gen=torch.Generator().manual_seed(992)
    images=torch.randn(5,count,*c['image_shape'],generator=data_gen);labels=torch.arange(count)%1000
    input_conditions={}
    if name.startswith('mdm'):input_conditions=dict(lm_outputs=torch.randn(count,8,32,generator=data_gen),lm_mask=torch.ones(count,8))
    if name=='dfm_repa':
        args=unpack_model(c['model'])['args'];input_conditions['teacher_features']=[torch.randn(count,(args['input_size']//2)**2,dim,generator=data_gen) for dim in args['z_dims']]
    original=nonzero_output(build('model',c['model']));port_model=copy.deepcopy(original)
    class Port:
        config=c
        def training(self,net,x,t,stream,noise=None,model_kwargs=None,active=None,context=None):
            _,_,fn,_,_=setup(name,c,x,model_kwargs['labels'],model_kwargs['conditions'])
            loss=fn(net,context['step'])
            return TrainingResult(loss if recipe.get('loss_reduction')=='sum' else loss.mean(),None,None,None,{})
    trainer=Trainer(Port(),port_model,**recipe)
    optimizer={'adam':torch.optim.Adam,'adamw':torch.optim.AdamW}[recipe.get('optimizer_name','adamw')](original.parameters(),
        lr=float(recipe['lr']),betas=recipe.get('betas',(.9,.999)),eps=recipe.get('eps',1e-8),weight_decay=recipe.get('weight_decay',0.))
    ema=copy.deepcopy(original).eval().requires_grad_(False);step=0;images_seen=0
    if a.resume:
        state=torch.load(a.resume,weights_only=False);original.load_state_dict(state['original']['model']);ema.load_state_dict(state['original']['ema'])
        optimizer.load_state_dict(state['original']['optimizer']);step=state['original']['step'];images_seen=state['original']['images_seen']
        trainer.load(Path(a.resume).with_name('unified.pt'));restore_rng(state['original']['rng'])
    audit=ActivationAudit(a.profile,[('network',original,port_model)],Path(a.output)/'layers');rows=[]
    for _ in range(a.steps):
        x=images[step];before=snapshot_rng();audit.begin(step);audit.side('native');original.train();optimizer.zero_grad(set_to_none=True)
        schedule=recipe.get('lr_schedule')
        if schedule:
            for group in optimizer.param_groups:group['lr']=float(recipe['lr'])*lr_factor(step,schedule,images_seen=images_seen)
        losses=[]
        for j in range(2):
            lo=j*micro;hi=lo+micro;conditions={k:([v0[lo:hi] for v0 in v] if isinstance(v,list) else v[lo:hi]) for k,v in input_conditions.items()}
            _,fn,_,_,_=setup(name,c,x[lo:hi],labels[lo:hi],conditions)
            raw=fn(original,step);loss=raw.sum() if recipe.get('loss_reduction')=='sum' else raw.mean()/(2 if recipe.get('accumulation_normalization','mean')=='mean' else 1)
            loss=loss*recipe.get('loss_multiplier',1.);loss.backward();losses.append(loss.detach())
        audit.execution='forward'
        raw_gradients={k:None if p.grad is None else p.grad.detach().clone() for k,p in original.named_parameters()}
        if recipe.get('gradient_repair'):
            for p in original.parameters():
                if p.grad is not None:torch.nan_to_num(p.grad,out=p.grad,**recipe['gradient_repair'])
        if recipe.get('grad_clip'):torch.nn.utils.clip_grad_norm_(original.parameters(),recipe['grad_clip'],error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            decay=recipe.get('ema_decay',.999)
            if recipe.get('ema_schedule'):
                half=recipe['ema_schedule']['half_life_images'];ramp=recipe['ema_schedule'].get('rampup_ratio')
                if ramp is not None:half=min(half,images_seen*ramp)
                decay=.5**(count*recipe.get('image_count_multiplier',1)/max(half,1e-8))
            for q,p in zip(ema.parameters(),original.parameters()):
                if recipe.get('ema_style')=='scaled_add':q.mul_(decay).add_((1-decay)*p)
                elif recipe.get('ema_style')=='source_lerp':q.copy_(p.detach().lerp(q,decay))
                else:q.mul_(decay).add_(p,alpha=1-decay)
            for q,p in zip(ema.buffers(),original.buffers()):q.copy_(p)
        after=snapshot_rng();audit.side('unified');restore_rng(before)
        def raw_gate(net):
            for k,p in net.named_parameters():
                expected=raw_gradients[k]
                if (expected is None)!=(p.grad is None) or (expected is not None and not torch.equal(byte_view(expected),byte_view(p.grad))):raise AssertionError('Pre-clip/repair parameter gradient '+k)
        trainer.step(x,None,model_kwargs={'labels':labels,'conditions':input_conditions},before_gradient_processing=raw_gate)
        audit.parameters(original,port_model);audit.finish_segment()
        checks={k:compare_state(u,v) for k,u,v in [('model',original.state_dict(),port_model.state_dict()),('ema',ema.state_dict(),trainer.ema.state_dict()),('optimizer',optimizer.state_dict(),trainer.optimizer.state_dict())]}
        if any(s['exact']!=s['tensors'] for s in checks.values()) or fingerprint(after)!=fingerprint(snapshot_rng()):raise AssertionError('Lifecycle state or RNG differs')
        rows.append(dict(step=step,state=checks,rng_byte_exact=True,pre_clip_repair_parameter_gradients_byte_exact=True));step+=1;images_seen+=count*recipe.get('image_count_multiplier',1)
    act=audit.close()
    if act['structural_mismatches'] or any(v['pairs']!=v['byte_exact'] for v in act['totals'].values()):raise AssertionError('Raw layer/gradient mismatch')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    native_state=dict(model=original.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),step=step,images_seen=images_seen,rng=snapshot_rng())
    torch.save(dict(original=native_state),out/'original.pt');trainer.save(out/'unified.pt')
    report=dict(status='passed',profile=a.profile,effective_config=c,deviations=notes,steps=rows,activation=act,
        source_executions=native.AUDIT,original_state=manifest(native_state),unified_state=manifest(trainer.state_dict()),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),gpu_used=False,
        scope='Independent source-core optimizer/EMA versus Trainer;fresh-process checkpoint lifecycle,all parameter groups retained;CPU FP32,reduced nonzero author model,synthetic inputs;not full author CLI.')
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--worker',action='store_true');p.add_argument('--profile');p.add_argument('--steps',type=int,default=4);p.add_argument('--resume',type=Path);a=p.parse_args()
    if a.worker:worker(a);return 0
    entries=json.loads((ROOT/'configs/paired_experiments/manifest.json').read_text())['profiles']
    names=[r['id'] for r in entries if r['paper'] not in ('rmd','diffusability')]
    if a.profile:names=[a.profile]
    a.output.mkdir(parents=True,exist_ok=True);rows=[];started=time.time()
    for name in names:
        with tempfile.TemporaryDirectory(prefix='umdiff_resume_'+name+'_') as tmp:
            tmp=Path(tmp);base=a.output/name;base.mkdir(parents=True,exist_ok=True)
            def run(kind,steps,resume=None):
                cmd=[sys.executable,__file__,'--worker','--profile',name,'--output',str(tmp/kind),'--steps',str(steps)]
                if resume:cmd+=['--resume',str(resume)]
                with (base/(kind+'.log')).open('w') as log:result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=300)
                if result.returncode:raise RuntimeError(f'{name}/{kind} failed')
                report=json.loads((tmp/kind/'result.json').read_text());(base/(kind+'.json')).write_text(json.dumps(report,indent=2)+'\n');return report
            try:
                full=run('full',4);first=run('first',2);resumed=run('resumed',2,tmp/'first/original.pt')
                for key in ('original_state','unified_state'):
                    if full[key]!=resumed[key]:raise AssertionError('Resumed '+key+' differs')
                rows.append(dict(profile=name,status='passed',updates=4,fresh_process_original_exact=True,fresh_process_unified_exact=True,source_core=True,author_cli=False))
            except Exception:rows.append(dict(profile=name,status='failed',failure=traceback.format_exc()))
            print(json.dumps(rows[-1]),flush=True)
            (a.output/'SUMMARY.json').write_text(json.dumps(dict(status='running',profiles=rows,remaining=names[len(rows):]),indent=2)+'\n')
    report=dict(status='passed' if all(r['status']=='passed' for r in rows) else 'failed',profiles=rows,remaining=[],gpu_used=False,formal_training=False,seconds=time.time()-started)
    (a.output/'SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n');return report['status']!='passed'
if __name__=='__main__':raise SystemExit(main())
