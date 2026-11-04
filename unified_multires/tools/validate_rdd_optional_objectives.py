#!/usr/bin/env python3
"""RDD optional objectives versus actual local source; no teachers downloaded."""
import argparse,copy,hashlib,itertools,json,os,subprocess,sys,tempfile,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'_deps')]
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1'
import numpy as np
import torch
from umdiff import build
from umdiff.engine import Method
from umdiff.training import Trainer
from umdiff.rdd_native import diffusion_backend
from umdiff.randomness import GlobalRandomStream
from activation_trace import ActivationAudit,byte_view
from validate_training import nonzero_output,seed
from validate_full_forward_backward import snapshot_rng,restore_rng,fingerprint
from validate_fresh_process_lifecycle import manifest


def exact(a,b,name):
    if manifest(a)!=manifest(b):raise AssertionError(name+' differs (shape/dtype/bytes/typed state)')


def fixture(weighting='p2',repa='all',clock='adaptive'):
    backend=diffusion_backend()
    from diffusion.timestep_sampler import UniformSampler,LossSecondMomentResampler
    from diffusion.rdd_improvements import weighted_diffusion_loss,pool_teacher_tokens,representation_alignment_loss
    diffs=[backend.create_diffusion_with_drift('',diffusion_steps=16,noise_schedule='squaredcos_cap_v2',c=0 if i==0 else None,T_c=8*i,T_c_plus_1=8*(i+1),sample_scale=2) for i in range(2)]
    spec={'name':'native_rdd_unified','args':dict(patch_size=2,in_channels=4,hidden_size=32,depth=2,num_heads=4,class_dropout_prob=.1,learn_sigma=True,base_size=8,repa_dim=8 if repa!='off' else None,repa_depth=1,repa_projector_dim=32)}
    stages=[];samplers=[]
    for i,d in enumerate(diffs):
        align=repa=='all' or (repa=='finest' and i==0)
        sampler={'name':'discrete','args':{'steps':d.num_timesteps}} if clock=='torch' else {'name':'loss_second_moment','args':dict(steps=d.num_timesteps,adaptive=clock=='adaptive',history_per_term=2)}
        samplers.append(None if clock=='torch' else LossSecondMomentResampler(d,history_per_term=2) if clock=='adaptive' else UniformSampler(d))
        stages.append(dict(input={'name':'average_pool','args':dict(factor=2**i,detach=i>0)},time_sampler=sampler,
            method={'batch':{'name':'rdd_ddpm','args':dict(path=dict(betas=d.betas.tolist(),timestep_map=d.timestep_map,drift_start=0 if i==0 else None,drift_end=8 if i==0 else None),learned_variance=True,loss_weighting=weighting,repa_weight=(.5/(2 if repa=='all' else 1)) if align else 0.)}}))
    config=dict(training_stages=stages,training_stage_reduction='sum_stage_means')
    method=Method(config);model=nonzero_output(build('model',spec))
    gen=torch.Generator().manual_seed(661)
    images=torch.randn(3,4,16,16,generator=gen);labels=torch.arange(3);teacher=torch.randn(3,64,8,generator=gen)
    if clock=='adaptive':
        for source,(_,_,port,_) in zip(samplers,method.training_stages):
            # Exercise nonuniform weights immediately, including FIFO updates.
            for repeat in range(2):
                times=torch.arange(source.diffusion.num_timesteps);losses=(times.float()+1)*(repeat+1)/7
                source.update_with_all_losses(times.tolist(),losses.tolist());port.update(times,losses)
    captured=[]
    def original(net):
        total=0.;captured.clear()
        for i,d in enumerate(diffs):
            x=images if i==0 else images.detach().reshape(3,4,8,2,8,2).mean((3,5))
            t,w=(torch.randint(d.num_timesteps,(3,)),None) if samplers[i] is None else samplers[i].sample(3,'cpu')
            features={};align=repa=='all' or (repa=='finest' and i==0)
            def forward(x,t,**kw):
                prediction,features['value']=net(x,t,return_features=True,**kw)
                return prediction
            terms=d.training_losses(forward if align else net,x,t,{'y':labels})
            if clock=='adaptive':samplers[i].update_with_all_losses(t.tolist(),terms['loss'].detach().tolist())
            loss,_=weighted_diffusion_loss(terms,d,t,weighting,importance_weights=w)
            scalar=loss.mean()
            if align:scalar=scalar+(.5/(2 if repa=='all' else 1))*representation_alignment_loss(features['value'],pool_teacher_tokens(teacher,features['value'].shape[1])).mean()
            total=total+scalar;captured.append(dict(t=t,weight=w,terms=terms))
        return total
    return model,method,original,samplers,images,labels,teacher,spec


def matrix(out):
    rows=[]
    for weighting,repa,clock in itertools.product(('none','min-snr','p2'),('off','finest','all'),('torch','numpy','adaptive')):
        name='_'.join((weighting,repa,clock));seed(884)
        a,method,original,samplers,x,y,teacher,spec=fixture(weighting,repa,clock);b=copy.deepcopy(a)
        oa=torch.optim.AdamW(a.parameters(),lr=1e-4);ob=torch.optim.AdamW(b.parameters(),lr=1e-4)
        audit=ActivationAudit(name,[('network',a,b)],out/'layers')
        for step in range(2):
            audit.begin(step);before=snapshot_rng();oa.zero_grad(set_to_none=True);ob.zero_grad(set_to_none=True)
            audit.side('native');la=original(a);la.backward();after=snapshot_rng()
            audit.side('unified');restore_rng(before)
            lb=method.training(b,x,None,GlobalRandomStream(),model_kwargs={'y':y},context={'teacher_features':teacher}).loss
            lb.backward();exact(la,lb,name+'/loss');exact(fingerprint(after),fingerprint(snapshot_rng()),name+'/RNG')
            audit.parameters(a,b);audit.finish_segment()
            if clock=='adaptive':
                for source,(_,_,port,_) in zip(samplers,method.training_stages):
                    exact(source._loss_history,port.history,name+'/history');exact(source._loss_counts,port.counts,name+'/counts');exact(source.weights(),port.weights(),name+'/probabilities')
            oa.step();ob.step();exact(a.state_dict(),b.state_dict(),name+'/weights');exact(oa.state_dict(),ob.state_dict(),name+'/AdamW')
        act=audit.close()
        if act['structural_mismatches'] or any(v['pairs']!=v['byte_exact'] for v in act['totals'].values()):raise AssertionError(name+'/layer values or gradients')
        row=dict(name=name,status='passed',updates=2,activation=act,scope='Independent objective assembly versus local RDD training_losses + actual rdd_improvements/timestep_sampler helpers; shared real reduced DiTMultiRes; precomputed synthetic teacher features, no teacher encoder claim.')
        (out/(name+'.json')).write_text(json.dumps(row,indent=2)+'\n');rows.append(dict(name=name,status='passed'));print(name+' passed',flush=True)
    return rows


def worker(a):
    seed(884);net,method,_,_,x,y,teacher,_=fixture();trainer=Trainer(method,net,lr=1e-4,randomness='native_global')
    if a.resume:trainer.load(a.resume)
    for _ in range(a.steps):trainer.step(x,None,model_kwargs={'y':y},context={'teacher_features':teacher})
    a.output.mkdir(parents=True,exist_ok=True);trainer.save(a.output/'state.pt')
    (a.output/'state.json').write_text(json.dumps(manifest(trainer.state_dict()),indent=2)+'\n')


def guards(out):
    seed(884);model,method,*_=fixture();trainer=Trainer(method,model,randomness='native_global')
    original=trainer.state_dict();cases=[]
    def reject(name,state):
        p=out/'invalid.pt';torch.save(state,p)
        try:trainer.load(p)
        except ValueError as e:cases.append(dict(name=name,status='passed',rejection=str(e)))
        else:raise AssertionError('Invalid checkpoint accepted: '+name)
        finally:p.unlink(missing_ok=True)
    state=copy.deepcopy(original);state.pop('method_state');reject('missing_method_history',state)
    state=copy.deepcopy(original);state['method_state']['stages'][0]['sampler']=None;reject('missing_stage_history',state)
    state=copy.deepcopy(original);state['method_state']['stages'][0]['sampler']['history']=np.zeros((1,2));reject('wrong_history_shape',state)
    state=copy.deepcopy(original);state['method_state']['stages'][0]['sampler']['counts'][0]=999;reject('invalid_counts',state)
    trainer.time_sampler=build('time_sampler',{'name':'loss_second_moment','args':{'steps':8}})
    reject('missing_external_sampler_history',original)
    return cases


def cli_fixture(out):
    """The distributed-neutral CLI consumes local precomputed teacher fields."""
    config=ROOT/'configs/native_fixtures/rdd_optional_objectives.json'
    with tempfile.TemporaryDirectory(prefix='rdd_optional_cli_') as tmp:
        tmp=Path(tmp);gen=torch.Generator().manual_seed(117)
        torch.save(dict(latents=torch.randn(9,4,16,16,generator=gen),labels=torch.arange(9),conditions={'teacher_features':torch.randn(9,64,8,generator=gen)}),tmp/'data.pt')
        for name,steps,resume in [('full',4,None),('first',2,None),('resumed',2,tmp/'first/checkpoint.pt')]:
            cmd=[sys.executable,'-m','umdiff.cli','train','--config',str(config),'--data',str(tmp/'data.pt'),'--batch-size','3','--device','cpu','--seed','33','--steps',str(steps),'--output',str(tmp/name)]
            if resume:cmd+=['--resume',str(resume)]
            with (out/('cli_'+name+'.log')).open('w') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=180)
        full=torch.load(tmp/'full/checkpoint.pt',weights_only=False);resumed=torch.load(tmp/'resumed/checkpoint.pt',weights_only=False)
        exact(full,resumed,'Actual CLI optional RDD resume')
        return dict(status='passed',updates=4,split=2,all_checkpoint_fields_exact=True,config=str(config.relative_to(ROOT)),teacher='local synthetic precomputed features; encoder deliberately outside this objective contract')


def drift_weight_policy():
    from umdiff.gaussian_objective import GaussianBatch
    from diffusion.rdd_improvements import weighted_diffusion_loss
    d=diffusion_backend().create_diffusion_with_drift('',diffusion_steps=16,noise_schedule='squaredcos_cap_v2',c=0,T_c=0,T_c_plus_1=8,sample_scale=2)
    x=torch.randn(3,4,16,16);noise=torch.randn_like(x);t=torch.tensor([0,3,7]);prediction=torch.randn(3,12,16,16,requires_grad=True)
    args=dict(path=dict(betas=d.betas.tolist(),timestep_map=d.timestep_map,drift_start=0,drift_end=8),learned_variance=True,decoupled=True)
    base=GaussianBatch(**args)(x,t,GlobalRandomStream(),noise);raw,terms=base.loss(prediction)
    expected,_=weighted_diffusion_loss({**terms,'loss':raw},d,t,'p2')
    compat,_=GaussianBatch(**args,loss_weighting='p2',omit_drift_for_native_weighting=True)(x,t,GlobalRandomStream(),noise).loss(prediction)
    corrected,_=GaussianBatch(**args,loss_weighting='p2')(x,t,GlobalRandomStream(),noise).loss(prediction)
    exact(compat,expected,'native weighting omission policy');exact(corrected,expected+terms['mse_d'],'explicit retained drift loss')
    if torch.equal(compat,corrected):raise AssertionError('Negative control did not expose drift-loss omission')
    return dict(status='passed',native_helper_omits_drift_for_non_none_weighting=True,native_exact_option='omit_drift_for_native_weighting=true',default_policy='retain mse_d; intentional correction only for decoupled+non-none weighting',affects_base_rdd=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--worker',action='store_true');p.add_argument('--steps',type=int,default=4);p.add_argument('--resume',type=Path);a=p.parse_args();torch.set_num_threads(2)
    if a.worker:worker(a);return 0
    a.output.mkdir(parents=True,exist_ok=True);started=time.time();report={}
    try:
        report['cases']=matrix(a.output);report['negative_controls']=guards(a.output);report['drift_weight_policy']=drift_weight_policy();report['cli']=cli_fixture(a.output)
        with tempfile.TemporaryDirectory(prefix='rdd_optional_resume_') as tmp:
            tmp=Path(tmp)
            for kind,steps,resume in [('full',4,None),('first',2,None),('resumed',2,tmp/'first/state.pt')]:
                cmd=[sys.executable,__file__,'--worker','--output',str(tmp/kind),'--steps',str(steps)]
                if resume:cmd+=['--resume',str(resume)]
                with (a.output/(kind+'.log')).open('w') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=180)
            full=json.loads((tmp/'full/state.json').read_text());resumed=json.loads((tmp/'resumed/state.json').read_text())
            if full!=resumed:raise AssertionError('Fresh-process adaptive-history resume differs')
            report['resume']=dict(status='passed',all_checkpoint_fields_exact=True,updates=4,split=2,state_manifest=full)
        report['status']='passed'
    except Exception:report.update(status='failed',failure=traceback.format_exc());print(report['failure'],flush=True)
    report.update(seconds=time.time()-started,gpu_used=False,models_downloaded=False,harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256={str(f.relative_to(ROOT.parent)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [ROOT.parent/'2d/train_RDD_imagenet256_unified.py',ROOT.parent/'diffusion/rdd_improvements.py',ROOT.parent/'diffusion/timestep_sampler.py']})
    (a.output/'SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n');print(report['status'],flush=True);return report['status']!='passed'
if __name__=='__main__':raise SystemExit(main())
