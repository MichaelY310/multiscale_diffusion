#!/usr/bin/env python3
"""Native RMD equations, actual random Wan, sequential two-optimizer parity."""
import argparse,json,os,sys,time,traceback
from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'_deps'),str(ROOT/'sources/rmd/src')]
os.environ.setdefault('CUDA_VISIBLE_DEVICES','');os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import numpy as np
import torch
from umdiff import build,native
from umdiff.distillation import RMDClock,RMDPredictor,RMDObjective,score_loss,revision_loss
from umdiff.randomness import GlobalRandomStream
import conformance
compare=conformance.compare
AUDIT_FACTORY=None
CHECKPOINT_ROOT=ROOT/'runs'/'native_parity'
MODEL={'name':'wan_transformer','args':dict(patch_size=[1,2,2],num_attention_heads=2,attention_head_dim=16,in_channels=4,out_channels=4,text_dim=8,freq_dim=16,ffn_dim=64,num_layers=1)}

def fixtures(transition=1,warmup_step=3,target_size=(8,12)):
    from diffusers import FlowMatchEulerDiscreteScheduler
    source=native.load('rmd/src/rmd/engines/predictor.py')
    loss=native.load('rmd/src/rmd/losses/distillation.py')
    env=native.extract('rmd/src/rmd/engines/train.py',['_sample_timesteps','_sample_ind_t','_phase1','_phase2'],dict(torch=torch,gather_trajectory_batch=source.gather_trajectory_batch,extract_into_tensor=source.extract_into_tensor,TrainConfig=object,**{k:getattr(loss,k) for k in ('compute_huber_c','compute_revised_latents','compute_score_loss','compute_student_loss','compute_weighting_factor')}))
    source_clock=source.EulerSolver(FlowMatchEulerDiscreteScheduler(shift=3.).sigmas.numpy()[::-1],1000,1000)
    source_high=source.EulerSolver(FlowMatchEulerDiscreteScheduler(shift=5.).sigmas.numpy()[::-1],1000,1000)
    channels=MODEL['args'].get('in_channels',4);text_dim=MODEL['args'].get('text_dim',8)
    model=build('model',MODEL);negative=torch.randn(1,3,text_dim);text=torch.randn(2,3,text_dim);template=torch.empty(2,channels,2,4,6)
    args=dict(k_step=6,low_rs_step=3,flow_shift_trans=transition,warmup_step=warmup_step,split_timestep=True,t_offset_factor=.25,timestep_offset_mode='scaled',cfg=5.,eta=.5,score_weighting_mode='inverse_sigma',upsampler_mode='bilinear')
    cfg=SimpleNamespace(**args,gan=False)
    predictor=source.Predictor(cfg,None,None,'cpu',torch.float32,source_clock,source_high,uncond_prompt_embeds=negative)
    objective=RMDObjective(negative,GlobalRandomStream(),**args,target_size=target_size)
    return source,loss,env,cfg,predictor,objective,model,template,text,source_clock,source_high

def components():
    src,loss,env,cfg,p,objective,model,template,text,clock,high=fixtures()
    compare('rmd/sigma',objective.clock.sigmas,clock.sigmas,atol=0,rtol=0)
    compare('rmd/high_sigma',objective.clock.high_sigmas,high.sigmas,atol=0,rtol=0)
    state=torch.randn_like(template);index=torch.tensor([400,700])
    for higher in (False,True):
        for guidance in (None,1.,5.):
            a=p.predict(model,state,index,text,cfg=guidance,return_double=True,higher_shift=higher)
            b=objective.predictor.predict(model,state,index,text,cfg=guidance,return_double=True,high=higher)
            compare('rmd/predict_clean_noise',b,a,atol=0,rtol=0)
        start=torch.tensor([100,200]);noise=torch.randn_like(state)
        compare('rmd/add_noise',objective.predictor.add_noise(state,noise,start,index,higher),p.add_noise(state,noise,start,index,higher),atol=0,rtol=0)
    for dtype in (torch.float32,torch.float64):
        a=state.to(dtype);b=torch.randn_like(a);c=torch.randn_like(a);sigma=torch.tensor([.2,.8],dtype=dtype).reshape(2,1,1,1,1)
        for mode in ('inverse_sigma','legacy_snr'):compare('rmd/score_loss',score_loss(a,b,sigma,mode),loss.compute_score_loss(a,b,sigma,mode=mode),atol=0,rtol=0)
        revised=loss.compute_revised_latents(a,b,c);weight=loss.compute_weighting_factor(a,b)
        compare('rmd/revision_loss',revision_loss(a,b,c,a.shape),loss.compute_student_loss(a,revised,weight,loss.compute_huber_c(a.shape)),atol=0,rtol=0)
    for transition in (0,1,2):
        for eta in (0.,.5,1.):
            objective=RMDObjective(p.uncond_prompt_embeds,GlobalRandomStream(),eta=eta,flow_shift_trans=transition,target_size=(8,12))
            torch.manual_seed(4701);expected=p.generate_new_upsample(model,None,state,state,text,None,steps=6,eta=eta,return_mid=True,flow_shift_trans=transition,relusion_shift=2,target_size=(8,12))
            torch.manual_seed(4701);actual,clean,noisy=objective.predictor.generate(model,state,text,6,transition,2,(8,12))
            compare(f'rmd/transition{transition}/eta{eta}/clean',clean,expected[0],atol=0,rtol=0,scope='native_inference_trajectory')
            compare(f'rmd/transition{transition}/eta{eta}/noisy',noisy,expected[1],atol=0,rtol=0,scope='native_inference_trajectory')
    return {'scope':'Actual pinned source Predictor, helpers and loss functions, random local WanTransformer3DModel. All transitions0/1/2 and eta0/.5/1. No pretrained assets.'}

def train(steps,warmup_step=3,target_size=(8,12),transitions=(1,2),save_checkpoints=True):
    records=[]
    for transition in transitions:
        src,loss,env,cfg,p,objective,model,template,text,clock,high=fixtures(transition,warmup_step,target_size)
        a=deepcopy(model);b=deepcopy(model);af=deepcopy(model);bf=deepcopy(model);real=deepcopy(model).eval().requires_grad_(False)
        opts=[torch.optim.AdamW(net.parameters(),lr=rate,betas=(0.,.95),weight_decay=1e-4) for net,rate in [(a,5e-6),(b,5e-6),(af,2e-5),(bf,2e-5)]]
        audit=AUDIT_FACTORY(f'rmd/transition{transition}',[('generator',a,b),('fake',af,bf),('teacher',real,real)]) if AUDIT_FACTORY else None
        trace=[]
        for step in range(steps):
            for net in (a,b,af,bf):net.train()
            if audit:audit.begin(step,'trajectory');audit.side('native')
            torch.manual_seed(1100+step)
            torch.randn(template.shape);noise=torch.randn_like(template)
            a.eval()
            with torch.no_grad():
                clean,trajectory=p.generate_new_upsample(a,None,noise,noise,text,None,steps=6,eta=.5,return_mid=True,flow_shift_trans=transition,relusion_shift=2,target_size=target_size)
            trajectory.reverse();indices=torch.randint(4,6,(2,)).long();example=torch.randn_like(clean[int(indices[0])])
            for i in range(2):example[i]=clean[int(indices[i])][i]
            a.train()
            if audit:audit.side('unified')
            torch.manual_seed(1100+step);port_example,port_trajectory=objective.trajectory(b,template,text)
            compare('rmd/train/trajectory',port_trajectory,trajectory,atol=0,rtol=0)
            compare('rmd/train/example',port_example,example,atol=0,rtol=0)
            if audit:audit.finish_segment()
            values=[]
            for phase in (1,2):
                na,nb=(af,bf) if phase==1 else (a,b);oa,ob=(opts[2],opts[3]) if phase==1 else (opts[0],opts[1])
                oa.zero_grad(set_to_none=True);ob.zero_grad(set_to_none=True)
                args=(cfg,p,a,af)+((real,) if phase==2 else ())+(clock,high,example,trajectory,text,None,6,4,objective.scales,1000,torch.float32,'cpu')
                if phase==2:args+=(None,)
                args+=(step,)
                if audit:audit.begin(step,f'phase{phase}');audit.side('native')
                torch.manual_seed(2000+step*2+phase);expected=env[f'_phase{phase}'](*args,target_size=target_size)
                if audit:audit.side('unified')
                torch.manual_seed(2000+step*2+phase);actual=objective.phase(phase,b,bf,real,port_example,port_trajectory,text,step)
                name=f'rmd/trans{transition}/step{step}/phase{phase}'
                compare(name+'/loss',actual,expected,atol=1e-7,rtol=1e-5,scope='native_full_training_loss')
                if audit:audit.side('native','backward_recompute')
                expected.backward()
                if audit:audit.side('unified','backward_recompute')
                actual.backward()
                if audit:audit.parameters(na,nb,'fake' if phase==1 else 'generator');audit.finish_segment()
                for (key,pa),(_,pb) in zip(na.named_parameters(),nb.named_parameters()):
                    if (pa.grad is None)!=(pb.grad is None):raise AssertionError('Gradient presence differs: '+key)
                    if pa.grad is not None:compare(name+'/gradient/'+key,pb.grad,pa.grad,atol=1e-7,rtol=1e-4,scope='native_full_training_gradient')
                if phase==2:
                    compare(name+'/clip_norm',torch.nn.utils.clip_grad_norm_(nb.parameters(),1.),torch.nn.utils.clip_grad_norm_(na.parameters(),1.),atol=1e-7,rtol=1e-4)
                oa.step();ob.step()
                for (key,pa),(_,pb) in zip(na.named_parameters(),nb.named_parameters()):compare(name+'/updated_weight/'+key,pb,pa,atol=2e-7,rtol=1e-4,scope='native_full_training_optimizer')
                values.append([expected.item(),actual.item()])
            trace.append(dict(step=step,score=values[0],generator=values[1]))
        if audit:audit.close()
        initial=torch.randn_like(template)
        torch.manual_seed(8701);expected=p.generate_new_upsample(a.eval(),None,initial,initial,text,None,steps=6,eta=.5,return_mid=True,flow_shift_trans=transition,relusion_shift=2,target_size=target_size)
        torch.manual_seed(8701);actual,clean,noisy=objective.predictor.generate(b.eval(),initial,text,6,transition,2,target_size)
        compare('rmd/trained_clean_trajectory',clean,expected[0],atol=0,rtol=0,scope='native_trained_model_inference_latents')
        compare('rmd/trained_noisy_trajectory',noisy,expected[1],atol=0,rtol=0,scope='native_trained_model_inference_latents')
        from diffusers import AutoencoderKLWan
        channels=MODEL['args'].get('in_channels',4)
        vae=AutoencoderKLWan(base_dim=8,z_dim=channels,dim_mult=[1,1],num_res_blocks=1,temperal_downsample=[False],latents_mean=[0.]*channels,latents_std=[1.]*channels).eval()
        with torch.no_grad():
            e=vae.decode(expected[0][-1],return_dict=False)[0];v=vae.decode(actual,return_dict=False)[0]
        compare('rmd/trained_decoder_pixels',v,e,atol=0,rtol=0,scope='native_trained_model_inference_pixels')
        directory=CHECKPOINT_ROOT/f'rmd_transition{transition}'
        if save_checkpoints:
            directory.mkdir(parents=True,exist_ok=True)
            for label,g,f,og,of in [('native',a,af,opts[0],opts[2]),('unified',b,bf,opts[1],opts[3])]:torch.save(dict(model=g.state_dict(),fake=f.state_dict(),optimizer=og.state_dict(),fake_optimizer=of.state_dict(),steps=steps),directory/f'{label}.pt')
        from compare_released_profiles import compare_state,named_optimizer_state
        state_checks={'generator_optimizer':compare_state(named_optimizer_state(opts[0],a),named_optimizer_state(opts[1],b)),
                      'fake_optimizer':compare_state(named_optimizer_state(opts[2],af),named_optimizer_state(opts[3],bf))}
        if any(v['exact']!=v['tensors'] for v in state_checks.values()):raise AssertionError('RMD optimizer states differ')
        records.append(dict(transition=transition,steps=steps,parameters=sum(p.numel() for p in model.parameters()),warmup_step=warmup_step,target_size=target_size,losses=trace,state_checks=state_checks,checkpoint_pair=str(directory.relative_to(ROOT)) if save_checkpoints else None))
    return {'variants':records,'scope':'Actual Wan Transformer, native phase1/phase2 bodies, sequential fake AdamW then generator AdamW, native generator-only clip, both flow transition modes, frozen random teacher and random Wan VAE. No optional experimental GAN branch. No real-data quality claim.'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--component-only',action='store_true');parser.add_argument('--steps',type=int,default=128);args=parser.parse_args();torch.set_num_threads(2);torch.manual_seed(20261008);started=time.time();failure=None;details={}
    try:
        with torch.no_grad():component_details=components()
        component_checks=list(conformance.CHECKS)
        (ROOT/'evidence/rmd_conformance.json').write_text(json.dumps(dict(method='rmd',status='passed',device='cpu',gpu_used=False,checks=component_checks,details=component_details,source_executions=native.AUDIT),indent=2))
        conformance.CHECKS.clear()
        if not args.component_only:details=train(args.steps)
    except Exception:failure=traceback.format_exc();print(failure,file=sys.stderr)
    report=dict(method='rmd',status='failed' if failure else 'passed',device='cpu',gpu_used=False,seconds=time.time()-started,details=details,checks=conformance.CHECKS,source_executions=native.AUDIT,failure=failure,quality_claim='Source-native synthetic conformance; no paper FID or model quality guarantee.')
    if not args.component_only:(ROOT/'evidence/rmd_training.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ('method','status','seconds')}));return failure is not None
if __name__=='__main__':raise SystemExit(main())
