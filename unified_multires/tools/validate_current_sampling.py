#!/usr/bin/env python3
"""Fresh source-backed checks for sampler branches missed by old fixtures.

CPU, full published time grids/guidance, reduced nonzero author networks.
This checks equations and assembly, not pretrained image quality.
"""
import argparse,json,os,sys,time,traceback
from pathlib import Path
from copy import deepcopy
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'_deps'),str(ROOT/'tools')]
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch
from umdiff import build,native
from umdiff.randomness import GlobalRandomStream,RandomStream
from umdiff.native_models import activate
import validate_training as training
import conformance
CHECK=conformance.compare
OUT=ROOT/'evidence/current_recipe_parity_20261009/sampling'

def ppflow():
    activate('ppflow',['models','transport','modules'])
    from transport.transport import Transport,ModelType,PathType,WeightType,Sampler
    from umdiff.samplers import PPFlowNativeSampler
    source=Sampler(Transport(model_type=ModelType.VELOCITY,path_type=PathType.LINEAR,
                             loss_type=WeightType.NONE,train_eps=0.,sample_eps=0.))
    records=[]
    for levels,cfg in [(2,[1.,3.]),(3,[1.,3.5,3.75])]:
        a=training.nonzero_output(build('model',{'name':'native_ppflow','args':{
            'input_size':16,'patch_size':2,'patch_level':levels,'in_channels':4,
            'hidden_size':32,'depth':2,'num_heads':4,'class_dropout_prob':.1,'learn_sigma':True}})).eval()
        b=deepcopy(a);x=torch.randn(1,4,16,16);y=torch.tensor([7])
        for mode,method,points,extra in [('ode','heun2',250,{}),('ode','dopri5',5,{}),
                                        ('sde','Euler',9,{'diffusion_form':'sigma','last_step':'Mean','last_step_size':.04}),
                                        ('sde','Heun',9,{'diffusion_form':'sigma','last_step':'Euler','last_step_size':.04})]:
            kwargs={'y':torch.cat([y,torch.full_like(y,1000)]),'cfg_scales':cfg}
            function=source.sample_ode(sampling_method=method,num_steps=points) if mode=='ode' else source.sample_sde(sampling_method=method,num_steps=points,**extra)
            training.seed(7341)
            expected=function(torch.cat([x,x]),a.forward_with_cfg,**kwargs)
            rng=torch.get_rng_state().clone()
            training.seed(7341)
            value,trace=PPFlowNativeSampler(points=points,mode=mode,sampling_method=method,cfg_scales=cfg,**extra)(b,x,GlobalRandomStream(),{'y':y},True)
            assert len(trace)==len(expected)
            for i,(p,q) in enumerate(zip(trace,expected)):
                CHECK(f'ppflow/{levels}/{mode}/{method}/{points}/step{i}',p,q[:len(x)],atol=0,rtol=0)
            CHECK(f'ppflow/{levels}/{mode}/{method}/rng',torch.get_rng_state(),rng,atol=0,rtol=0)
            # Serializable private RNG must consume the same draws without
            # changing the surrounding global stream.
            outer=torch.get_rng_state().clone()
            private=RandomStream(7341)
            isolated=PPFlowNativeSampler(points=points,mode=mode,sampling_method=method,cfg_scales=cfg,**extra)(b,x,private,{'y':y})
            CHECK(f'ppflow/{levels}/{mode}/{method}/private_output',isolated,value,atol=0,rtol=0)
            CHECK(f'ppflow/{levels}/{mode}/{method}/outer_rng',torch.get_rng_state(),outer,atol=0,rtol=0)
            CHECK(f'ppflow/{levels}/{mode}/{method}/private_rng',private.generator.get_state(),rng,atol=0,rtol=0)
            records.append(dict(levels=levels,mode=mode,method=method,points=points,cfg=cfg))
    return {'variants':records,'backend':'Unchanged pinned author transport solver reused explicitly; unified configuration and RNG isolation checked independently.'}

def mdm():
    from umdiff.samplers import MDMSampler
    a=training.nonzero_output(build('model',{'name':'native_mdm'})).eval();b=deepcopy(a)
    from ml_mdm import diffusion as author,samplers
    text=torch.randn(1,5,8);negative=torch.randn_like(text);mask=torch.ones(1,5)
    x=torch.randn(1,4,16,16);records=[]
    for index,(power,shift,eta,steps) in enumerate([(1.,True,0.,250),(2.,True,1.,50),(1.,False,None,50),(2.,True,0.,250)]):
        if index==3:
            inner={'resolution_channels':'32','attention_levels':'','num_resnets_per_resolution':'1','conditioning_feature_dim':8,'nesting':True}
            middle={'resolution_channels':'32,32,32','attention_levels':'','num_resnets_per_resolution':'1','conditioning_feature_dim':8,'nesting':True,'inner_config':inner}
            a=training.nonzero_output(build('model',{'name':'native_mdm','args':{'config':{'resolution_channels':'32,32,32','attention_levels':'','num_resnets_per_resolution':'1','conditioning_feature_dim':8,'inner_config':middle}}})).eval();b=deepcopy(a)
            x=torch.randn(1,4,128,128)
        scales=a.nest_ratio+[1]
        cfg=author.NestedDiffusionConfig(sampler_config=samplers.SamplerConfig(
            num_diffusion_steps=1000,prediction_type=samplers.PredictionType.V_PREDICTION,
            loss_target_type=samplers.PredictionType.DDPM,schedule_type=samplers.ScheduleType.DEEPFLOYD,
            schedule_shifted=shift,schedule_shifted_power=power,threshold_function=samplers.ThresholdType.CLIP),
            use_double_loss=True,no_use_residual=True,use_vdm_loss_weights=False)
        pipe=author.NestedDiffusion(a,cfg)
        times=pipe.sampler.set_timesteps(steps)
        state=x.clone();expected=[]
        training.seed(5712)
        for t,s in zip(times[:-1],times[1:]):
            state=pipe.sampler.get_xt_minus_1(pipe.model,int(t),state,torch.cat([negative,text]),
                     torch.cat([mask,mask]),time_step_last=int(s),guidance_scale=7.5,ddim_eta=eta)
            expected.append(state)
        rng=torch.get_rng_state().clone();training.seed(5712)
        value,trace=MDMSampler(pipe.sampler.gammas.tolist(),scales=scales,power=power,
                     schedule_shifted=shift,eta=eta,guidance_scale=7.5,threshold='clip',
                     resample_steps=True,inference_steps=steps,clip_output=True)(b,x,GlobalRandomStream(),
                     {'conditioning':text,'cond_mask':mask,'micros':{},
                      'unconditional':{'conditioning':negative,'cond_mask':mask,'micros':{}}},True)
        assert len(trace)==len(times)
        for i,(p,q) in enumerate(zip(trace[1:],expected)):
            CHECK(f'mdm/levels{len(scales)}/power{power}/shift{shift}/eta{eta}/resample{steps}/step{i}',p,q,atol=0,rtol=0)
        CHECK(f'mdm/levels{len(scales)}/shift{shift}/resample{steps}/output',value,(expected[-1][0]*(1 if shift else scales[0])).clamp(-1,1),atol=0,rtol=0)
        CHECK(f'mdm/shift{shift}/resample{steps}/rng',torch.get_rng_state(),rng,atol=0,rtol=0)
        # Also execute the complete unchanged native dispatcher/postprocessor.
        training.seed(5712)
        final=pipe.sampler.sample(pipe.model,x,torch.cat([negative,text]),torch.cat([mask,mask]),{},
                     num_inference_steps=steps,resample_steps=True,guidance_scale=7.5,ddim_eta=eta)
        CHECK(f'mdm/shift{shift}/resample{steps}/native_dispatcher',value,final,atol=0,rtol=0)
        records.append(dict(scales=scales,actual_grid=list(x.shape),power=power,schedule_shifted=shift,eta=eta,inference_steps=steps,first_index=int(times[0]),last_index=int(times[-1])))
    return {'variants':records,'backend':'Independent posterior/resampling implementation vs unchanged author get_xt_minus_1 and complete sample dispatcher.'}

def main():
    global OUT
    p=argparse.ArgumentParser();p.add_argument('method',choices=['ppflow','mdm']);p.add_argument('--output',type=Path);args=p.parse_args()
    if args.output:OUT=args.output
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);training.seed(20261009)
    started=time.time();details=None;failure=None
    try:details=globals()[args.method]()
    except Exception:failure=traceback.format_exc();print(failure,flush=True)
    report=dict(method=args.method,status='failed' if failure else 'passed',device='cpu',gpu_used=False,
                seconds=time.time()-started,details=details,failure=failure,checks=conformance.CHECKS,
                source_executions=native.AUDIT,scope='Full configured sampling time grids/CFG; reduced nonzero author network, synthetic input, no quality claim.')
    (OUT/(args.method+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('method','status','seconds','failure')}),flush=True)
    return bool(failure)
if __name__=='__main__':raise SystemExit(main())
