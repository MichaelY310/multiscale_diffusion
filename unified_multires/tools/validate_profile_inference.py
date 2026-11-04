#!/usr/bin/env python3
"""Current released-profile solver/CFG/grid assembly vs author inference code.

The backbone width/depth is explicitly reduced, with nonzero heads. Keep
published schedules, guidance and input grids except Relay's documented fixture.
No local/pretrained codec/text assets are substituted with a quality claim.
"""
import argparse,json,os,sys,time,traceback,math
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'_deps'),str(ROOT/'tools')]
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np
import torch
import torch.nn.functional as F
from umdiff import build,native
from umdiff.engine import Method
from umdiff.randomness import GlobalRandomStream
from umdiff.native_models import activate
from compare_released_profiles import PROFILES,unpack_model
import validate_training as training
import conformance
CHECK=conformance.compare
OUT=ROOT/'evidence/current_recipe_parity_20261009/sampling_profiles'
CONFIG_OVERRIDE=None

def pair(actual,expected,name):
    CHECK(name,actual,expected,atol=0,rtol=0,scope='current_profile_sampling_tensors')
    file=OUT/(name.replace('/','_')+'.pt')
    torch.save({'native':expected,'unified':actual},file)
    return str(file.relative_to(ROOT))

@torch.no_grad()
def run(name):
    c=json.loads((CONFIG_OVERRIDE or ROOT/'configs/paper_recipes'/(PROFILES[name]+'.json')).read_text())
    original=deepcopy(c['model']);spec=unpack_model(c['model']);a=spec['args']
    shape=c['image_shape'];labels=torch.tensor([7]);stream=GlobalRandomStream();notes=[]
    if name.startswith('dctdiff'):
        a.update(embed_dim=32,depth=2,num_heads=4)
    elif name=='dimr':
        for stage in a['config']['stage_configs']:
            stage.update(dim=32,hidden_dim=64,num_attention_heads=4,num_blocks=1)
    elif name=='pixelflow':a.update(depth=2,num_attention_heads=2)
    elif name.startswith('dfm'):a.update(hidden_size=32,decoder_hidden_size=32,depth=2,encoder_depth=1,num_heads=4)
    elif name=='lapflow_paper':a.update(hidden_size=32,depth=2,num_heads=4)
    elif name=='ssd':a.update(model_channels=32,num_res_blocks=1)
    elif name.startswith('relay'):
        shape=[3,8 if name=='relay_low' else 16,8 if name=='relay_low' else 16]
        a.update(img_resolution=shape[-1],model_channels=32,channel_mult=[1,2],num_blocks=1,attn_resolutions=[4 if name=='relay_low' else 8])
        notes.append('Relay fixture grid8/16 replaces64/256; get_alpha_t repeat_dims=H/patch on native instance, all sampling constants retained.')
    elif name=='rdd':a.update(hidden_size=32,depth=2,num_heads=4)
    else:raise ValueError(name)
    method=Method(c)
    a=training.nonzero_output(build('model',c['model'])).eval();b=deepcopy(a)
    notes.append('Reduced shared author backbone with nonzero output heads; synthetic untrained parameters. Full released architectures are tested separately for training.')
    artifact=None
    if name.startswith('dctdiff'):
        x=method.representation.encode(torch.randn(1,*shape).clamp(-1,1));kwargs={'y':labels} if unpack_model(c['model'])['args']['num_classes']>0 else {}
        s=native.load('dctdiff/sde.py');settings=c['sampler']['args']
        if c['sampler']['name']=='dctdiff_dpm':
            dpm=native.load('dctdiff/dpm_solver_pytorch.py');schedule=dpm.NoiseScheduleVP(schedule='linear',SNR_scale=settings['snr_scale'])
            score=s.ScoreModel(a,'noise_pred',s.VPSDE(SNR_scale=settings['snr_scale']))
            fn=dpm.model_wrapper(score.noise_pred,schedule,time_input_type='0',model_kwargs=kwargs)
            expected=dpm.DPM_Solver(fn,schedule).sample(x,steps=settings['steps'],eps=1e-4,adaptive_step_size=False,fast_version=True)
        else:
            score=s.ScoreModel(a,'noise_pred',s.VPSDE(SNR_scale=settings['snr_scale']))
            process=(s.ReverseSDE if settings.get('stochastic',False) else s.ODE)(score)
            training.seed(6241);expected=s.euler_maruyama(process,x,settings['steps'],eps=1e-3,**kwargs)
        rng=torch.get_rng_state().clone();training.seed(6241)
        actual=method.sampler(b,x,stream,kwargs)
        artifact=pair(actual,expected,name+'/tokens')
        pair(method.representation.decode(actual),method.representation.decode(expected),name+'/decoded_uint8')
        if c['sampler']['name']!='dctdiff_dpm':CHECK(name+'/rng',torch.get_rng_state(),rng,atol=0,rtol=0)
    elif name=='dimr':
        x=torch.randn(1,*shape);settings=c['sampler']['args'];source=native.load('dimr/dpm_solver.py')
        env=native.statements('dimr/eval.py',93,115,dict(nnet=lambda x,log_snr,**kw:a(x,log_snr,**kw),dataset=SimpleNamespace(K=1000),device='cpu',cfg_scale=settings['guidance_scale'],cfg_scale_pow=settings['guidance_power'],torch=torch,math=math))
        env.update(N=1000,kwargs={'y':labels});env=native.statements('dimr/eval.py',134,142,env)
        schedule=source.NoiseScheduleVP(schedule='discrete',betas=torch.tensor(settings['betas']).float())
        expected=source.DPM_Solver(env['model_fn'],schedule,predict_x0=True,thresholding=False).sample(x,steps=settings['steps'],eps=.001,T=1.)
        actual=method.sample(b,x,stream=stream,model_kwargs={'y':labels});artifact=pair(actual,expected,name+'/latent')
    elif name=='pixelflow':
        activate('pixelflow',['pixelflow'])
        from pixelflow.pipeline_pixelflow import PixelFlowPipeline
        import pixelflow.pipeline_pixelflow as module
        from pixelflow.scheduling_pixelflow import PixelFlowScheduler
        from diffusers.models.embeddings import _get_2d_rotary_pos_embed_np
        module.get_2d_rotary_pos_embed=_get_2d_rotary_pos_embed_np
        pipe=PixelFlowPipeline(PixelFlowScheduler(1000,4),a.model)
        raw=[];old_step=pipe.scheduler.step
        def observed_step(*args,**kwargs):
            value=old_step(*args,**kwargs);raw.append(value.detach().clone());return value
        pipe.scheduler.step=observed_step
        training.seed(6241);expected=pipe([7],256,256,num_inference_steps=[30]*4,guidance_scale=2.4,device=torch.device('cpu'),use_ode_dopri5=c['sampler']['args'].get('solver')=='native_dopri5')
        rng=torch.get_rng_state().clone();training.seed(6241);x=torch.randn(1,3,32,32)
        actual,trace=method.sample(b,x,stream=stream,model_kwargs={'class_labels':labels},return_trajectory=True)
        if raw:
            cursor=1
            for stage in range(4):
                if stage:cursor+=1
                for j in range(30):CHECK(f'pixelflow/raw_stage{stage}/step{j}',trace[cursor+j],raw[stage*30+j],atol=0,rtol=0)
                cursor+=30
            pair(actual,raw[-1],name+'/unclipped_pixels')
        pixels=(actual/2+.5).clamp(0,1).cpu().permute(0,2,3,1)
        artifact=pair(pixels,expected,name+'/pixels');CHECK(name+'/rng',torch.get_rng_state(),rng,atol=0,rtol=0)
    elif name.startswith('dfm'):
        activate('dfm',['models','dfm_utils','loss'])
        from dfm_utils.samplers_dfm import dfm_euler_sampler
        x={i:torch.randn_like(t) for i,t in enumerate(method.representation.encode(torch.randn(1,*shape)))}
        steps=c['clock']['args']['steps'];thresholds=c['clock']['args']['thresholds'];cfg=c['guidance']['args']
        expected=dfm_euler_sampler(lambda x,t,**kw:(a(x,t,**kw),None),x,labels,num_steps_per_scale=steps,stage_thresholds=thresholds,cfg_scale=cfg['scale'],guidance_low=cfg['low'],guidance_high=cfg['high'])
        actual=method.sample(b,x,stream=stream,model_kwargs={'y':labels})
        artifact=pair(actual,method.representation.decode(expected),name+'/latent')
    elif name=='lapflow_paper':
        activate('lapflow',['models','sampler']);from sampler import ours_sample as author
        x=torch.randn(1,*shape);settings=c['sampler']['args'];cfg=settings['guidance_scale']
        args=SimpleNamespace(noise_var_factor=None,path_type='linear',use_gvp=False,epsilon=None,variance_aware=False,use_mot=True,loss_weight_scale_0=1.,loss_weight_scale_1=1.,num_levels=2,method='euler',step_size=settings['step_size'],perturb=False,compute_nfe=False,cfg_scale=cfg,time_seg_small=.5,time_segments=[.33,.67],atol=1e-5,rtol=1e-5)
        old=torch.tensor
        def tensor(*args,**kw):
            if kw.get('device')=='cuda':kw['device']='cpu'
            return old(*args,**kw)
        training.seed(6241)
        with patch.object(author.torch,'tensor',tensor):expected=author.ours_sample_from_model([a.model],0,torch.cat([x,x]),{'y':torch.cat([labels,torch.full_like(labels,1000)]),'cfg_scale':cfg},args,False,largest_only=True)[:1]
        rng=torch.get_rng_state().clone();training.seed(6241)
        actual=method.sample(b,x,stream=stream,model_kwargs={'y':labels});artifact=pair(actual,expected,name+'/latent')
        CHECK(name+'/rng',torch.get_rng_state(),rng,atol=0,rtol=0)
    elif name=='ssd':
        activate('ssd',['guided_diffusion']);from guided_diffusion import scale_space_diffusion as author
        author.dist=SimpleNamespace(get_rank=lambda:0,barrier=lambda:None)
        settings=c['batch']['args'];mapping={int(k):v for k,v in settings['unet_res_layer_map'].items()}
        source=author.ScaleSpaceDiffusion(betas=np.array(settings['betas']),model_mean_type=author.ModelMeanType.START_X_RS,model_var_type=author.ModelVarType.FIXED_SMALL,loss_type=author.LossType.MSE,
             ssd_config=dict(resolution_schedule='ConvexDecay_0.5',t_res_sampling_mode='one_res_many_t_except_one_t_for_res_change',multires_training_mode='flexi_unet',unet_res_layer_map=mapping,scale_min=8),resolutions_array=[8,16,32,64],mse_loss_weight_type='min_snr_5')
        for op in source.ops.values():op.H_out,op.W_out=op.size_out
        notes.append('Missing BilinearOp H_out/W_out metadata assigned on native instances only; source files unchanged.')
        x=torch.randn(1,3,8,8);training.seed(6241)
        expected=source.p_sample_loop(a,(1,3,64,64),noise=x,device=torch.device('cpu'),progress=False)
        rng=torch.get_rng_state().clone();training.seed(6241)
        actual=method.sample(b,x,stream=stream);artifact=pair(actual,expected,name+'/pixels');CHECK(name+'/rng',torch.get_rng_state(),rng,atol=0,rtol=0)
    elif name.startswith('relay'):
        blur=native.load('relay/blurring.py');env=native.extract('relay/generate_imagenet.py',['edm_sampler','blur_sampler'],dict(torch=torch,np=np,block_noise=blur.block_noise,dct_2d=blur.dct_2d,idct_2d=blur.idct_2d,get_alpha_t=lambda t,patch,device,**kw:blur.get_alpha_t(t,patch,device,repeat_dims=shape[-1]//patch,**kw)))
        x=torch.randn(1,*shape);onehot=F.one_hot(labels,1000).float();settings=c['sampler']['args'];training.seed(6241)
        if name=='relay_low':expected=env['edm_sampler'](a,x,class_labels=onehot,**settings)
        else:
            expected=env['blur_sampler'](a,x,class_labels=onehot,num_steps=settings['steps'],rho=settings['rho'],cfg_scale=settings['cfg'],s_block=settings['block_scale'])
        rng=torch.get_rng_state().clone();training.seed(6241)
        actual=method.sample(b,x,stream=stream,model_kwargs={'class_labels':onehot});artifact=pair(actual,expected,name+'/pixels');CHECK(name+'/rng',torch.get_rng_state(),rng,atol=0,rtol=0)
    elif name=='rdd':
        import ast
        from diffusion.gaussian_diffusion_with_drift import _extract_into_tensor
        file=ROOT.parent/'2d/sample_fid_unified.py';tree=ast.parse(file.read_text())
        env=dict(torch=torch,_extract_into_tensor=_extract_into_tensor)
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_redraw_upsample','_run_stage')],type_ignores=[]),str(file),'exec'),env)
        x=torch.randn(1,4,8,8);kwargs={'y':torch.cat([labels,torch.full_like(labels,1000)]),'cfg_scale':1.5};training.seed(6241)
        state=torch.cat([x,torch.randn_like(x)])
        for position,index in enumerate((2,1,0)):
            d=method.sampler.diffusions[index]
            if position:
                state,_=env['_redraw_upsample'](out['pred_xstart'][:1].float(),torch.full((1,),d.num_timesteps-1,dtype=torch.long),d,method.sampler.resolutions[index]);state=torch.cat([state,state])
            out=env['_run_stage'](d,a.forward_with_cfg,state,kwargs,'cpu','dpm2');state=out['sample']
        rng=torch.get_rng_state().clone();training.seed(6241)
        actual=method.sample(b,x,stream=stream,model_kwargs={'y':labels});artifact=pair(actual,state[:1].float(),name+'/latent');CHECK(name+'/rng',torch.get_rng_state(),rng,atol=0,rtol=0)
    return dict(template=PROFILES[name],original_model=original,effective_model=c['model'],sampler=c.get('sampler'),clock=c.get('clock'),guidance=c.get('guidance'),parameters=sum(p.numel() for p in a.parameters()),artifact=artifact,deviations=notes)

def main():
    global OUT,CONFIG_OVERRIDE
    p=argparse.ArgumentParser();p.add_argument('profile',choices=['dctdiff','dctdiff_imagenet','dimr','pixelflow','dfm','dfm_yaml','lapflow_paper','ssd','relay_low','relay_high','rdd']);p.add_argument('--config',type=Path);p.add_argument('--output',type=Path);args=p.parse_args()
    if args.config:CONFIG_OVERRIDE=args.config
    if args.output:OUT=args.output.resolve()
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);training.seed(20261009)
    started=time.time();details=None;failure=None
    try:details=run(args.profile)
    except Exception:failure=traceback.format_exc();print(failure,flush=True)
    report=dict(profile=args.profile,status='failed' if failure else 'passed',seconds=time.time()-started,
        device='cpu',gpu_used=False,details=details,failure=failure,checks=conformance.CHECKS,source_executions=native.AUDIT,
        scope='Released-profile solver schedule/CFG and synthetic output parity; reduced nonzero author backbone. No pretrained decoded quality guarantee.')
    (OUT/(args.profile+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('profile','status','seconds','failure')}),flush=True)
    return bool(failure)
if __name__=='__main__':raise SystemExit(main())
