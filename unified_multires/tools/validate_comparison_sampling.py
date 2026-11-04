"""Fresh CPU sampling conformance for the common-backbone experiments.

Author samplers see the same nonzero random common DiT as the unified sampler.
Adapters change the model API/domain only, never replace author solver algebra.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace, MethodType
from typing import Optional, Union, List
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'_deps'),str(ROOT/'tools')]
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from umdiff import build
from umdiff.cli import make_method
from umdiff.native_models import activate
from umdiff.randomness import GlobalRandomStream, RandomStream
from activation_trace import byte_view

OUT=ROOT/'evidence/sampling_parity_20261015/common'
PF=ROOT/'sources/pyramid_flow'
CHECKS=[]
NOISE_SEED=641
BATCH=1


def check(name,a,b):
    if torch.is_tensor(a):
        assert torch.isfinite(a).all() and torch.isfinite(b).all(),name+' nonfinite tensor'
        same=a.shape==b.shape and a.dtype==b.dtype
        exact=same and torch.equal(byte_view(a),byte_view(b))
        error=float((a.double()-b.double()).abs().max()) if a.shape==b.shape and a.numel() else 0.
        CHECKS.append(dict(name=name,shape=list(a.shape),dtype=str(a.dtype),byte_exact=exact,max_abs=error))
        if not exact:raise AssertionError(f'{name}: {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}, max_abs={error}')
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),name
        for k in a:check(name+'/'+str(k),a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),(name,len(a),len(b))
        for i,(p,q) in enumerate(zip(a,b)):check(name+'/'+str(i),p,q)
    else:assert a==b,(name,a,b)


def clone(value):
    if torch.is_tensor(value):return value.detach().clone()
    if isinstance(value,dict):return {k:clone(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return type(value)(clone(v) for v in value)
    return value


def recording(net):
    original=net.forward;calls=[]
    def forward(x,t,*args,**kwargs):
        output=original(x,t,*args,**kwargs)
        # stage is an API routing hint; the common comparison DiT ignores it.
        if getattr(net,'_sampling_trace_enabled',True):
            calls.append(clone(dict(x=x,t=t,args=args,kwargs={k:v for k,v in kwargs.items() if k!='stage'},output=output)))
        return output
    net.forward=forward
    return calls


def sample_checked(method,model,initial,**kwargs):
    """Public production API: private seeded stream must match native global RNG.

    Also verify that replay doesn't consume the caller's global RNG state.
    """
    result=method.sample(model,clone(initial),stream=GlobalRandomStream(),**kwargs)
    outer=torch.get_rng_state().clone()
    model._sampling_trace_enabled=False
    try:replay=method.sample(model,clone(initial),stream=RandomStream(NOISE_SEED),**kwargs)
    finally:model._sampling_trace_enabled=True
    check('private_stream_output_and_trajectory',result,replay)
    check('private_stream_preserves_outer_RNG',outer,torch.get_rng_state())
    return result


def extract(path,names,env,cls=None):
    tree=ast.parse(path.read_text())
    nodes=tree.body if cls is None else next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls).body
    selected=[n for n in nodes if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert len(selected)==len(names),(path,names)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),env)
    return env


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def models(config):
    c=copy.deepcopy(config)
    spec=c['model']
    if spec['name']=='ssd_grid':spec=spec['args']['model']
    architecture=spec['args'] if spec['name']=='native_rdd_unified' else spec['args']['architecture']
    architecture.update(hidden_size=32,depth=2,num_heads=4)
    torch.manual_seed(3841+NOISE_SEED-641)
    a=build('model',c['model']).eval()
    with torch.no_grad():
        for p in a.parameters():p.normal_(0,.03)
    b=copy.deepcopy(a)
    return c,a,b,recording(a),recording(b)


def decode_check(a,b):
    from diffusers import AutoencoderKL
    torch.manual_seed(510)
    vae=AutoencoderKL(in_channels=3,out_channels=3,latent_channels=4,
        down_block_types=('DownEncoderBlock2D',)*4,up_block_types=('UpDecoderBlock2D',)*4,
        block_out_channels=(8,8,8,8),layers_per_block=1,norm_num_groups=4,sample_size=256).eval()
    pa=vae.decode(a.float()/.18215).sample;pb=vae.decode(b.float()/.18215).sample
    assert torch.isfinite(pa).all() and torch.isfinite(pb).all()
    check('decoded_RGB_float',pa,pb)
    return dict(shape=list(pa.shape),sha256=hashlib.sha256(pa.numpy().tobytes()).hexdigest(),
                codec='Identical locally initialized tiny AutoencoderKL, scale0.18215; not pretrained quality')


def pixelflow(c,a,b,solver):
    activate('pixelflow',['pixelflow'])
    from pixelflow.pipeline_pixelflow import PixelFlowPipeline
    from pixelflow.scheduling_pixelflow import PixelFlowScheduler
    from pixelflow.solver_ode_wrapper import ODE
    from diffusers.models.embeddings import _get_2d_rotary_pos_embed_np
    import pixelflow.pipeline_pixelflow as module
    args=c['process']['args'];args['solver']=solver;args['guidance_scale']=args.get('guidance_scale',1.)
    method=make_method(c);p=method.process
    class Proxy(torch.nn.Module):
        patch_size=2;attention_head_dim=8
        def __init__(self,net):super().__init__();self.net=net
        def forward(self,hidden_states,timestep,class_labels,latent_size,pos_embed):
            return self.net(hidden_states,timestep/1000,y=class_labels)
    # Find the model class rather than hardcoding its spelling.
    tree=ast.parse((ROOT/'sources/pixelflow/pixelflow/model.py').read_text())
    cls=next(n.name for n in tree.body if isinstance(n,ast.ClassDef) and any(isinstance(q,ast.FunctionDef) and q.name=='c2i_forward_cfg_torchdiffq' for q in n.body))
    env=extract(ROOT/'sources/pixelflow/pixelflow/model.py',{'c2i_forward_cfg_torchdiffq'},{'torch':torch},cls)
    Proxy.c2i_forward_cfg_torchdiffq=env['c2i_forward_cfg_torchdiffq']
    scheduler=PixelFlowScheduler(1000,len(c['grids']))
    # Controlled partitions are not a published recipe. Override only the
    # scheduler tables to the requested partition; solver remains author code.
    if args.get('stage_range') is not None:
        import pixelflow.scheduling_pixelflow as scheduling
        tree=ast.parse((ROOT/'sources/pixelflow/pixelflow/scheduling_pixelflow.py').read_text())
        init=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PixelFlowScheduler').body[0]
        replaced=0
        for node in ast.walk(init):
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='stage_range' for t in node.targets):
                node.value=ast.Name(id='controlled_stage_range',ctx=ast.Load());replaced+=1
        assert replaced==1
        env=dict(vars(scheduling),controlled_stage_range=args['stage_range'])
        exec(compile(ast.fix_missing_locations(ast.Module(body=[init],type_ignores=[])),'author_controlled_partition','exec'),env)
        env['__init__'](scheduler,1000,len(c['grids']))
    check('author_network_clock_tables',list(scheduler.Timesteps_per_stage.values()),p.clock.network_times)
    pipe=PixelFlowPipeline(scheduler,Proxy(a))
    x=torch.randn(BATCH,4,*c['grids'][0]);labels=torch.tensor([7]*BATCH,dtype=torch.int32)
    raw=[];stage_values=[];old_step=scheduler.step;old_ode=ODE.sample
    def step(*args,**kw):
        value=old_step(*args,**kw);raw.append(value.clone());return value
    def ode(self,*args,**kw):
        value=old_ode(self,*args,**kw);stage_values.append(value.clone());return value
    scheduler.step=step
    torch.manual_seed(NOISE_SEED)
    with patch.object(module,'randn_tensor',lambda *args,**kw:x.clone()),patch.object(module,'get_2d_rotary_pos_embed',_get_2d_rotary_pos_embed_np),patch.object(ODE,'sample',ode):
        pixels=pipe([7]*BATCH,c['grids'][-1][0],c['grids'][-1][1],num_inference_steps=p.steps,
                    guidance_scale=p.cfg,device=torch.device('cpu'),shift=p.shift,use_ode_dopri5=solver=='native_dopri5')
    expected=raw[-1] if solver=='euler' else stage_values[-1][-1][:BATCH]
    rng=torch.get_rng_state();torch.manual_seed(NOISE_SEED)
    author_ode_values=stage_values.copy();stage_values.clear()
    with patch.object(ODE,'sample',ode):actual,trace=sample_checked(method,b,x.clone(),model_kwargs={'y':labels},return_trajectory=True)
    check('RNG',rng,torch.get_rng_state())
    check('unclipped_latent',expected,actual)
    check('author_display_pixels',torch.from_numpy(pixels),(actual/2+.5).clamp(0,1).permute(0,2,3,1))
    if solver=='native_dopri5':
        check('all_saved_ODE_stage_states',author_ode_values,stage_values[:len(author_ode_values)])
        check('private_all_saved_ODE_stage_states',author_ode_values,stage_values[len(author_ode_values):])
    else:
        cursor=1
        for i,count in enumerate(p.steps):
            if i:cursor+=1
            check(f'Euler_stage{i}',raw[sum(p.steps[:i]):sum(p.steps[:i+1])],trace[cursor:cursor+count]);cursor+=count
    return expected,actual,dict(solver=solver,steps=p.steps,cfg=p.cfg,
        author='PixelFlowPipeline.__call__, PixelFlowScheduler, unmodified c2i_forward_cfg_torchdiffq',
        adaptations=['Initial supplied tensor has4 latent channels instead of pipeline hardcoded3 RGB; both model functions use identical common DiT and RoPE',
                     'Diffusers rotary API compatibility alias; old RoPE values unused by common-model adapter'],
        solver_backend_reused=solver=='native_dopri5')


def dfm(c,a,b,solver):
    activate('dfm',['models','dfm_utils','loss'])
    from dfm_utils.samplers_dfm import dfm_euler_sampler
    method=make_method(c);x={i:torch.randn_like(v) for i,v in enumerate(method.representation.encode(torch.randn(BATCH,4,32,32)))}
    labels=torch.tensor([7]*BATCH);settings=c['clock']['args']
    expected=dfm_euler_sampler(lambda x,t,**kw:(a(x,t,**kw),None),clone(x),labels,
                              num_steps_per_scale=settings['steps'],stage_thresholds=settings['thresholds'])
    expected=method.representation.decode(expected)
    actual=sample_checked(method,b,clone(x),model_kwargs={'y':labels})
    check('unclipped_latent',expected,actual)
    return expected,actual,dict(solver='author DFM Euler',steps=settings['steps'],cfg=1,solver_backend_reused=False)


def lapflow(c,a,b,solver):
    activate('lapflow',['models','sampler'])
    from sampler import ours_sample as author
    method=make_method(c);settings=c['sampler']['args'];x=torch.randn(BATCH,4,32,32);labels=torch.tensor([7]*BATCH)
    class Proxy(torch.nn.Module):
        def __init__(self,net):super().__init__();self.net=net
        def forward(self,t,x,var_scale=None,**kwargs):return tuple(self.net(x,t,**kwargs))
    args=SimpleNamespace(noise_var_factor=None,path_type='linear',use_gvp=False,epsilon=None,
         variance_aware=False,use_mot=True,loss_weight_scale_0=1.,loss_weight_scale_1=1.,num_levels=2,
         method=settings['solver'],step_size=settings['step_size'],perturb=False,compute_nfe=False,
         cfg_scale=1.,time_seg_small=.5,time_segments=[.33,.67],atol=settings['atol'],rtol=settings['rtol'])
    old=torch.tensor
    def cpu_tensor(*args,**kwargs):
        if kwargs.get('device')=='cuda':kwargs['device']='cpu'
        return old(*args,**kwargs)
    torch.manual_seed(NOISE_SEED)
    with patch.object(author.torch,'tensor',cpu_tensor):expected=author.ours_sample_from_model([Proxy(a)],0,x.clone(),{'y':labels},args,False,largest_only=True)
    rng=torch.get_rng_state();torch.manual_seed(NOISE_SEED)
    actual=sample_checked(method,b,x.clone(),model_kwargs={'y':labels})
    check('RNG',rng,torch.get_rng_state());check('unclipped_latent',expected,actual)
    return expected,actual,dict(solver=args.method,atol=args.atol,rtol=args.rtol,solver_backend_reused=True,
                              adaptations=['Author hardcoded cuda time-tensor device mapped to cpu; model API t,x mapped to x,t'])


def pyramid_flow(c,a,b,solver):
    c['process']['args']['boundary_policy']='native_psd'
    method=make_method(c);p=method.process
    schedule=load_module('sampling_pf_scheduler',PF/'diffusion_schedulers/scheduling_flow_matching.py')
    scheduler=schedule.PyramidFlowMatchEulerDiscreteScheduler(stage_range=c['process']['args']['stage_range'])
    env=extract(PF/'pyramid_dit/pyramid_dit_for_video_gen_pipeline.py',{'generate_one_unit'},
                dict(torch=torch,F=F,math=math,rearrange=rearrange,Optional=Optional,Union=Union,List=List,
                     is_sequence_parallel_initialized=lambda:False),cls='PyramidDiTForVideoGeneration')
    compat=load_module('pf_psd_source_compat',ROOT.parent/'experiments/official_video_baselines/pyramid_compat.py')
    labels=torch.tensor([7]*BATCH);x=torch.randn(BATCH,4,*c['grids'][0]);raw=[]
    class Proxy:
        def __call__(self,sample,timestep_ratio,**kwargs):
            return [a(sample[0][-1].squeeze(2),timestep_ratio/1000,y=labels).unsqueeze(2)]
    instance=SimpleNamespace(stages=list(range(len(c['grids']))),scheduler=scheduler,
         do_classifier_free_guidance=False,dit=Proxy(),sample_block_noise=lambda *shape:compat.sample_psd_block_noise(1/3,*shape))
    old=scheduler.step
    def step(*args,**kwargs):
        value=old(*args,**kwargs);raw.append(value.prev_sample.squeeze(2).clone());return value
    scheduler.step=step
    torch.manual_seed(NOISE_SEED)
    result=env['generate_one_unit'](instance,x.unsqueeze(2),[[] for _ in c['grids']],None,None,None,
                                   p.steps,*c['grids'][0],1,torch.device('cpu'),torch.float32)
    expected=result[-1].squeeze(2);rng=torch.get_rng_state();torch.manual_seed(NOISE_SEED)
    actual,trace=sample_checked(method,b,x.clone(),model_kwargs={'y':labels},return_trajectory=True)
    check('RNG',rng,torch.get_rng_state())
    cursor=1
    for i,count in enumerate(p.steps):
        check(f'Euler_stage{i}',raw[sum(p.steps[:i]):sum(p.steps[:i+1])],trace[cursor:cursor+count]);cursor+=count+1
    check('unclipped_latent',expected,actual)
    return expected,actual,dict(solver='author Euler',steps=p.steps,boundary='native_psd',
        adaptations=['Same published singular gamma1/3 Gaussian with established PSD factor on native side; unpatched Cholesky is not defined',
                     'One-frame spatial unit, common DiT replaces original video/text model'],solver_backend_reused=False)


def ssd(c,a,b,solver):
    activate('ssd',['guided_diffusion'])
    from guided_diffusion import scale_space_diffusion as author
    author.dist=SimpleNamespace(get_rank=lambda:0,barrier=lambda:None)
    settings=c['batch']['args'];mapping={int(k):v for k,v in settings['unet_res_layer_map'].items()}
    grids=sorted(mapping)
    cls=author.ScaleSpaceDiffusion
    controlled='shared' in c['name']
    if controlled:
        class ControlledSSD(cls):
            def get_resolution(self,t_idx):return settings['resolutions'][t_idx]
        cls=ControlledSSD
    source=cls(betas=np.array(settings['betas']),model_mean_type=author.ModelMeanType.START_X_RS,
         model_var_type=author.ModelVarType.FIXED_SMALL,loss_type=author.LossType.MSE,
         ssd_config=dict(resolution_schedule='ConvexDecay_0.5',t_res_sampling_mode='one_res_many_t_except_one_t_for_res_change',
                         multires_training_mode='flexi_unet',unet_res_layer_map=mapping,scale_min=grids[0]),
         resolutions_array=grids,mse_loss_weight_type='min_snr_5')
    assert [source.get_resolution(i) for i in range(1000)]==settings['resolutions']
    for op in source.ops.values():op.H_out,op.W_out=op.size_out
    method=make_method(c);x=torch.randn(BATCH,4,grids[0],grids[0]);labels=torch.tensor([7]*BATCH)
    torch.manual_seed(NOISE_SEED)
    # Explicit adapter keeps the author's positional cond_rs calling convention.
    fn=lambda x,t,cond_rs,**kwargs:a(x,t,cond_rs=cond_rs,**kwargs)
    raw=[v['sample'].clone() for v in source.p_sample_loop_progressive(fn,(BATCH,4,32,32),noise=x.clone(),
        device=torch.device('cpu'),model_kwargs={'y':labels},clip_denoised=False,progress=False)]
    expected=raw[-1];rng=torch.get_rng_state();torch.manual_seed(NOISE_SEED)
    actual,trace=sample_checked(method,b,x.clone(),model_kwargs={'y':labels},return_trajectory=True)
    check('RNG',rng,torch.get_rng_state());check('all_posterior_steps',raw,trace);check('unclipped_latent',expected,actual)
    return expected,actual,dict(solver='author1000-step operator DDPM',steps=1000,
        adaptations=['BilinearOp missing H_out/W_out metadata set on native instances',
                     'Both sides disable clean clipping for common unbounded latent representation; native RGB clipping tested separately'],solver_backend_reused=False)


def rdd(c,a,b,solver):
    from umdiff.rdd_native import diffusion_backend
    backend=diffusion_backend()
    from diffusion.gaussian_diffusion_with_drift import _extract_into_tensor
    env=extract(ROOT.parent/'2d/sample_fid_unified.py',{'_redraw_upsample','_run_stage'},
                dict(torch=torch,_extract_into_tensor=_extract_into_tensor))
    method=make_method(c);s=c['sampler']['args'];limits=np.concatenate([[0],np.cumsum(s['stage_lengths'])]);ds=[]
    for i,r in enumerate(s['resolutions']):
        ds.append(backend.create_diffusion_with_drift(str(s['steps'][-1-i]),diffusion_steps=int(limits[-1]),
            c=i if i<len(s['resolutions'])-1 else None,T_c=int(limits[i]),T_c_plus_1=int(limits[i+1]),sample_scale=2))
    x=torch.randn(BATCH,4,s['resolutions'][-1],s['resolutions'][-1]);labels=torch.tensor([7]*BATCH);state=x.clone()
    torch.manual_seed(NOISE_SEED)
    for j,i in enumerate(range(len(ds)-1,-1,-1)):
        if j:state,_=env['_redraw_upsample'](out['pred_xstart'].float(),torch.full((BATCH,),ds[i].num_timesteps-1),ds[i],s['resolutions'][i])
        if s['solver']=='ancestral':out=ds[i].p_sample_loop(a.forward,state.shape,noise=state,clip_denoised=False,model_kwargs={'y':labels},device='cpu',progress=False)
        else:out=env['_run_stage'](ds[i],a.forward,state,{'y':labels},'cpu',s['solver'])
        state=out['sample']
    expected=state.float();rng=torch.get_rng_state();torch.manual_seed(NOISE_SEED)
    actual=sample_checked(method,b,x.clone(),model_kwargs={'y':labels})
    check('RNG',rng,torch.get_rng_state());check('unclipped_latent',expected,actual)
    return expected,actual,dict(solver=s['solver'],steps=s['steps'],solver_backend_reused=True,
        author='Original sample_fid_unified.py _run_stage/_redraw_upsample and independently built original diffusions')


def paper_equations(c,a,b,solver):
    import sampling_paper_oracles as ref
    method=make_method(c);name=c.get('process',{}).get('name','flow')
    shape=[4,*c['grids'][0]] if 'grids' in c else c['image_shape']
    x=torch.randn(BATCH,*shape);labels=torch.tensor([7]*BATCH)
    torch.manual_seed(NOISE_SEED);expected,raw=getattr(ref,name)(c,a,x.clone(),labels)
    rng=torch.get_rng_state();torch.manual_seed(NOISE_SEED)
    actual,trace=sample_checked(method,b,x.clone(),steps=250,model_kwargs={'y':labels},return_trajectory=True)
    check('RNG',rng,torch.get_rng_state());check('all_equation_steps',raw,trace);check('unclipped_latent',expected,actual)
    return expected,actual,dict(solver=name,oracle='Independent equation implementation, NOT original repository',
        author_code_parity=False,solver_backend_reused=False,
        limitations='Finite endpoints and selected clock/grid are declared numerical policies; no unpublished author sampling recipe is certified')


def main():
    global NOISE_SEED,BATCH
    parser=argparse.ArgumentParser();parser.add_argument('config');parser.add_argument('--solver',default='euler');parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--seed',type=int,default=641);parser.add_argument('--batch',type=int,default=1)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    NOISE_SEED=args.seed;BATCH=args.batch;torch.set_num_threads(2)
    path=Path(args.config);spec=json.loads(path.read_text());c,a,b,ra,rb=models(spec['method'])
    name=c.get('process',c.get('batch',{})).get('name','flow')
    if c.get('sampler',{}).get('name')=='rdd_native_stages':name='rdd'
    if name in ('fdm','pyramidal_wan_path','nonuniform_haar','flow'):name='paper_equations'
    report=dict(config=str(path),config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),device='cpu',seed=NOISE_SEED,batch=BATCH,
        scope='Same common DiT on both sides, width32/depth2, nonzero random weights; synthetic latent input. Not author architecture or quality equality.')
    started=time.monotonic();torch.set_num_threads(2)
    try:
        with torch.no_grad():
            expected,actual,details=globals()[name](c,a,b,args.solver)
            check('all_model_calls',ra,rb)
            report.update(status='passed',details=details,model_calls=len(ra),decoded=decode_check(expected,actual))
    except Exception:
        report.update(status='failed',failure=traceback.format_exc(),native_calls=len(ra),unified_calls=len(rb))
    report.update(seconds=time.monotonic()-started,checks=CHECKS,cuda_initialized=torch.cuda.is_initialized())
    out=args.output/(spec['suite']+'_'+spec['id']+'_'+args.solver+'.json')
    if out.exists():out.rename(out.with_name(out.stem+'_attempt'+str(time.time_ns())+'.json'))
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','seconds','failure') if k in report}),flush=True)
    return report['status']!='passed'


if __name__=='__main__':raise SystemExit(main())
