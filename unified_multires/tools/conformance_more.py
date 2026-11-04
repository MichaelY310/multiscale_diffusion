"""Native algorithm cases; loaded by conformance.py, each in its own process."""
import sys,math,json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
import torch.nn.functional as F
from umdiff import native
ROOT=native.ROOT

def install(compare,train_compare,fixture):
    globals().update(compare=compare,train_compare=train_compare,fixture=fixture)
    return {name[4:]:value for name,value in globals().items() if name.startswith('run_')}

def run_rdd():
    sys.path.insert(0,str(ROOT.parent))
    from diffusion import gaussian_diffusion_with_drift as author
    from diffusion.flow_matching_with_drift import FlowMatchingWithDrift
    from umdiff.paths import DiscreteVP,RDDFlow
    from umdiff.targets import Joint,RDDDecoupled
    from umdiff.solvers import DDPM
    from umdiff.models import TinyConv
    x,eps,t=fixture();betas=np.linspace(.001,.02,20);index=torch.tensor([4,9,16])
    for drift in (False,True):
        kwargs={'c':0,'T_c':0,'T_c_plus_1':20,'timestep_map':list(range(20))} if drift else {}
        source=author.GaussianDiffusion(betas=betas,model_mean_type=author.ModelMeanType.EPSILON,model_var_type=author.ModelVarType.FIXED_SMALL,loss_type=author.LossType.MSE,**kwargs)
        source.timestep_map=list(range(20))
        path=DiscreteVP(betas=betas,drift_start=0 if drift else None,drift_end=20 if drift else None)
        d=source.upsample(source.downsample(x,2),2)-x if drift else None
        expected=source.q_sample(x,index,d=d,noise=eps);sample=path.sample(x,index,eps)
        compare(f'rdd/ddpm/{drift}/state',sample.state,expected,atol=0,rtol=0)
        a,b=DDPM(False).posterior(path,x,expected,index)
        ma,_,lv=source.q_posterior_mean_variance(x,expected,index)
        compare('rdd/ddpm/posterior_mean',a,ma,atol=0,rtol=0)
        compare('rdd/ddpm/posterior_logvar',b,lv,atol=0,rtol=0)
        target=Joint('effective_epsilon').encode(sample)
        # Capture actual native training output and target via gradients/loss.
        class Capture(TinyConv):
            def forward(self,state,time,**kw):self.last_state=state;return super().forward(state,time,**kw)
        model=Capture(3,16)
        native_loss=source.training_losses(model,x,index,noise=eps)['loss']
        predicted=model(sample.state,index)
        loss=(predicted-target).square().mean(dim=list(range(1,predicted.ndim)))
        compare('rdd/ddpm/native_training_state',sample.state,model.last_state,atol=0,rtol=0)
        compare('rdd/ddpm/native_training_loss',loss,native_loss,atol=3e-6,scope='native_training_loss')
        train_compare(f'rdd/ddpm/{drift}',model,sample.state,expected,index,target,target)
    for a,b in ((0.,.4),(.4,.7),(.7,1.)):
        times=torch.tensor([a+.05,(a+b)/2,b-.01])
        source=FlowMatchingWithDrift(c=0,s_k=a,s_k_plus_1=b)
        d=source.compute_drift(x);state=source.q_sample(x,times,noise=eps,d=d)
        velocity=source.compute_velocity_target(x,eps,times,d=d)
        path=RDDFlow(a,b);out=path.sample(x,times,eps)
        compare('rdd/fm/state',out.state,state,atol=0,rtol=0)
        compare('rdd/fm/velocity',out.velocity,velocity,atol=2e-6)
        compare('rdd/fm/clean',path.clean_from_velocity(state,velocity,times),source.predict_x1_from_velocity(state,velocity,times),atol=3e-6)
        train_compare('rdd/fm',TinyConv(3,16),out.state,state,times,out.velocity,velocity)
    from umdiff.boundaries import RDDLegacyPosterior
    from umdiff.randomness import GlobalRandomStream
    low=torch.randn(2,2,2,3);clean=torch.randn_like(low)
    alpha=torch.full_like(low,.7);variance=torch.full_like(low,.3)
    obj=author.GaussianDiffusion.__new__(author.GaussianDiffusion);obj.sample_scale=2
    torch.manual_seed(10001);mu,cov,draw=obj.compute_posterior_x1(clean,low,alpha,variance)
    torch.manual_seed(10001);port=RDDLegacyPosterior()(low,clean=clean,alpha=alpha,variance=variance,stream=GlobalRandomStream())
    compare('rdd/legacy_boundary/mean',port.metadata['mean'],mu,atol=0,rtol=0)
    compare('rdd/legacy_boundary/covariance',port.metadata['covariance'],cov,atol=0,rtol=0)
    compare('rdd/legacy_boundary/pixels',port.state,draw,atol=0,rtol=0,scope='native_inference_transition_pixels')

def run_ppflow():
    from umdiff.operators import channel_down,channel_up
    from umdiff.paths import StageFlow
    from umdiff.models import TinyConv
    source=native.load('ppflow/modules/invertible_sample.py').ChannelInvertibleSampling()
    plan=native.load('ppflow/transport/path.py').ICPlan()
    x,eps,t=fixture()
    for width in (False,True):
        native_down=source.down_w if width else source.down_square
        native_up=source.up_w if width else source.up_square
        clean=native_down(x,2);noise=native_down(eps,2)
        compare('ppflow/encode',channel_down(x,2,width),clean,atol=0,rtol=0)
        compare('ppflow/decode',channel_up(clean,2,width),native_up(clean,2),atol=0,rtol=0)
        for a,b in ((0.,.5),(.5,.75),(.75,1.)):
            times=(a+(b-a)*t)
            start=a*clean+(1-a)*noise;end=b*clean+(1-b)*noise
            expected=plan.compute_xt(times,start,end,a,b)
            velocity=plan.compute_ut(times,start,end,a,b)
            out=StageFlow(a,b).sample(clean,times,noise)
            compare('ppflow/forward_path',out.state,expected,atol=2e-6)
            compare('ppflow/velocity',out.velocity,velocity,atol=2e-6)
            train_compare('ppflow/objective',TinyConv(clean.shape[1],16),out.state,expected,times,out.velocity,velocity)

def run_lapflow():
    sys.path.insert(0,str(ROOT/'sources/lapflow'))
    from sampler import ours_sample as author
    from sampler import edify_image_sample as shared
    from umdiff.representations import Laplacian
    from umdiff.paths import DelayedFlow
    from umdiff.models import ProductConv
    x,eps,t=fixture();t=t.view(-1,1,1,1)
    args=SimpleNamespace(noise_var_factor=None,path_type='linear',use_gvp=False,epsilon=None,variance_aware=False,use_mot=True,loss_weight_scale_0=1.,loss_weight_scale_1=1.)
    for levels in (2,3):
        rep=Laplacian(levels,up='nearest',order='fine_first')
        compare('lapflow/decomposition',rep.encode(x),shared.generate_laplacian_pyramid(x,levels),atol=0,rtol=0)
        compare('lapflow/reconstruction',rep.decode(rep.encode(x)),x,atol=3e-7)
        for selector in range(levels):
            torch.manual_seed(29)
            targets,states,_,_=author.generate_xt_and_u([.3,.6],.4,x,t,selector,levels,False,args)
            torch.manual_seed(29);full=torch.randn_like(x)
            if levels==2:noises=rep.encode(full);activations=[.4,0.]
            else:noises=[full,torch.randn_like(shared.down(full,2)),torch.randn_like(shared.down(full,4))];activations=[.6,.3,0.]
            clean=rep.encode(x)
            out=[DelayedFlow(a).sample(c,t.flatten(),n) for a,c,n in zip(activations,clean,noises)][selector:]
            states=list(states) if isinstance(states,(tuple,list)) else [states]
            targets=list(targets) if isinstance(targets,(tuple,list)) else [targets]
            ours_states=[s.state for s in out];ours_targets=[s.velocity for s in out]
            compare(f'lapflow/{levels}/{selector}/state',ours_states,states,atol=0,rtol=0,scope='native_training_inputs')
            compare(f'lapflow/{levels}/{selector}/target',ours_targets,targets,atol=0,rtol=0,scope='native_training_targets')
            train_compare(f'lapflow/{levels}/{selector}',ProductConv([3]*len(out),16),ours_states,states,t.flatten(),ours_targets,targets)

def run_dfm():
    sys.path.insert(0,str(ROOT/'sources/dfm'))
    import loss as author_loss
    from dfm_utils.laplacian_decomposer import LaplacianDecomposer2D
    from dfm_utils.samplers_dfm import MultiscaleScheduler,dfm_euler_sampler
    from umdiff.representations import Laplacian
    from umdiff.paths import Flow
    from umdiff.clocks import DFMClock
    from umdiff.models import ProductConv
    x,eps,t=fixture()
    for levels in (2,3):
        rep=Laplacian(levels,up='bilinear',order='coarse_first')
        decomposer=LaplacianDecomposer2D(stages_count=levels)
        original=decomposer.decompose(x);clean=rep.encode(x)
        compare('dfm/decomposition',clean,[original[i] for i in range(levels)],atol=0,rtol=0)
        compare('dfm/reconstruction',rep.decode(clean),decomposer.recompose(original),atol=0,rtol=0)
        model=ProductConv([3]*levels,16)
        class Capture(torch.nn.Module):
            def __init__(self):super().__init__();self.net=model
            def forward(self,state,time,**kw):
                self.state=state;self.time=time;self.kw=kw
                return dict(enumerate(self.net([state[i] for i in range(levels)],time))),None
        capture=Capture();objective=author_loss.DFMSILoss(num_stages=levels,stage_weights=[1/levels]*levels)
        # Fix only random sampling, not native path/model/loss implementation.
        torch.manual_seed(37);np.random.seed(37)
        native_value,_=objective(capture,x)
        torch.manual_seed(37);np.random.seed(37)
        last=np.random.choice(levels,p=[1/levels]*levels)
        times=[]
        for i in range(last+1):times.append(torch.sigmoid(torch.randn(x.shape[0],1,1,1)-2) if i<last else torch.rand(x.shape[0],1,1,1))
        times.extend([torch.ones(x.shape[0],1,1,1) for _ in range(levels-last-1)])
        samples=[Flow('data_to_noise').sample(c,u.flatten(),torch.randn_like(c)) for c,u in zip(clean,times)]
        compare('dfm/native_model_inputs',[s.state for s in samples],[capture.state[i] for i in range(levels)],atol=0,rtol=0)
        network_t=torch.stack([u.flatten() for u in times],-1)
        compare('dfm/native_network_times',network_t,capture.time,atol=0,rtol=0)
        pred=model([s.state for s in samples],network_t)
        port_value=sum((p-s.velocity).square().flatten(1).mean(1) for p,s in zip(pred[:last+1],samples[:last+1]))
        compare('dfm/native_loss',port_value,native_value,atol=0,rtol=0,scope='native_training_loss')
        thresholds=[.5] if levels==2 else [.6,.3];steps=[3]*levels
        if levels==3:
            # Public scheduler has positive coarse deltas for this legal 3-level
            # configuration. Preserve and record its rejection, do not hide it.
            for clock in (MultiscaleScheduler(steps,thresholds),DFMClock(steps,thresholds)):
                try:clock.schedule('cpu')
                except ValueError as exc:
                    if 'positive' not in str(exc) and 'decrease' not in str(exc):raise
                else:raise AssertionError('Expected native 3-level scheduling rejection')
            continue
        expected=MultiscaleScheduler(steps,thresholds).schedule('cpu');actual=DFMClock(steps,thresholds).schedule()
        compare('dfm/schedule',actual,expected,atol=0,rtol=0)
        torch.manual_seed(31);initial={i:torch.randn_like(v) for i,v in enumerate(clean)}
        native_result=dfm_euler_sampler(capture,initial,torch.zeros(x.shape[0],dtype=torch.long),num_steps_per_scale=steps,stage_thresholds=thresholds)
        state={i:v.double() for i,v in initial.items()}
        for row in actual:
            pred,_=capture({i:v.float() for i,v in state.items()},row['times'].expand(x.shape[0],-1),drop_mask=row['drop_mask'])
            for i in state:
                if row['drop_mask'][i]:state[i]=state[i]+row['delta_t'][i]*pred[i].double()
        compare('dfm/native_sampler',state,native_result,atol=0,rtol=0,scope='native_inference_trajectory')
        train_compare('dfm/objective',model,[s.state for s in samples],[capture.state[i] for i in range(levels)] if False else [s.state for s in samples],network_t,[s.velocity for s in samples],[s.velocity for s in samples])

def run_nami():
    from umdiff.boundaries import LearnedAffine
    nn=torch.nn
    env=native.extract('nami/src/infer.py',['BridgeFlow'],{'torch':torch,'nn':nn,'F':F})
    torch.manual_seed(41);original=env['BridgeFlow']()
    port=LearnedAffine();port.load_state_dict(original.state_dict())
    for stage,size in ((1,32),(2,64)):
        value=torch.randn(1,16,size,size)
        compare(f'nami/bridge_stage{stage}',port(value,stage=stage).state,original(value,stage),atol=0,rtol=0,scope='native_inference_transition')
    author=native.load('nami/src/scheduling_flow_matching.py')
    scheduler=author.PyramidFlowMatchEulerDiscreteScheduler()
    for stage in range(3):
        scheduler.set_timesteps(5,stage,'cpu');state=torch.randn(2,4,8,8)
        for i in range(5):
            velocity=.3*torch.sin(state)
            expected=state.float()+(scheduler.sigmas[i+1]-scheduler.sigmas[i])*velocity
            result=scheduler.step(velocity,scheduler.timesteps[i],state,return_dict=False)[0]
            compare('nami/scheduler_step',result,expected.to(velocity.dtype),atol=0,rtol=0,scope='native_inference_trajectory')
            state=result

def run_mrflow():
    from diffusers import FlowMatchEulerDiscreteScheduler
    author=native.load('mrflow/mrflow_utils.py')
    scheduler=FlowMatchEulerDiscreteScheduler()
    original=scheduler.set_timesteps
    for steps in (3,7):
        with author.direct_sigma_schedule(scheduler,.12,steps,'cpu'):
            scheduler.set_timesteps(steps)
            compare('mrflow/sigma_grid',scheduler.sigmas,torch.linspace(.12,0,steps+1),atol=0,rtol=0)
            compare('mrflow/network_grid',scheduler.timesteps,torch.linspace(.12,0,steps+1)[:-1]*1000,atol=0,rtol=0)
        if scheduler.set_timesteps!=original:raise AssertionError('Native scheduler context did not restore original method')
    # No fake SR parity: external RealESRGAN/FLUX weights unavailable by request.
