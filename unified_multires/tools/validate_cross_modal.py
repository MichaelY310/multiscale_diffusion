"""CPU source/math/backbone audit for the new common-model implementation.

Run after activate_cpu.sh. Writes JSON evidence, not reusable weight assets.
No training data, pretrained weights, network access or GPU are required.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from activation_trace import ActivationAudit
from umdiff.cross_modal import recipe,STAGE_METHODS
from umdiff.cli import setup,make_method
from umdiff.registry import build
from umdiff.randomness import RandomStream
from umdiff.multires_process import PyramidFlowProcess,PyramidalWanProcess,FDMProcess,EdifyProcess,PDDPMProcess,PixelFlowProcess
from umdiff.grids import GridMap

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT.parent
PF=ROOT/'sources/pyramid_flow'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def comparison(a,b,atol=1e-6,rtol=1e-5):
    torch.testing.assert_close(a,b,atol=atol,rtol=rtol)
    return {'shape':list(a.shape),'dtype':str(a.dtype),'exact':torch.equal(a,b),
            'max_abs':float((a.detach().double()-b.detach().double()).abs().max()),'atol':atol,'rtol':rtol}


def backbone_audit(directory):
    reports=[]
    for backbone,frames,frame_clock in [('dit_grid',0,False),('dit_grid',1,False),('wan_grid',0,False),('wan_grid',3,False),('wan_grid',3,True)]:
        config=recipe('flow',[2,*([frames] if frames else []),8,8],backbone,width=24,depth=2,attention_heads=3)
        wrapper=build('model',config['model']);native=copy.deepcopy(wrapper.model)
        # Exercise all layers under nonzero weights, including DiT zero-init gates.
        with torch.no_grad():
            for p in native.parameters():
                if p.requires_grad:p.normal_(0,.08)
        wrapper.model.load_state_dict(native.state_dict());native.eval();wrapper.eval()
        x=torch.randn(2,*config['image_shape'],requires_grad=True);xx=x.detach().clone().requires_grad_()
        t=torch.tensor([[.23,.31,.47],[.61,.52,.4]]) if frame_clock else torch.tensor([.23,.61]);text=torch.randn(2,3,16)
        audit=ActivationAudit(f'{backbone}_T{frames}'+('_frame_clocks' if frame_clock else ''),[('backbone',native,wrapper.model)],directory)
        audit.begin(0);audit.side('native')
        if backbone=='dit_grid':
            y=native(x if not frames else x.squeeze(2),t*1000,torch.zeros(2,dtype=torch.long))
            if frames:y=y.unsqueeze(2)
        else:
            native_t=t.repeat_interleave((8//native.config.patch_size[1])*(8//native.config.patch_size[2]),dim=1) if frame_clock else t
            y=native(hidden_states=x.unsqueeze(2) if not frames else x,timestep=native_t*1000,encoder_hidden_states=text,return_dict=False)[0]
            if not frames:y=y.squeeze(2)
        target=torch.randn_like(y);(y-target).square().mean().backward()
        audit.side('unified');yy=wrapper(xx,t,**({'text':text} if backbone=='wan_grid' else {}))
        (yy-target).square().mean().backward()
        audit.parameters(native,wrapper.model,'backbone');audit.finish_segment();result=audit.close()
        result['final_output']=comparison(y,yy,0,0);result['input_gradient']=comparison(x.grad,xx.grad,0,0)
        assert not result['structural_mismatches']
        assert all(v['exact']==v['pairs'] for v in result['totals'].values())
        reports.append(result)
    return reports


def pyramid_flow_audit(directory=None):
    scheduler=load('crossmodal_pf_scheduler',PF/'diffusion_schedulers/scheduling_flow_matching.py').PyramidFlowMatchEulerDiscreteScheduler()
    path=PF/'pyramid_dit/pyramid_dit_for_video_gen_pipeline.py'
    tree=ast.parse(path.read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='PyramidDiTForVideoGeneration')
    fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='add_pyramid_noise')
    u=torch.tensor([.233,.719]);namespace={'torch':torch,'F':F,'rearrange':rearrange,'compute_density_for_timestep_sampling':lambda **kwargs:u}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),namespace)
    instance=SimpleNamespace(stages=[0,1,2],scheduler=scheduler,corrupt_ratio=.1)
    clean=[torch.randn(6,2,2,size,size,requires_grad=True) for size in (2,4,8)]
    noise=torch.randn_like(clean[-1]);program=PyramidFlowProcess([[2,2,2],[2,4,4],[2,8,8]],steps=4)
    with patch('torch.randn_like',lambda x:noise):original=namespace['add_pyramid_noise'](instance,clean)
    noises=program.noise_pyramid(noise);rows=[]
    for i in range(3):
        selected=[v[i::3] for v in clean]
        state,target,clock,_=program.prepare(selected,noises[i][i::3],u,i)
        ns=original[0][i][0];nt=original[3][i];nc=original[2][i]/1000
        row={'stage':i,'state':comparison(ns,state),'target':comparison(nt,target),'clock':comparison(nc,clock)}
        # The official method is decorated no_grad: no encoder/input backward
        # is claimed. Compare the actual trainable network on both objectives.
        model=build('model',{'name':'grid_dit','args':dict(channels=2,width=24,depth=1,num_heads=3,patch=[1,1,1],max_stages=3)})
        clone=copy.deepcopy(model)
        audit=ActivationAudit(f'pyramid_flow_stage{i}',[('common_backbone',model,clone)],directory or ROOT/'evidence/cross_modal_20261013')
        audit.begin(0);audit.side('native')
        pred=model(ns,nc,stage=i);la=(pred-nt).square().mean();la.backward()
        audit.side('unified');port=clone(state.detach(),clock,stage=i)
        lb=(port-target.detach()).square().mean();lb.backward()
        audit.parameters(model,clone,'common_backbone');audit.finish_segment();trace=audit.close()
        assert not trace['structural_mismatches'] and all(v['exact']==v['pairs'] for v in trace['totals'].values())
        row['module_trace']=trace
        row['model_output']=comparison(pred,port);row['loss']=comparison(la,lb)
        row['parameter_gradients']=[comparison(a.grad,b.grad) for a,b in zip(model.parameters(),clone.parameters()) if a.grad is not None]
        assert all(row[key]['exact'] for key in ('state','target','clock','model_output','loss'))
        assert all(v['exact'] for v in row['parameter_gradients'])
        row['preprocessing_gradient']='disabled by unchanged author @torch.no_grad decorator; network backward measured'
        rows.append(row)
    return {'source':str(path),'execution':'unchanged extracted author method plus imported author scheduler; no model/assets',
            'training':rows,'boundary':'orthogonal PSD equivalent for gamma=1/3, not Cholesky RNG identity',
            'batch_policy':'native selects interleaved examples; common trainer evaluates all stages per minibatch; comparison uses identical selected examples'}


def pyramidal_wan_audit():
    path=WORK/'experiments/official_video_baselines/pyramidal_wan_paper_math.py'
    ref=load('crossmodal_pwan_reference',path)
    grids=[[3,2,2],[5,4,4],[9,8,8]];proc=PyramidalWanProcess(grids,first_frame=True)
    clean=[torch.randn(2,2,*g,dtype=torch.float64,requires_grad=True) for g in grids]
    noise=torch.randn_like(clean[-1]);noises=proc.noise_pyramid(noise);u=torch.tensor([.27,.71],dtype=torch.float64);rows=[]
    for i in range(3):
        natural=proc.bounds[i+1]+u*(proc.bounds[i]-proc.bounds[i+1])
        original=ref.prepare(clean[::-1],noise,natural,2-i)
        state,target,clock,_=proc.prepare(clean,noises[i],u,i)
        row={'stage':i,'state':comparison(original.noisy,state,1e-12,1e-12),'target':comparison(original.target,target,1e-12,1e-12),
             'clock':comparison(original.sigma[:,0,:,0,0],clock[:,None].expand(-1,grids[i][0]) if clock.ndim==1 else clock,1e-12,1e-12)}
        ga=torch.autograd.grad((original.noisy.square()+original.target.square()).sum(),clean[max(0,i-1):i+1],retain_graph=True)
        gb=torch.autograd.grad((state.square()+target.square()).sum(),clean[max(0,i-1):i+1],retain_graph=True)
        row['input_gradients']=[comparison(a,b,1e-10,1e-12) for a,b in zip(ga,gb)]
        if i<2:
            innovation=torch.randn(2,2,*grids[i+1],dtype=torch.float64)
            row['boundary']=comparison(ref.transition(state,2-i,innovation),proc.transition(state,i,SimpleNamespace(normal=lambda x:innovation)),1e-12,1e-12)
        rows.append(row)
    return {'source':str(path),'evidence_type':'comparison to independently implemented local paper equations, NOT an author repo','stages':rows}


def math_audit():
    rows=[]
    for video in (False,True):
        grids=[[*([2] if video else []),s,s] for s in (2,4,8)]
        clean=[torch.randn(2,2,*g,dtype=torch.float64) for g in grids]
        u=torch.tensor([.26,.72],dtype=torch.float64)
        # f-DM reconstruction and Eq.9 gradients, independent symbolic oracle.
        for mode in ('sp','vp'):
            p=FDMProcess(grids,rescaling=mode)
            for i in range(3):
                e=torch.randn_like(clean[i]);state,target,t,meta=p.prepare(clean,e,u,i)
                previous=clean[i] if i==0 else F.interpolate(clean[i-1],size=grids[i],mode='trilinear' if video else 'bilinear',align_corners=False)
                fraction=((t-p.bounds[i+1])/(p.bounds[i]-p.bounds[i+1])).reshape(2,*([1]*(state.ndim-1)))
                signal=(1-fraction)*clean[i]+fraction*previous
                a=torch.cos(torch.pi*t/2).reshape(2,*([1]*(state.ndim-1)))
                s=torch.sin(torch.pi*t/2).reshape_as(a)/(4/2**i)
                if mode=='vp':norm=(a*a+s*s).sqrt();a,s=a/norm,s/norm
                row={'method':'fdm','video':video,'rescaling':mode,'stage':i,'state':comparison(a*signal+s*e,state,1e-13,1e-13)}
                pred={'epsilon':torch.randn_like(e,requires_grad=True),'residual':torch.randn_like(e,requires_grad=True)}
                expected=(((state-s*pred['epsilon'])/a-signal).square()+(pred['residual']-(clean[i]-signal)).square())/(1+a*a/(s*s))
                loss=p.loss(pred,target,meta)[0]
                row['loss']=comparison(expected.flatten(1).mean(1),loss,1e-10,1e-12)
                row['gradient']=[comparison(x,y,1e-10,1e-12) for x,y in zip(torch.autograd.grad(expected.mean(),list(pred.values()),retain_graph=True),torch.autograd.grad(loss.mean(),list(pred.values())))]
                rows.append(row)
        # EDM velocity equals the derivative of the specified signal/noise path.
        p=EdifyProcess(grids)
        for i in range(3):
            sigma=torch.tensor([(p.starts[i]+p.ends[i])/2]*2,dtype=torch.float64,requires_grad=True)
            e=torch.randn_like(clean[i]);s=sigma.reshape(2,*([1]*(e.ndim-1)))
            low=clean[i] if not i else p.maps[i-1].project(clean[i]);high=clean[i]-low
            alpha,_=p.attenuation(s,i);state=low+alpha*high+s*e
            v=p.velocity(clean[i],state,sigma,i);probe=torch.randn_like(state)
            actual=torch.autograd.grad((state*probe).sum(),sigma)[0]
            expected=(v*probe).flatten(1).sum(1)
            rows.append({'method':'edify_laplacian','video':video,'stage':i,'path_derivative':comparison(expected,actual,1e-11,1e-11)})
        # PDDPM state and epsilon target match the original DDPM marginal.
        p=PDDPMProcess(grids,steps=64,train_steps=64)
        for i in range(3):
            e=torch.randn_like(clean[i]);state,target,t,_=p.prepare(clean,e,u,i)
            idx=(64*u).long();ab=(1-torch.linspace(.0001,.02,64,dtype=torch.float64)).cumprod(0)[idx].reshape(2,*([1]*(e.ndim-1)))
            rows.append({'method':'pddpm','video':video,'stage':i,'marginal':comparison(ab.sqrt()*clean[i]+(1-ab).sqrt()*e,state,1e-12,1e-12),'target':comparison(e,target,0,0)})
        # PixelFlow endpoint/clock against the pre-existing native-certified path.
        from umdiff.paths import EndpointFlow
        p=PixelFlowProcess(grids)
        clean=p.clean_pyramid(clean[-1],{})
        for i in range(3):
            e=torch.randn_like(clean[i]);state,target,t,_=p.prepare(clean,e,u,i)
            # For stage zero the signal coefficient is zero, so projection is immaterial.
            path=EndpointFlow(p.clock.starts[i],p.clock.ends[i],{'down':'bilinear','up':'nearest'})
            original=path.sample(clean[i],(u*1000).long().double()/1000,e)
            rows.append({'method':'pixelflow','video':video,'stage':i,'state':comparison(original.state,state,1e-12,1e-12),'target':comparison(original.velocity,target,1e-12,1e-12)})
    return rows


def tiny_learning():
    rows=[]
    for name in STAGE_METHODS:
        config=recipe(name,[2,3,8,8],width=24,depth=1,attention_heads=3,patch=(1,1,1),steps=4)
        config['training']['optimizer']['lr']=.003
        method,model,trainer=setup(config,torch.device('cpu'),23)
        x=torch.randn(2,*config['image_shape']);e=torch.randn_like(x);t=torch.tensor([.32,.66]);losses=[]
        tape=trainer.stream.state_dict()
        for _ in range(128):
            trainer.stream.load_state_dict(tape)
            losses.append(float(trainer.step(x,t,noise=e).loss.detach().mean()))
        assert losses[-1]<losses[0]*.7,(name,losses[0],losses[-1])
        rows.append({'method':name,'updates':128,'initial':losses[0],'final':losses[-1],'ratio':losses[-1]/losses[0],
                     'interpretation':'fixed synthetic video/noise/time overfit; verifies trainability, not generation quality'})
    return rows


def speed_boundary_audit():
    source=load('crossmodal_speed_reference',ROOT/'sources/speed/utils.py')
    from umdiff.boundaries import SpectralBoundary
    rows=[]
    for video in (False,True):
        state=torch.randn(2,3,*([3] if video else []),4,4)
        for transform in ('dct','dwt','fft'):
            a,ta=source.spectral_expand_and_align(state,(8,8),2.,.41,transform,37)
            b=SpectralBoundary(transform)(state,time=.41,seed=37,target_hw=(8,8),ratio=2.)
            assert ta==b.time
            rows.append({'video':video,'transform':transform,'output':comparison(a,b.state,0,0),'time_equal':True,
                         'backward':'not applicable: unchanged source spectral expansion uses detached NumPy arrays'})
    return rows


def nonuniform_loss_audit():
    from umdiff.multires_process import NonUniformProcess
    rows=[]
    for video in (False,True):
        grids=[[*([2] if video else []),s,s] for s in (2,4,8)]
        program=NonUniformProcess(grids)
        x=torch.randn(2,2,*grids[-1],dtype=torch.float64);noise=torch.randn_like(x);u=torch.tensor([.21,.67],dtype=torch.float64)
        class Linear(torch.nn.Module):
            def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.3,dtype=torch.float64))
            def forward(self,state,t,stage=0):return {j:v*self.weight for j,v in state.items()}
        model=Linear();result=program.training(model,x,u,RandomStream(1),noise,{},{});terms=[]
        for stage,state in enumerate(result.state):
            high=program.horizons[stage];low=program.horizons[stage+1] if stage+1<len(grids) else program.terminal
            t=low+(high-low)*u;size=sum(v[0].numel() for v in state.values());term=0.
            for j,value in state.items():
                tt=t.clone().requires_grad_();slope=(program.min_snr-program.max_snr)/program.horizons[j]
                alpha=torch.sigmoid(program.max_snr+slope*tt).sqrt()
                beta=-2*torch.autograd.grad(alpha.log().sum(),tt)[0]
                sigma=(1-alpha.square()).sqrt();shape=(-1,*([1]*(value.ndim-1)))
                error=(-model.weight*value/sigma.reshape(shape)+result.target[stage][j]/sigma.reshape(shape)).square()
                term+=(error*beta.reshape(shape)).flatten(1).sum(1)/(2*size)
            terms.append(term*program.weights[stage])
        loss=sum(terms)
        row={'video':video,'weighted_score_objective':comparison(loss,result.loss,1e-11,1e-11),
             'parameter_gradient':comparison(torch.autograd.grad(loss.mean(),model.weight,retain_graph=True)[0],torch.autograd.grad(result.loss.mean(),model.weight)[0],1e-11,1e-11)}
        rows.append(row)
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default=str(ROOT/'evidence/cross_modal_20261013'));args=parser.parse_args()
    directory=Path(args.output);directory.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(121)
    report={'device':'cpu','pretrained_assets':False,'formal_training':False,'checks':{}}
    try:
        for name,fn in [('backbones',lambda:backbone_audit(directory)),('pyramid_flow',lambda:pyramid_flow_audit(directory)),('pyramidal_wan_path',pyramidal_wan_audit),('math',math_audit),('nonuniform_weighted_dsm',nonuniform_loss_audit),('speed_boundary',speed_boundary_audit),('fixed_fixture_learning',tiny_learning)]:
            report['checks'][name]=fn();print(name,'PASS',flush=True)
            (directory/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
        paths=list((ROOT/'umdiff').glob('*.py'))+[Path(__file__)]+list((ROOT/'tests').glob('test_cross_modal*.py'))
        paths += [PF/'diffusion_schedulers/scheduling_flow_matching.py',PF/'pyramid_dit/pyramid_dit_for_video_gen_pipeline.py',WORK/'experiments/official_video_baselines/pyramidal_wan_paper_math.py']
        report['sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        report['pass']=True
        report['cuda_initialized']=torch.cuda.is_initialized()
        assert not report['cuda_initialized']
    except Exception as exc:
        report['pass']=False;report['error']=repr(exc);raise
    finally:(directory/'audit.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
