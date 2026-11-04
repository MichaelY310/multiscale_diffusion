#!/usr/bin/env python3
"""CPU-first native source conformance, with numeric failures retained in JSON."""
import argparse,json,os,sys,time,traceback,hashlib,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'_deps')]
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import numpy as np
import torch
from umdiff import native

CHECKS=[]
ARTIFACT_ROOT=ROOT/'runs/parity_outputs'
def compare(name,actual,expected,atol=2e-6,rtol=2e-5,scope='component'):
    if isinstance(actual,(dict,list,tuple)):
        if isinstance(actual,dict):
            if set(actual)!=set(expected):raise AssertionError('Tree keys mismatch')
            keys=actual
        else:
            if len(actual)!=len(expected):raise AssertionError('Tree length mismatch')
            keys=range(len(actual))
        for key in keys:compare(f'{name}/{key}',actual[key],expected[key],atol,rtol,scope)
        return
    a=torch.as_tensor(actual).detach().cpu();e=torch.as_tensor(expected).detach().cpu()
    if a.shape!=e.shape:raise AssertionError(f'{name}: shape {a.shape} != {e.shape}')
    from activation_trace import byte_view
    diagnostic_dtype=torch.complex128 if a.is_complex() or e.is_complex() else torch.float64
    error=torch.where(a==e,0.,(a.to(diagnostic_dtype)-e.to(diagnostic_dtype)).abs())
    error=torch.nan_to_num(error,nan=float('inf'),posinf=float('inf'))
    compatible=a.dtype==e.dtype
    passed=compatible and torch.allclose(a.to(diagnostic_dtype),e.to(diagnostic_dtype),atol=atol,rtol=rtol)
    bitwise=compatible and torch.equal(byte_view(a),byte_view(e))
    CHECKS.append({'name':name,'scope':scope,'shape':list(a.shape),'dtype':str(a.dtype),'expected_dtype':str(e.dtype),'max_abs':error.max().item() if error.numel() else 0.,'rms':error.square().mean().sqrt().item() if error.numel() else 0.,'atol':atol,'rtol':rtol,'passed':passed,'values_exact':compatible and torch.equal(a,e),'bitwise':bitwise})
    if 'inference' in scope:
        directory=ARTIFACT_ROOT/name.split('/')[0]
        directory.mkdir(parents=True,exist_ok=True)
        stem=re.sub(r'[^a-zA-Z0-9_.-]','_',name)[:130]+f'_{len(CHECKS):06d}'
        path=directory/(stem+'.pt')
        torch.save(dict(name=name,scope=scope,native=e,unified=a,bitwise=bitwise,max_abs=CHECKS[-1]['max_abs']),path)
        artifact=dict(tensor_pair=str(path.relative_to(ROOT)))
        # Preview only explicit pixel-space outputs; latent channels stay tensors.
        ep,ap=e,a
        if 'pixels' in scope and ep.ndim==5 and ep.shape[1]==3:ep,ap=ep[:,:,0],ap[:,:,0]
        if 'pixels' in scope and ep.ndim==3:ep,ap=ep[None],ap[None]
        if 'pixels' in scope and ep.ndim==4:
            if ep.shape[1]!=3 and ep.shape[-1]==3:ep,ap=ep.permute(0,3,1,2),ap.permute(0,3,1,2)
            if ep.shape[1]==3:
                from PIL import Image
                def pixels(value):
                    return value if value.dtype==torch.uint8 else ((value.float().clamp(-1,1)+1)*127.5).round().to(torch.uint8)
                native_pixel=pixels(ep)[0];unified_pixel=pixels(ap)[0]
                difference=(native_pixel.short()-unified_pixel.short()).abs().to(torch.uint8)
                for label,value in [('native',native_pixel),('unified',unified_pixel),('difference',difference)]:
                    image_path=directory/(stem+'_'+label+'.png')
                    Image.fromarray(value.permute(1,2,0).numpy()).save(image_path)
                    artifact[label+'_png']=str(image_path.relative_to(ROOT))
                artifact['preview']='batch0, frame0 for BCTHW; uint8 preserved, float preview clamps [-1,1] then rounds; raw tensors retained'
        CHECKS[-1]['artifact']=artifact
    if not passed:raise AssertionError(f'{name}: max error {CHECKS[-1]["max_abs"]}')

def fixture():
    torch.manual_seed(20261008);np.random.seed(20261008)
    return torch.randn(3,3,16,16),torch.randn(3,3,16,16),torch.tensor([.57,.71,.89])

def train_compare(name,model,states_a,states_b,t,targets_a,targets_b,steps=3,loss_fn=None):
    """Nonzero random shared backbone, independent inputs/targets/optimizer paths."""
    from copy import deepcopy
    from umdiff.tree import map_tree,leaves
    a=deepcopy(model);b=deepcopy(model)
    oa=torch.optim.AdamW(a.parameters(),lr=1e-4,betas=(.9,.99),weight_decay=.01)
    ob=torch.optim.AdamW(b.parameters(),lr=1e-4,betas=(.9,.99),weight_decay=.01)
    if loss_fn is None:
        def loss_fn(pred,target):
            errors=map_tree(lambda p,q:(p-q).square().flatten(1).mean(1),pred,target)
            return sum(v.mean() for v in leaves(errors))
    for index in range(steps):
        oa.zero_grad();ob.zero_grad()
        pa=a(states_a,t);pb=b(states_b,t)
        compare(f'{name}/step{index}/prediction',pa,pb,scope='training')
        la=loss_fn(pa,targets_a);lb=loss_fn(pb,targets_b)
        compare(f'{name}/step{index}/loss',la,lb,scope='training')
        la.backward();lb.backward()
        for (key,wa),(_,wb) in zip(a.named_parameters(),b.named_parameters()):
            if wa.grad is not None:compare(f'{name}/step{index}/gradient/{key}',wa.grad,wb.grad,atol=4e-6,rtol=5e-4,scope='training')
        oa.step();ob.step()
        for (key,wa),(_,wb) in zip(a.named_parameters(),b.named_parameters()):compare(f'{name}/step{index}/weight/{key}',wa,wb,atol=5e-6,rtol=5e-4,scope='training')

def run_speed():
    from umdiff.boundaries import SpectralBoundary
    from umdiff.clocks import speed_activation
    author=native.load('speed/utils.py');x,_,_=fixture();x=x[:2,:2,:8,:8]
    for dtype in (torch.float32,torch.float64):
        for basis in ('dct','dwt','fft'):
            for t in (.12,.57,.91):
                expected,expected_t=author.spectral_expand_and_align(x.to(dtype),(16,16),2,t,basis,19)
                result=SpectralBoundary(basis)(x.to(dtype),time=t,seed=19)
                compare(f'speed/{dtype}/{basis}/{t}/state',result.state,expected,atol=0,rtol=0,scope='native_inference_transition')
                compare(f'speed/{basis}/{t}/clock',result.time,expected_t,atol=0,rtol=0)
    for p in (.1,1,10):compare('speed/activation',speed_activation(p,.03),author.activation_time(p,.03),atol=0,rtol=0)

def run_relay():
    import scipy.stats
    from umdiff.operators import heat_blur,block_dct,rolling_noise
    author=native.load('relay/blurring.py');x,eps,t=fixture()
    for dtype in (torch.float32,torch.float64):
        value=x.to(dtype)
        compare(f'relay/dct/{dtype}',block_dct(value,4),author.dct_2d(value,4,norm='ortho'),atol=3e-6 if dtype==torch.float32 else 2e-14)
        compare(f'relay/rolling_noise/{dtype}',rolling_noise(eps.to(dtype),4),author.block_noise(value,randn_like=lambda _:eps.to(dtype),block_size=4),atol=0,rtol=0)
        blur=torch.tensor([.2,1.,2.5],dtype=dtype).view(-1,1,1,1)
        freq=torch.pi*torch.linspace(0,3,4)/4
        damping=torch.exp(-(freq[:,None]**2+freq[None,:]**2).repeat(4,4)*(blur**2/2))*(1-.001)+.001
        expected=author.idct_2d(author.dct_2d(value,4,norm='ortho')*damping,4,norm='ortho')
        actual=heat_blur(value,blur,4,.001)
        compare(f'relay/heat/{dtype}',actual,expected,atol=4e-6 if dtype==torch.float32 else 2e-14)
    # Independent heat path is exercised through nonzero trainable prediction.
    from umdiff.models import TinyConv
    train_compare('relay/heat_objective',TinyConv(3,16),actual.float()+eps,expected.float()+eps,t,x,x)

def run_pixelflow():
    import math
    from einops import rearrange
    from umdiff.clocks import PixelFlowClock
    from umdiff.paths import EndpointFlow
    from umdiff.boundaries import PixelFlowBoundary
    from umdiff.noise import equicorrelated_blocks
    author=native.load('pixelflow/pixelflow/scheduling_pixelflow.py')
    raw=native.extract('pixelflow/pixelflow/pipeline_pixelflow.py',[],{}) if False else None
    x,eps,t=fixture()
    for count in (2,4):
        scheduler=author.PixelFlowScheduler(1000,count);clock=PixelFlowClock(1000,count)
        for stage in range(count):
            compare(f'pixelflow/{count}/{stage}/start',clock.starts[stage],scheduler.start_t[stage],atol=0,rtol=0)
            compare(f'pixelflow/{count}/{stage}/network_training_grid',clock.network_times[stage],scheduler.Timesteps_per_stage[stage],atol=0,rtol=0)
            for shift in (1.,2.):
                scheduler.set_timesteps(5,stage,shift=shift)
                grid,net=clock.grid(5,stage,shift)
                compare('pixelflow/local_grid',grid,scheduler.t,atol=0,rtol=0)
                compare('pixelflow/network_grid',net,scheduler.Timesteps,atol=1e-12,rtol=1e-12)
                state=x.clone();port=x.clone()
                for i in range(5):
                    velocity=.2*torch.sin(state)+.001*net[i].float()
                    state=scheduler.step(velocity,state)
                    port=(port.float()+(grid[i+1].float()-grid[i].float())*velocity).to(velocity.dtype)
                    compare(f'pixelflow/sampler_step{i}',port,state,atol=0,rtol=0,scope='native_inference_trajectory')
            a=clock.starts[stage];b=clock.ends[stage]
            px=torch.nn.functional.interpolate(torch.nn.functional.interpolate(x,scale_factor=.5,mode='bilinear',align_corners=False),scale_factor=2,mode='nearest')
            start=a*px+(1-a)*eps;end=b*x+(1-b)*eps
            expected=t[:,None,None,None]*end+(1-t[:,None,None,None])*start
            actual=EndpointFlow(a,b,projection={'down':'bilinear','up':'nearest'}).sample(x,t,eps)
            compare('pixelflow/training_state',actual.state,expected,atol=0,rtol=0)
            compare('pixelflow/velocity',actual.velocity,end-start,atol=0,rtol=0)
    # Execute the author's method verbatim, bypassing the unrelated text pipeline.
    import ast
    path=ROOT/'sources/pixelflow/pixelflow/pipeline_pixelflow.py';tree=ast.parse(path.read_text())
    method=next(n for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='sample_block_noise')
    env={'torch':torch,'rearrange':rearrange};exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),env)
    from types import SimpleNamespace
    torch.manual_seed(17);noise=env['sample_block_noise'](SimpleNamespace(scheduler=SimpleNamespace(gamma=-1/3)),3,3,16,16)
    torch.manual_seed(17);ours=equicorrelated_blocks(x)
    compare('pixelflow/block_noise',ours,noise,atol=0,rtol=0)
    for time_ in (.25,.5,.75):
        a=1/(math.sqrt(4)*(1-time_)+time_);b=a*(1-time_)*math.sqrt(3)
        original=a*torch.nn.functional.interpolate(x,scale_factor=2,mode='nearest')+b*torch.nn.functional.interpolate(noise,scale_factor=2,mode='nearest')
        result=PixelFlowBoundary()(x,time=time_,noise=torch.nn.functional.interpolate(noise,scale_factor=2,mode='nearest'))
        compare('pixelflow/boundary',result.state,original,atol=2e-6)

RUNNERS={'speed':run_speed,'relay':run_relay,'pixelflow':run_pixelflow}
import conformance_more
RUNNERS.update(conformance_more.install(compare,train_compare,fixture))
import conformance_additional
RUNNERS.update(conformance_additional.install(compare,train_compare,fixture))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('method',choices=RUNNERS);parser.add_argument('--output');parser.add_argument('--threads',type=int,default=2);args=parser.parse_args()
    torch.set_num_threads(args.threads);started=time.time();status='passed';failure=None
    try:RUNNERS[args.method]()
    except Exception:
        status='failed';failure=traceback.format_exc();print(failure,file=sys.stderr)
    report={'method':args.method,'status':status,'device':'cpu','gpu_used':False,'torch':torch.__version__,'seconds':time.time()-started,'checks':CHECKS,'source_executions':native.AUDIT,'failure':failure,'quality_claim':'Synthetic conformance only; no trained-data quality or FID claim.'}
    output=Path(args.output) if args.output else ROOT/'evidence'/f'{args.method}_conformance.json'
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2))
    print(json.dumps({'method':args.method,'status':status,'checks':len(CHECKS),'report':str(output),'seconds':report['seconds']}))
    return 0 if status=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
