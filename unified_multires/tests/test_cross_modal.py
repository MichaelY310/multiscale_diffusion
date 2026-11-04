"""CPU regression and cross-modal contracts; no external training assets."""
import copy
import random
import numpy as np
import pytest
import torch
from umdiff.cross_modal import recipe,METHODS,STAGE_METHODS,PRODUCT_METHODS
from umdiff.cli import make_method,setup
from umdiff.registry import build
from umdiff.randomness import RandomStream
from umdiff.grids import GridMap,haar_split,haar_merge

torch.set_num_threads(2)


def small(method,video=False,backbone='grid_dit',**kwargs):
    shape=[3,*([3] if video else []),16,16]
    return recipe(method,shape,backbone,width=24,depth=1,attention_heads=3,patch=(1,1,1),steps=4,**kwargs)


def initial(method,config,stream,batch=2):
    shape=method.initial_shape() if hasattr(method,'initial_shape') else list(config['image_shape'])
    if config.get('sampler',{}).get('name')=='ssd':shape[-2:]=[method.sampler.batch.resolutions[-1]]*2
    if getattr(method,'sampler',None) is not None and hasattr(method.sampler,'initial_shape'):shape=method.sampler.initial_shape(shape)
    value=stream.normal(torch.empty(batch,*shape))
    if config['evidence']['source_method']=='dfm':value=dict(enumerate(stream.normal(x) for x in method.representation.encode(value)))
    if config['evidence']['source_method']=='dctdiff':value=stream.normal(method.representation.encode(value))
    return value


@pytest.mark.parametrize('name',METHODS)
@pytest.mark.parametrize('video',[False,True])
def test_training_sampling(name,video):
    config=small(name,video)
    method,model,trainer=setup(config,torch.device('cpu'),42)
    images=torch.randn(2,*config['image_shape'])
    sampler=build('time_sampler',config['training']['time_sampler'])
    result=trainer.step(images,None,time_sampler=sampler)
    assert torch.isfinite(result.loss).all()
    assert any(p.grad is not None and p.grad.abs().max()>0 for p in model.parameters())
    with torch.no_grad():out=method.sample(model,initial(method,config,trainer.stream),stream=trainer.stream)
    assert out.shape==images.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize('name',[n for n in METHODS if n not in PRODUCT_METHODS])
@pytest.mark.parametrize('backbone',['dit_grid','wan_grid','grid_conv'])
@pytest.mark.parametrize('video',[False,True])
def test_backbone_matrix(name,backbone,video):
    config=small(name,video,backbone)
    method,model,trainer=setup(config,torch.device('cpu'),19)
    images=torch.randn(1,*config['image_shape'])
    kwargs={'text':torch.randn(1,3,16)} if backbone=='wan_grid' else {}
    result=trainer.step(images,torch.tensor([.51]),model_kwargs=kwargs)
    assert torch.isfinite(result.loss).all()
    with torch.no_grad():out=method.sample(model,initial(method,config,trainer.stream,1),stream=trainer.stream,model_kwargs=kwargs)
    assert out.shape==images.shape and torch.isfinite(out).all()


@pytest.mark.parametrize('position',['sincos','rope'])
def test_video_is_joint_and_single_frame_matches_image(position):
    model=build('model',{'name':'grid_dit','args':dict(channels=2,width=24,depth=2,num_heads=3,patch=[1,2,2],max_stages=3,position=position)})
    # Zero-gated blocks at initialization hide temporal dependencies. Test
    # nonzero learned weights so a framewise implementation cannot pass.
    with torch.no_grad():
        for block in model.blocks:block.modulation[-1].weight.normal_(0,.05)
    x=torch.randn(2,2,8,8,requires_grad=True);t=torch.tensor([.3,.7])
    y=model(x,t);v=model(x.unsqueeze(2),t).squeeze(2)
    torch.testing.assert_close(y,v,rtol=0,atol=0)
    grad=torch.autograd.grad(y.square().sum(),x,retain_graph=True)[0]
    gv=torch.autograd.grad(v.square().sum(),x)[0]
    torch.testing.assert_close(grad,gv,rtol=0,atol=0)
    video=torch.randn(1,2,3,8,8,requires_grad=True)
    out=model(video,torch.tensor([[.2,.4,.6]])) # T == max_stages regression
    cross=torch.autograd.grad(out[:,:,0].square().sum(),video)[0][:,:,1:]
    assert cross.abs().max()>1e-7


@pytest.mark.parametrize('fine,coarse,first',[( (8,8),(4,4),False),((4,8,8),(2,4,4),False),((5,8,8),(3,4,4),True)])
def test_projection_and_noise_covariance(fine,coarse,first):
    op=GridMap(fine,coarse,first);x=torch.randn(2,1,*fine,dtype=torch.float64,requires_grad=True)
    low=op.down(x)
    torch.testing.assert_close(op.down(op.up(low)),low,rtol=0,atol=0)
    p=op.project(x);q=x-p
    torch.testing.assert_close((p*q).sum(),torch.zeros((),dtype=x.dtype),rtol=0,atol=1e-12)
    eps=torch.randn_like(low);lifted=op.lift_noise(eps,torch.randn_like(x))
    torch.testing.assert_close(op.down_noise(lifted),eps,rtol=1e-14,atol=1e-14)
    # Exact linear covariance, not a noisy Monte Carlo variance claim.
    n=int(np.prod(fine));m=int(np.prod(coarse))
    basis=torch.eye(n+m,dtype=torch.float64)
    A=op.lift_noise(basis[:,:m].reshape(n+m,1,*coarse),basis[:,m:].reshape(n+m,1,*fine)).flatten(1).T
    torch.testing.assert_close(A@A.T,torch.eye(n,dtype=x.dtype),rtol=1e-14,atol=1e-14)


def test_haar_invertible_and_energy():
    for axes in ((3,4),(2,3,4)):
        x=torch.randn(2,3,4,8,8,dtype=torch.float64,requires_grad=True)
        parts=haar_split(x,axes);out=haar_merge(parts,axes)
        torch.testing.assert_close(x,out,rtol=1e-14,atol=1e-14)
        torch.testing.assert_close(sum(v.square().sum() for v in parts),x.square().sum(),rtol=1e-14,atol=1e-14)
        torch.testing.assert_close(torch.autograd.grad(out.sum(),x)[0],torch.ones_like(x),rtol=1e-14,atol=1e-14)


@pytest.mark.parametrize('name',STAGE_METHODS)
def test_exact_resume(name,tmp_path):
    config=small(name,True)
    method,model,trainer=setup(config,torch.device('cpu'),34)
    images=torch.randn(2,*config['image_shape'])
    trainer.step(images,None)
    path=tmp_path/'state.pt';trainer.save(path)
    expected=trainer.step(images,None)
    _,model2,trainer2=setup(config,torch.device('cpu'),999)
    trainer2.load(path);actual=trainer2.step(images,None)
    torch.testing.assert_close(actual.loss,expected.loss,rtol=0,atol=0)
    def compare(a,b):
        if torch.is_tensor(a):torch.testing.assert_close(a,b,rtol=0,atol=0)
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:compare(a[k],b[k])
        elif isinstance(a,(list,tuple)):
            for v,w in zip(a,b):compare(v,w)
        else:assert a==b
    for key in ('model','ema','optimizer','stream'):compare(trainer.state_dict()[key],trainer2.state_dict()[key])


def test_spacetime_first_frame_and_independent_models():
    for name in ('pyramidal_wan_path','rdd','fdm','pddpm'):
        config=recipe(name,[2,9,8,8],grids=[[3,2,2],[5,4,4],[9,8,8]],width=24,depth=1,attention_heads=3,patch=(1,1,1),first_frame=True,separate_stages=True,steps=4)
        method,model,trainer=setup(config,torch.device('cpu'),7)
        result=trainer.step(torch.randn(1,*config['image_shape']),torch.tensor([.5]))
        assert torch.isfinite(result.loss).all()
        for child in model.models:assert any(p.grad is not None and p.grad.abs().max()>0 for p in child.parameters())
        with torch.no_grad():out=method.sample(model,initial(method,config,trainer.stream,1),stream=trainer.stream)
        assert out.shape==(1,*config['image_shape']) and torch.isfinite(out).all()


def test_rdd_decoupled():
    config=small('rdd',True,decoupled=True)
    method,model,trainer=setup(config,torch.device('cpu'),7)
    result=trainer.step(torch.randn(2,*config['image_shape']),torch.tensor([.2,.7]))
    assert set(result.prediction[0])=={'main','drift'}
    with torch.no_grad():out=method.sample(model,initial(method,config,trainer.stream),stream=trainer.stream)
    assert torch.isfinite(out).all()


def test_invalid_combinations_fail_explicitly():
    with pytest.raises(ValueError,match='joint multiband'):small('dfm',True,'wan_grid')
    with pytest.raises(ValueError,match='frame'):recipe('pyramid_flow',[3,5,8,8],grids=[[3,4,4],[5,8,8]],first_frame=True)
    with pytest.raises(ValueError,match='three stages'):recipe('pyramidal_wan_path',[3,8,8],grids=[[4,4],[8,8]])


@pytest.mark.parametrize('policy',['speed','bottleneck'])
@pytest.mark.parametrize('video',[False,True])
def test_inference_policies_on_same_trained_model(policy,video):
    config=small('flow',video,inference_policy=policy)
    method,model,trainer=setup(config,torch.device('cpu'),13)
    result=trainer.step(torch.randn(2,*config['image_shape']),torch.tensor([.3,.7]))
    assert torch.isfinite(result.loss).all()
    out=method.sample(model,initial(method,config,trainer.stream),stream=trainer.stream)
    assert out.shape==(2,*config['image_shape']) and torch.isfinite(out).all()
    # An exact constant-clean oracle integrates to clean even after resizing,
    # renoising, and shifted times. Also checks the velocity sign convention.
    class Oracle(torch.nn.Module):
        def forward(self,x,t,stage=0):return (x-.31)/t.reshape(-1,*([1]*(x.ndim-1)))
    stream=RandomStream(43)
    out,trace=method.sample(Oracle(),initial(method,config,stream),stream=stream,return_trajectory=True)
    torch.testing.assert_close(out,torch.full_like(out,.31),rtol=0,atol=2e-6)
    spec=config['sampler']['args']
    count=sum(spec['steps'])-sum(spec.get('start_indices',[0]*3))
    assert len(trace)==1+count+2


def test_existing_spatial_operators_preserve_frame_arithmetic():
    from umdiff.operators import resize,block_average,block_dct,channel_down,channel_up
    x=torch.randn(2,3,8,8,requires_grad=True)
    for fn in (lambda v:resize(v,scale=.5),block_average,lambda v:block_dct(v,4),channel_down):
        a=fn(x);b=fn(x.unsqueeze(2)).squeeze(2)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        ga=torch.autograd.grad(a.square().sum(),x,retain_graph=True)[0]
        gb=torch.autograd.grad(b.square().sum(),x,retain_graph=True)[0]
        torch.testing.assert_close(ga,gb,rtol=0,atol=0)


def test_lanczos_constant_preservation_and_derivative():
    from umdiff.grid_sampling import lanczos_resize
    x=torch.full((2,3,3,8,8),.37,dtype=torch.float64,requires_grad=True)
    for size in ((4,4),(12,12)):
        y=lanczos_resize(x,size)
        torch.testing.assert_close(y,torch.full_like(y,.37),rtol=0,atol=1e-15)
        assert torch.isfinite(torch.autograd.grad(y.sum(),x,retain_graph=True)[0]).all()


def test_sampling_budget_does_not_change_ddpm_training_process():
    from umdiff.multires_process import PDDPMProcess
    a=PDDPMProcess([[4,4],[8,8]],steps=4);b=PDDPMProcess([[4,4],[8,8]],steps=40)
    assert torch.equal(a.abar,b.abar) and a.abar[-1]<1e-4
    clean=a.clean_pyramid(torch.randn(2,3,8,8),{});noise=torch.randn_like(clean[1]);t=torch.tensor([.2,.9])
    for x,y in zip(a.prepare(clean,noise,t,1)[:3],b.prepare(clean,noise,t,1)[:3]):assert torch.equal(x,y)


def test_conditional_checkpointing_and_axis_positions():
    from umdiff.grid_models import GridDiT,grid_position
    coords=torch.tensor([[.1,.4,.7],[.4,.1,.7]])
    pe=grid_position(coords,24)
    assert not torch.equal(pe[0],pe[1])
    torch.manual_seed(76)
    a=GridDiT(channels=2,width=24,depth=2,num_heads=3,num_classes=5,text_dim=7)
    with torch.no_grad():
        for block in a.blocks:block.modulation[-1].weight.normal_(0,.05)
    b=copy.deepcopy(a);b.gradient_checkpointing=True
    x=torch.randn(2,2,3,8,8);t=torch.tensor([.3,.6]);kw=dict(y=torch.tensor([1,3]),text=torch.randn(2,4,7),text_mask=torch.tensor([[1,1,1,0],[1,1,0,0]],dtype=torch.bool))
    aa=a(x,t,**kw);bb=b(x,t,**kw);torch.testing.assert_close(aa,bb,rtol=0,atol=0)
    aa.square().mean().backward();bb.square().mean().backward()
    for p,q in zip(a.parameters(),b.parameters()):
        if p.grad is None:assert q.grad is None
        else:torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)


@pytest.mark.parametrize('name',STAGE_METHODS)
def test_sampling_trace_is_optional_and_gradients_are_disabled(name):
    from umdiff.trajectories import Trace
    trace=Trace(False);trace.append(object());assert not trace
    config=small(name,True);method,model,_=setup(config,torch.device('cpu'),77)
    x=initial(method,config,RandomStream(32),1).requires_grad_()
    a=method.sample(model,x,stream=RandomStream(12))
    b,history=method.sample(model,x,stream=RandomStream(12),return_trajectory=True)
    assert not a.requires_grad and not b.requires_grad and len(history)>0
    assert type(history) is list
    torch.testing.assert_close(a,b,rtol=0,atol=0)
