import copy
import torch
from umdiff.registry import build
from umdiff.clocks import PixelFlowClock


ARCH=dict(patch_size=2,in_channels=4,hidden_size=32,depth=2,num_heads=4,
          num_classes=1000,class_dropout_prob=.1,learn_sigma=True,base_size=16)


def nonzero(net):
    with torch.no_grad():
        for p in net.parameters():
            p.normal_(0,.03)


def test_scalar_adapter_preserves_native_forward_and_backward():
    torch.manual_seed(29)
    model=build('model',{'name':'comparison_dit','args':{'architecture':ARCH,'heads':['epsilon','residual'],'time_scale':1.}})
    nonzero(model)
    native=copy.deepcopy(model.core)
    for resolution in (8,16,32):
        model.zero_grad();native.zero_grad()
        x=torch.randn(2,4,resolution,resolution);t=torch.tensor([12.,653.]);y=torch.tensor([9,23])
        torch.manual_seed(99);out=model(x,t,y=y)
        torch.manual_seed(99);ref=native(x,t,y)
        assert torch.equal(torch.cat([out['epsilon'],out['residual']],1),ref)
        torch.cat([out['epsilon'],out['residual']],1).square().mean().backward()
        ref.square().mean().backward()
        for a,b in zip(model.core.parameters(),native.parameters()):
            assert (a.grad is None)==(b.grad is None)
            if a.grad is not None:assert torch.equal(a.grad,b.grad)


def test_product_mask_blocks_inactive_band_and_retains_joint_gradients():
    model=build('model',{'name':'comparison_dit','args':{'architecture':ARCH,'channels':[4,4]}}).eval()
    nonzero(model)
    x={0:torch.randn(2,4,8,8,requires_grad=True),1:torch.randn(2,4,16,16,requires_grad=True)}
    t=torch.tensor([[.4,.8],[.3,.7]]);y=torch.tensor([0,1])
    out=model(x,t,y=y,drop_mask=torch.tensor([1.,0.]))
    changed=model({0:x[0],1:x[1]+17},t,y=y,drop_mask=torch.tensor([1.,0.]))
    assert torch.equal(out[0],changed[0])
    out[0].square().mean().backward()
    assert x[0].grad.abs().sum()>0 and x[1].grad.abs().sum()==0
    active=model(x,t,y=y,drop_mask=torch.ones(2))
    grad=torch.autograd.grad(active[0].sum(),x[1])[0]
    assert grad.abs().sum()>0


def test_causal_lapflow_and_haar_ports():
    model=build('model',{'name':'comparison_dit','args':{'architecture':ARCH,'channels':[4,4],'product_order':'fine_first','causal_scales':True}}).eval()
    nonzero(model)
    fine,coarse=torch.randn(2,4,16,16),torch.randn(2,4,8,8)
    t=torch.full((2,),.7);y=torch.tensor([1,2])
    a=model([fine,coarse],t,y=y);b=model([fine+1,coarse],t,y=y)
    assert torch.equal(a[1],b[1]) and not torch.equal(a[0],b[0])
    haar=build('model',{'name':'comparison_dit','args':{'architecture':ARCH,'channels':[4,12,12]}})
    values={0:coarse,1:torch.randn(2,12,8,8),2:torch.randn(2,12,16,16)}
    out=haar(values,torch.full((2,3),.4),y=y)
    assert all(out[k].shape==v.shape for k,v in values.items())
    sum(v.square().mean() for v in out.values()).backward()


def test_pixelflow_custom_partition_keeps_default_and_changes_only_requested_clock():
    a=PixelFlowClock();b=PixelFlowClock(stage_range=[0,.25,.5,.75,1])
    assert a.starts==b.starts and a.ends==b.ends
    assert all(torch.equal(x,y) for x,y in zip(a.network_times,b.network_times))
    c=PixelFlowClock(stages=3,stage_range=[0,.4,.7,1])
    assert abs(c.ends[0]-.4)<1e-7 and abs(c.ends[1]-.7)<1e-7
    # It remains a renormalized PixelFlow path, not a forced continuous sigma.
    assert c.starts[1]<c.ends[0]
