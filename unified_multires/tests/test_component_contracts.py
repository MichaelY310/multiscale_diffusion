import pytest
import torch
from umdiff import build
from umdiff.paths import Flow,RDDFlow,EndpointFlow
from umdiff.targets import Joint,DecoupledClean
from umdiff.engine import Method
from umdiff.operators import Projection,BilinearOperator
from umdiff.noise import coupled_lift

@pytest.mark.parametrize('path',[Flow(),Flow('data_to_noise'),Flow(curve='cosine'),RDDFlow(),EndpointFlow(.25,.5)])
def test_velocity_is_path_derivative(path):
    torch.manual_seed(123);x=torch.randn(2,3,8,8,dtype=torch.float64);eps=torch.randn_like(x);t=torch.tensor([.35,.6],dtype=x.dtype);h=1e-6
    actual=path.sample(x,t,eps);finite=(path.sample(x,t+h,eps).state-path.sample(x,t-h,eps).state)/(2*h)
    torch.testing.assert_close(actual.velocity,finite,atol=2e-9,rtol=1e-8)

@pytest.mark.parametrize('quantity',['epsilon','effective_epsilon','clean','effective_clean','v','velocity'])
def test_parameterization_inverse(quantity):
    torch.manual_seed(1);x=torch.randn(2,3,8,8,dtype=torch.float64);eps=torch.randn_like(x);t=torch.tensor([.25,.75],dtype=x.dtype);path=RDDFlow();target=Joint(quantity)
    sample=path.sample(x,t,eps);restored=target.clean(target.encode(sample),sample.state,t,path)
    torch.testing.assert_close(restored,x,atol=1e-12,rtol=1e-12)

def test_decoupled_sampling_uses_band_drift_derivative():
    x=torch.randn(2,3,8,8);eps=torch.randn_like(x);t=torch.tensor([.3,.7]);path=RDDFlow();out=path.sample(x,t,eps)
    method=Method({'path':{'name':'rdd_flow'},'target':{'name':'decoupled_clean'}})
    actual=method._velocity(DecoupledClean().encode(out),out.state,t)
    torch.testing.assert_close(actual,out.velocity,atol=2e-6,rtol=1e-6)

def test_zero_endpoint_has_correct_limit_and_decoder_rejects_unidentifiable_band():
    x=torch.randn(1,3,8,8);eps=torch.randn_like(x);path=RDDFlow();t=torch.zeros(1);out=path.sample(x,t,eps)
    torch.testing.assert_close(out.effective_clean,path.projection(x),atol=0,rtol=0)
    with pytest.raises(ValueError,match='identifiable'):Joint('velocity').clean(out.velocity,out.state,t,path)

def test_adjoint_identity_and_nonorthogonal_inverse_rejection():
    torch.manual_seed(2);op=BilinearOperator((8,8),(4,4));x=torch.randn(2,3,8,8,dtype=torch.float64);y=torch.randn(2,3,4,4,dtype=torch.float64)
    torch.testing.assert_close((op(x)*y).sum(),(x*op.transpose(y)).sum(),atol=2e-14,rtol=1e-12)
    path=EndpointFlow(.25,.5,projection={'down':'bilinear','up':'nearest'})
    with pytest.raises(ValueError,match='orthogonal'):path.clean_from_velocity(x,x,torch.tensor([.3,.6]))

def test_rdd_projection_inverse_rejects_nonidempotent_resize():
    torch.manual_seed(481)
    x=torch.randn(2,3,8,8,dtype=torch.float64);eps=torch.randn_like(x)
    t=torch.tensor([.45,.6],dtype=x.dtype)
    path=RDDFlow(start=.4,end=.7,projection={'down':'average','up':'bilinear'})
    assert (path.projection(path.projection(x))-path.projection(x)).abs().max()>1e-3
    sample=path.sample(x,t,eps)
    with pytest.raises(ValueError,match='orthogonal'):
        path.clean_from_velocity(sample.state,sample.velocity,t)
    # The operator is irrelevant when there is no drift.
    plain=RDDFlow(start=.4,end=.7,drift=False,projection={'up':'bilinear'})
    sample=plain.sample(x,t,eps)
    torch.testing.assert_close(plain.clean_from_velocity(sample.state,sample.velocity,t),x,atol=1e-12,rtol=1e-12)

def test_joint_coupling_has_iid_fine_covariance():
    # Check the linear map exactly, without a statistical pass/fail threshold.
    # Input consists of independent coarse, low innovation, high innovation.
    coarse=torch.eye(4,dtype=torch.float64).reshape(4,1,2,2)
    high=torch.eye(16,dtype=torch.float64).reshape(16,1,4,4)
    zeros=torch.zeros(4,1,4,4,dtype=torch.float64);coarse_zero=torch.zeros(16,1,2,2,dtype=torch.float64)
    rho=.6
    a=coupled_lift(coarse,torch.zeros_like(coarse),zeros,2,rho).flatten(1).T
    b=coupled_lift(torch.zeros_like(coarse),coarse,zeros,2,rho).flatten(1).T
    c=coupled_lift(coarse_zero,coarse_zero,high,2,rho).flatten(1).T
    transform=torch.cat([a,b,c],1)
    torch.testing.assert_close(transform@transform.T,torch.eye(16,dtype=torch.float64),atol=2e-15,rtol=0)

def test_unknown_components_and_incompatible_sampler_fail():
    with pytest.raises(KeyError):build('path',{'name':'typo'})
    with pytest.raises(TypeError):build('path',{'name':'flow','args':{'silently_ignored':1}})
    method=Method({'batch':{'name':'relay'}})
    with pytest.raises(ValueError,match='explicit compatible sampler'):method.sample(None,torch.randn(1,3,8,8))

def test_pixelflow_packing_must_match_full_network_config():
    config={'batch':{'name':'pixelflow'},'model':{'name':'native_pixelflow','args':{'patch_size':4,'attention_head_dim':72}}}
    with pytest.raises(ValueError,match='packing patch and rotary dimension'):Method(config)
    config['batch']['args']={'patch':4,'attention_head_dim':72}
    Method(config)

def test_mdm_resampling_rejects_silently_ignored_or_duplicate_step_requests():
    from umdiff.samplers import MDMSampler
    with pytest.raises(ValueError,match='requires resample_steps'):MDMSampler([1.,.7,.2],inference_steps=1)
    with pytest.raises(ValueError,match='requires 1..training_steps'):MDMSampler([1.,.7,.2],resample_steps=True,inference_steps=3)

def test_relay_nonfinite_gradient_policy_precedes_optimizer_and_is_checkpointed():
    from umdiff.training import Trainer
    from umdiff.engine import TrainingResult
    class Nonfinite(torch.autograd.Function):
        @staticmethod
        def forward(ctx,x):return x.sum()
        @staticmethod
        def backward(ctx,g):return g*torch.tensor([float('nan'),float('inf'),-float('inf'),2.])
    class Network(torch.nn.Module):
        def __init__(self):super().__init__();self.weights=torch.nn.Parameter(torch.zeros(4))
    class Objective:
        config={}
        def training(self,net,*args,**kwargs):return TrainingResult(Nonfinite.apply(net.weights),None,None,None,{})
    net=Network();policy={'nan':0.,'posinf':1e5,'neginf':-1e5}
    trainer=Trainer(Objective(),net,optimizer_name='adam',gradient_repair=policy)
    trainer.step(torch.zeros(1,1),None)
    torch.testing.assert_close(net.weights.grad,torch.tensor([0.,1e5,-1e5,2.]),atol=0,rtol=0)
    assert torch.isfinite(net.weights).all()
    assert trainer.state_dict()['recipe']['gradient_repair']==policy
    with pytest.raises(ValueError,match='non-scaled precision'):Trainer(Objective(),Network(),precision='float16',gradient_repair=policy)

def test_recipe_local_asset_names_can_contain_checkpoint_resolution(tmp_path,monkeypatch):
    import json,runpy,sys
    from pathlib import Path
    checkpoint=tmp_path/'inner.pt';checkpoint.write_bytes(b'local asset placeholder for binding test')
    recipe=tmp_path/'recipe.json';recipe.write_text(json.dumps({'model':{'path':'@mdm_inner256@'},'evidence':{'source_pins':[]}}))
    output=tmp_path/'bound.json'
    monkeypatch.setattr(sys,'argv',['render_paper_recipe',str(recipe),'--asset',f'mdm_inner256={checkpoint}','--data',str(tmp_path),'--output',str(output)])
    runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/render_paper_recipe.py'),run_name='__main__')
    assert json.loads(output.read_text())['model']['path']==str(checkpoint.resolve())

def test_packed_pixel_loss_matches_author_single_reduction_and_gradient():
    from umdiff.batches import Prepared
    from umdiff import native
    torch.manual_seed(42);lengths=[4096,1024,256,64]
    pred=torch.randn(sum(lengths),48,requires_grad=True);target=torch.randn_like(pred)
    expected=native.statements('pixelflow/train.py',120,126,{'model_output':pred,'target':target,'batch':{'seqlen_list_q':lengths},'torch':torch})['loss']
    actual=Prepared(None,target,torch.zeros(4),segment_lengths=lengths,reduction='native_packed').loss(pred)[0].mean()
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    torch.testing.assert_close(torch.autograd.grad(actual,pred)[0],torch.autograd.grad(expected,pred)[0],atol=0,rtol=0)

def test_dfm_loss_preserves_source_reduction_on_channels_last_output():
    from umdiff.batches import Prepared
    from umdiff import native
    torch.manual_seed(7);pred=torch.randn(3,4,32,32).to(memory_format=torch.channels_last).requires_grad_();target=torch.randn_like(pred)
    source=native.extract('dfm/loss.py',['mean_flat'],{'torch':torch})['mean_flat']
    expected=source((pred-target)**2)
    actual=Prepared(None,target,torch.zeros(3),reduction='native_dims').loss(pred)[0]
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    torch.testing.assert_close(torch.autograd.grad(actual.mean(),pred)[0],torch.autograd.grad(expected.mean(),pred)[0],atol=0,rtol=0)

def test_gaussian_conditioning_preserves_exact_coarse_observation():
    from umdiff.boundaries import GaussianConditional
    D=torch.tensor([[.5,.5,0.,0.],[0.,0.,.5,.5]],dtype=torch.float64)
    C=torch.eye(4,dtype=torch.float64);mean=torch.randn(3,1,2,2,dtype=C.dtype)
    observation=torch.randn(3,1,1,2,dtype=C.dtype);noise=torch.randn_like(mean)
    result=GaussianConditional()(observation,mean=mean,covariance=C,observation=D,noise=noise)
    torch.testing.assert_close(result.state.flatten(2)@D.T,observation.flatten(2),atol=2e-15,rtol=0)
    torch.testing.assert_close(D@result.metadata['covariance'],torch.zeros(2,4,dtype=C.dtype),atol=0,rtol=0)

@pytest.mark.parametrize('quantity',['velocity','epsilon','clean','v'])
def test_stage_solver_decodes_configured_parameterization(quantity):
    # Analytic clean/noise oracle: every parameterization must integrate the
    # same linear path, including through the stage-specific path/clock API.
    clean=torch.randn(2,3,8,8);eps=torch.randn_like(clean)
    method=Method({'target':{'name':'joint','args':{'quantity':quantity}},'stages':[{'path':{'name':'flow'},'clock':{'name':'linear','args':{'start':.1,'end':.9}},'steps':8}]})
    path=Flow();target=Joint(quantity);initial=path.sample(clean,torch.full((2,),.1),eps).state
    def oracle(state,time,**kwargs):return target.encode(path.sample(clean,time,eps))
    actual=method.sample_stages(oracle,initial,[])
    expected=path.sample(clean,torch.full((2,),.9),eps).state
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=1e-6)
