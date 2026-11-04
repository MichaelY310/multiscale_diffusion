#!/usr/bin/env python3
"""Compare actual native loss bodies to the configured unified training engine."""
import argparse,json,os,sys,time,traceback,random
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'_deps')]
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import torch
import torch.nn.functional as F
import conformance
from umdiff import native,build
from umdiff.engine import Method
from umdiff.randomness import GlobalRandomStream
compare=conformance.compare
STREAM=GlobalRandomStream()
AUDIT_FACTORY=None
CURRENT_AUDIT=None
CHECKPOINT_ROOT=ROOT/'runs'/'native_parity'

def seed(value):torch.manual_seed(value);np.random.seed(value);random.seed(value)

def verify(name,model,original,port,steps,clip=None):
    """Independent native vs port loss, backward, AdamW and EMA trajectories."""
    global CURRENT_AUDIT
    a=deepcopy(model);b=deepcopy(model);ema_a=deepcopy(model);ema_b=deepcopy(model)
    oa=torch.optim.AdamW(a.parameters(),lr=1e-4,betas=(.9,.99),weight_decay=.01)
    ob=torch.optim.AdamW(b.parameters(),lr=1e-4,betas=(.9,.99),weight_decay=.01)
    audit=AUDIT_FACTORY(name,[('network',a,b)]) if AUDIT_FACTORY else None
    CURRENT_AUDIT=audit
    losses=[]
    for step in range(steps):
        a.train();b.train();oa.zero_grad(set_to_none=True);ob.zero_grad(set_to_none=True)
        if audit:audit.begin(step);audit.side('native')
        seed(1100+step);la=original(a,step)
        if audit:audit.side('unified')
        seed(1100+step);lb=port(b,step)
        compare(f'{name}/step{step}/native_vs_unified_loss',lb.mean(),la.mean(),atol=4e-6,rtol=5e-5,scope='native_full_training_loss')
        if audit:audit.side('native','backward_recompute')
        la.mean().backward()
        if audit:audit.side('unified','backward_recompute')
        lb.mean().backward()
        if audit:audit.parameters(a,b);audit.finish_segment()
        max_gradient=0.
        for (key,pa),(_,pb) in zip(a.named_parameters(),b.named_parameters()):
            if (pa.grad is None)!=(pb.grad is None):raise AssertionError(f'Gradient presence differs: {key}')
            if pa.grad is not None:
                compare(f'{name}/step{step}/gradient/{key}',pb.grad,pa.grad,atol=1e-5,rtol=5e-4,scope='native_full_training_gradient')
                max_gradient=max(max_gradient,float((pb.grad-pa.grad).abs().max()))
        if clip is not None:
            na=torch.nn.utils.clip_grad_norm_(a.parameters(),clip);nb=torch.nn.utils.clip_grad_norm_(b.parameters(),clip)
            compare(f'{name}/step{step}/clip_norm',nb,na,scope='native_full_training_optimizer')
        oa.step();ob.step()
        with torch.no_grad():
            for ea,eb,pa,pb in zip(ema_a.parameters(),ema_b.parameters(),a.parameters(),b.parameters()):
                ea.mul_(.99).add_(pa,alpha=.01);eb.mul_(.99).add_(pb,alpha=.01)
            for (key,pa),(_,pb) in zip(a.named_parameters(),b.named_parameters()):compare(f'{name}/step{step}/updated_weight/{key}',pb,pa,atol=2e-5,rtol=5e-4,scope='native_full_training_optimizer')
            for (key,pa),(_,pb) in zip(ema_a.named_parameters(),ema_b.named_parameters()):compare(f'{name}/step{step}/ema/{key}',pb,pa,atol=5e-6,rtol=5e-4,scope='native_full_training_ema')
        losses.append({'step':step,'native_loss':la.mean().item(),'unified_loss':lb.mean().item(),'max_gradient_abs':max_gradient})
    if audit:audit.close()
    CURRENT_AUDIT=None
    artifact=CHECKPOINT_ROOT/name.replace('/','_');artifact.mkdir(parents=True,exist_ok=True)
    for label,net,ema,opt in [('native',a,ema_a,oa),('unified',b,ema_b,ob)]:
        torch.save({'model':net.state_dict(),'ema':ema.state_dict(),'optimizer':opt.state_dict(),'steps':steps},artifact/f'{label}.pt')
    return a,b,{'steps':steps,'parameters':sum(p.numel() for p in model.parameters()),'backbone':type(model).__module__+'.'+type(model).__name__,'losses':losses,'optimizer':'AdamW(lr=1e-4,betas=(.9,.99),weight_decay=.01), shared controlled optimizer for operator conformance','ema_decay':.99,'locally_trained_checkpoint_pair':str(artifact.relative_to(ROOT))}

def nonzero_output(model):
    # Test a nontrivial common initial state; do not pass only because all final
    # heads are zero. This is a synthetic initial checkpoint, not paper weights.
    with torch.no_grad():
        for name,p in model.named_parameters():
            if p.ndim>=2 and not torch.any(p) and ('final' in name or 'proj_out' in name or 'out' in name):p.normal_(std=.01)
    return model

def run_relay(steps):
    import scipy.stats as stats
    blur=native.load('relay/blurring.py')
    lossenv=native.extract('relay/training/loss.py',['DCTBlur','BlurLoss'],dict(torch=torch,np=np,st=stats,persistence=SimpleNamespace(persistent_class=lambda cls:cls),block_noise=blur.block_noise,dct_2d=blur.dct_2d,idct_2d=blur.idct_2d))
    model=build('model',{'name':'native_relay','args':{'img_resolution':16,'img_channels':3,'model_channels':64,'channel_mult':[1,2],'num_blocks':1,'dropout':0.,'attn_resolutions':[8]}})
    model=nonzero_output(model);images=torch.randn(2,3,16,16)
    objective=lossenv['BlurLoss'](up_scale=4,block_scale=.15)
    method=Method({'name':'relay_native','batch':{'name':'relay','args':{'patch':4,'block_scale':.15}},'target':{'name':'joint','args':{'quantity':'clean'}}})
    def original(net,step):return objective(net,images)
    def port(net,step):
        u=torch.rand(2,1,1,1)
        return method.training(net,images,u.flatten(),STREAM).loss[:,None,None,None].expand_as(images)
    # Native returns per-pixel loss; reduction must happen identically before
    # comparison (fp64 sigma weights).
    def native_reduced(net,step):return original(net,step).flatten(1).mean(1)
    def port_reduced(net,step):return method.training(net,images,torch.rand(2),STREAM).loss
    a,b,record=verify('relay',model,native_reduced,port_reduced,steps)
    from umdiff.relay import RelaySampler
    env=native.extract('relay/generate_imagenet.py',['blur_sampler'],dict(torch=torch,np=np,block_noise=blur.block_noise,dct_2d=blur.dct_2d,idct_2d=blur.idct_2d,get_alpha_t=lambda t,patch,device,**kw:blur.get_alpha_t(t,patch,device,repeat_dims=16//patch,**kw)))
    initial=torch.randn(2,3,16,16)
    seed(1301);expected=env['blur_sampler'](a.eval(),initial,num_steps=12,sigma_max=2.,truncation_sigma=1.,up_scale=4,s_block=.15,s_noise=.2)
    seed(1301);actual=RelaySampler(steps=12,sigma_max=2.,truncation_sigma=1.)(b.eval(),initial)
    compare('relay/trained_model_final_pixels',actual,expected,atol=1e-5,rtol=1e-4,scope='native_trained_model_inference_pixels')
    record['native_compatibility']='get_alpha_t repeat_dims=H/patch for the small fixture; original default 64 assumes 256x256 with patch4. Persistence decorator omitted, xformers unused.'
    return record

def run_dctdiff(steps):
    source=native.load('dctdiff/sde.py')
    model=build('model',{'name':'native_dctdiff','args':{'embed_dim':32,'depth':2,'num_heads':4,'tokens':16,'low_freqs':4,'num_classes':-1}})
    clean=torch.randn(3,16,24)
    from umdiff.paths import VP
    method=Method({'name':'dctdiff_vp','path':{'name':'vp','args':{'snr_scale':4.}},'target':{'name':'joint','args':{'quantity':'epsilon'}},'clock':{'name':'linear','args':{'network_scale':999.}}})
    def original(net,step):return source.LSimple(source.ScoreModel(net,'noise_pred',source.VPSDE(SNR_scale=4.)),clean,reweight=1.)
    def port(net,step):
        t=torch.rand(3);return method.training(net,clean,t,STREAM).loss
    a,b,record=verify('dctdiff',model,original,port,steps)
    # Actual native DPM-Solver runs on both independently trained weights; the
    # solver reuse is explicit, while VP/path/training are independently ported.
    solver=native.load('dctdiff/dpm_solver_pytorch.py')
    schedule=solver.NoiseScheduleVP(schedule='linear',SNR_scale=4.)
    score=source.ScoreModel(a.eval(),'noise_pred',source.VPSDE(SNR_scale=4.))
    fn=solver.model_wrapper(score.noise_pred,schedule,time_input_type='0',model_kwargs={})
    initial=torch.randn(3,16,24)
    expected=solver.DPM_Solver(fn,schedule).sample(initial,steps=12,eps=1e-4,adaptive_step_size=False,fast_version=True)
    from umdiff.samplers import DCTDPMSampler
    actual=DCTDPMSampler(steps=12,snr_scale=4.)(b.eval(),initial)
    compare('dctdiff/trained_model_tokens',actual,expected,atol=1e-5,rtol=1e-4,scope='native_trained_model_inference_tokens')
    from umdiff.dctcodec import DCTYCbCr
    codec=DCTYCbCr(32,4,4,256.)
    compare('dctdiff/trained_model_decoded_pixels',codec.decode(actual),codec.decode(expected),atol=1.,rtol=0.,scope='native_trained_model_inference_pixels')
    record['sampler_reuse']='Author DPM-Solver backend reused explicitly; trained weights, path/loss assembly independent; token and decoded uint8 pixel comparison.'
    return record

def run_lapflow(steps):
    from umdiff.native_models import activate
    activate('lapflow',['models','sampler'])
    from sampler.ours_sample import generate_xt_and_u
    spec={'name':'adapter','args':{'time_first':True,'model':{'name':'native_lapflow','args':{'input_size':[16,8],'patch_size':2,'in_channels':4,'hidden_size':32,'depth':2,'num_heads':4,'class_dropout_prob':0.,'use_mot':True,'use_mask':True}}}}
    model=nonzero_output(build('model',spec));images=torch.randn(3,4,16,16);labels=torch.tensor([0,1,2])
    args=SimpleNamespace(noise_var_factor=None,path_type='linear',use_gvp=False,epsilon=None,variance_aware=False,use_mot=True,loss_weight_scale_0=1.,loss_weight_scale_1=1.,num_levels=2)
    method=Method({'name':'lapflow_native','batch':{'name':'lapflow','args':{'levels':2}}})
    t=torch.tensor([.57,.71,.89])
    def original(net,step):
        u,xt,weights,var=generate_xt_and_u([.3,.6],.4,images,t[:,None,None,None],0,2,False,args)
        # Execute the actual source loop loss, with source model wrapped only to
        # normalize its (time,state) call signature.
        class NativeCall:
            def __call__(self,time,x,var_scale=None,y=None):return net(x,time,y=y,var_scale=var_scale)
        env=native.statements('lapflow/feb12_train_ours.py',415,427,dict(training_model=NativeCall(),t=t[:,None,None,None],x_t=xt,u=u,loss_scales=weights,var_scale=var,y=labels,args=args,scale_selector=0,F=F,epoch=0))
        return env['loss']
    def port(net,step):return method.training(net,images,t,STREAM,model_kwargs={'y':labels}).loss
    a,b,record=verify('lapflow',model,original,port,steps,clip=1.)
    from sampler import ours_sample as sampling
    from unittest.mock import patch
    from umdiff.samplers import LapFlowSampler
    args.__dict__.update(method='euler',step_size=.1,perturb=False,compute_nfe=False,cfg_scale=1.,num_levels=2,time_seg_small=.4,time_segments=[.3,.6],atol=1e-5,rtol=1e-5)
    initial=torch.randn(3,4,16,16);original_tensor=torch.tensor
    def cpu_tensor(*args,**kwargs):
        if kwargs.get('device')=='cuda':kwargs['device']='cpu'
        return original_tensor(*args,**kwargs)
    class Call(torch.nn.Module):
        def __init__(self,net):super().__init__();self.net=net
        def forward(self,t,x,**kwargs):return tuple(self.net(x,t,**kwargs))
    with patch.object(sampling.torch,'tensor',cpu_tensor):expected=sampling.ours_sample_from_model([Call(a.eval())],0,initial,{'y':labels},args,False,largest_only=True)
    actual=LapFlowSampler(step_size=.1)(b.eval(),initial,STREAM,{'y':labels})
    compare('lapflow/trained_model_final_pixels',actual,expected,atol=2e-5,rtol=2e-4,scope='native_trained_model_inference_pixels')
    record['native_compatibility']='Preserves released MoT final-head index behavior; original time-first signature adapted only.'
    return record

def run_dfm(steps,dropout=0.):
    from umdiff.native_models import activate
    activate('dfm',['models','dfm_utils','loss'])
    import loss as author
    spec={'name':'adapter','args':{'output_index':0,'model':{'name':'native_dfm','args':{'input_size':16,'in_channels':4,'hidden_size':32,'decoder_hidden_size':32,'encoder_depth':1,'depth':2,'num_heads':4,'class_dropout_prob':dropout,'enable_repa':False,'qk_norm':False}}}}
    model=nonzero_output(build('model',spec));images=torch.randn(3,4,16,16);labels=torch.tensor([0,1,2])
    method=Method({'name':'dfm_native','batch':{'name':'dfm'}})
    objective=author.DFMSILoss(stage_weights=[.9,.1])
    def original(net,step):
        def wrapper(x,t,**kwargs):return net(x,t,**kwargs),None
        return objective(wrapper,images,model_kwargs={'y':labels})[0]
    def port(net,step):
        stage=int(np.random.choice(2,p=[.9,.1]));times=[]
        for i in range(stage+1):times.append(torch.sigmoid(torch.randn(3,1,1,1)-2) if i<stage else torch.rand(3,1,1,1))
        times.extend([torch.ones(3,1,1,1) for _ in range(1-stage)])
        t=torch.stack([v.flatten() for v in times],-1)
        return method.training(net,images,t,STREAM,model_kwargs={'y':labels},context={'stage':stage}).loss
    a,b,record=verify('dfm',model,original,port,steps)
    from dfm_utils.samplers_dfm import dfm_euler_sampler
    from umdiff.representations import Laplacian
    rep=Laplacian(2,up='bilinear',order='coarse_first');initial=dict(enumerate(torch.randn_like(x) for x in rep.encode(images)))
    expected=dfm_euler_sampler(lambda x,t,**kwargs:(a.eval()(x,t,**kwargs),None),initial,labels,num_steps_per_scale=[4,4],stage_thresholds=[.5])
    sample_method=Method({'name':'dfm','representation':{'name':'laplacian','args':{'levels':2,'up':'bilinear','order':'coarse_first'}},'clock':{'name':'dfm'}})
    actual=sample_method.sample(b.eval(),initial,model_kwargs={'y':labels})
    compare('dfm/trained_model_final_pixels',actual,rep.decode(expected),atol=1e-5,rtol=1e-4,scope='native_trained_model_inference_pixels')
    repa_spec={'name':'native_dfm','args':{**spec['args']['model']['args'],'enable_repa':True,'z_dims':[8],'projector_dim':32}}
    repa_model=nonzero_output(build('model',repa_spec));features=[torch.randn(3,64,8)]
    repa_method=Method({'name':'dfm_repa','batch':{'name':'dfm','args':{'repa_weight':.3}}})
    def original_repa(net,step):
        denoising,projection=objective(net,images,model_kwargs={'y':labels},zs=features)
        return denoising.mean()+projection.mean()*.3
    def port_repa(net,step):
        stage=int(np.random.choice(2,p=[.9,.1]));times=[]
        for i in range(stage+1):times.append(torch.sigmoid(torch.randn(3,1,1,1)-2) if i<stage else torch.rand(3,1,1,1))
        times.extend([torch.ones(3,1,1,1) for _ in range(1-stage)])
        t=torch.stack([v.flatten() for v in times],-1)
        return repa_method.training(net,images,t,STREAM,model_kwargs={'y':labels},context={'stage':stage,'teacher_features':features}).loss
    ra,rb,repa_record=verify('dfm_repa',repa_model,original_repa,port_repa,steps)
    repa_expected=dfm_euler_sampler(ra.eval(),initial,labels,num_steps_per_scale=[4,4],stage_thresholds=[.5])
    repa_actual=sample_method.sample(rb.eval(),initial,model_kwargs={'y':labels})
    compare('dfm_repa/trained_model_final_pixels',repa_actual,rep.decode(repa_expected),atol=1e-5,rtol=1e-4,scope='native_trained_model_inference_pixels')
    record['repa_variant']=repa_record
    record['repa_teacher_scope']='Synthetic local teacher feature tensors; author cosine projection objective and model projector gradients executed, no pretrained DINO download or paper representation-quality claim.'
    record['native_compatibility']='Native README calls this released SiT/REPA reimplementation architecture different from paper; debugging np.random logging draws affect later global stage RNG unless explicit stage/noise inputs are coupled.'
    return record

RUNNERS={'relay':run_relay,'dctdiff':run_dctdiff,'lapflow':run_lapflow,'dfm':run_dfm}
import validate_training_more
RUNNERS.update(validate_training_more.install(verify,nonzero_output,compare))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('method',choices=RUNNERS);parser.add_argument('--steps',type=int,default=12);parser.add_argument('--seed',type=int,default=20261008);parser.add_argument('--output');args=parser.parse_args()
    if args.steps<1:parser.error('steps must be positive')
    torch.set_num_threads(2);seed(args.seed);started=time.time();failure=None;details={};status='passed'
    try:details=RUNNERS[args.method](args.steps)
    except Exception:status='failed';failure=traceback.format_exc();print(failure,file=sys.stderr)
    report={'method':args.method,'status':status,'device':'cpu','gpu_used':False,'initialization_seed':args.seed,'step_noise_seeds':'1100 + step','seconds':time.time()-started,'details':details,'checks':conformance.CHECKS,'source_executions':native.AUDIT,'failure':failure,'quality_claim':'Conditional synthetic training/operator and sampling conformance, not a real-data image quality/FID result.'}
    path=Path(args.output) if args.output else ROOT/'evidence'/f'{args.method}_training.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2))
    print(json.dumps({'method':args.method,'status':status,'checks':len(conformance.CHECKS),'seconds':report['seconds'],'report':str(path)}))
    return status!='passed'
if __name__=='__main__':raise SystemExit(main())
