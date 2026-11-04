#!/usr/bin/env python3
"""Final CPU branch checks, with original functions and explicit test scopes."""
import os,sys,json,math,ast,traceback,contextlib,io,hashlib
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'_deps'),str(ROOT/'tools')]
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1'
import numpy as np
import torch
import torch.nn.functional as F
import conformance,validate_training as training
from umdiff import build,native
from umdiff.randomness import RandomStream,GlobalRandomStream
C=conformance.compare;OUT=ROOT/'evidence/recipe_extensions';RUN=ROOT/'runs/recipe_extensions/auxiliary';RUN.mkdir(parents=True,exist_ok=True)

def dct():
    import cv2
    from umdiff.samplers import DCTEulerMaruyama
    from umdiff.dctcodec import DCTBlockUpsample
    from prepare_dct_statistics import prepare
    model=build('model',{'name':'native_dctdiff','args':{'embed_dim':32,'depth':2,'num_heads':4,'tokens':16,'low_freqs':4,'num_classes':-1}})
    model.load_state_dict(torch.load(ROOT/'runs/recipe_extensions/native_pairs/dctdiff_EBFR/unified.pt',weights_only=False)['model']);model.eval()
    source=native.load('dctdiff/sde.py');initial=torch.randn(2,16,24)
    for stochastic in (False,True):
        training.seed(303);stream=RandomStream(981);outer=torch.get_rng_state().clone();expected_trace=[]
        with stream.native_rng('cpu'):
            score=source.ScoreModel(model,'noise_pred',source.VPSDE(SNR_scale=4.))
            expected=source.euler_maruyama((source.ReverseSDE if stochastic else source.ODE)(score),initial,12,trace=expected_trace)
        expected_rng=stream.generator.get_state().clone();C('dct/native_scope_restores_outer_rng',torch.get_rng_state(),outer,atol=0,rtol=0)
        stream=RandomStream(981);actual,trace=DCTEulerMaruyama(12,snr_scale=4.,stochastic=stochastic)(model,initial,stream,return_trajectory=True)
        C('dct/EM_'+str(stochastic)+'/trace',trace,expected_trace,atol=0,rtol=0);C('dct/EM_'+str(stochastic)+'/final',actual,expected,atol=0,rtol=0);C('dct/EM_rng',stream.generator.get_state(),expected_rng,atol=0,rtol=0)
    env=native.extract('dctdiff/DCT_analysis_upsampling.py',['split_into_blocks','combine_blocks','dct_transform','idct_transform','DCT_upsampling'],dict(np=np,cv2=cv2,math=math))
    pixels=torch.randint(0,256,(2,3,8,12),dtype=torch.uint8)
    expected=torch.from_numpy(np.stack([env['DCT_upsampling'](x,block_sz=4) for x in pixels.permute(0,2,3,1).numpy()])).permute(0,3,1,2)
    C('dct/auxiliary_block_upsampling_pixels',DCTBlockUpsample()(pixels).state,expected,atol=0,rtol=0)
    paths=[]
    for index in range(3):
        path=RUN/f'dct_coefficients_{index}.npy';np.save(path,np.random.default_rng(index).normal(size=(1000,4))*[120,40,30,15]);paths.append(path)
    stats=prepare(paths,2);eta=np.round(max(abs(np.percentile(np.load(paths[0])[:,0],1.75)),abs(np.percentile(np.load(paths[0])[:,0],98.25))),3)
    C('dct/statistics_bound',np.asarray(stats['Y_bound'],dtype=np.float64),np.asarray([eta],dtype=np.float64),atol=0,rtol=0)
    for key,path in zip(('Y_std','Cb_std','Cr_std'),paths):
        values=[]
        for data in np.load(path).T:
            low,high=np.percentile(data,[1.75,98.25]);counts,_=np.histogram(data[(data>low)&(data<high)]/eta,bins=100,range=(-1,1));probability=counts/counts.sum();values.append(np.around(-np.sum(probability*np.log2(probability+1e-9)),3))
        C('dct/native_entropy_'+key,np.asarray(stats[key],dtype=np.float64),np.asarray(values,dtype=np.float64),atol=0,rtol=0)
    return {'samplers':'12-step ODE/SDE: full trajectories and RNG state exact on the 128-step trained reduced UViT','upsampler':'actual author RGB uint8 auxiliary, not a diffusion bridge','statistics':'native percentile bound and histogram entropy exact on local synthetic coefficient arrays'}

def codec():
    import einops
    from typing import Optional
    from umdiff.codec_training import CodecMethod
    from umdiff.input_transforms import LocalVAE
    from umdiff.regularizers import native_dct_utilities
    args={'block_out_channels':[32,32],'down_block_types':['DownEncoderBlock2D']*2,'up_block_types':['UpDecoderBlock2D']*2,'latent_channels':4,'norm_num_groups':32,'layers_per_block':1,'mid_block_add_attention':False}
    model=build('model',{'name':'autoencoder_kl','args':args});images=torch.randn(2,3,16,16)
    transform=LocalVAE({'name':'autoencoder_kl','args':args});transform.model.load_state_dict(model.state_dict())
    with torch.no_grad():
        for stream_type in (GlobalRandomStream,RandomStream):
            training.seed(141);stream=stream_type() if stream_type is GlobalRandomStream else stream_type(141)
            expected=model.encode(images).latent_dist.sample(generator=stream.for_device('cpu'))*.18215
            training.seed(141);stream=stream_type() if stream_type is GlobalRandomStream else stream_type(141)
            C('codec/local_VAE_'+stream_type.__name__,transform(images,stream),expected,atol=0,rtol=0)
        from umdiff.cli import decode
        model.save_pretrained(RUN/'local_vae_fixture',safe_serialization=True);latents=torch.randn(2,4,8,8);torch.save(latents,RUN/'latents.pt')
        decode(SimpleNamespace(vae=str(RUN/'local_vae_fixture'),latents=str(RUN/'latents.pt'),scale=.18215,output=str(RUN/'pixels.pt'),device='cpu'))
        C('codec/local_decoder_pixels',torch.load(RUN/'pixels.pt',weights_only=True),model.decode(latents/.18215,return_dict=False)[0],atol=0,rtol=0)
    weights=native.extract('diffusability/src/utils/training_utils.py',['compute_scheduled_weight','compute_annealed_weight'],{})
    kl=native.extract('diffusability/src/utils/autoencoder_utils.py',['compute_kl'],dict(torch=torch,math=math,Optional=Optional))['compute_kl']
    r1=native.extract('diffusability/src/training/loss.py',['compute_r1_reg'],dict(torch=torch,einops=einops))['compute_r1_reg']
    utilities=native_dct_utilities();scope=[]
    for policy in ('scheduled','annealed'):
        settings={'reconstruction_style':'native','mse_weight':.7,'mae_weight':.2,'pseudo_huber_weight':.1,'pseudo_huber_breadth':.01,'kl_weight':0. if policy=='scheduled' else .01,'decoder_r1_weight':.001,'decoder_r1_frequency':2,'high_frequency':{'weight':.2,'block_size':4,'power':2}}
        if policy=='scheduled':settings['kl_schedule']=[[1,0.],[3,.001],[10,.01]]
        else:settings['kl_anneal_steps']=3
        method=CodecMethod({'task':'codec','codec':settings});a=deepcopy(model);b=deepcopy(model)
        for step in (0,1,2,3,10):
            a.zero_grad(set_to_none=True);b.zero_grad(set_to_none=True);training.seed(178+step)
            posterior=a.encode(images).latent_dist;mean=posterior.mean.contiguous();logvar=posterior.logvar.contiguous();z=mean+torch.randn_like(mean)*torch.exp(.5*logvar);z5=z[:,None];pred=a.decode(z5[:,0],return_dict=False)[0].contiguous();diff=pred[:,None]-images[:,None]
            expected=.7*diff.square().flatten(1).mean(1,keepdim=True)+.2*diff.abs().flatten(1).mean(1,keepdim=True)+.1*(diff.square()+.01**2).sqrt().sub(.01).flatten(1).mean(1,keepdim=True)
            weight=weights['compute_scheduled_weight'](step,settings['kl_schedule']) if policy=='scheduled' else weights['compute_annealed_weight'](step,0.,.01,3)
            if weight:expected=expected+weight*kl(mean[:,None],logvar[:,None],None,None,z[:,None])[0]
            if step%2==0:expected=expected+.002*r1(z5,pred,reduce_op='mean')
            expected=expected+.2*utilities['reg_dc_dct2d_high_freqs'](z[:,None],block_size=4,power=2)
            expected.mean().backward();training.seed(178+step);actual=method.training(b,images,None,GlobalRandomStream(),context={'step':step});actual.loss.mean().backward()
            C('codec/'+policy+'/loss/'+str(step),actual.loss,expected,atol=0,rtol=0)
            for (name,p),(other,q) in zip(a.named_parameters(),b.named_parameters()):
                if p.grad is not None:C('codec/'+policy+'/gradient/'+str(step)+'/'+name,q.grad,p.grad,atol=0,rtol=0)
        scope.append(policy)
    return {'policies':scope,'steps_checked':[0,1,2,3,10],'scope':'Actual reduced attention-free AutoencoderKL. Native KL/schedule/annealing, R1 and high-frequency primitives; combined objective loss/gradients. Source full separate R1 optimizer phase and temporal codecs not certified.'}

def rdd():
    from umdiff.rdd_native import NativeRDDStages
    from diffusion.gaussian_diffusion_with_drift import _extract_into_tensor
    from diffusion.stage_transition import lift_noise_with_nullspace_2d
    file=ROOT.parent/'2d/sample_fid_unified.py';tree=ast.parse(file.read_text());names={'_redraw_upsample','_noise_filling_upsample','_run_stage'};env=dict(torch=torch,_extract_into_tensor=_extract_into_tensor,lift_noise_with_nullspace_2d=lift_noise_with_nullspace_2d)
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),str(file),'exec'),env)
    model=build('model',{'name':'native_rdd_unified','args':{'patch_size':2,'in_channels':4,'hidden_size':32,'depth':2,'num_heads':4,'num_classes':1000,'base_size':8}})
    model.load_state_dict(torch.load(ROOT/'runs/recipe_extensions/native_pairs/rdd_multires_training_mapped_time/unified.pt',weights_only=False)['model']);model.eval();initial=torch.randn(3,4,8,8);labels=torch.arange(3)
    tested=[]
    for boundary in ('fresh','isometric'):
        for solver in ('ddpm','dpm1','dpm2','ddim','ancestral'):
            sampler=NativeRDDStages(resolutions=[16,8],stage_lengths=[8,8],steps=[4,4],noise_schedule='squaredcos_cap_v2',solver=solver,boundary=boundary,guidance_scale=2.)
            training.seed(371);state=torch.cat([initial,torch.randn_like(initial)]);coarse=initial;trace=[];kw={'y':torch.cat([labels,torch.full_like(labels,1000)]),'cfg_scale':2.}
            for position,index in enumerate((1,0)):
                diffusion=sampler.diffusions[index]
                if position:
                    t=torch.full((3,),diffusion.num_timesteps-1,dtype=torch.long)
                    if boundary=='fresh':state,coarse=env['_redraw_upsample'](out['pred_xstart'][:3].float(),t,diffusion,16)
                    else:state,coarse=env['_noise_filling_upsample'](out['pred_xstart'][:3].float(),coarse,t,diffusion,16)
                    state=torch.cat([state,state]);trace.append(state[:3])
                if solver in ('ddpm','dpm1','dpm2'):out=env['_run_stage'](diffusion,model.forward_with_cfg,state,kw,'cpu',solver)
                else:
                    common=dict(noise=state,clip_denoised=False,model_kwargs=kw,device='cpu',progress=False)
                    out=diffusion.ddim_sample_loop(model.forward_with_cfg,state.shape,eta=0.,**common) if solver=='ddim' else diffusion.p_sample_loop(model.forward_with_cfg,state.shape,**common)
                state=out['sample'];trace.append(state[:3].clone())
            training.seed(371);actual,observed=sampler(model,initial,GlobalRandomStream(),{'y':labels},True)
            C('rdd/'+boundary+'/'+solver+'/trace',observed,trace,atol=0,rtol=0);C('rdd/'+boundary+'/'+solver+'/final',actual,state[:3].float(),atol=0,rtol=0);tested.append([boundary,solver])
    return {'cases':tested,'scope':'Native RDD source stage/transition helpers. DDPM label preserves source jump dispatcher; ancestral/DDIM explicitly reuse native solver backends.'}

def runtime_contracts():
    from umdiff.training import Trainer,data_binding
    from umdiff.engine import Method
    from umdiff.cli import read_config,TensorShards
    model=build('model',{'name':'native_mdm','args':{'config':{'resolution_channels':'32,32','attention_levels':'','num_resnets_per_resolution':'1','conditioning_feature_dim':8,'resnet_config':{'num_groups_norm':16,'use_attention_ffn':True},'inner_config':{'resolution_channels':'32','attention_levels':'','num_resnets_per_resolution':'1','conditioning_feature_dim':8,'nesting':True,'resnet_config':{'num_groups_norm':16}}}}})
    x=torch.randn(3,4,16,16);model([x,F.avg_pool2d(x,2)],torch.rand(3),conditioning=torch.randn(3,5,8),cond_mask=torch.ones(3,5),micros={})
    assert model.config.resnet_config.num_groups_norm==16
    controlled=build('model',{'name':'tiny_conv','args':{'channels':3,'width':8}});method=Method({'path':{'name':'flow'}});trainer=Trainer(method,controlled,accumulation_steps=2)
    images=torch.randn(4,3,8,8);result=trainer.step(images,torch.rand(4),noise=torch.randn_like(images));assert len(result.prediction)==4 and len(result.target)==4 and len(result.state)==4 and len(result.microbatches)==2
    trainer.data_binding=data_binding(None,4,0,synthetic=True);checkpoint=RUN/'binding.pt';trainer.save(checkpoint);trainer.load(checkpoint);trainer.data_binding=data_binding(None,2,0,synthetic=True)
    try:trainer.load(checkpoint)
    except ValueError as exc:assert 'data identity' in str(exc)
    else:raise AssertionError('Different batch resume accepted')
    path=RUN/'invalid_config.json';path.write_text(json.dumps({'training':{'precision':'float16'}}))
    try:read_config(path)
    except ValueError as exc:assert 'training.optimizer' in str(exc)
    else:raise AssertionError('Ignored precision accepted')
    weighted_model=build('model',{'name':'tiny_conv','args':{'channels':3,'width':8}});reference=deepcopy(weighted_model);weighted=Trainer(method,weighted_model,accumulation_steps=2,loss_reduction='normalized_weights',lr=.001,ema_decay=.999)
    opt=torch.optim.AdamW(reference.parameters(),lr=.001,weight_decay=0.);times=torch.rand(4);noise=torch.randn_like(images);weights=torch.tensor([.3,.8,1.1,.6]);opt.zero_grad(set_to_none=True)
    for a,z in ((0,2),(2,4)):(method.training(reference,images[a:z],times[a:z],GlobalRandomStream(),noise=noise[a:z]).loss*weights[a:z]).sum().div(weights.sum()).backward()
    opt.step();weighted.step(images,times,noise=noise,context={'loss_weights':weights})
    for name,p in reference.state_dict().items():C('runtime/weighted_microbatch/'+name,weighted.model.state_dict()[name],p,atol=0,rtol=0)
    shards=RUN/'shards';shards.mkdir(exist_ok=True);entries=[];data=torch.randn(5,3,8,8);labels=torch.arange(5)
    for i,(a,z) in enumerate(((0,2),(2,5))):
        filename=f'{i}.pt';torch.save({'images':data[a:z],'labels':labels[a:z]},shards/filename);entries.append({'file':filename,'count':z-a})
    (shards/'manifest.json').write_text(json.dumps({'format':'umdiff_shards_v1','shards':entries}));dataset=TensorShards(shards)
    for i in (4,0,1,2,3):C('runtime/sharded_images/'+str(i),dataset[i]['images'],data[i],atol=0,rtol=0);C('runtime/sharded_labels/'+str(i),dataset[i]['labels'],labels[i],atol=0,rtol=0)
    # Native Relay sample-count LR and source.lerp(EMA, beta) arithmetic.
    reference=deepcopy(controlled);ema=deepcopy(reference);native_opt=torch.optim.Adam(reference.parameters(),lr=.001)
    actual=Trainer(method,deepcopy(controlled),lr=.001,optimizer_name='adam',weight_decay=0.,ema_style='source_lerp',ema_schedule={'name':'image_halflife','half_life_images':500.,'rampup_ratio':.05},lr_schedule={'name':'image_warmup','warmup_images':200.},loss_multiplier=3.)
    for step in range(128):
        native_opt.zero_grad(set_to_none=True);method.training(reference,images,times,GlobalRandomStream(),noise=noise).loss.mean().mul(3.).backward()
        native_opt.param_groups[0]['lr']=.001*min(step*4/200.,1.);native_opt.step();beta=.5**(4/max(min(500.,step*4*.05),1e-8))
        with torch.no_grad():
            for p,q in zip(ema.parameters(),reference.parameters()):p.copy_(q.detach().lerp(p,beta))
        actual.step(images,times,noise=noise)
        for name,p in reference.state_dict().items():C('runtime/Relay_image_clock_weights/'+str(step)+'/'+name,actual.model.state_dict()[name],p,atol=0,rtol=0)
        for name,p in ema.state_dict().items():C('runtime/Relay_image_clock_EMA/'+str(step)+'/'+name,actual.ema.state_dict()[name],p,atol=0,rtol=0)
        if step==63:
            actual.save(RUN/'image_clock_resume.pt');restored=Trainer(method,deepcopy(controlled),**{**actual.recipe,'ema_decay':actual.decay});restored.load(RUN/'image_clock_resume.pt');actual=restored
    assert actual.images_seen==512
    from umdiff.recipes import lr_factor
    for step in range(12):C('runtime/epoch_cosine/'+str(step),lr_factor(step,{'name':'cosine_epoch','steps_per_epoch':3,'epochs':4,'minimum_ratio':.02}),.02+.98*.5*(1+math.cos(math.pi*(step//3)/4)),atol=0,rtol=0)
    return {'mdm_nondefault_resnet':'nested typed configs instantiate and execute','microbatch_diagnostics':'full-batch predictions/targets/state and explicit microbatch records','resume':'wrong batch identity rejected','precision':'misplaced ordinary precision rejected','image_clock':'128-step native Relay LR/EMA arithmetic on a shared controlled objective; checkpoint resume at64','epoch_cosine':'Source epoch clock support; full Accelerate scheduler wrapper not certified'}

def main():
    torch.set_num_threads(1);training.seed(1029);details={};failure=None
    try:
        details['dct']=dct();details['codec']=codec()
        from umdiff.rdd_native import diffusion_backend
        diffusion_backend();details['rdd']=rdd();details['runtime']=runtime_contracts()
    except Exception:failure=traceback.format_exc();print(failure,flush=True)
    report={'status':'failed' if failure else 'passed','failure':failure,'details':details,'checks':conformance.CHECKS,'source_executions':native.AUDIT,'device':'cpu','gpu_used':False,'synthetic':True,'quality_verified':False}
    (OUT/'auxiliary.json').write_text(json.dumps(report,indent=2));print(json.dumps({'status':report['status'],'checks':len(report['checks']),'failure':failure}));return int(failure is not None)
if __name__=='__main__':raise SystemExit(main())
