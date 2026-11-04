"""CPU native image RDD-FM recipe parity against frozen pre-renaming source.

No pretrained assets or datasets. Tests use shared original DiT and per-stage
FM backend, but independent old/new stage assembly and backward/update loops.
"""
import argparse, ast, copy, hashlib, importlib.util, inspect, json, os, sys, tempfile, time
from collections import OrderedDict
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'_deps'), str(ROOT/'tools')]
import numpy as np
import torch
from umdiff import build
from umdiff.cli import make_method
from umdiff.training import Trainer, LocalImages
from umdiff.randomness import RandomStream, GlobalRandomStream
from activation_trace import ActivationAudit, tensors, byte_view
REF = ROOT/'evidence/rdd_reproduction_20261104/reference_before'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(x): return hashlib.sha256(byte_view(x.detach().cpu()).numpy().tobytes()).hexdigest()
def exact(a, b, label=''):
    if torch.is_tensor(a):
        assert torch.is_tensor(b) and a.shape == b.shape and a.dtype == b.dtype, label
        assert torch.equal(byte_view(a), byte_view(b)), (label, float((a-b).abs().max()))
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), label
        for k in a: exact(a[k], b[k], label+'/'+str(k))
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b), label
        for i, (x,y) in enumerate(zip(a,b)): exact(x,y,label+'/'+str(i))
    else: assert a == b, (label,a,b)


def source():
    spec=importlib.util.spec_from_file_location('rdd_fm_frozen_reference',REF/'flow_matching_with_drift.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    from PIL import Image
    tree=ast.parse((REF/'train_RDD_FM_imagenet256_unified.py').read_text())
    names={'center_crop_arr','downsample_latent','update_ema'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    ns=dict(torch=torch,OrderedDict=OrderedDict,Image=Image,np=np)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(REF/'train_RDD_FM_imagenet256_unified.py'),'exec'),ns)
    return mod,ns


def config(bounds, boundary='noise_filling', solver='euler', cfg=1., learned=True):
    return dict(task='rdd_fm_native',name='native_fm_reproduction_cpu',image_shape=[4,32,32],
        model=dict(name='native_rdd_unified',args=dict(patch_size=2,in_channels=4,hidden_size=32,
            depth=2,num_heads=4,class_dropout_prob=.1,learn_sigma=learned,base_size=16)),
        process=dict(resolutions=[32,16,8],bounds=bounds,steps=[80,80,90] if solver=='euler' else [40,40,45],
            solver=solver,boundary=boundary,cfg_scale=cfg,train_eps=1e-5),
        training=dict(time_sampler=dict(name='none'),optimizer=dict(lr=1e-4,weight_decay=0.,
            betas=[.9,.999],ema_decay=.9999,randomness='native_global',accumulation_steps=2)))


def matchers(old,bounds):
    return [old.FlowMatchingWithDrift(c=i if i<2 else None,s_k=bounds[2-i],s_k_plus_1=bounds[3-i]) for i in range(3)]


def model(c):
    torch.manual_seed(731)
    net=build('model',c['model'])
    with torch.no_grad():
        for p in net.parameters():p.normal_(0.,.03)
    return net


def verify_audit(audit):
    value=audit.close()
    assert not value['structural_mismatches'], value['structural_mismatches']
    assert not value['first_differences'], value['first_differences'][:2]
    assert all(v['pairs']==v['byte_exact'] for v in value['totals'].values())
    return value


def training_case(old, ns, out, bounds, learned=True):
    name='training_'+str(bounds[1]).replace('.','p')+('_sigma' if learned else '_velocity')
    c=config(bounds,learned=learned);native=model(c);port=copy.deepcopy(native)
    method=make_method(c);trainer=Trainer(method,port,**c['training']['optimizer'])
    opt=torch.optim.AdamW(native.parameters(),lr=1e-4,weight_decay=0.)
    ema=copy.deepcopy(native).eval().requires_grad_(False)
    fm=matchers(old,bounds)
    generator=torch.Generator().manual_seed(382)
    images=torch.randn(4,4,32,32,generator=generator);labels=torch.tensor([1,33,998,7])
    audit=ActivationAudit(name,[('network',native,port)],out)
    rows=[]
    for step in range(4):
        native.train();opt.zero_grad(set_to_none=True)
        audit.begin(step);audit.side('native');torch.manual_seed(1100+step)
        losses=[]
        for offset in (0,2):
            x=images[offset:offset+2];y=labels[offset:offset+2];total=0.0
            for i,(res,obj) in enumerate(zip([32,16,8],fm)):
                xstage=x if i==0 else ns['downsample_latent'](x.detach(),res)
                total=total+obj.training_losses(native,xstage,model_kwargs={'y':y})['loss'].mean()
            (total/2).backward();losses.append(total.detach())
        rng_expected=torch.get_rng_state().clone();opt.step();ns['update_ema'](ema,native,decay=.9999)
        audit.side('unified');torch.manual_seed(1100+step)
        got=trainer.step(images,None,model_kwargs={'y':labels})
        exact(torch.stack(losses),got.loss,'loss');exact(rng_expected,torch.get_rng_state(),'rng')
        audit.parameters(native,port);audit.finish_segment()
        exact(native.state_dict(),port.state_dict(),'updated_parameters')
        exact(opt.state_dict(),trainer.optimizer.state_dict(),'AdamW')
        exact(ema.state_dict(),trainer.ema.state_dict(),'EMA')
        rows.append(dict(step=step,losses=[v.item() for v in losses],all_byte_exact=True))
    summary=verify_audit(audit)
    (out/(name+'_LAYERS.json')).write_text(json.dumps(summary,indent=2)+'\n')
    # Resuming the new format must preserve the next stochastic update exactly.
    with tempfile.TemporaryDirectory(prefix='rdd_fm_native_resume_') as tmp:
        file=Path(tmp)/'checkpoint.pt';trainer.save(file)
        restored=Trainer(make_method(c),model(c),**c['training']['optimizer']);restored.load(file)
        rng=torch.get_rng_state().clone()
        a=trainer.step(images,None,model_kwargs={'y':labels});arng=torch.get_rng_state().clone()
        torch.set_rng_state(rng);b=restored.step(images,None,model_kwargs={'y':labels})
        exact(a.loss,b.loss,'resume_loss');exact(arng,torch.get_rng_state(),'resume_rng')
        for key in ('model','ema','optimizer','step','images_seen'):exact(trainer.state_dict()[key],restored.state_dict()[key],key)
        payload=dict(sha256=sha(file),bytes=file.stat().st_size,deleted_after_verification=True)
    return dict(name=name,status='passed',bounds=bounds,learned_sigma=learned,
        updates=4,microbatches_per_update=2,batch_size=4,layers=summary['totals'],rows=rows,
        AdamW_and_EMA_byte_exact=True,resume_next_update_byte_exact=True,temporary_checkpoint=payload),trainer.ema


class SamplingHashes:
    """Record every executed module tensor boundary without large tensor dumps."""
    def __init__(self,net):
        self.records=[];self.calls={};self.handles=[];self.root=[]
        for name,module in net.named_modules():
            signature=inspect.signature(module.forward)
            def pre(mod,args,kwargs,name=name,sig=signature):
                self.calls[name]=self.calls.get(name,0)+1
                bound=sig.bind(*args,**kwargs);bound.apply_defaults()
                self.record(name,self.calls[name],'input',bound.arguments)
                if not name:self.root.append({k:v.detach().clone() for k,v in bound.arguments.items() if torch.is_tensor(v)})
            def post(mod,args,kwargs,value,name=name):self.record(name,self.calls[name],'output',value)
            self.handles.extend([module.register_forward_pre_hook(pre,with_kwargs=True),module.register_forward_hook(post,with_kwargs=True)])
    def record(self,name,index,direction,value):
        for key,x in tensors(value):self.records.append(dict(module=name,call=index,direction=direction,tensor=key,shape=list(x.shape),dtype=str(x.dtype),sha256=digest(x)))
    def close(self):
        for h in self.handles:h.remove()
        return self.records


def sampling_case(old, net, out, bounds, boundary, solver, cfg):
    c=config(bounds,boundary,solver,cfg);method=make_method(c);net=copy.deepcopy(net).eval()
    name='sample_'+str(bounds[1]).replace('.','p')+'_'+boundary+'_'+solver+'_cfg'+str(cfg).replace('.','p')
    kwargs={'y':torch.tensor([17])};seed=641
    recorder=SamplingHashes(net);torch.manual_seed(seed)
    with torch.no_grad():
        expected,metrics=old.sample_rdd_fm_multistage(net,list(reversed(matchers(old,bounds))),[8,16,32],
            c['process']['steps'],1,torch.device('cpu'),model_kwargs=kwargs,cfg_scale=cfg,
            upsample_mode='nearest' if boundary=='fresh' else boundary,solver=solver,return_metrics=True)
    rng_expected=torch.get_rng_state().clone();records_a=recorder.close();roots_a=recorder.root
    recorder=SamplingHashes(net);torch.manual_seed(1991);outer=torch.get_rng_state().clone()
    stream=RandomStream(seed);initial=stream.normal(torch.empty(1,4,8,8))
    actual=method.sample(net,initial,stream=stream,model_kwargs=kwargs)
    records_b=recorder.close();roots_b=recorder.root
    exact(expected,actual,name+'/final_latent');exact(rng_expected,stream.generator.get_state(),name+'/rng')
    exact(outer,torch.get_rng_state(),name+'/outer_rng')
    exact(roots_a,roots_b,name+'/all_model_inputs')
    assert records_a==records_b, next((a,b) for a,b in zip(records_a,records_b) if a!=b)
    assert metrics['nfe']==len(roots_a)==method.last_sample_metrics['nfe']
    assert metrics['stage_nfe']==method.last_sample_metrics['stage_nfe']
    # Every integration-step input and intermediate midpoint state is included
    # among root calls; final output plus endpoint calls cover terminal states.
    row=dict(name=name,status='passed',bounds=bounds,boundary=boundary,solver=solver,cfg=cfg,
        steps_coarse_to_fine=c['process']['steps'],nfe=metrics['nfe'],stage_nfe=metrics['stage_nfe'],
        model_input_calls=len(roots_a),layer_tensor_pairs=len(records_a),all_byte_exact=True,
        final_latent_sha256=digest(actual),rng_byte_exact=True,outer_rng_unchanged=True,
        trace_digest=hashlib.sha256(json.dumps(records_a,sort_keys=True).encode()).hexdigest())
    (out/(name+'.json')).write_text(json.dumps(dict(**row,module_boundary_records=records_a),indent=2)+'\n')
    return row,actual


def rgb_codec(ns,out):
    from PIL import Image
    from torchvision import transforms
    from diffusers import AutoencoderKL
    from umdiff.input_transforms import LocalVAE
    with tempfile.TemporaryDirectory(prefix='rdd_fm_rgb_') as tmp:
        tmp=Path(tmp);rng=np.random.default_rng(484)
        sizes=[(270,260),(519,777),(79,96),(256,256)]
        for i,(h,w) in enumerate(sizes):Image.fromarray(rng.integers(0,256,(h,w,3),dtype=np.uint8)).save(tmp/f'{i}.png')
        original=transforms.Compose([transforms.Lambda(lambda x:ns['center_crop_arr'](x,256)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3,inplace=True)])
        port=LocalImages(tmp,256,crop='adm_center',horizontal_flip=True,flip_rng='torch',normalization='torchvision')
        rgb=[]
        for i in range(len(sizes)):
            torch.manual_seed(73+i)
            with Image.open(tmp/f'{i}.png') as im:a=original(im.convert('RGB'))
            after=torch.get_rng_state().clone();torch.manual_seed(73+i);b=port[i]
            exact(a,b,'preprocessing');exact(after,torch.get_rng_state(),'preprocess_rng');rgb.append(a)
        torch.manual_seed(41)
        vae=AutoencoderKL(in_channels=3,out_channels=3,down_block_types=('DownEncoderBlock2D',)*4,
            up_block_types=('UpDecoderBlock2D',)*4,block_out_channels=(8,8,8,8),layers_per_block=1,
            latent_channels=4,norm_num_groups=4,sample_size=256).eval().requires_grad_(False)
        vae.save_pretrained(tmp/'vae',safe_serialization=True)
        transform=LocalVAE(model=dict(name='local_autoencoder',args=dict(path=str(tmp/'vae'))),scale=.18215)
        audit=ActivationAudit('rgb_vae_encode',[('vae',vae,transform.model)],out);audit.begin(0)
        x=torch.stack(rgb[:2]);torch.manual_seed(332);audit.side('native')
        with torch.no_grad():a=vae.encode(x).latent_dist.sample().mul_(.18215)
        after=torch.get_rng_state().clone();torch.manual_seed(332);audit.side('unified');b=transform(x,GlobalRandomStream())
        exact(a,b,'vae_latents');exact(after,torch.get_rng_state(),'vae_rng');audit.finish_segment()
        audit.begin(1,'decode');audit.side('native')
        with torch.no_grad():pixels_a=vae.decode(a/.18215).sample
        audit.side('unified')
        with torch.no_grad():pixels_b=transform.model.decode(b/.18215).sample
        exact(pixels_a,pixels_b,'decoded_pixels');audit.finish_segment();summary=verify_audit(audit)
        (out/'RGB_CODEC_LAYERS.json').write_text(json.dumps(summary,indent=2)+'\n')
    return dict(status='passed',preprocessing_cases=len(sizes),input_rgb_shapes=sizes,
        latent_shape=list(a.shape),layers=summary['totals'],latent_sha256=digest(a),decoded_rgb_sha256=digest(pixels_a),
        note='Random small-channel AutoencoderKL, full256 RGB and32 latent grid; frozen VAE has no training gradient. Original ImageNet folder transform, posterior sample, .18215 scale, decode; not real VAE quality or data-loader DDP order.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--threads',type=int,default=2)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(args.threads)
    old,ns=source();report=dict(status='running',cpu_only=True,training=[],sampling=[],
        scope='Frozen pre-renaming old FM source vs current native FM frontend; shared original DiT definition and per-stage backend, independently assembled stage loops. CPU FP32 random nonzero weights, full32/16/8 grids; no real datasets or pretrained assets.')
    started=time.monotonic()
    def save(): (args.output/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    for bounds in ([0.,.3,.6,1.],[0.,.4,.7,1.]):
        row,ema=training_case(old,ns,args.output,bounds);report['training'].append(row);save();print(json.dumps({'training':row['name'],'status':'passed'}),flush=True)
        for boundary in ('fresh','noise_filling'):
            for solver in ('euler','midpoint'):
                for cfg in (1.,1.5):
                    row,_=sampling_case(old,ema,args.output,bounds,boundary,solver,cfg)
                    report['sampling'].append(row);save();print(json.dumps(row),flush=True)
    row,_=training_case(old,ns,args.output,[0.,.3,.6,1.],False);report['training'].append(row);save()
    report['rgb_codec']=rgb_codec(ns,args.output);save()
    report['source_sha256']={str(p.relative_to(ROOT)):sha(p) for p in [REF/'flow_matching_with_drift.py',REF/'train_RDD_FM_imagenet256_unified.py',ROOT/'umdiff/rdd_fm_native.py',ROOT/'umdiff/training.py',ROOT/'umdiff/cli.py',ROOT/'tools/validate_rdd_fm_native_20261104.py']}
    report['cuda_initialized']=torch.cuda.is_initialized();assert not report['cuda_initialized']
    report['status']='passed';report['seconds']=time.monotonic()-started;save();print(json.dumps({'status':'passed','seconds':report['seconds']}),flush=True)

if __name__=='__main__':main()
