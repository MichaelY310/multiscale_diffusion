"""Old three-optimizer native MultiStage DiT vs unified routed stage trainer."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'_deps'),str(ROOT/'tools')]
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import numpy as np
import torch
from umdiff.rdd_multistage import NativeRDDMultistage
from umdiff.cli import setup
from umdiff.rdd_native import diffusion_backend
from activation_trace import ActivationAudit

OUT=ROOT/'evidence/rdd_reproduction_20261104/ddpm_boundary/multistage_training'


def equal(a,b,path='value'):
    if torch.is_tensor(a):
        assert a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8)),path
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:equal(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+'/'+str(i))
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b),path
    else:assert a==b,(path,a,b)


def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    c=json.loads((ROOT/'configs/comparison_imagenet256/main/rdd_ddpm_xl_legacy.json').read_text())['method']
    c=deepcopy(c);c['name']='native_multistage_three_optimizer_conformance'
    c['model']={'name':'native_rdd_multistage','args':{'models':{str(r):{'name':'native_rdd','args':dict(
        input_size=r,hidden_size=w,depth=d,num_heads=4,patch_size=2,in_channels=4,
        num_classes=1000,class_dropout_prob=.1,learn_sigma=True)}
        for r,w,d in [(32,64,3),(16,48,2),(8,32,2)]}}}
    c['training'].pop('input_transform',None)
    c['training']['optimizer']['accumulation_steps']=2
    method,net,trainer=setup(c,torch.device('cpu'),617)
    with torch.no_grad():
        for p in net.parameters():p.normal_(0,.03)
    trainer.ema.load_state_dict(net.state_dict())
    original=deepcopy(net).train();ema=deepcopy(original).eval().requires_grad_(False)
    optimizers={r:torch.optim.AdamW(original.for_resolution(r).parameters(),lr=1e-4,weight_decay=0.) for r in (32,16,8)}
    backend=diffusion_backend();limits=[0,300,600,1000]
    ds=[backend.create_diffusion_with_drift('',diffusion_steps=1000,c=i if i<2 else None,
        T_c=limits[i],T_c_plus_1=limits[i+1],sample_scale=2) for i in range(3)]
    audit=ActivationAudit('independent_three_models',[(str(r),original.for_resolution(r),net.for_resolution(r)) for r in (32,16,8)],OUT)
    losses=[]
    for update in range(4):
        images=torch.randn(4,4,32,32);labels=torch.tensor([3,7,11,19]);start_rng=torch.get_rng_state().clone()
        audit.begin(update);audit.side('native');expected_losses=[]
        for opt in optimizers.values():opt.zero_grad(set_to_none=True)
        for micro in range(2):
            x=images[micro*2:(micro+1)*2];y=labels[micro*2:(micro+1)*2];total=0.
            for i,(res,ds_i) in enumerate(zip((32,16,8),ds)):
                factor=32//res
                stage=x if i==0 else x.detach().view(len(x),4,res,factor,res,factor).mean((3,5))
                t=torch.randint(0,ds_i.num_timesteps,(len(stage),))
                loss=ds_i.training_losses(original.for_resolution(res),stage,t,{'y':y})['loss'].mean()
                (loss/2).backward();total=total+loss.detach()
            expected_losses.append(total)
        for opt in optimizers.values():opt.step()
        with torch.no_grad():
            for p,q in zip(ema.parameters(),original.parameters()):p.mul_(.9999).add_(q,alpha=1-.9999)
        expected_rng=torch.get_rng_state().clone();torch.set_rng_state(start_rng);audit.side('unified')
        got=trainer.step(images,None,model_kwargs={'y':labels})
        for res in (32,16,8):audit.parameters(original.for_resolution(res),net.for_resolution(res),role=str(res))
        audit.finish_segment()
        equal(torch.stack(expected_losses),got.loss,'loss');equal(expected_rng,torch.get_rng_state(),'RNG')
        equal(original.state_dict(),net.state_dict(),'weights');equal(ema.state_dict(),trainer.ema.state_dict(),'EMA')
        for res in (32,16,8):
            for (n,a),(m,b) in zip(original.for_resolution(res).named_parameters(),net.for_resolution(res).named_parameters()):
                assert n==m
                equal(optimizers[res].state.get(a,{}),trainer.optimizer.state.get(b,{}),f'Adam/{res}/{n}')
        losses.append(got.loss.tolist())
    layers=audit.close();(OUT/'LAYERS.json').write_text(json.dumps(layers,indent=2)+'\n')
    assert not layers['structural_mismatches'] and not layers['first_differences']
    assert all(x['pairs']==x['byte_exact'] for x in layers['totals'].values())
    # Exact local unified checkpoint continuation and legacy per-stage imports.
    cleanup=[]
    with tempfile.TemporaryDirectory(prefix='rdd_multistage_validation_') as temp:
        path=Path(temp)/'resume.pt';trainer.save(path)
        got=trainer.step(images,None,model_kwargs={'y':labels});expected=deepcopy(trainer.state_dict())
        _,_,resumed=setup(c,torch.device('cpu'),919);resumed.load(path)
        replay=resumed.step(images,None,model_kwargs={'y':labels})
        equal(got.loss,replay.loss,'resume_loss');equal(expected,resumed.state_dict(),'resume_state')
        imported=NativeRDDMultistage(**c['model']['args'])
        files={}
        for res in (32,16,8):
            p=Path(temp)/f'{res}.pt';torch.save({'ema':ema.for_resolution(res).state_dict()},p);files[str(res)]=p
        imported.load_legacy_checkpoints(files,key='ema');equal(ema.state_dict(),imported.state_dict(),'legacy_weight_import')
        for p in Path(temp).glob('*.pt'):cleanup.append({'filename':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'removed':True})
    files=[Path(__file__),ROOT/'umdiff/rdd_multistage.py',ROOT/'umdiff/engine.py',ROOT/'umdiff/training.py',ROOT.parent/'DiT/models.py',ROOT.parent/'2d/train_RDD_imagenet256.py']
    report=dict(status='passed',updates=4,microbatches_per_update=2,images_per_update=4,
        grids=[32,16,8],widths=[64,48,32],depths=[3,2,2],totals=layers['totals'],losses=losses,
        optimizer='Three original per-stage AdamW optimizers vs one unified AdamW over disjoint children; every per-parameter optimizer state byte-identical',
        exact_resume=True,strict_legacy_ema_import=True,temporary_payload_cleanup=cleanup,
        source_sha256={str(p.relative_to(ROOT.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        scope='Original fixed-grid DiT per resolution, independent original RDD losses, separate stage backward calls vs unified stage-sum graph; all module activations/activation gradients/parameter gradients, optimizer, EMA and RNG. CPU FP32 random nonzero models, synthetic latents; no historical dataloader/multirank/real-checkpoint-quality claim.',
        cuda_initialized=torch.cuda.is_initialized())
    (OUT/'SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
