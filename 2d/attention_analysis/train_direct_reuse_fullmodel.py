"""Train original shared backbone, no adapters; paired teacher rollout inputs.

Uses the audited 4096-image /128-image corpus. This is off-policy prediction
distillation, NOT native ImageNet diffusion retraining or full unrolled BPTT.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from direct_reuse_fullmodel import DirectReuseStudent, MASKS
from train_rdd_csra_adapter import checkpoint_weights, atomic_save
from models_multires import DiTMultiRes_models


class PairDataset(Dataset):
    """Read only inputs/targets, never old learned/frozen hidden feature caches."""
    def __init__(self, root, sources, base, split, max_files=0):
        fields={k:[] for k in ('x','t','labels','target','image_index')}
        src={f'source{r}_{k}':[] for r in (8,16) for k in ('x','t','target')}
        self.seeds=set();offset=0
        manifests=sorted(Path(root).glob('shard_*/manifest.json'))
        if len(manifests)!=4: raise ValueError('Expected all4 completed shards')
        self.hashes={}
        for mpath in manifests:
            m=json.loads(mpath.read_text());digest=hashlib.sha256(mpath.read_bytes()).hexdigest()
            self.hashes[str(mpath)]=digest
            smpath=Path(sources)/mpath.parent.name/'manifest.json'
            side=json.loads(smpath.read_text())
            if (not m['complete'] or m['parity_max_abs']!=0 or not side['complete']
                or side['source_manifest_sha256']!=digest or Path(m['base_checkpoint']).resolve()!=Path(base).resolve()):
                raise ValueError('Corpus provenance/parity mismatch')
            count=0
            for item in m['files']:
                if item['split']!=split: continue
                if max_files and count>=max_files: break
                name=item['name']
                if Path(name).name!=name: raise ValueError('Invalid path')
                d=torch.load(mpath.parent/name,map_location='cpu',weights_only=True)
                s=torch.load(smpath.parent/name,map_location='cpu',weights_only=True)
                if (d['seed']!=s['seed'] or d['seed'] in self.seeds or s['cache_parity_max_abs']!=0
                    or s['source_manifest_sha256']!=digest): raise ValueError('Source parity/seed mismatch')
                self.seeds.add(d['seed']);n,b=d['x'].shape[:2]
                if n!=8 or d['x'].shape[2:]!=(4,32,32): raise ValueError('Invalid states')
                fields['x'].append(d['x'].flatten(0,1));fields['target'].append(d['target'].flatten(0,1))
                fields['t'].append(d['t'].flatten().long());fields['labels'].append(d['labels'].repeat(n).long())
                fields['image_index'].append(torch.arange(b).repeat(n)+offset)
                for r in (8,16):
                    sr=s['sources'][str(r)]
                    if not torch.equal(sr['labels'],d['labels']): raise ValueError('Unpaired labels')
                    if not torch.all(sr['t']==(600 if r==8 else 300)): raise ValueError('Source time mismatch')
                    for k in ('x','t','target'): src[f'source{r}_{k}'].append(sr[k])
                offset+=b;count+=1
        self.fields={k:torch.cat(v) for k,v in fields.items()};self.sources={k:torch.cat(v) for k,v in src.items()}
        self.images=offset

    def __len__(self): return len(self.fields['t'])

    def __getitem__(self, i):
        im=int(self.fields['image_index'][i])
        return {**{k:v[i] for k,v in self.fields.items() if k!='image_index'},**{k:v[im] for k,v in self.sources.items()}}


def loss(outputs,batch):
    totals=[];mse=[]
    for p,k in zip(outputs,('target','source16_target','source8_target')):
        e=(p.float()-batch[k].float()).square()
        main=e[:,:4].mean();var=e[:,4:].mean()
        totals.append(main+.05*var);mse.append(main.detach())
    return totals[0]+.5*(totals[1]+totals[2]),torch.stack(mse)


def device_batch(b,dev): return {k:v.to(dev,non_blocking=True) for k,v in b.items()}


@torch.no_grad()
def validate(net,loader,dev,distributed):
    sums=torch.zeros(4,device=dev,dtype=torch.float64)
    for b in loader:
        b=device_batch(b,dev);_,ms=loss(net(b),b);n=len(b['t'])
        sums[:3]+=ms*n;sums[3]+=n
    if distributed: dist.all_reduce(sums)
    return dict(zip(('fine','middle','coarse'),(sums[:3]/sums[3]).tolist()))


def main(a):
    multi='RANK' in os.environ;rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    local=int(os.environ.get('LOCAL_RANK',0));dev=torch.device('cuda',local);torch.cuda.set_device(dev)
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if any(v not in ('0','1','2','3') for v in visible) or world>4: raise ValueError('GPU0-3 only')
    if torch.cuda.mem_get_info(dev)[0]<30*2**30: raise RuntimeError('Need30GiB free per GPU')
    torch.cuda.set_per_process_memory_fraction(.30,dev)
    if multi: dist.init_process_group('nccl',device_id=dev)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists(): return
    if (out/'latest.pt').exists() and not a.resume: raise FileExistsError('Resume explicitly; never overwrite old run')
    weights=checkpoint_weights(Path(a.base))
    m=DiTMultiRes_models['DiT-L/2'](base_size=16,num_classes=weights['y_embedder.embedding_table.weight'].shape[0]-1)
    m.load_state_dict(weights,strict=True);del weights
    net=DirectReuseStudent(m.to(dev).float().eval().requires_grad_(True),MASKS[a.mask],True)
    if not all(p.requires_grad for p in net.parameters()): raise AssertionError('Entire backbone must train')
    train=PairDataset(a.data,a.sources,a.base,'train',max_files=1 if a.smoke else 0)
    val=PairDataset(a.data,a.sources,a.base,'validation',max_files=1 if a.smoke else 0)
    if train.seeds & val.seeds: raise ValueError('Train/validation leakage')
    sampler=DistributedSampler(train,world,rank,shuffle=True,seed=a.seed,drop_last=True)
    vs=DistributedSampler(val,world,rank,shuffle=False,drop_last=False)
    if len(val)%world: raise ValueError('Validation duplication')
    loader=DataLoader(train,batch_size=a.per_gpu,sampler=sampler,num_workers=0,pin_memory=True,drop_last=True)
    vl=DataLoader(val,batch_size=a.per_gpu,sampler=vs,num_workers=0,pin_memory=True)
    optim=torch.optim.AdamW(net.parameters(),lr=a.lr,betas=(.9,.999),weight_decay=.01,foreach=False)
    startstep=0
    if a.resume:
        saved=torch.load(out/'latest.pt',map_location=dev,weights_only=False)
        if saved['mask']!=a.mask or saved['per_gpu']!=a.per_gpu or saved['world']!=world: raise ValueError('Resume protocol changed')
        net.backbone.load_state_dict(saved['model']);optim.load_state_dict(saved['optimizer']);startstep=saved['step'];del saved
    if multi: net=DDP(net,device_ids=[local],broadcast_buffers=False,find_unused_parameters=False)
    print(json.dumps(dict(rank=rank,phase='DDP construction complete',multi=multi)),flush=True)
    core=net.module if multi else net
    protocol=dict(method='direct coarse residual reuse + whole shared backbone finetuning',mask=a.mask,
        layers=list(MASKS[a.mask]),parameters=sum(p.numel() for p in core.parameters()),new_parameters=0,
        base=str(Path(a.base).resolve()),global_batch=a.per_gpu*world,per_gpu=a.per_gpu,world=world,
        accumulation=1,lr=a.lr,weight_decay=.01,precision='FP32 master/Adam + BF16 autocast; TF32 off',
        checkpoint_blocks=True,train_images=train.images,validation_images=val.images,
        loss='fine epsilonMSE+.05variance + .5*(16anchor+8anchor), anchors same epsilon/variance weights',
        input_distribution='original teacher trajectories; student live coarse residuals, gradients not detached',
        train_data_sha256=train.hashes,max_steps=a.steps,seed=a.seed,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('direct_reuse_fullmodel.py'))})
    if rank==0: (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print(json.dumps(dict(rank=rank,phase='initial validation start')),flush=True)
    initial=validate(core,vl,dev,multi)
    if rank==0: print(json.dumps(dict(initial=initial,protocol=protocol)),flush=True)
    begin=time.perf_counter();epoch=startstep//len(loader);sampler.set_epoch(epoch);it=iter(loader)
    for _ in range(startstep%len(loader)): next(it)
    for step in range(startstep+1,a.steps+1):
        try: batch=next(it)
        except StopIteration:
            epoch+=1;sampler.set_epoch(epoch);it=iter(loader);batch=next(it)
        batch=device_batch(batch,dev);optim.zero_grad(set_to_none=True)
        objective,ms=loss(net(batch),batch)
        if not torch.isfinite(objective): raise RuntimeError('Nonfinite loss')
        objective.backward()
        if step==startstep+1:
            missing=[n for n,p in core.named_parameters() if p.grad is None]
            if missing: raise AssertionError(f'No gradient: {missing}')
            skipped_grads={l:core.backbone.blocks[l].attn.qkv.weight.grad.float().norm().item() for l in core.layers}
            if any(v==0 for v in skipped_grads.values()): raise AssertionError('Coarse producer frozen/disconnected')
            if rank==0: print(json.dumps(dict(coarse_producer_grad_norm=skipped_grads,all_parameters_have_grad=True)),flush=True)
        grad=torch.nn.utils.clip_grad_norm_(core.parameters(),1.,error_if_nonfinite=True);optim.step()
        if step%10==0 or a.smoke:
            if rank==0: print(json.dumps(dict(step=step,loss=objective.item(),mse=ms.tolist(),gradient_norm=float(grad),
                elapsed_s=time.perf_counter()-begin,peak_gib=torch.cuda.max_memory_allocated()/2**30)),flush=True)
        if step%a.validate_every==0 or step==a.steps:
            values=validate(core,vl,dev,multi)
            if rank==0:
                with (out/'validation.jsonl').open('a') as f: f.write(json.dumps(dict(step=step,**values))+'\n')
                if not a.smoke:
                    atomic_save(dict(model=core.backbone.state_dict(),optimizer=optim.state_dict(),step=step,
                        mask=a.mask,per_gpu=a.per_gpu,world=world,protocol=protocol),out/'latest.pt')
                    atomic_save(dict(model=core.backbone.state_dict(),step=step,mask=a.mask,protocol=protocol),out/'sample.pt')
                print(json.dumps(dict(validated_step=step,validation=values)),flush=True)
            if multi: dist.barrier()
    if rank==0: (out/'complete.json').write_text(json.dumps(dict(complete=True,step=a.steps,initial=initial,validation=values,
        elapsed_s=time.perf_counter()-begin,peak_gib=torch.cuda.max_memory_allocated()/2**30)))
    if multi: dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',default='checkpoints/RDD-Unified-L/0400000.pt')
    p.add_argument('--data',default='eval_results/csra_rollout_data4k_20260912')
    p.add_argument('--sources',default='eval_results/csra_joint_sources_20260913')
    p.add_argument('--output',required=True);p.add_argument('--mask',choices=MASKS,default='reuse6')
    p.add_argument('--steps',type=int,default=200);p.add_argument('--validate-every',type=int,default=100)
    p.add_argument('--per-gpu',type=int,default=64);p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--seed',type=int,default=9152026);p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true')
    main(p.parse_args())
