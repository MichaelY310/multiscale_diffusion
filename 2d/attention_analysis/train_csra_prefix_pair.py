#!/usr/bin/env python3
"""Matched source/zero prefix pilot; original8/16 weights stay unchanged."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler
from csra_prefix_student import prefix_student_class,prefix_checkpoint_format,check_prefix_initializer
from csra_output_distillation import prediction_loss
from train_csra_rollout_output import RolloutDataset,to_device
from train_rdd_csra_adapter import atomic_save,load_frozen_teacher


@torch.no_grad()
def validate(model,loader,device):
    sums = torch.zeros(3,device=device,dtype=torch.float64)
    for batch in loader:
        batch = to_device(batch,device)
        _,metrics = prediction_loss(model(batch),batch['target'],.05)
        n = len(batch['t'])
        sums[0] += metrics['main_mse'].double()*n
        sums[1] += metrics['variance_mse'].double()*n
        sums[2] += n
    dist.all_reduce(sums)
    return {'main_mse':float(sums[0]/sums[2]),'variance_mse':float(sums[1]/sums[2]),
            'objective':float((sums[0]+.05*sums[1])/sums[2]),'states':int(sums[2])}


def main(args):
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    device = torch.device('cuda',local)
    dist.init_process_group('nccl',device_id=device)
    rank,world = dist.get_rank(),dist.get_world_size()
    if world != 4: raise ValueError('This pilot requires4cards x64, global256; no accumulation')
    if torch.cuda.mem_get_info(device)[0] < 20*2**30: raise RuntimeError('Insufficient headroom; never stop other jobs')
    out = Path(args.output_dir)
    if not args.resume and out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    initializer = Path(args.initializer)
    init_sha = hashlib.sha256(initializer.read_bytes()).hexdigest()
    initial = torch.load(initializer,map_location='cpu',weights_only=True)
    check_prefix_initializer(initial,args.prefix_depth)
    teacher = load_frozen_teacher(Path(args.base_ckpt),device,'DiT-L/2','rdd')
    torch.manual_seed(args.seed)  # same new-bridge random weights in both arms/ranks
    model = prefix_student_class(args.bridge_kind)(teacher,initial['fits']['fine_only'],args.coarse_source,
                              prefix_depth=args.prefix_depth).to(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    del initial
    datasets = [RolloutDataset(args.data_root,s,[13,15,17,20],args.base_ckpt,args.hidden_root)
                for s in ('train','validation')]
    for data in datasets: data.coarse = {13:data.coarse[13]}
    train,val = datasets
    if train.seeds & val.seeds or len(val)%world: raise ValueError('Split overlap/validation duplicates')
    sampler = DistributedSampler(train,world,rank,seed=args.seed,drop_last=True)
    val_sampler = DistributedSampler(val,world,rank,shuffle=False,drop_last=False)
    loader = DataLoader(train,batch_size=64,sampler=sampler,drop_last=True,pin_memory=True,num_workers=0)
    validation = DataLoader(val,batch_size=64,sampler=val_sampler,pin_memory=True,num_workers=0)
    bridge_params = [p for n,p in model.named_parameters() if not n.startswith('backbone.')]
    optimizer = torch.optim.AdamW([{'params':bridge_params,'lr':1e-4},
        {'params':[p for p in model.backbone.parameters() if p.requires_grad],'lr':1e-5}],weight_decay=.01)
    step,best = 0,float('inf')
    if args.resume:
        payload = torch.load(args.resume,map_location='cpu',weights_only=True)
        if payload['format'] != prefix_checkpoint_format(args.bridge_kind) or payload['initializer_sha256'] != init_sha:
            raise ValueError('Incompatible checkpoint/initializer')
        if payload['args'].get('bridge_kind','linear') != args.bridge_kind:
            raise ValueError('Resume cannot change bridge type')
        if payload['prefix_depth'] != args.prefix_depth or payload['source_layer'] != 13:
            raise ValueError('Resume cannot change prefix depth')
        for key in ('coarse_source','seed','base_ckpt','data_root','hidden_root','initializer'):
            if payload['args'][key] != vars(args)[key]: raise ValueError(f'Resume changed {key}')
        model.load_state_dict(payload['student']); optimizer.load_state_dict(payload['optimizer'])
        step,best = payload['step'],payload['best_objective']
    torch.manual_seed(args.seed+rank)
    network = DDP(model,device_ids=[local],broadcast_buffers=False)
    params = [p for p in model.parameters() if p.requires_grad]
    def save(name):
        atomic_save({'format':prefix_checkpoint_format(args.bridge_kind),'student':model.state_dict(),
            'optimizer':optimizer.state_dict(),'step':step,'best_objective':best,'args':vars(args),
            'initializer_sha256':init_sha,'prefix_depth':args.prefix_depth,'source_layer':13,
            'base_checkpoint':str(Path(args.base_ckpt).resolve()),
            'note':f'Original8/16 separately preserved. Fine skips{args.prefix_depth}blocks; trains lift, patch/time/class embedding, last{24-args.prefix_depth}blocks and final layer. Offline teacher trajectories, noton-policy.'},out/name)
    baseline = validate(model,validation,device)
    best = min(best,baseline['objective'])
    if rank == 0:
        print(json.dumps({'event':'initialized','step':step,'coarse_source':args.coarse_source,
            'validation':baseline,'per_gpu_batch':64,'global_batch':256,'accumulation':1,
            'trainable_backbone':sum(p.numel() for p in model.backbone.parameters() if p.requires_grad),
            'trainable_lift':sum(p.numel() for p in bridge_params),'config':vars(args)}),flush=True)
        if not args.resume and not args.smoke: save('best.pt')
    dist.barrier()
    start = time.perf_counter()
    epoch,skip = divmod(step,len(loader))
    while step < args.max_steps:
        sampler.set_epoch(epoch)
        for i,batch in enumerate(loader):
            if i < skip: continue
            batch = to_device(batch,device)
            optimizer.zero_grad(set_to_none=True)
            loss,metrics = prediction_loss(network(batch),batch['target'],.05)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            if step == 0:
                absent = [n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]
                if absent: raise RuntimeError(f'Missing trainable gradients: {absent}')
            optimizer.step(); step += 1
            if rank == 0 and (step <= 3 or step%10 == 0):
                print(json.dumps({'event':'train','step':step,'main_mse':float(metrics['main_mse']),
                    'loss':float(loss),'grad_norm':float(norm),'seconds':time.perf_counter()-start,
                    'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                    'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30}),flush=True)
            if step%100 == 0 or step == args.max_steps:
                score = validate(model,validation,device)
                improved = score['objective'] < best
                if improved: best = score['objective']
                if rank == 0:
                    print(json.dumps({'event':'validation','step':step,**score,'new_best':improved}),flush=True)
                    if not args.smoke:
                        if improved: save('best.pt')
                        save('latest.pt')
                dist.barrier()
            if step >= args.max_steps: break
        epoch,skip = epoch+1,0
    if rank == 0:
        (out/'complete.json').write_text(json.dumps({'complete':True,'step':step,'best_objective':best,
            'smoke':args.smoke,'config':vars(args),'initializer_sha256':init_sha},indent=2))
    dist.barrier(); dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--base-ckpt',default='checkpoints/RDD-Unified-L/0400000.pt')
    p.add_argument('--data-root',default='eval_results/csra_rollout_data4k_20260912')
    p.add_argument('--hidden-root',default='eval_results/csra_hidden_sources_20260913')
    p.add_argument('--initializer',default='eval_results/csra_prefix_information_20260913/selected_ridge.pt')
    p.add_argument('--coarse-source',choices=('aligned','zero'),required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--resume',default='')
    p.add_argument('--max-steps',type=int,default=800)
    p.add_argument('--prefix-depth',type=int,default=14)
    p.add_argument('--bridge-kind',choices=('linear','conditional_attention'),default='linear')
    p.add_argument('--seed',type=int,default=20279000)
    p.add_argument('--smoke',action='store_true')
    main(p.parse_args())
