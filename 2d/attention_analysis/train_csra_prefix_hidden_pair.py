#!/usr/bin/env python3
"""Matched2x2 auxiliary-hidden-loss fine-tune, not unchanged-objective resume.

All arms start from their own fixed3200 parent. Original coarse generation
stays unchanged. Extra teacher prefix is used ONLY during training/evaluation.
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
from torch.utils.data import DataLoader,DistributedSampler
from csra_prefix_student import prefix_student_class,prefix_checkpoint_format,check_prefix_initializer
from csra_prefix_hidden_loss import teacher_prefix_hidden,hidden_loss_per_image
from csra_output_distillation import prediction_loss
from train_csra_rollout_output import RolloutDataset,to_device
from train_rdd_csra_adapter import atomic_save,load_frozen_teacher


@torch.no_grad()
def validate(model,teacher,loader,device):
    total=torch.zeros(4,device=device,dtype=torch.float64)
    for batch in loader:
        batch=to_device(batch,device)
        out,hidden=model(batch,return_hidden=True)
        _,scores=prediction_loss(out,batch['target'],.05)
        target=teacher_prefix_hidden(teacher,batch)
        n=len(batch['t'])
        total[0]+=scores['main_mse'].double()*n
        total[1]+=scores['variance_mse'].double()*n
        total[2]+=hidden_loss_per_image(hidden,target).double().sum()
        total[3]+=n
    dist.all_reduce(total)
    return dict(main_mse=float(total[0]/total[3]),variance_mse=float(total[1]/total[3]),
                hidden_nmse=float(total[2]/total[3]),states=int(total[3]),
                objective=float((total[0]+.05*total[1])/total[3]))


def main(args):
    local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local)
    device=torch.device('cuda',local);dist.init_process_group('nccl',device_id=device)
    rank,world=dist.get_rank(),dist.get_world_size()
    if world!=4:raise ValueError('Requires4cards x64; no accumulation')
    if torch.cuda.mem_get_info(device)[0]<25*2**30:raise RuntimeError('Need25GiB/card; never stop unrelated jobs')
    if not 1<=args.extra_steps<=800:raise ValueError('Bounded at800additional updates/arm')
    out=Path(args.output_dir)
    if not args.resume and out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    parent_path=Path(f'checkpoints/RDD-CSRA-prefix8-conditional3200-{args.coarse_source}-L400K-20260913/best.pt')
    parent=torch.load(parent_path,map_location='cpu',mmap=True,weights_only=True)
    audit_path=Path('eval_results/csra_prefix8_hidden_audit_20260913/metrics.json')
    audit=json.loads(audit_path.read_text());audit_sha=hashlib.sha256(audit_path.read_bytes()).hexdigest()
    if (not audit['complete'] or audit['calibration_role']!='fit_only'
        or parent['step']!=3200 or parent['prefix_depth']!=8 or parent['source_layer']!=13
        or parent['format']!='csra_prefix_conditional_v8'
        or parent['args']['coarse_source']!=args.coarse_source):raise ValueError('Wrong parent or audit')
    cfg=dict(parent['args'])
    if cfg['bridge_kind']!='conditional_attention' or cfg['seed']!=20279000:raise ValueError('Wrong architecture/seed')
    coefficient=float(audit['coefficient']) if args.aux=='on' else 0.
    if not 0<float(audit['coefficient'])<1:raise ValueError('Invalid fixed calibration')
    init_path=Path(cfg['initializer'])
    init_sha=hashlib.sha256(init_path.read_bytes()).hexdigest()
    if init_sha!=parent['initializer_sha256']:raise ValueError('Changed initializer')
    initial=torch.load(init_path,map_location='cpu',weights_only=True);check_prefix_initializer(initial,8)
    original=load_frozen_teacher(Path(cfg['base_ckpt']),device,'DiT-L/2','rdd')
    torch.manual_seed(cfg['seed'])
    model=prefix_student_class(cfg['bridge_kind'])(original,initial['fits']['fine_only'],
        args.coarse_source,prefix_depth=8).to(device)
    model.load_state_dict(parent['student'])
    teacher=load_frozen_teacher(Path(cfg['base_ckpt']),device,'DiT-L/2','rdd')
    torch.backends.cuda.matmul.allow_tf32=False
    bridge=[p for n,p in model.named_parameters() if not n.startswith('backbone.')]
    optimizer=torch.optim.AdamW([{'params':bridge,'lr':1e-4},
        {'params':[p for p in model.backbone.parameters() if p.requires_grad],'lr':1e-5}],weight_decay=.01)
    optimizer.load_state_dict(parent['optimizer'])
    for expected,g in zip((1e-4,1e-5),optimizer.param_groups):
        if g['lr']!=expected or g['weight_decay']!=.01:raise ValueError('Changed optimizer')
    del initial,parent
    data=[RolloutDataset(cfg['data_root'],split,[13,15,17,20],cfg['base_ckpt'],cfg['hidden_root'])
          for split in ('train','validation')]
    train,val=data
    for d in data:d.coarse={13:d.coarse[13]}
    if train.seeds & val.seeds or train.images!=4096 or val.images!=128:raise ValueError('Dataset mismatch')
    sampler=DistributedSampler(train,world,rank,seed=cfg['seed'],drop_last=True)
    val_sampler=DistributedSampler(val,world,rank,shuffle=False,drop_last=False)
    loader=DataLoader(train,batch_size=64,sampler=sampler,drop_last=True,pin_memory=True,num_workers=0)
    validation=DataLoader(val,batch_size=64,sampler=val_sampler,pin_memory=True,num_workers=0)
    step=3200;best=float('inf')
    experiment=dict(name='prefix_hidden_auxiliary_2x2',parent=str(parent_path),parent_step=3200,
        aux=args.aux,coefficient=coefficient,audit_sha256=audit_sha,extra_steps=args.extra_steps,
        target='ORIGINAL frozen teacher h_after7 on samex/t/y',
        hidden_loss='mean_image(MSE(hidden,target)/mean(target^2))',selection='main_mse+.05variance_mse')
    if args.resume:
        ck=torch.load(args.resume,map_location='cpu',weights_only=True)
        if ck.get('hidden_supervision')!=experiment or ck['args']['coarse_source']!=args.coarse_source:
            raise ValueError('Resume changed experiment')
        if ck['initializer_sha256']!=init_sha:raise ValueError('Resume initializer mismatch')
        model.load_state_dict(ck['student']);optimizer.load_state_dict(ck['optimizer'])
        step,best=ck['step'],ck['best_objective'];del ck
    cfg.update(output_dir=str(out),max_steps=3200+args.extra_steps,resume=args.resume,smoke=args.smoke)
    baseline=validate(model,teacher,validation,device)
    if not args.resume:
        expected={'aligned':(.014664345071651042,.009859663114184514),
                  'zero':(.01591951650334522,.01060521072940901)}[args.coarse_source]
        for key,value in zip(('main_mse','variance_mse'),expected):
            if abs(baseline[key]-value)>1e-10:raise RuntimeError('Parent prediction parity failed')
    best=min(best,baseline['objective'])
    params=[p for p in model.parameters() if p.requires_grad]
    torch.manual_seed(cfg['seed']+rank)
    network=DDP(model,device_ids=[local],broadcast_buffers=False)
    def save(name):
        atomic_save(dict(format=prefix_checkpoint_format(cfg['bridge_kind']),student=model.state_dict(),
            optimizer=optimizer.state_dict(),step=step,best_objective=best,args=cfg,
            initializer_sha256=init_sha,prefix_depth=8,source_layer=13,
            base_checkpoint=str(Path(cfg['base_ckpt']).resolve()),hidden_supervision=experiment,
            note='Extra teacher prefix training-only; inference unchanged, original8/16 separately preserved.'),out/name)
    if rank==0:
        print(json.dumps(dict(event='initialized',step=step,experiment=experiment,validation=baseline,
            per_gpu_batch=64,global_batch=256,accumulation=1)),flush=True)
        if not args.smoke and not args.resume:save('best.pt')
    dist.barrier();start=time.perf_counter();epoch,skip=divmod(step,len(loader))
    initial_step=step
    while step<3200+args.extra_steps:
        sampler.set_epoch(epoch)
        for i,batch in enumerate(loader):
            if i<skip:continue
            batch=to_device(batch,device);optimizer.zero_grad(set_to_none=True)
            target=teacher_prefix_hidden(teacher,batch)
            pred,hidden=network(batch,return_hidden=True)
            primary,score=prediction_loss(pred,batch['target'],.05)
            aux_loss=hidden_loss_per_image(hidden,target).mean()
            loss=primary+coefficient*aux_loss
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward()
            if step==initial_step:
                absent=[n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]
                if absent:raise RuntimeError(f'Missing gradients: {absent}')
                if any(p.grad is not None for p in teacher.parameters()):raise RuntimeError('Teacher updated')
            norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            optimizer.step();step+=1
            if rank==0 and (step<=initial_step+2 or step%10==0):
                print(json.dumps(dict(event='train',step=step,main_mse=float(score['main_mse']),
                    hidden_nmse=float(aux_loss),loss=float(loss),grad_norm=float(norm),
                    seconds=time.perf_counter()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)),flush=True)
            if step%100==0 or step==3200+args.extra_steps:
                score=validate(model,teacher,validation,device);improved=score['objective']<best
                if improved:best=score['objective']
                if rank==0:
                    print(json.dumps(dict(event='validation',step=step,**score,new_best=improved)),flush=True)
                    if not args.smoke:
                        if improved:save('best.pt')
                        save('latest.pt')
                dist.barrier()
            if step>=3200+args.extra_steps:break
        epoch,skip=epoch+1,0
    if rank==0:
        (out/'complete.json').write_text(json.dumps(dict(complete=True,step=step,smoke=args.smoke,
            experiment=experiment,best_objective=best,config=cfg,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30),indent=2))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--coarse-source',choices=('aligned','zero'),required=True)
    p.add_argument('--aux',choices=('on','off'),required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--extra-steps',type=int,default=800);p.add_argument('--resume',default='')
    p.add_argument('--smoke',action='store_true');main(p.parse_args())
