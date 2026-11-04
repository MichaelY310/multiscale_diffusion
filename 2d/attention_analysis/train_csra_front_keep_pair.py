#!/usr/bin/env python3
"""Fixed800 front1 source/zero pilot, real4x64 batches, original8/16 preserved."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler
from csra_front_keep_student import FORMAT,FrontKeepStudent,check_initializer,check_resume
from csra_output_distillation import prediction_loss
from train_csra_rollout_output import RolloutDataset,to_device
from train_csra_prefix_pair import validate
from train_rdd_csra_adapter import atomic_save,load_frozen_teacher
from probe_csra_front_keep import BASE_SHA,file_sha,write_json


def main(args):
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    device = torch.device('cuda',local)
    if torch.cuda.mem_get_info(device)[0]<26*2**30 or shutil.disk_usage('.').free<40*2**30:
        raise RuntimeError('Need26GiB free/card and40GiB disk; do not stop other jobs')
    torch.cuda.set_per_process_memory_fraction(.24,device)
    dist.init_process_group('nccl',device_id=device)
    rank,world = dist.get_rank(),dist.get_world_size()
    if world!=4 or args.max_steps not in (2,800) or args.smoke!=(args.max_steps==2):
        raise ValueError('Only fixed800 or2-step smoke,4x64 no accumulation')
    out = Path(args.output_dir)
    if rank==0:
        if not args.resume and out.exists() and any(out.iterdir()): raise FileExistsError(out)
        out.mkdir(parents=True,exist_ok=True)
    dist.barrier()
    initial = torch.load(args.initializer,map_location='cpu',weights_only=True)
    check_initializer(initial)
    init_sha = file_sha(args.initializer)
    if file_sha(args.base_ckpt)!=BASE_SHA: raise ValueError('Base checkpoint SHA changed')
    backbone = load_frozen_teacher(Path(args.base_ckpt),device,'DiT-L/2','rdd')
    torch.manual_seed(args.seed)
    model = FrontKeepStudent(backbone,initial['fits']['fine_only'],args.coarse_source,
                             prefix_depth=6,source_layer=13,checkpoint_blocks=True).to(device)
    del initial
    torch.backends.cuda.matmul.allow_tf32 = False
    datasets = [RolloutDataset(args.data_root,s,[13,15,17,20],args.base_ckpt,args.hidden_root)
                for s in ('train','validation')]
    for data in datasets: data.coarse={13:data.coarse[13]}
    train,val = datasets
    if train.seeds&val.seeds or len(val)%world: raise ValueError('Split overlap/duplicate validation')
    sampler = DistributedSampler(train,world,rank,seed=args.seed,drop_last=True)
    vs = DistributedSampler(val,world,rank,shuffle=False,drop_last=False)
    loader = DataLoader(train,batch_size=64,sampler=sampler,drop_last=True,pin_memory=True,num_workers=0)
    validation = DataLoader(val,batch_size=64,sampler=vs,pin_memory=True,num_workers=0)
    bridge = [p for n,p in model.named_parameters() if not n.startswith('backbone.')]
    optimizer = torch.optim.AdamW([{'params':bridge,'lr':1e-4},
        {'params':[p for p in model.backbone.parameters() if p.requires_grad],'lr':1e-5}],weight_decay=.01)
    code_sha = {name:file_sha(Path(__file__).parent/name) for name in
                ('train_csra_front_keep_pair.py','csra_front_keep_student.py','csra_prefix_student.py')}
    step,best = 0,float('inf')
    if args.resume:
        ck=torch.load(args.resume,map_location='cpu',weights_only=True)
        check_resume(ck,args,init_sha)
        if ck['code_sha256']!=code_sha: raise ValueError('Resume training code changed')
        model.load_state_dict(ck['student']); optimizer.load_state_dict(ck['optimizer'])
        step,best=ck['step'],ck['best_objective']
    torch.manual_seed(args.seed+rank)
    network=DDP(model,device_ids=[local],broadcast_buffers=False)
    params=[p for p in model.parameters() if p.requires_grad]
    def save(name):
        atomic_save(dict(format=FORMAT,student=model.state_dict(),optimizer=optimizer.state_dict(),
            step=step,best_objective=best,args=vars(args),initializer_sha256=init_sha,code_sha256=code_sha,
            front_keep=1,prefix_depth=6,source_layer=13,base_checkpoint=str(Path(args.base_ckpt).resolve()),
            base_sha256=BASE_SHA,note='Original8/16 unchanged. Execute0, skip1..6, execute7..23. Offline teacher states; original MSE+.05variance. Fixedlatest800 for pilot.'),out/name)
    baseline=validate(model,validation,device)
    best=min(best,baseline['objective'])
    if rank==0:
        write_json(out/'experiment.json',dict(format=FORMAT,args=vars(args),base_sha256=BASE_SHA,
            initializer_sha256=init_sha,code_sha256=code_sha,validation=baseline,
            executed_layers=[0]+list(range(7,24)),skipped_layers=list(range(1,7)),
            trainable_backbone=sum(p.numel() for p in model.backbone.parameters() if p.requires_grad),
            trainable_bridge=sum(p.numel() for p in bridge),per_gpu_batch=64,global_batch=256,accumulation=1))
        print(json.dumps({'event':'initialized','step':step,'validation':baseline,'source':args.coarse_source,
                          'per_gpu_batch':64,'global_batch':256}),flush=True)
        if not args.resume and not args.smoke: save('best.pt')
    dist.barrier()
    start=time.perf_counter()
    epoch,skip=divmod(step,len(loader))
    trace=[]
    while step<args.max_steps:
        sampler.set_epoch(epoch)
        indices=list(sampler)
        for i,batch in enumerate(loader):
            if i<skip: continue
            batch=to_device(batch,device)
            if len(trace)<2:
                trace.append({'step':step,'epoch':epoch,'batch_index':i,'dataset_indices':indices[i*64:(i+1)*64],
                              't':batch['t'].cpu().tolist(),'labels':batch['labels'].cpu().tolist()})
            optimizer.zero_grad(set_to_none=True)
            loss,metrics=prediction_loss(network(batch),batch['target'],.05)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            if step==0:
                absent=[n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]
                if absent: raise RuntimeError(f'Missing gradients:{absent}')
            optimizer.step();step+=1
            if rank==0 and (step<=3 or step%10==0):
                print(json.dumps({'event':'train','step':step,'main_mse':float(metrics['main_mse']),
                    'loss':float(loss),'grad_norm':float(norm),'seconds':time.perf_counter()-start,
                    'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                    'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30}),flush=True)
            if step%100==0 or step==args.max_steps:
                score=validate(model,validation,device)
                improved=score['objective']<best
                if improved: best=score['objective']
                if rank==0:
                    print(json.dumps({'event':'validation','step':step,**score,'new_best':improved}),flush=True)
                    if not args.smoke:
                        if improved:save('best.pt')
                        save('latest.pt')
                dist.barrier()
            if step>=args.max_steps:break
        epoch,skip=epoch+1,0
    write_json(out/f'start_trace_rank{rank}.json',{'rank':rank,'trace':trace})
    dist.barrier()
    if rank==0:write_json(out/'complete.json',dict(complete=True,step=step,smoke=args.smoke,args=vars(args),
        initializer_sha256=init_sha,best_objective=best,final_validation=score,format=FORMAT,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-ckpt',default='checkpoints/RDD-Unified-L/0400000.pt')
    p.add_argument('--data-root',default='eval_results/csra_rollout_data4k_20260912')
    p.add_argument('--hidden-root',default='eval_results/csra_hidden_sources_20260913')
    p.add_argument('--initializer',default='eval_results/csra_front_keep6_information_20260914/front1/selected_ridge.pt')
    p.add_argument('--coarse-source',choices=('aligned','zero'),required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--resume',default='')
    p.add_argument('--max-steps',type=int,default=800)
    p.add_argument('--seed',type=int,default=20279000)
    p.add_argument('--smoke',action='store_true')
    main(p.parse_args())
