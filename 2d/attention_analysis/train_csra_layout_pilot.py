#!/usr/bin/env python3
"""Fixed800-update layout pilot, same original coarse, 4x64, full fine training.

No best-checkpoint selection. Saves latest optimizer/RNG for recovery plus final
student-only weights. A separate paired rollout is required after every arm.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler,Subset
from csra_layout_student import LayoutStudent,FORMAT
from csra_layer_layouts import LAYOUTS,layout
from csra_output_distillation import prediction_loss
from train_csra_rollout_output import RolloutDataset,to_device
from train_csra_prefix_pair import validate
from train_rdd_csra_adapter import atomic_save,load_frozen_teacher
from probe_csra_front_keep import BASE_SHA,file_sha
from probe_csra_prefix_information import BASE,write_json

ROOT='/home/yuheng/rdd_experiments/csra_skip_layout_20260915'


def hashes():
    folder=Path(__file__).parent
    return {str(p):file_sha(p) for p in [Path(__file__),folder/'csra_layout_student.py',
        folder/'csra_layer_layouts.py',folder/'csra_output_distillation.py',folder/'train_csra_rollout_output.py',
        folder/'train_csra_prefix_pair.py',folder/'train_rdd_csra_adapter.py']}


def main(args):
    local=int(os.environ['LOCAL_RANK']); torch.cuda.set_device(local)
    dev=torch.device('cuda',local); dist.init_process_group('nccl',device_id=dev)
    rank,world=dist.get_rank(),dist.get_world_size()
    if world!=4 or args.steps not in (2,800): raise ValueError('Predeclared4x64, two-step smoke or800-update pilot')
    if args.smoke != (args.steps==2): raise ValueError('Smoke budget mismatch')
    if torch.cuda.mem_get_info(dev)[0]<26*2**30: raise RuntimeError('Need26GiB free; do not stop other jobs')
    torch.cuda.set_per_process_memory_fraction(.24,dev); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    root=Path(args.root); initpath=root/'layouts'/args.layout/'selected_ridge.pt'
    out=root/('train_smoke' if args.smoke else 'training')/f'{args.layout}_{args.source}'
    out.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(out).free<24*2**30: raise RuntimeError('Need24GiB disk for checkpoint transactions')
    init=torch.load(initpath,map_location='cpu',weights_only=True)
    if file_sha(BASE)!=BASE_SHA: raise ValueError('Base checkpoint changed')
    contract=dict(format=FORMAT,layout=layout(args.layout),coarse_source=args.source,
        base_checkpoint=str(BASE.resolve()),base_sha256=BASE_SHA,initializer_sha256=file_sha(initpath),
        source_sha256=hashes(),data_manifests_sha256=init['provenance'],seed=41579915,
        global_batch=256,batch_per_gpu=64,accumulation=1,checkpoint_blocks=True,
        bridge_lr=1e-4,backbone_lr=1e-5,weight_decay=.01,variance_loss_weight=.05,clip=1.,
        precision='BF16 autocast, FP32 master, TF32 disabled',
        nccl_transport={k:os.environ.get(k,'') for k in ('NCCL_P2P_DISABLE','NCCL_IB_DISABLE')},
        updates=args.steps,smoke=args.smoke)
    for path,digest in contract['data_manifests_sha256'].items():
        if file_sha(path)!=digest: raise ValueError('Dataset manifest changed')
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())!=contract:
        raise ValueError('Incompatible existing run; not overwritten')
    if rank==0: write_json(out/'protocol.json',contract)
    if (out/'complete.json').exists():
        dist.destroy_process_group(); return
    torch.manual_seed(contract['seed'])
    model=LayoutStudent(load_frozen_teacher(BASE,dev,'DiT-L/2','rdd'),init,args.layout,args.source).to(dev)
    del init
    # Dataset keeps teacher outputs and the original, independently frozen
    # stage16 source. No fine-training gradient can change the coarse producer.
    data=[RolloutDataset(args.data_root,s,[13,15,17,20],str(BASE),args.hidden_root) for s in ('train','validation')]
    for ds in data: ds.coarse={13:ds.coarse[13]}
    if data[0].seeds & data[1].seeds: raise ValueError('Train/validation overlap')
    if args.smoke: data=[Subset(data[0],range(256)),Subset(data[1],range(256))]
    ts=DistributedSampler(data[0],world,rank,seed=contract['seed'],drop_last=True)
    vs=DistributedSampler(data[1],world,rank,shuffle=False)
    loader=DataLoader(data[0],batch_size=64,sampler=ts,drop_last=True,num_workers=0,pin_memory=True)
    vl=DataLoader(data[1],batch_size=64,sampler=vs,num_workers=0,pin_memory=True)
    optimizer=torch.optim.AdamW([dict(params=model.readers.parameters(),lr=1e-4),
        dict(params=[p for p in model.backbone.parameters() if p.requires_grad],lr=1e-5)],weight_decay=.01)
    step=0; resume=out/'latest.pt'
    if resume.exists():
        ck=torch.load(resume,map_location='cpu',weights_only=True)
        if ck['contract']!=contract: raise ValueError('Incompatible resume')
        model.load_state_dict(ck['student']); optimizer.load_state_dict(ck['optimizer']); step=ck['step']
        torch.set_rng_state(ck['rng'][rank]['cpu']);torch.cuda.set_rng_state(ck['rng'][rank]['cuda'],dev)
        del ck
    else: torch.manual_seed(contract['seed']+rank)
    net=DDP(model,device_ids=[local],broadcast_buffers=False)
    params=[p for p in model.parameters() if p.requires_grad]
    baseline=validate(model,vl,dev)
    if rank==0: print(json.dumps(dict(event='initialized',step=step,validation=baseline,contract=contract)),flush=True)
    epoch,skip=divmod(step,len(loader)); begin=time.monotonic()
    while step<args.steps:
        ts.set_epoch(epoch)
        for i,batch in enumerate(loader):
            if i<skip: continue
            optimizer.zero_grad(set_to_none=True)
            batch=to_device(batch,dev)
            loss,metrics=prediction_loss(net(batch),batch['target'],.05)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            loss.backward(); norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            if step==0:
                missing=[n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]
                if missing: raise RuntimeError(f'Missing gradients: {missing}')
            optimizer.step();step+=1
            if rank==0 and (step<=3 or step%20==0):
                print(json.dumps(dict(event='train',step=step,loss=float(loss),main_mse=float(metrics['main_mse']),
                    grad_norm=float(norm),elapsed=time.monotonic()-begin,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)),flush=True)
            if step%200==0 or step==args.steps:
                score=validate(model,vl,dev)
                states=[None]*world
                dist.all_gather_object(states,dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(dev)))
                if rank==0:
                    print(json.dumps(dict(event='validation',step=step,**score)),flush=True)
                    if not args.smoke:
                        if shutil.disk_usage(out).free<12*2**30: raise RuntimeError('Insufficient disk for atomic checkpoint')
                        atomic_save(dict(contract=contract,step=step,student=model.state_dict(),
                            optimizer=optimizer.state_dict(),rng=states,validation=score),resume)
                        if step==args.steps:
                            atomic_save(dict(contract=contract,step=step,student=model.state_dict(),validation=score),out/'final.pt')
                dist.barrier()
            if step>=args.steps: break
        epoch,skip=epoch+1,0
    if rank==0:
        write_json(out/'complete.json',dict(complete=True,contract=contract,step=step,validation=score,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            warning='Fixed-budget offline teacher-output distillation; no best selection. Only final paired rollout measures image behavior. This is not trained video.'))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=ROOT)
    p.add_argument('--layout',choices=tuple(LAYOUTS),required=True)
    p.add_argument('--source',choices=('aligned','zero'),default='aligned')
    p.add_argument('--steps',type=int,default=800);p.add_argument('--smoke',action='store_true')
    p.add_argument('--data-root',default='eval_results/csra_rollout_data4k_20260912')
    p.add_argument('--hidden-root',default='eval_results/csra_hidden_sources_20260913')
    main(p.parse_args())
