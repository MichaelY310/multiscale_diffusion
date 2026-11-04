"""One predeclared800->1600 extension, new output/protocol; old800 immutable.

Restore each arm's exact optimizer and RNG, not weights-only extra training.
Both source controls and front0 have identical800 additional updates. No best.
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
from torch.utils.data import DataLoader,DistributedSampler
from csra_layout_student import LayoutStudent,FORMAT
from csra_output_distillation import prediction_loss
from train_csra_rollout_output import RolloutDataset,to_device
from train_csra_prefix_pair import validate
from train_rdd_csra_adapter import atomic_save,load_frozen_teacher
from train_csra_layout_pilot import hashes,ROOT
from csra_layer_layouts import layout
from probe_csra_front_keep import BASE_SHA,file_sha
from probe_csra_prefix_information import BASE,write_json

ARMS=('front0_6_aligned','split3_3_aligned','split3_3_zero')
VERSION='csra_layout_fixed1600_continuation_v1'


def continuation_hashes():
    return {**hashes(),str(Path(__file__)):file_sha(__file__)}


def main(a):
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if visible!='0,1,2,3':raise ValueError('Only physical0-3 authorized')
    local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local);dev=torch.device('cuda',local)
    dist.init_process_group('nccl',device_id=dev);rank,world=dist.get_rank(),dist.get_world_size()
    if world!=4:raise ValueError('Four-card batch contract')
    if torch.cuda.mem_get_info(dev)[0]<26*2**30:raise RuntimeError('Need26GiB free; do not displace jobs')
    torch.cuda.set_per_process_memory_fraction(.24,dev);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    root=Path(a.root);name,source=a.arm.rsplit('_',1)
    out=root/('continuation1600_smoke' if a.smoke else 'continuation1600')/a.arm
    out.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(out).free<24*2**30:raise RuntimeError('Need24GiB disk')
    parent=root/'training'/a.arm/'latest.pt';parent_sha=file_sha(parent)
    ck=torch.load(parent,map_location='cpu',weights_only=True,mmap=True);pc=ck['contract']
    ip=root/'layouts'/name/'selected_ridge.pt'
    if (ck['step']!=800 or pc['format']!=FORMAT or pc['layout']!=layout(name)
        or pc['coarse_source']!=source or pc['smoke'] or pc['source_sha256']!=hashes()
        or pc['base_sha256']!=BASE_SHA or file_sha(BASE)!=BASE_SHA
        or pc['initializer_sha256']!=file_sha(ip)):
        raise ValueError('Wrong parent checkpoint/code/initializer')
    for path,digest in pc['data_manifests_sha256'].items():
        if file_sha(path)!=digest:raise ValueError('Changed training data')
    goal=802 if a.smoke else 1600
    contract={**pc,'format':VERSION,'model_format':FORMAT,'parent_contract':pc,
        'parent_checkpoint':str(parent),'parent_latest_sha256':parent_sha,'source_sha256':continuation_hashes(),
        'start_step':800,'updates':goal,'additional_updates':goal-800,'smoke':a.smoke,
        'optimizer_policy':'exact parent optimizer/RNG; fixed LR unchanged',
        'selection':'fixed final only; paired quality required; no automatic further extension'}
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())!=contract:
        raise ValueError('Existing different continuation not overwritten')
    if rank==0:write_json(out/'protocol.json',contract)
    if (out/'complete.json').exists():dist.destroy_process_group();return
    init=torch.load(ip,map_location='cpu',weights_only=True)
    model=LayoutStudent(load_frozen_teacher(BASE,dev,'DiT-L/2','rdd'),init,name,source).to(dev);del init
    data=[RolloutDataset('eval_results/csra_rollout_data4k_20260912',s,[13,15,17,20],str(BASE),
        'eval_results/csra_hidden_sources_20260913') for s in ('train','validation')]
    for ds in data:ds.coarse={13:ds.coarse[13]}
    if data[0].seeds & data[1].seeds:raise ValueError('Overlapping splits')
    ts=DistributedSampler(data[0],world,rank,seed=pc['seed'],drop_last=True)
    vs=DistributedSampler(data[1],world,rank,shuffle=False)
    loader=DataLoader(data[0],batch_size=64,sampler=ts,drop_last=True,num_workers=0,pin_memory=True)
    vl=DataLoader(data[1],batch_size=64,sampler=vs,num_workers=0,pin_memory=True)
    optimizer=torch.optim.AdamW([dict(params=model.readers.parameters(),lr=1e-4),
        dict(params=[p for p in model.backbone.parameters() if p.requires_grad],lr=1e-5)],weight_decay=.01)
    resume=out/'latest.pt'
    if resume.exists():
        del ck;ck=torch.load(resume,map_location='cpu',weights_only=True,mmap=True)
        if ck['contract']!=contract:raise ValueError('Continuation resume mismatch')
    model.load_state_dict(ck['student']);optimizer.load_state_dict(ck['optimizer']);step=ck['step']
    torch.set_rng_state(ck['rng'][rank]['cpu']);torch.cuda.set_rng_state(ck['rng'][rank]['cuda'],dev);del ck
    if not 800<=step<goal:raise ValueError('Unexpected resume step')
    net=DDP(model,device_ids=[local],broadcast_buffers=False)
    params=[p for p in model.parameters() if p.requires_grad]
    initial=validate(model,vl,dev)
    if rank==0:print(json.dumps(dict(event='restored',step=step,validation=initial,contract=contract)),flush=True)
    epoch,skip=divmod(step,len(loader));begin=time.monotonic();first=True
    while step<goal:
        ts.set_epoch(epoch)
        for i,batch in enumerate(loader):
            if i<skip:continue
            optimizer.zero_grad(set_to_none=True);batch=to_device(batch,dev)
            loss,metrics=prediction_loss(net(batch),batch['target'],.05)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            if first:
                missing=[n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]
                if missing:raise RuntimeError(f'Missing gradients:{missing}')
                first=False
            optimizer.step();step+=1
            if rank==0 and (step<=803 or step%20==0):
                print(json.dumps(dict(event='train',step=step,loss=float(loss),main_mse=float(metrics['main_mse']),
                    grad_norm=float(norm),elapsed=time.monotonic()-begin,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)),flush=True)
            if step%200==0 or step==goal:
                score=validate(model,vl,dev);rng=[None]*world
                dist.all_gather_object(rng,dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(dev)))
                if rank==0:
                    print(json.dumps(dict(event='validation',step=step,**score)),flush=True)
                    if not a.smoke:
                        if shutil.disk_usage(out).free<12*2**30:raise RuntimeError('Checkpoint disk budget')
                        atomic_save(dict(contract=contract,step=step,student=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng,validation=score),resume)
                        if step==goal:atomic_save(dict(contract=contract,step=step,student=model.state_dict(),validation=score),out/'final.pt')
                dist.barrier()
            if step>=goal:break
        epoch,skip=epoch+1,0
    if rank==0:write_json(out/'complete.json',dict(complete=True,contract=contract,step=step,validation=score,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=ROOT);p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--smoke',action='store_true');main(p.parse_args())
