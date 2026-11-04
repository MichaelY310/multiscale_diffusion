"""Same inference graph/parent/update budget; teacher-only vs50% snapshot replay.

400 further updates, actual4x64 batch256, identical Adam reset. No best-model
selection. Original8/16 never trained. Snapshot replay is NOT live on-policy KD.
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
from torch.utils.data import DataLoader, DistributedSampler

from csra_state_replay_data import PROTOCOL, ROOT, PARENT, PARENT_SHA, BASE_SHA, SEED, PairedReplayDataset
from csra_neighbor_protocol import source_paths
from sample_csra_neighbor_quality import checked_loader
from csra_output_distillation import prediction_loss
from train_csra_prefix_pair import validate
from train_csra_rollout_output import to_device
from train_rdd_csra_adapter import atomic_save
from probe_csra_front_keep import file_sha, write_json


def source_hashes():
    paths = source_paths() + ['2d/attention_analysis/train_csra_state_replay.py',
                              '2d/attention_analysis/csra_state_replay_data.py',
                              '2d/attention_analysis/collect_csra_state_replay.py']
    return {p: file_sha(p) for p in paths}


def dataset_hashes():
    result = {}
    for path in sorted((ROOT/'data').glob('shard_*/manifest.json')):
        manifest = json.loads(path.read_text())
        if not manifest.get('complete') or manifest.get('smoke') or manifest.get('protocol') != PROTOCOL:
            raise ValueError('Require completed formal replay data')
        result[str(path)] = file_sha(path)
        for item in manifest['files']:
            if Path(item['name']).name != item['name']:
                raise ValueError('Unsafe data filename')
            if file_sha(path.parent/item['name']) != item['sha256']:
                raise ValueError('Replay file checksum mismatch')
    if len(result) != 4:
        raise ValueError('Missing replay data shards')
    return result


def make_model(device, trainable=False):
    if file_sha(PARENT) != PARENT_SHA:
        raise ValueError('Fixed2400 snapshot changed')
    model, parent = checked_loader('aligned', device, PARENT, 6, 2400)
    if trainable:
        model.requires_grad_(True)
        model.neighbor.requires_grad_(False)
        for block in model.backbone.blocks[1:7]:
            block.requires_grad_(False)
        model.checkpoint_blocks = True
    return model, parent


def load_result(path, device, expected_arm):
    ck = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
    if (ck.get('protocol') != PROTOCOL or ck.get('updates') != 400 or ck.get('smoke')
            or ck.get('arm') != expected_arm or ck.get('parent_sha256') != PARENT_SHA
            or ck.get('base_sha256') != BASE_SHA or ck.get('code_sha256') != source_hashes()
            or ck.get('data_manifests_sha256') != dataset_hashes()):
        raise ValueError('Wrong fixed400 replay result/provenance')
    model, parent = make_model(device)
    model.load_state_dict(ck['student'], strict=True)
    return model, dict(checkpoint=str(path), checkpoint_sha256=file_sha(path),
                       arm=expected_arm, updates=400, parent=parent, protocol=PROTOCOL,
                       validation=ck['validation'], code_sha256=ck['code_sha256'])


def train(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1,2,3':
        raise ValueError('Only authorizedGPUs0-3')
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    dev = torch.device('cuda', local)
    torch.set_num_threads(2)
    if torch.cuda.mem_get_info(dev)[0] < 26*2**30 or shutil.disk_usage('.').free < 20*2**30:
        raise RuntimeError('Need26GiB GPU/20GiB disk; never stop/delete another job')
    torch.cuda.set_per_process_memory_fraction(.24, dev)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group('nccl', device_id=dev)
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 4:
        raise ValueError('Requireactual4x64, no accumulation')
    out = Path(args.output_dir)
    if rank == 0:
        if not args.resume and out.exists() and any(out.iterdir()):
            raise FileExistsError(out)
        out.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    code, data_sha = source_hashes(), dataset_hashes()
    model, parent = make_model(dev, trainable=True)
    trainset = PairedReplayDataset(ROOT/'data', 'train', args.arm)
    valsets = {r:PairedReplayDataset(ROOT/'data','validation',fixed_role=r) for r in ('teacher','student')}
    if trainset.images != 4096 or any(v.images != 128 or trainset.seeds & v.seeds for v in valsets.values()):
        raise ValueError('Wrong split sizes or seed leakage')
    sampler = DistributedSampler(trainset, world, rank, seed=SEED, drop_last=True)
    loader = DataLoader(trainset, batch_size=64, sampler=sampler, pin_memory=True, num_workers=0, drop_last=True)
    val = {r:DataLoader(v,batch_size=64,sampler=DistributedSampler(v,world,rank,shuffle=False),
                      pin_memory=True,num_workers=0) for r,v in valsets.items()}
    bridge = [p for n,p in model.named_parameters() if not n.startswith('backbone.') and p.requires_grad]
    fine = [p for p in model.backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([dict(params=bridge,lr=1e-4),dict(params=fine,lr=1e-5)],weight_decay=.01)
    torch.manual_seed(SEED+rank)
    updates, saved_rng = 0, None
    if args.resume:
        ck = torch.load(args.resume,map_location='cpu',mmap=True,weights_only=True)
        if (ck.get('protocol')!=PROTOCOL or ck.get('arm')!=args.arm or ck.get('smoke')
                or ck.get('parent_sha256')!=PARENT_SHA or ck.get('code_sha256')!=code
                or ck.get('data_manifests_sha256')!=data_sha or not 0<=ck.get('updates',-1)<=400):
            raise ValueError('Resume contract mismatch')
        model.load_state_dict(ck['student'],strict=True);optimizer.load_state_dict(ck['optimizer'])
        updates,saved_rng=ck['updates'],ck['rank_rng_states'];del ck
    network = DDP(model,device_ids=[local],broadcast_buffers=False)
    if saved_rng is not None:
        torch.set_rng_state(saved_rng[rank]['cpu']);torch.cuda.set_rng_state(saved_rng[rank]['cuda'],dev)
    baseline = {r:validate(model,l,dev) for r,l in val.items()}
    if rank==0:
        write_json(out/'experiment.json',dict(protocol=PROTOCOL,arm=args.arm,smoke=args.smoke,parent=parent,
            parent_sha256=PARENT_SHA,code_sha256=code,data_manifests_sha256=data_sha,initial_validation=baseline,
            global_batch=256,per_gpu_batch=64,accumulation=1,updates_planned=400,seed=SEED,
            optimizer='AdamW reset in both arms; bridge1e-4 fine1e-5 WD.01 clip1',
            objective='mainMSE+.05variance; teacher targets recomputed at the selected state',
            warning='Fixed snapshot replay; not live on-policy, no backward through sampling, no new inference cost.'))
        print(json.dumps(dict(event='initialized',arm=args.arm,updates=updates,validation=baseline)),flush=True)
    traces=[]
    start=time.perf_counter();stop=2 if args.smoke else 400
    epoch,skip=divmod(updates,len(loader))
    score=baseline
    while updates<stop:
        sampler.set_epoch(epoch);trainset.epoch=epoch
        for batch_index,batch in enumerate(loader):
            if batch_index<skip:continue
            if updates in (0,1):
                traces.append(dict(update=updates,indices=batch['index'].tolist(),labels=batch['labels'].tolist(),
                                   t=batch['t'].tolist(),state_role=batch['state_role'].tolist()))
                write_json(out/f'trace_rank{rank}.json',dict(rank=rank,trace=traces))
            batch=to_device(batch,dev)
            optimizer.zero_grad(set_to_none=True)
            loss,metrics=prediction_loss(network(batch),batch['target'],.05)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward()
            if updates==0:
                absent=[n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]
                if absent:raise RuntimeError(f'Missing trainable gradients:{absent}')
            norm=torch.nn.utils.clip_grad_norm_(bridge+fine,1.,error_if_nonfinite=True)
            optimizer.step();updates+=1
            if rank==0 and (updates<=2 or updates%25==0):
                print(json.dumps(dict(event='update',arm=args.arm,updates=updates,
                    main_mse=float(metrics['main_mse']),grad_norm=float(norm),
                    elapsed=time.perf_counter()-start,peak_gib=torch.cuda.max_memory_allocated()/2**30)),flush=True)
            if updates%100==0 or updates==stop:
                score={r:validate(model,l,dev) for r,l in val.items()}
                if rank==0:print(json.dumps(dict(event='validation',updates=updates,scores=score)),flush=True)
                if not args.smoke:
                    rng=dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(dev))
                    states=[None]*world;dist.all_gather_object(states,rng)
                    # Collectively fail before writing: do not leave other ranks stuck.
                    disk_ok=torch.tensor(int(shutil.disk_usage('.').free>=16*2**30),device=dev)
                    dist.all_reduce(disk_ok,op=dist.ReduceOp.MIN)
                    if not int(disk_ok):raise RuntimeError('Insufficient save headroom')
                    if rank==0:
                        atomic_save(dict(protocol=PROTOCOL,arm=args.arm,smoke=False,updates=updates,
                            student=model.state_dict(),optimizer=optimizer.state_dict(),rank_rng_states=states,
                            parent_sha256=PARENT_SHA,base_sha256=BASE_SHA,code_sha256=code,
                            data_manifests_sha256=data_sha,validation=score),out/'latest.pt')
                    dist.barrier()
            if updates>=stop:break
        epoch,skip=epoch+1,0
    if rank==0:
        write_json(out/'complete.json',dict(complete=True,protocol=PROTOCOL,arm=args.arm,smoke=args.smoke,
            updates=updates,parent_sha256=PARENT_SHA,final_validation=score,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=('teacher_only','mixed50'),required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--resume',default='');p.add_argument('--smoke',action='store_true')
    train(p.parse_args())
