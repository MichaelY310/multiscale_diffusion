"""Bridge warmup800 + joint800 versus matched direct-joint1600.

Model, source, loss and inference cost are unchanged. During warmup backbone
gradients are computed by the same DDP graph, then discarded BEFORE clipping
and AdamW: no weight decay or Adam-state update on the backbone. This deliberately
avoids changing DDP graphs; it is not advertised as training-compute saving.
"""
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
from torch.utils.data import DataLoader, DistributedSampler
from csra_front_keep_student import FORMAT, FrontKeepStudent, check_initializer
from csra_staged_protocol import PROTOCOL, phase, check_parent
from csra_output_distillation import prediction_loss
from train_csra_rollout_output import RolloutDataset, to_device
from train_csra_prefix_pair import validate
from train_rdd_csra_adapter import atomic_save, load_frozen_teacher
from probe_csra_front_keep import BASE_SHA, file_sha, write_json


def backbone_digest(model):
    result = hashlib.sha256()
    for name, tensor in model.backbone.state_dict().items():
        result.update(name.encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def main(args):
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    if torch.cuda.mem_get_info(device)[0] < 26*2**30 or shutil.disk_usage('.').free < 50*2**30:
        raise RuntimeError('Need26GiB free/card and50GiB disk; do not stop another job')
    torch.cuda.set_per_process_memory_fraction(.24, device)
    torch.set_num_threads(2)
    dist.init_process_group('nccl', device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 4: raise ValueError('Fixed4x64=256 actual batch, no accumulation')
    out = Path(args.output_dir)
    if rank == 0:
        if not args.resume and out.exists() and any(out.iterdir()): raise FileExistsError(out)
        out.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    initial = torch.load(args.initializer, map_location='cpu', weights_only=True)
    check_initializer(initial)
    init_sha = file_sha(args.initializer)
    if file_sha(args.base_ckpt) != BASE_SHA: raise ValueError('Base checkpoint changed')
    backbone = load_frozen_teacher(Path(args.base_ckpt), device, 'DiT-L/2', 'rdd')
    torch.manual_seed(args.seed)
    model = FrontKeepStudent(backbone, initial['fits']['fine_only'], args.coarse_source,
                             prefix_depth=6,source_layer=13,checkpoint_blocks=True).to(device)
    del initial
    torch.backends.cuda.matmul.allow_tf32 = False
    bridge = [p for n,p in model.named_parameters() if not n.startswith('backbone.')]
    fine = [p for p in model.backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([dict(params=bridge,lr=1e-4),dict(params=fine,lr=1e-5)],weight_decay=.01)
    step, best, parent, warmup_verified = 0, float('inf'), {}, False
    if args.mode == 'direct_control' and not args.resume:
        path = Path(f'checkpoints/RDD-CSRA-front1-skip6-{args.coarse_source}-L400K-20260914/latest.pt')
        ck = torch.load(path,map_location='cpu',weights_only=True)
        check_parent(ck,args.coarse_source,BASE_SHA,init_sha)
        model.load_state_dict(ck['student']);optimizer.load_state_dict(ck['optimizer'])
        step, best = ck['step'], ck['best_objective']
        parent = dict(path=str(path),sha256=file_sha(path),step=step)
        del ck
    code_sha = {name:file_sha(Path(__file__).parent/name) for name in
        ('train_csra_staged_front_pair.py','csra_staged_protocol.py','csra_front_keep_student.py','csra_prefix_student.py')}
    if args.resume:
        ck = torch.load(args.resume,map_location='cpu',weights_only=True)
        if (ck.get('training_protocol') != PROTOCOL or ck['code_sha256'] != code_sha
                or ck['format'] != FORMAT or ck['base_sha256'] != BASE_SHA or ck['initializer_sha256'] != init_sha):
            raise ValueError('Wrong staged resume protocol/code/initializer')
        for key in ('mode','coarse_source','seed','data_root','hidden_root','base_ckpt','initializer'):
            if ck['args'][key] != vars(args)[key]: raise ValueError(f'Resume changed {key}')
        model.load_state_dict(ck['student']);optimizer.load_state_dict(ck['optimizer'])
        step,best,parent,warmup_verified = ck['step'],ck['best_objective'],ck['parent'],ck['warmup_backbone_verified']
        del ck
    warmup_hash = backbone_digest(model) if args.mode == 'staged' and step < 800 else None
    if args.mode == 'staged' and step >= 800 and not warmup_verified:
        raise ValueError('Unverified warmup checkpoint')
    datasets = [RolloutDataset(args.data_root,s,[13,15,17,20],args.base_ckpt,args.hidden_root)
                for s in ('train','validation')]
    for data in datasets: data.coarse = {13:data.coarse[13]}
    train,val = datasets
    if train.seeds & val.seeds or len(val)%world: raise ValueError('Split overlap or duplicated validation')
    sampler = DistributedSampler(train,world,rank,seed=args.seed,drop_last=True)
    vs = DistributedSampler(val,world,rank,shuffle=False)
    loader = DataLoader(train,batch_size=64,sampler=sampler,drop_last=True,pin_memory=True,num_workers=0)
    validation = DataLoader(val,batch_size=64,sampler=vs,pin_memory=True,num_workers=0)
    torch.manual_seed(args.seed+rank)
    network = DDP(model,device_ids=[local],broadcast_buffers=False)
    stop = step+2 if args.smoke else 1600
    trace_path = out/f'start_trace_rank{rank}.json'
    traces = json.loads(trace_path.read_text())['trace'] if args.resume and trace_path.exists() else []
    def save(name):
        atomic_save(dict(format=FORMAT,training_protocol=PROTOCOL,student=model.state_dict(),
            optimizer=optimizer.state_dict(),step=step,best_objective=best,args=vars(args),
            initializer_sha256=init_sha,base_sha256=BASE_SHA,code_sha256=code_sha,
            front_keep=1,prefix_depth=6,source_layer=13,base_checkpoint=str(Path(args.base_ckpt).resolve()),
            parent=parent,warmup_backbone_verified=warmup_verified),out/name)
    baseline = validate(model,validation,device)
    best = min(best,baseline['objective'])
    if rank == 0:
        write_json(out/'experiment.json',dict(protocol=PROTOCOL,args=vars(args),parent=parent,
            base_sha256=BASE_SHA,initializer_sha256=init_sha,code_sha256=code_sha,initial_validation=baseline,
            actual_global_batch=256,per_gpu_batch=64,accumulation=1,total_updates=1600,
            note='Same inference graph. Staged first800 discards fine gradients before clipping/AdamW; next800 joint. Direct control restores full optimizer800 and continues to1600. This is offline teacher-state output distillation, not on-policy.'))
        print(json.dumps(dict(event='initialized',mode=args.mode,source=args.coarse_source,step=step,validation=baseline)),flush=True)
    start = time.perf_counter()
    epoch,skip = divmod(step,len(loader))
    while step < stop:
        sampler.set_epoch(epoch)
        indices = list(sampler)
        for i,batch in enumerate(loader):
            if i < skip: continue
            current_phase = phase(args.mode,step)
            batch = to_device(batch,device)
            if step in (0,1,800,801) and not any(row['step'] == step for row in traces):
                traces.append(dict(step=step,phase=current_phase,indices=indices[i*64:(i+1)*64],
                                   t=batch['t'].cpu().tolist(),labels=batch['labels'].cpu().tolist()))
                write_json(trace_path,dict(rank=rank,trace=traces))
            optimizer.zero_grad(set_to_none=True)
            loss,metrics = prediction_loss(network(batch),batch['target'],.05)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            loss.backward()
            if step in (0,800):
                absent = [n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]
                if absent: raise RuntimeError(f'Missing gradients before phase masking:{absent}')
            if current_phase == 'bridge_only':
                for p in fine: p.grad = None
            active = bridge if current_phase == 'bridge_only' else bridge+fine
            norm = torch.nn.utils.clip_grad_norm_(active,1.,error_if_nonfinite=True)
            optimizer.step();step += 1
            if current_phase == 'bridge_only' and (step == 800 or args.smoke and step == stop):
                if backbone_digest(model) != warmup_hash: raise RuntimeError('Warmup changed original fine backbone')
                if any(p in optimizer.state for p in fine): raise RuntimeError('Warmup changed fine optimizer moments')
                warmup_verified = True
                if rank == 0: print(json.dumps(dict(event='warmup_backbone_exact',step=step,sha256=warmup_hash)),flush=True)
            if rank == 0 and (step <= 3 or step%10 == 0 or step in (801,802)):
                print(json.dumps(dict(event='train',phase=current_phase,step=step,loss=float(loss),
                    main_mse=float(metrics['main_mse']),grad_norm=float(norm),seconds=time.perf_counter()-start,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)),flush=True)
            if step%100 == 0 or step == stop:
                score = validate(model,validation,device)
                improved = score['objective'] < best
                if improved: best = score['objective']
                if rank == 0:
                    print(json.dumps(dict(event='validation',step=step,**score)),flush=True)
                    if not args.smoke:
                        # Fixed1600 evaluation only. One recoverable latest per
                        # arm avoids duplicating multi-GB optimizer states on /scr.
                        save('latest.pt')
                dist.barrier()
            if step >= stop: break
        epoch,skip = epoch+1,0
    write_json(out/f'start_trace_rank{rank}.json',dict(rank=rank,trace=traces))
    if rank == 0:
        write_json(out/'complete.json',dict(complete=True,step=step,smoke=args.smoke,args=vars(args),
            protocol=PROTOCOL,final_validation=score,warmup_backbone_verified=warmup_verified,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
    dist.barrier();dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('staged','direct_control'),required=True)
    p.add_argument('--coarse-source',choices=('aligned','zero'),required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--base-ckpt',default='checkpoints/RDD-Unified-L/0400000.pt')
    p.add_argument('--data-root',default='eval_results/csra_rollout_data4k_20260912')
    p.add_argument('--hidden-root',default='eval_results/csra_hidden_sources_20260913')
    p.add_argument('--initializer',default='eval_results/csra_front_keep6_information_20260914/front1/selected_ridge.pt')
    p.add_argument('--seed',type=int,default=20279000)
    p.add_argument('--resume',default='')
    p.add_argument('--smoke',action='store_true')
    main(p.parse_args())
