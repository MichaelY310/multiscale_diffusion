"""Fixed800 additional joint updates, neighbor on/off x source/zero.

Start each source arm at its own fixed direct-control1600 weights. Reset ALL
AdamW moments identically in four arms. Frozen original8/16 are not trained.
This is offline teacher-trajectory distillation, not on-policy training.
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

from csra_neighbor_protocol import (PROTOCOL, FORMAT, OPTIMIZER_POLICY, FIXED_ARGS,
                                    check_parent, check_new, source_paths)
from csra_neighbor_student import NeighborFrontStudent
from csra_front_keep_student import check_initializer
from csra_output_distillation import prediction_loss
from train_csra_rollout_output import RolloutDataset, to_device
from train_csra_prefix_pair import validate
from train_rdd_csra_adapter import atomic_save, load_frozen_teacher
from probe_csra_front_keep import BASE_SHA, file_sha, write_json


def main(args):
    if any(getattr(args, k) != v for k, v in FIXED_ARGS.items()):
        raise ValueError('This pilot has fixed seed/data/base/initializer')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1,2,3':
        raise ValueError('Only the authorized four physical GPUs0,1,2,3')
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    if torch.cuda.mem_get_info(device)[0] < 26*2**30 or shutil.disk_usage('.').free < 30*2**30:
        raise RuntimeError('Need26GiB free/card and30GiB disk; never stop/delete another job')
    torch.cuda.set_per_process_memory_fraction(.24, device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group('nccl', device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 4: raise ValueError('Require actual4x64=256, no accumulation')
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
    model = NeighborFrontStudent(backbone, initial['fits']['fine_only'], args.coarse_source,
        prefix_depth=6, source_layer=13, checkpoint_blocks=True,
        neighbor_enabled=args.neighbor == 'on', neighbor_rank=128).to(device)
    del initial
    parent_path = Path(f'checkpoints/RDD-CSRA-front1-direct_control-{args.coarse_source}-L400K-20260914/latest.pt')
    digest = file_sha(parent_path)
    parent_ck = torch.load(parent_path, map_location='cpu', mmap=True, weights_only=True)
    check_parent(parent_ck, args.coarse_source, digest, BASE_SHA, init_sha)
    model.initialize_from_front_parent(parent_ck['student'])
    parent = dict(path=str(parent_path), sha256=digest, step=1600)
    del parent_ck
    bridge = [p for n, p in model.named_parameters() if not n.startswith('backbone.') and p.requires_grad]
    fine = [p for p in model.backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([dict(params=bridge, lr=1e-4), dict(params=fine, lr=1e-5)], weight_decay=.01)
    if len(optimizer.state) != 0: raise AssertionError('All four arms must reset Adam moments')
    code_sha = {name: file_sha(Path(name)) for name in source_paths()}
    step, best, saved_rng = 1600, float('inf'), None
    if args.resume:
        ck = torch.load(args.resume, map_location='cpu', mmap=True, weights_only=True)
        check_new(ck, vars(args), code_sha, BASE_SHA, init_sha)
        model.load_state_dict(ck['student'], strict=True)
        optimizer.load_state_dict(ck['optimizer'])
        step, best, saved_rng = ck['step'], ck['best_objective'], ck['rank_rng_states']
        del ck
    datasets = [RolloutDataset(args.data_root, split, [13,15,17,20], args.base_ckpt, args.hidden_root)
                for split in ('train','validation')]
    for data in datasets: data.coarse = {13:data.coarse[13]}
    train, val = datasets
    if train.seeds & val.seeds or len(val) % world: raise ValueError('Split overlap/duplicated validation')
    sampler = DistributedSampler(train, world, rank, seed=args.seed, drop_last=True)
    vs = DistributedSampler(val, world, rank, shuffle=False)
    loader = DataLoader(train, batch_size=64, sampler=sampler, drop_last=True, pin_memory=True, num_workers=0)
    validation = DataLoader(val, batch_size=64, sampler=vs, pin_memory=True, num_workers=0)
    # Actual64-image fine forward before training: new correction on/off must
    # preserve exactly the loaded parent's function at zero initialization.
    initial_parity = None
    if not args.resume:
        batch = to_device(next(iter(validation)), device)
        model.neighbor_enabled = False
        with torch.no_grad(): expected = model(batch)
        model.neighbor_enabled = args.neighbor == 'on'
        with torch.no_grad(): actual = model(batch)
        initial_parity = (actual.float()-expected.float()).abs().max().item()
        if initial_parity != 0: raise AssertionError('Actual batch64 initial function changed')
        del actual, expected, batch
    torch.manual_seed(args.seed + rank)
    network = DDP(model, device_ids=[local], broadcast_buffers=False)
    if saved_rng is not None:
        torch.set_rng_state(saved_rng[rank]['cpu'])
        torch.cuda.set_rng_state(saved_rng[rank]['cuda'], device)
    stop = step + 2 if args.smoke else 2400
    trace_path = out/f'start_trace_rank{rank}.json'
    traces = json.loads(trace_path.read_text())['trace'] if args.resume and trace_path.exists() else []
    baseline = validate(model, validation, device)
    best = min(best, baseline['objective'])
    if rank == 0:
        write_json(out/'experiment.json', dict(protocol=PROTOCOL, format=FORMAT, args=vars(args),
            parent=parent, code_sha256=code_sha, base_sha256=BASE_SHA, initializer_sha256=init_sha,
            optimizer_policy=OPTIMIZER_POLICY, initial_validation=baseline, initial_branch_max_abs=initial_parity,
            per_gpu_batch=64, actual_global_batch=256, accumulation=1, fixed_final_step=2400,
            additional_updates=800, note='No warmup. Joint bridge+executed fine weights. Same offline teacher data and paired sample ordering; original8/16 stay independent and frozen.'))
        print(json.dumps(dict(event='initialized',step=step,neighbor=args.neighbor,source=args.coarse_source,
            validation=baseline,initial_branch_max_abs=initial_parity,optimizer_policy=OPTIMIZER_POLICY)),flush=True)
    start = time.perf_counter()
    epoch, skip = divmod(step, len(loader))
    while step < stop:
        sampler.set_epoch(epoch)
        indices = list(sampler)
        for i, batch in enumerate(loader):
            if i < skip: continue
            batch = to_device(batch, device)
            if step in (1600,1601) and not any(row['step'] == step for row in traces):
                traces.append(dict(step=step, indices=indices[i*64:(i+1)*64],
                                   t=batch['t'].cpu().tolist(), labels=batch['labels'].cpu().tolist()))
                write_json(trace_path, dict(rank=rank, trace=traces))
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = prediction_loss(network(batch), batch['target'], .05)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            loss.backward()
            if step in (1600,1601):
                absent = [n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]
                if absent: raise RuntimeError(f'Missing gradients:{absent}')
            norm = torch.nn.utils.clip_grad_norm_(bridge+fine, 1., error_if_nonfinite=True)
            optimizer.step(); step += 1
            if rank == 0 and (step <= 1603 or step % 10 == 0):
                print(json.dumps(dict(event='train',step=step,loss=float(loss),main_mse=float(metrics['main_mse']),
                    grad_norm=float(norm),seconds=time.perf_counter()-start,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)),flush=True)
            if step % 100 == 0 or step == stop:
                score = validate(model, validation, device)
                best = min(best, score['objective'])
                if rank == 0: print(json.dumps(dict(event='validation',step=step,**score)),flush=True)
                if not args.smoke:
                    rng = dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device))
                    all_rng = [None] * world
                    dist.all_gather_object(all_rng, rng)
                    if rank == 0:
                        if shutil.disk_usage('.').free < 16*2**30:
                            raise RuntimeError('Too little space for atomic checkpoint; no old file deleted')
                        atomic_save(dict(format=FORMAT, training_protocol=PROTOCOL,
                            student=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                            best_objective=best, args=vars(args), initializer_sha256=init_sha,
                            base_sha256=BASE_SHA, code_sha256=code_sha, front_keep=1, prefix_depth=6,
                            source_layer=13, neighbor_rank=128, base_checkpoint=str(Path(args.base_ckpt).resolve()),
                            optimizer_policy=OPTIMIZER_POLICY, parent=parent, rank_rng_states=all_rng), out/'latest.pt')
                dist.barrier()
            if step >= stop: break
        epoch, skip = epoch+1, 0
    write_json(trace_path, dict(rank=rank, trace=traces))
    if rank == 0:
        write_json(out/'complete.json', dict(complete=True,step=step,smoke=args.smoke,args=vars(args),
            protocol=PROTOCOL,format=FORMAT,parent=parent,final_validation=score,
            initial_branch_max_abs=initial_parity,optimizer_policy=OPTIMIZER_POLICY,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
    dist.barrier(); dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--neighbor',choices=('off','on'),required=True)
    p.add_argument('--coarse-source',choices=('aligned','zero'),required=True)
    p.add_argument('--output-dir',required=True)
    for key, value in FIXED_ARGS.items():
        p.add_argument('--'+key.replace('_','-'),default=value,type=int if key=='seed' else str)
    p.add_argument('--resume',default='')
    p.add_argument('--smoke',action='store_true')
    main(p.parse_args())
