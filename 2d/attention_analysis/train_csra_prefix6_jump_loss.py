"""Four matched 1600->2400 continuations: old/jump loss x aligned/zero.

Separate output directories and opt-in trainer. Parent checkpoints, architecture,
sampler, training corpus and existing trainer defaults are never modified.
Compare fixed latest2400, not different validation-selected training budgets.
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
from torch.utils.data import DataLoader, DistributedSampler

from csra_jump_weighted_loss import (TRAIN_TIMES, load_verified_calibration,
                                    weight_table, weighted_prediction_loss)
from csra_output_distillation import prediction_loss
from csra_prefix_student import prefix_student_class, check_prefix_initializer
from train_csra_rollout_output import RolloutDataset, to_device
from train_rdd_csra_adapter import atomic_save, load_frozen_teacher


PARENT_SHA = {
    'aligned': 'bed405396a43aa01e987c9e2ae5cbb3cdcca14f158846d40a31f56a1e6dad689',
    'zero': '9dcd2b80501d5bb3593fd73f5e55f7dcb33d1ddc85839ba41782debfb22b2156',
}
BASE_SHA = 'cade56f41317b2d04045431c139b2dd0097346e6392142a39f189274187a0c9f'
PARENT_VALIDATION = {
    'aligned': (.009811078547500074, .012350292992778122),
    'zero': (.011241913365665823, .012923073372803628),
}


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def check_parent(parent, source):
    cfg = parent['args']
    if (parent['step'] != 1600 or parent['prefix_depth'] != 6 or parent['source_layer'] != 13
            or parent['format'] != 'csra_prefix_conditional_v8'
            or cfg['coarse_source'] != source or cfg['bridge_kind'] != 'conditional_attention'
            or cfg['seed'] != 20279000
            or cfg['base_ckpt'] != 'checkpoints/RDD-Unified-L/0400000.pt'
            or cfg['data_root'] != 'eval_results/csra_rollout_data4k_20260912'
            or cfg['hidden_root'] != 'eval_results/csra_hidden_sources_20260913'
            or cfg['initializer'] != 'eval_results/csra_prefix6_information_20260913/selected_ridge.pt'):
        raise ValueError('Wrong parent, source, architecture or saved corpus')


def check_resume(ck, experiment, cfg, initializer_sha):
    if (ck.get('jump_supervision') != experiment or ck['initializer_sha256'] != initializer_sha
            or ck['format'] != 'csra_prefix_conditional_v8' or ck['prefix_depth'] != 6
            or ck['source_layer'] != 13 or not 1600 <= ck['step'] <= 2400):
        raise ValueError('Resume changed experiment')
    for k in ('coarse_source', 'seed', 'base_ckpt', 'data_root', 'hidden_root', 'initializer', 'bridge_kind'):
        if ck['args'][k] != cfg[k]:
            raise ValueError(f'Resume changed {k}')


def check_optimizer(optimizer):
    if len(optimizer.param_groups) != 2:
        raise ValueError('Changed optimizer groups')
    for lr, group in zip((1e-4, 1e-5), optimizer.param_groups):
        if group['lr'] != lr or group['weight_decay'] != .01 or tuple(group['betas']) != (.9, .999):
            raise ValueError('Changed optimizer configuration')


@torch.no_grad()
def validate(model, loader, device, table):
    keys = ('main_mse', 'variance_mse', 'weighted_main_mse', 'low_mse', 'high_mse')
    total = torch.zeros(len(keys)+1, device=device, dtype=torch.float64)
    for batch in loader:
        batch = to_device(batch, device)
        _, scores = weighted_prediction_loss(model(batch), batch['target'], table[batch['t']])
        n = len(batch['t'])
        for i, key in enumerate(keys):
            total[i] += scores[key].double()*n
        total[-1] += n
    dist.all_reduce(total)
    scores = {k: float(total[i]/total[-1]) for i, k in enumerate(keys)}
    scores.update(states=int(total[-1]), objective=float((total[0]+.05*total[1])/total[-1]),
                  jump_objective=float((total[2]+.05*total[1])/total[-1]))
    return scores


def main(args):
    if not 1 <= args.extra_steps <= 800 or (args.smoke and args.extra_steps != 2):
        raise ValueError('Bounded 800 updates; smoke must be two real optimizer updates')
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    dist.init_process_group('nccl', device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 4 or os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1,2,3':
        raise ValueError('Exactly physical0-3 x64 required, no accumulation')
    if torch.cuda.mem_get_info(device)[0] < 26*2**30:
        raise RuntimeError('Need26GiB/card; do not stop others or reduce batch')
    torch.cuda.set_per_process_memory_fraction(.24, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(args.output_dir)
    if not args.resume and out.exists() and any(out.iterdir()):
        raise FileExistsError('Refusing to overwrite existing experiment output')
    out.mkdir(parents=True, exist_ok=True)
    calibration = load_verified_calibration(args.calibration)
    table = weight_table(calibration, device)
    parent_path = Path(f'checkpoints/RDD-CSRA-prefix6-conditional1600-{args.coarse_source}-L400K-20260913/best.pt')
    parent = torch.load(parent_path, map_location='cpu', mmap=True, weights_only=True)
    check_parent(parent, args.coarse_source)
    cfg = dict(parent['args'])
    hashes = [None]
    if rank == 0:
        hashes[0] = dict(parent=sha_file(parent_path), base=sha_file(cfg['base_ckpt']))
    dist.broadcast_object_list(hashes, src=0, device=device)
    if hashes[0] != dict(parent=PARENT_SHA[args.coarse_source], base=BASE_SHA):
        raise ValueError('Parent/base bytes changed')
    initializer_sha = sha_file(cfg['initializer'])
    if initializer_sha != parent['initializer_sha256']:
        raise ValueError('Initializer changed')
    initial = torch.load(cfg['initializer'], map_location='cpu', weights_only=True)
    check_prefix_initializer(initial, 6)
    backbone = load_frozen_teacher(Path(cfg['base_ckpt']), device, 'DiT-L/2', 'rdd')
    torch.manual_seed(cfg['seed'])
    model = prefix_student_class(cfg['bridge_kind'])(backbone, initial['fits']['fine_only'],
        args.coarse_source, prefix_depth=6).to(device)
    model.load_state_dict(parent['student'])
    bridge = [p for n, p in model.named_parameters() if not n.startswith('backbone.')]
    fine = [p for p in model.backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{'params': bridge, 'lr': 1e-4}, {'params': fine, 'lr': 1e-5}], weight_decay=.01)
    optimizer.load_state_dict(parent['optimizer'])
    check_optimizer(optimizer)
    del parent, initial
    data = [RolloutDataset(cfg['data_root'], split, [13,15,17,20], cfg['base_ckpt'], cfg['hidden_root'])
            for split in ('train', 'validation')]
    train, val = data
    for dataset in data:
        dataset.coarse = {13: dataset.coarse[13]}
        times, counts = torch.unique(dataset.tensors['t'], return_counts=True)
        if set(times.tolist()) != set(TRAIN_TIMES) or len(set(counts.tolist())) != 1:
            raise ValueError('Expected uniform saved eight training times, no weighting of data sampler')
    if train.seeds & val.seeds or train.images != 4096 or val.images != 128:
        raise ValueError('Changed splits or leakage')
    sampler = DistributedSampler(train, world, rank, seed=cfg['seed'], drop_last=True)
    vs = DistributedSampler(val, world, rank, shuffle=False, drop_last=False)
    loader = DataLoader(train, batch_size=64, sampler=sampler, drop_last=True, pin_memory=True, num_workers=0)
    validation = DataLoader(val, batch_size=64, sampler=vs, pin_memory=True, num_workers=0)
    experiment = dict(name='prefix6_jump_loss_2x2_v1', loss_mode=args.loss_mode,
        parent=str(parent_path), parent_sha256=hashes[0]['parent'], parent_step=1600,
        base_sha256=BASE_SHA, calibration_sha256=sha_file(args.calibration), calibration=calibration,
        extra_steps=args.extra_steps, fixed_endpoint=1600+args.extra_steps,
        checkpoint_selection='Compare fixed latest2400 across arms; best archives use old validation objective',
        data_order='DistributedSampler(seed=20279000); restore epoch/offset via divmod(step,128)',
        per_gpu_batch=64, world_size=4, accumulation=1, allocator_cap_fraction=.24,
        trainer_sha256=sha_file(__file__))
    step, best = 1600, float('inf')
    if args.resume:
        ck = torch.load(args.resume, map_location='cpu', mmap=True, weights_only=True)
        check_resume(ck, experiment, cfg, initializer_sha)
        model.load_state_dict(ck['student']); optimizer.load_state_dict(ck['optimizer'])
        check_optimizer(optimizer)
        step, best = ck['step'], ck['best_objective']
        del ck
    cfg.update(output_dir=str(out), max_steps=1600+args.extra_steps, resume=args.resume,
               smoke=args.smoke, jump_loss_mode=args.loss_mode, jump_calibration=args.calibration)
    torch.manual_seed(cfg['seed']+rank)
    network = DDP(model, device_ids=[local], broadcast_buffers=False)
    params = [p for p in model.parameters() if p.requires_grad]
    baseline = validate(model, validation, device, table)
    if not args.resume:
        for key, expected in zip(('main_mse', 'variance_mse'), PARENT_VALIDATION[args.coarse_source]):
            if abs(baseline[key]-expected) > 1e-10:
                raise RuntimeError(f'Parent validation parity failed: {key} {baseline[key]} != {expected}')
    best = min(best, baseline['objective'])

    def save(name):
        atomic_save(dict(format='csra_prefix_conditional_v8', student=model.state_dict(),
            optimizer=optimizer.state_dict(), step=step, best_objective=best, args=cfg,
            initializer_sha256=initializer_sha, prefix_depth=6, source_layer=13,
            base_checkpoint=str(Path(cfg['base_ckpt']).resolve()), jump_supervision=experiment,
            note='Original8/16 separate and unchanged; inference architecture unchanged. Offline teacher states. '
                 'Use fixed latest endpoint, not best checkpoints, for the 2x2 comparison.'), out/name)

    if rank == 0:
        print(json.dumps(dict(event='initialized', step=step, experiment=experiment, validation=baseline,
            trainable_backbone=sum(p.numel() for p in fine), trainable_bridge=sum(p.numel() for p in bridge))), flush=True)
        (out/'experiment.json').write_text(json.dumps(dict(experiment=experiment, config=cfg), indent=2)+'\n')
        if not args.smoke and not args.resume:
            save('best.pt')
    dist.barrier()
    start = time.perf_counter()
    initial_step = step
    score = baseline  # a restart may already be at the fixed endpoint
    epoch, skip = divmod(step, len(loader))
    if len(loader) != 128:
        raise ValueError('Unexpected loader length; sampler resume would change')
    while step < 1600+args.extra_steps:
        sampler.set_epoch(epoch)
        indices = list(iter(sampler))
        for i, batch in enumerate(loader):
            if i < skip:
                continue
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = network(batch)
            if args.loss_mode == 'old':
                loss, scores = prediction_loss(output, batch['target'], .05)
            else:
                loss, scores = weighted_prediction_loss(output, batch['target'], table[batch['t']])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            if step == initial_step:
                absent = [n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]
                if absent:
                    raise RuntimeError(f'Missing gradients: {absent}')
                if any(p.grad is not None for b in model.backbone.blocks[:6] for p in b.parameters()):
                    raise RuntimeError('Skipped original prefix must stay frozen')
            norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            optimizer.step()
            step += 1
            if step <= initial_step+2:
                trace = dict(rank=rank, step=step, epoch=epoch, batch_index=i,
                    indices_sha256=hashlib.sha256(json.dumps(indices[i*64:(i+1)*64]).encode()).hexdigest(),
                    first_indices=indices[i*64:i*64+4], times=batch['t'].cpu().tolist(),
                    loss=float(loss), grad_norm=float(norm))
                (out/f'start_trace_rank{rank}_step{step}.json').write_text(json.dumps(trace, indent=2)+'\n')
            if rank == 0 and (step <= initial_step+2 or step % 10 == 0):
                print(json.dumps(dict(event='train', step=step, main_mse=float(scores['main_mse']),
                    loss=float(loss), grad_norm=float(norm), seconds=time.perf_counter()-start,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)), flush=True)
            if step % 100 == 0 or step == 1600+args.extra_steps:
                score = validate(model, validation, device, table)
                improved = score['objective'] < best
                if improved:
                    best = score['objective']
                if rank == 0:
                    print(json.dumps(dict(event='validation', step=step, **score, new_best=improved)), flush=True)
                    if not args.smoke:
                        if improved:
                            save('best.pt')
                        save('latest.pt')
                dist.barrier()
            if step >= 1600+args.extra_steps:
                break
        epoch, skip = epoch+1, 0
    peak = torch.tensor([torch.cuda.max_memory_allocated()/2**30,
                         torch.cuda.max_memory_reserved()/2**30], device=device)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    if rank == 0:
        (out/'complete.json').write_text(json.dumps(dict(complete=True, step=step, smoke=args.smoke,
            experiment=experiment, config=cfg, best_objective=best, validation=score,
            max_rank_peak_allocated_gib=float(peak[0]), max_rank_peak_reserved_gib=float(peak[1])), indent=2)+'\n')
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--coarse-source', choices=('aligned','zero'), required=True)
    p.add_argument('--loss-mode', choices=('old','jump'), required=True)
    p.add_argument('--calibration', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--extra-steps', type=int, default=800)
    p.add_argument('--resume', default='')
    p.add_argument('--smoke', action='store_true')
    main(p.parse_args())
