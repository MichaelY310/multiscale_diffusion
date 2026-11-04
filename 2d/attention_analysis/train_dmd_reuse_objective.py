"""Paired whole-model objective ablation from SAME A@5000 and optimizer/RNG.

control: original MSE continuation. dmd_lpips: RDD score distribution matching
plus online original-DMD-image LPIPS, moderate stage anchors. NOT a literal
DMD2 reproduction, no GAN, no new inference modules. Extra models TRAIN ONLY.
"""
import argparse
import contextlib
import copy
from datetime import timedelta, datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'2d')); sys.path.insert(0, str(ROOT/'DiT'))
from models_multires import DiTMultiRes_models
from dmd_cascade_reuse import CascadeDMD, coupled_noises
from dmd_objective_utils import (DenseScore, chunked_value_gradient,
                                value_with_gradient, schedule, score_candidates, choose_phase)
from dmd_hidden_distillation import (HIDDEN_LAYERS, TERM_NAMES, hidden_config,
                                     hidden_weight_at, hidden_distillation_loss)
from dmd_training_budget import budget_metadata, budget_reached, training_updates
from train_dmd_cascade_reuse import atomic_json, atomic_torch, clean_mse
from diffusion.rdd_phased_dmd import (downsample_clean, rdd_renoise,
                                    rdd_dmd_gradient, dmd_surrogate_loss, fresh_transition)
from train_rdd_phased_dmd3 import HistoricalCFGForward, predict_clean, decode_for_lpips
from diffusion.cross_scale_reuse_config import description


def model_from(state):
    m = DiTMultiRes_models['DiT-L/2'](num_classes=1000, base_size=16)
    m.load_state_dict(state, strict=True)
    return m.eval()


def main(a):
    budget = budget_metadata(a.steps)
    if a.smoke and a.steps == 0:
        raise ValueError('Smoke must have a finite update budget')
    hidden_protocol = hidden_config(a.hidden_weight, a.hidden_ramp_steps)
    use_hidden = hidden_protocol['enabled']
    if a.holdout_start < 0:
        raise ValueError('holdout_start must be nonnegative')
    rank = int(os.environ.get('RANK', 0)); world = int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0)); torch.cuda.set_device(local)
    dev = torch.device('cuda', local); torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(a.memory_fraction, dev)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(814017)
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    control_group = dist.new_group(backend='gloo', timeout=timedelta(minutes=30)) if world > 1 else None
    if a.global_batch % (world * a.batch_per_gpu):
        raise ValueError('global batch must divide world * per-GPU batch')
    accum = a.global_batch // (world * a.batch_per_gpu)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True); (out/'checkpoints').mkdir(exist_ok=True)
    if (out/'protocol.json').exists():
        previous = json.loads((out/'protocol.json').read_text())
        if previous.get('hidden_distillation', {'enabled': False}) != hidden_protocol:
            raise ValueError('Hidden objective changed in an existing run: use a NEW output directory')
    gate = json.loads(Path(a.speed_gate).read_text())
    assert gate['complete'] and gate['candidates']['high_heavy']['passed']
    source = torch.load(a.initial, map_location='cpu', mmap=True, weights_only=False)
    assert source['family'] == 'high_heavy' and source['step'] == 5000
    start_signature = hashlib.sha256(source['model']['final_layer.linear.weight'].numpy().tobytes()).hexdigest()
    student = CascadeDMD(model_from(source['model']).to(dev), 'high_heavy', True).requires_grad_(True)
    teacher_ck = torch.load(a.dmd_teacher, map_location='cpu', mmap=True, weights_only=False)
    teacher = CascadeDMD(model_from(teacher_ck['model']).to(dev), 'dense', False).requires_grad_(False)
    del teacher_ck
    optimizer = torch.optim.AdamW(student.parameters(), lr=a.lr, betas=(.9,.999), weight_decay=.01, fused=True)
    optimizer.load_state_dict(source['optimizer'])
    generator = torch.Generator(device=dev)
    generator.set_state(source['generator_states'][rank])
    auxgen = torch.Generator(device=dev).manual_seed(774410 + rank*100003)
    del source; gc.collect()
    wrapper = DDP(student, device_ids=[local], broadcast_buffers=False, gradient_as_bucket_view=True) if world > 1 else student
    amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16)
    candidates = score_candidates(student.diffusions)
    critic = critic_net = critic_optimizer = score_teacher = guided_teacher = vae = lpips = None
    if a.objective == 'dmd_lpips':
        from diffusers import AutoencoderKL
        import piq
        score_ck = torch.load(a.score_teacher, map_location='cpu', mmap=True, weights_only=False)
        score_weights = score_ck.get('ema', score_ck.get('model'))
        if not isinstance(score_weights, dict): raise ValueError('No diffusion teacher weights')
        score_teacher = model_from(score_weights).to(dev).requires_grad_(False)
        critic_net = DenseScore(model_from(score_weights).to(dev)).eval().requires_grad_(True)
        del score_ck, score_weights; gc.collect()
        critic = DDP(critic_net, device_ids=[local], broadcast_buffers=False,
                     gradient_as_bucket_view=True) if world > 1 else critic_net
        critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=a.critic_lr,
                                             betas=(.9,.999), weight_decay=.01, fused=True)
        guided_teacher = HistoricalCFGForward(score_teacher, 1000, 1.5)
        vae = AutoencoderKL.from_pretrained(a.vae, local_files_only=True).to(dev).eval().requires_grad_(False)
        lpips = piq.LPIPS(reduction='none').to(dev).eval().requires_grad_(False)

    step = 0; warm_step = 0
    resume = out/'checkpoints/latest.pt'
    if a.resume and resume.exists():
        saved = torch.load(resume, map_location='cpu', mmap=True, weights_only=False)
        assert saved['objective'] == a.objective and saved['family'] == 'high_heavy'
        assert saved['initial_signature'] == start_signature
        if saved['protocol'].get('hidden_distillation', {'enabled': False}) != hidden_protocol:
            raise ValueError('Cannot silently change hidden loss when resuming an existing checkpoint')
        student.backbone.load_state_dict(saved['model'], strict=True); optimizer.load_state_dict(saved['optimizer'])
        step = int(saved['step']); warm_step = int(saved['warm_step'])
        state = saved['rng'][rank]; generator.set_state(state['paired']); auxgen.set_state(state['aux'])
        torch.set_rng_state(state['cpu']); torch.cuda.set_rng_state(state['cuda'], dev)
        if critic is not None:
            critic_net.core.backbone.load_state_dict(saved['critic'], strict=True)
            critic_optimizer.load_state_dict(saved['critic_optimizer'])
        del saved; gc.collect()
    stopping = {'yes': False}
    def request_stop(signum, frame): stopping['yes'] = True
    signal.signal(signal.SIGTERM, request_stop); signal.signal(signal.SIGINT, request_stop)

    protocol = {'args': vars(a), 'objective': a.objective, 'source_compression_step': 5000,
                **budget,
                'family': description('high_heavy', 24), 'video_mapping': description('high_heavy',30),
                'initial_signature': start_signature, 'optimizer_restored_from_A5000': True,
                'paired_rng_restored_from_A5000': True, 'world_size': world, 'accumulation': accum,
                'trainable_generator_params': sum(p.numel() for p in student.parameters()),
                'added_inference_params': 0, 'nfe': 3, 'global_t': [999,599,299], 'noise_filling': True,
                'precision': 'FP32 trainable/frozen weights, BF16 autocast, TF32 off',
                'LPIPS_target': 'online frozen ORIGINAL DMD output with same independent noises, not fixed 250-step pairs',
                'score_teacher': 'original un-distilled RDD Unified-L 400k, historical CFG1.5 epsilon channels0:3',
                'critic_initialization': 'original diffusion teacher; trained dense on pooled FINAL student sample',
                'score_time': 'global uniform20..979; RDD exact drift operator gradient',
                'DMD_recipe': 'hybrid objective experiment, not exact original DMD or DMD2 replication',
                'common_lr': a.lr, 'validation': 'fixed historical64 plus 256 new held-out seeds at milestones; NOT FID',
                'storage_policy': 'latest full state plus generator snapshots at milestones; no existing runs removed'}
    protocol['hidden_distillation'] = hidden_protocol
    if a.latest_only:
        protocol['storage_policy'] = 'atomic latest full model/optimizer/critic/RNG only; validation evidence retained'
    if rank == 0:
        atomic_json(out/'protocol.json', protocol); print(json.dumps(protocol), flush=True)
    start_time = time.perf_counter()

    def log(row):
        if rank == 0:
            row.update(budget)
            row['time'] = datetime.now(timezone.utc).isoformat()
            print(json.dumps(row), flush=True)
            with (out/'train.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
            atomic_json(out/'progress.json', row)

    if step > 0:
        log(dict(step=step, total_compression_updates=5000+step, state='resumed', complete=False,
                 resumed_from=str(resume), optimizer_and_rng_restored=True,
                 critic_restored=critic is not None, warm_step=warm_step))

    def stopped():
        flag = torch.tensor(int(stopping['yes'] or (out/'STOP_REQUESTED').exists()), device=dev)
        if world > 1: dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag)

    def save(at_step, snapshot=False):
        rng = {'paired': generator.get_state(), 'aux': auxgen.get_state(),
               'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(dev)}
        all_rng = [None]*world if rank == 0 else None
        if world > 1: dist.gather_object(rng, all_rng, dst=0, group=control_group)
        else: all_rng = [rng]
        error = [None]
        if rank == 0:
            try:
                needed = 14 if critic is not None else 8
                if shutil.disk_usage(out).free < needed*2**30:
                    raise OSError('Insufficient disk for atomic checkpoint; do not delete other runs')
                state = {k:v.detach().cpu() for k,v in student.backbone.state_dict().items()}
                payload = dict(model=state, optimizer=optimizer.state_dict(), step=at_step, warm_step=warm_step,
                               objective=a.objective, family='high_heavy', rng=all_rng,
                               initial_signature=start_signature, protocol=protocol)
                if critic is not None:
                    payload['critic'] = {k:v.detach().cpu() for k,v in critic_net.core.backbone.state_dict().items()}
                    payload['critic_optimizer'] = critic_optimizer.state_dict()
                atomic_torch(resume, payload)
                if snapshot and not a.latest_only:
                    atomic_torch(out/f'checkpoints/step_{at_step:07d}.pt',
                                 dict(model=state, step=at_step, family='high_heavy', objective=a.objective, protocol=protocol))
                del payload, state
            except Exception as exc: error[0] = repr(exc)
        if world > 1: dist.broadcast_object_list(error, src=0, group=control_group)
        if error[0]: raise OSError(error[0])

    def validate(at_step, count=16, seed=9100000, prefix='validation', destination=None, oracle=False):
        folder = out if destination is None else destination
        folder.mkdir(exist_ok=True)
        rng = torch.Generator(device=dev).manual_seed(seed+rank*100003)
        sums = torch.zeros(10, device=dev, dtype=torch.float64); batches=[]
        with torch.no_grad(), amp():
            for offset in range(0, count, 8):
                b = min(8, count-offset); noises, y = coupled_noises(b, dev, rng)
                target = teacher(noises,y)
                pred = student(noises,y,hidden_layers=HIDDEN_LAYERS) if use_hidden else student(noises,y)
                if use_hidden:
                    ref_hidden=[{},{},{}]
                    for s in (1,2):
                        teacher.clean_at(pred['inputs'][s],y,s,capture_layers=HIDDEN_LAYERS[s],hidden_out=ref_hidden[s])
                    _,terms=hidden_distillation_loss(pred['hidden'],ref_hidden)
                    sums[7:]+=torch.stack([terms[k] for k in TERM_NAMES]).double()*b
                    del ref_hidden,terms
                for s in range(3): sums[s] += clean_mse(pred['cleans'][s],target['cleans'][s])*b
                final=pred['cleans'][-1]; ref=target['cleans'][-1]
                sums[3] += (final-ref).square().sum()/ref.square().sum().clamp_min(1e-9)*b
                sums[4] += final.std()*b; sums[5] += float(torch.isfinite(final).all())*b
                if oracle:
                    # Teacher *producer* must capture exactly the student's middle mask.
                    old=teacher.masks; teacher.masks={0:(),1:student.masks[1],2:()}
                    try: coarse,_,cache=teacher.clean_at(noises[0],y,0)
                    finally: teacher.masks=old
                    state=fresh_transition(coarse,16,student.diffusions[1],noise=noises[1])
                    middle,_,cache=student.clean_at(state,y,1,cache)
                    state=fresh_transition(middle,32,student.diffusions[2],noise=noises[2])
                    replacement=student.clean_at(state,y,2,cache)[0]
                    sums[6] += clean_mse(replacement,ref)*b
                batches.append({'labels':y.cpu(),'teacher':ref.float().cpu(),'student':final.float().cpu()})
        atomic_torch(folder/f'{prefix}_{at_step:07d}_rank{rank}.pt',batches)
        if world > 1: dist.all_reduce(sums)
        v=(sums/(world*count)).tolist()
        data=dict(step=at_step,total_compression_updates=5000+at_step,clean_mse_8_16_32=v[:3],
                  final_relative_l2=math.sqrt(v[3]),student_latent_std=v[4],finite_batch_fraction=v[5],
                  validation_images=world*count,NOT_FID=True,student_blend=1.,seed=seed,
                  oracle_coarse_teacher_final_mse=v[6] if oracle else None)
        if use_hidden: data['hidden_nmse']=dict(zip(TERM_NAMES,v[7:]))
        if rank==0: atomic_json(folder/f'{prefix}_{at_step:07d}.json',data); print('VALIDATION '+json.dumps(data),flush=True)
        return data

    def fake_update(final, labels, stage):
        critic_optimizer.zero_grad(set_to_none=True)
        diffusion=student.diffusions[stage]; target=downsample_clean(final.detach(),(8,16,32)[stage])
        t=torch.randint(diffusion.num_timesteps,(len(labels),),device=dev,generator=auxgen)
        noise=torch.randn(target.shape,device=dev,generator=auxgen)
        with amp(): loss=diffusion.training_losses(critic,target,t,model_kwargs={'y':labels},noise=noise)['loss'].mean()
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite fake-score loss')
        loss.backward(); grad=torch.nn.utils.clip_grad_norm_(critic.parameters(),10.,error_if_nonfinite=True)
        critic_optimizer.step()
        return loss.detach(), grad.detach()

    if step==0:
        validate(0,oracle=True)
        if a.objective=='dmd_lpips':
            # CPU/generator warm-up doesn't move paired RNG or student parameters.
            for i in range(warm_step,a.critic_warmup):
                with torch.no_grad(),amp():
                    noises,y=coupled_noises(a.batch_per_gpu,dev,auxgen); final=student(noises,y)['cleans'][-1]
                stage=(2,1,0)[i%3]  # warm all three shapes, including expensive fine phase
                value,grad=fake_update(final,y,stage); warm_step=i+1
                if i%10==0 or warm_step==a.critic_warmup:
                    log(dict(step=0,total_compression_updates=5000,state='critic_warmup',complete=False,
                             warm_step=warm_step,critic_loss=float(value),critic_grad=float(grad),
                             peak_gib=torch.cuda.max_memory_allocated()/2**30,elapsed_s=time.perf_counter()-start_time))
                if stopped():
                    save(0); log(dict(step=0,state='paused',complete=False)); return
            save(0)

    def perceptual_function(z, target):
        with amp():
            predicted=decode_for_lpips(vae,z)
            with torch.no_grad(): reference=decode_for_lpips(vae,target)
        # VGG features in FP32; differentiable decoder BF16 same for both images.
        return lpips(predicted.float(),reference.float())

    # A restart after the final checkpoint but before final telemetry must still
    # finish/export cleanly, not fail on an empty training loop.
    if budget_reached(step, a.steps):
        result=validate(step)
        if not a.smoke:
            validate(step,count=64,seed=27182818,destination=out/'holdout256')
        log(dict(step=step,total_compression_updates=5000+step,complete=True,
                 state='smoke_passed' if a.smoke else 'completed',validation=result,
                 peak_gib=torch.cuda.max_memory_allocated()/2**30))
        if world>1: dist.destroy_process_group()
        return

    final_validation=None
    for update in training_updates(step, a.steps):
        for group in optimizer.param_groups: group['lr']=a.lr
        optimizer.zero_grad(set_to_none=True); torch.cuda.reset_peak_memory_stats()
        stats=torch.zeros(17,device=dev); last_fake=[]
        weights=schedule(update,a.ramp_steps)
        hidden_weight=hidden_weight_at(update,a.hidden_weight,a.hidden_ramp_steps)
        # Smoke explicitly tests nonzero DMD and LPIPS plus every phase at full batch.
        stage=(2,1,0)[(update-1)%3] if a.smoke else choose_phase(candidates,update,174919)
        for micro in range(accum):
            noises,y=coupled_noises(a.batch_per_gpu,dev,generator)
            with torch.no_grad(),amp(): target=teacher(noises,y)
            sync=wrapper.no_sync() if world>1 and micro<accum-1 else contextlib.nullcontext()
            with sync:
                with amp():
                    current=wrapper(noises,y,hidden_layers=HIDDEN_LAYERS) if use_hidden else wrapper(noises,y)
                    pairs=[clean_mse(p,q) for p,q in zip(current['cleans'],target['cleans'])]
                    with torch.no_grad():
                        if use_hidden:
                            teacher_hidden=[{},{},{}]; local_targets=[]
                            # Reuse the already-required on-policy teacher forwards.
                            for s,x in enumerate(current['inputs']):
                                local_targets.append(teacher.clean_at(x.detach(),y,s,
                                    capture_layers=HIDDEN_LAYERS.get(s,()),hidden_out=teacher_hidden[s])[0])
                        else:
                            local_targets=[teacher.clean_at(x.detach(),y,s)[0] for s,x in enumerate(current['inputs'])]
                    onpolicy=sum(clean_mse(p,q) for p,q in zip(current['cleans'],local_targets))/3
                    var=sum(clean_mse(p[:,4:],q[:,4:]) for p,q in zip(current['raws'],target['raws']))/3
                final=current['cleans'][-1]; zero=final.new_zeros(())
                dmd=lpips_proxy=perceptual_value=gradient_rms=zero
                if a.objective=='control':
                    loss=pairs[2]+.5*(pairs[0]+pairs[1])+.25*onpolicy+.01*var
                else:
                    generated=downsample_clean(final,(8,16,32)[stage]); diffusion=student.diffusions[stage]
                    indices=candidates[stage].to(dev)
                    t=indices[torch.randint(len(indices),(len(y),),device=dev,generator=auxgen)]
                    with torch.no_grad(),amp():
                        noisy=rdd_renoise(diffusion,generated.detach(),t,torch.randn(generated.shape,device=dev,generator=auxgen))
                        real=predict_clean(diffusion,guided_teacher,noisy,t,y).float()
                        fake=predict_clean(diffusion,critic_net,noisy,t,y).float()
                        grad=rdd_dmd_gradient(diffusion,generated.detach().float(),real,fake,t)
                        if not torch.isfinite(grad).all(): raise FloatingPointError('Nonfinite exact RDD-DMD gradient')
                        gradient_rms=grad.square().mean().sqrt()
                    dmd=dmd_surrogate_loss(generated,grad)
                    perceptual_value,perceptual_grad=chunked_value_gradient(final,target['cleans'][-1],perceptual_function,a.perceptual_chunk)
                    lpips_proxy=value_with_gradient(final,perceptual_value,perceptual_grad)
                    loss=(weights['final_mse']*pairs[2]+.5*(pairs[0]+pairs[1])+
                          weights['onpolicy']*onpolicy+.01*var+weights['dmd']*dmd+weights['lpips']*lpips_proxy)
                    last_fake.append((final.detach(),y.detach()))
                if use_hidden:
                    hidden_loss,hidden_terms=hidden_distillation_loss(current['hidden'],teacher_hidden)
                    loss=loss+hidden_weight*hidden_loss
                    stats[13:]+=torch.stack([hidden_loss.detach(),*(hidden_terms[k].detach() for k in TERM_NAMES)])/accum
                if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite generator loss update={update}')
                (loss/accum).backward()
                stats[:10]+=torch.stack([loss.detach(),*(p.detach() for p in pairs),onpolicy.detach(),var.detach(),
                                         dmd.detach(),perceptual_value.detach(),gradient_rms.detach(),torch.tensor(float(stage),device=dev)])/accum
                del target,current,local_targets,loss,final,lpips_proxy,dmd
                if use_hidden: del teacher_hidden,hidden_loss,hidden_terms
                if a.objective=='dmd_lpips': del generated,perceptual_grad,real,fake,noisy,grad
        norm=torch.nn.utils.clip_grad_norm_(student.parameters(),1.,error_if_nonfinite=True)
        if a.smoke:
            for i in range(24):
                g=student.backbone.blocks[i].attn.qkv.weight.grad
                if g is None or not torch.isfinite(g).all() or g.abs().sum()==0: raise AssertionError(f'No producer gradient block{i}')
        optimizer.step(); stats[10]=norm
        if critic is not None:
            # Accumulation fallback concatenates only detached small latent tensors;
            # fake-score update uses microbatches below without increasing batch.
            critic_optimizer.zero_grad(set_to_none=True)
            for micro,(final,y) in enumerate(last_fake):
                diffusion=student.diffusions[stage]; clean=downsample_clean(final,(8,16,32)[stage])
                t=torch.randint(diffusion.num_timesteps,(len(y),),device=dev,generator=auxgen)
                sync=critic.no_sync() if world>1 and micro<accum-1 else contextlib.nullcontext()
                with sync,amp():
                    loss=diffusion.training_losses(critic,clean,t,model_kwargs={'y':y},noise=torch.randn(clean.shape,device=dev,generator=auxgen))['loss'].mean()
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite fake loss')
                    (loss/accum).backward(); stats[11]+=loss.detach()/accum
                del loss
            stats[12]=torch.nn.utils.clip_grad_norm_(critic.parameters(),10.,error_if_nonfinite=True)
            critic_optimizer.step()
        del last_fake
        if world>1: dist.all_reduce(stats); stats/=world
        peak=torch.tensor(torch.cuda.max_memory_allocated()/2**30,device=dev)
        if world>1: dist.all_reduce(peak,op=dist.ReduceOp.MAX)
        should_stop=stopped()
        if update%a.log_every==0 or update==1 or a.smoke:
            row=dict(step=update,total_compression_updates=5000+update,state='training',complete=False,
                     loss=float(stats[0]),pair_mse_8_16_32=stats[1:4].tolist(),onpolicy_mse=float(stats[4]),variance_mse=float(stats[5]),
                     dmd_surrogate=float(stats[6]),lpips=float(stats[7]),dmd_gradient_rms=float(stats[8]),score_stage=int(stats[9]),
                     student_grad=float(stats[10]),critic_loss=float(stats[11]),critic_grad=float(stats[12]),
                     weights=weights if critic is not None else 'original MSE',lr=a.lr,global_batch=a.global_batch,
                     max_rank_peak_gib=float(peak),elapsed_s=time.perf_counter()-start_time)
            if use_hidden:
                row['hidden_loss']=float(stats[13]); row['hidden_weight']=hidden_weight
                row['hidden_terms']=dict(zip(TERM_NAMES,stats[14:].tolist()))
            log(row)
        validate_due=update in (1,200,1000,2000) or update%a.eval_every==0 or budget_reached(update,a.steps) or should_stop
        save_due=validate_due or update%1000==0
        if save_due:
            save(update,snapshot=validate_due)
        if validate_due:
            final_validation=validate(update)
            if update>=a.holdout_start and not a.smoke:
                validate(update,count=64,seed=27182818,destination=out/'holdout256')
        if should_stop: break
    log(dict(step=update,total_compression_updates=5000+update,complete=budget_reached(update,a.steps),
             state='smoke_passed' if a.smoke else ('completed' if budget_reached(update,a.steps) else 'paused'),
             peak_gib=float(peak),validation=final_validation,elapsed_s=time.perf_counter()-start_time))
    if world>1: dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--objective',choices=['control','dmd_lpips'],required=True)
    p.add_argument('--initial',required=True); p.add_argument('--dmd-teacher',required=True)
    p.add_argument('--score-teacher',required=True); p.add_argument('--vae',required=True)
    p.add_argument('--output',required=True); p.add_argument('--speed-gate',required=True)
    p.add_argument('--steps',type=int,default=30000,help='Total additional updates since A@5000; 0 means run until explicitly stopped')
    p.add_argument('--global-batch',type=int,default=256)
    p.add_argument('--batch-per-gpu',type=int,default=64); p.add_argument('--perceptual-chunk',type=int,default=2)
    p.add_argument('--lr',type=float,default=1e-5); p.add_argument('--critic-lr',type=float,default=1e-5)
    p.add_argument('--critic-warmup',type=int,default=200); p.add_argument('--ramp-steps',type=int,default=1000)
    p.add_argument('--memory-fraction',type=float,default=.29); p.add_argument('--eval-every',type=int,default=5000)
    p.add_argument('--log-every',type=int,default=10); p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--hidden-weight',type=float,default=0.,help='Training-only hidden distillation, off by default')
    p.add_argument('--hidden-ramp-steps',type=int,default=500)
    p.add_argument('--latest-only',action='store_true',help='Keep atomic full latest, do not emit extra generator snapshots')
    p.add_argument('--holdout-start',type=int,default=1000,help='First eligible scheduled validation with additional 256 images')
    main(p.parse_args())
