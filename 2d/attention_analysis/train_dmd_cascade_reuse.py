"""Finetune a REAL 3-NFE RDD-DMD generator, no adapter or extra inference net.

Fresh coupled-noise teacher trajectories + differentiable student rollout.
Pair loss preserves the original generator; on-policy teacher supervision
reduces distribution mismatch; early input curriculum is removed after warmup.
This is compression distillation, NOT another adversarial/DMD critic training.
"""
import argparse
import copy
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'DiT'))
from models_multires import DiTMultiRes_models
from dmd_cascade_reuse import CascadeDMD,coupled_noises
from diffusion.cross_scale_reuse_config import description
from diffusion.rdd_phased_dmd import fresh_transition


def atomic_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2));os.replace(temp,path)


def atomic_torch(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');torch.save(value,temp);os.replace(temp,path)


def clean_mse(pred,target):
    # Equal per-sample averaging, no data-dependent gradient normalization.
    return (pred.float()-target.float()).square().flatten(1).mean(1).mean()


def main(a):
    rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'));local=int(os.environ.get('LOCAL_RANK','0'))
    torch.cuda.set_device(local);dev=torch.device('cuda',local)
    torch.cuda.set_per_process_memory_fraction(a.memory_fraction,dev)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if world>1:dist.init_process_group('nccl',timeout=timedelta(minutes=12))
    # RNG/checkpoint metadata is CPU data, not a gradient tensor. Keep object
    # collectives and disk-write barriers off NCCL: the installed Blackwell
    # stack failed in its first byte-object gather after 200 healthy updates.
    control_group=dist.new_group(backend='gloo',timeout=timedelta(minutes=12)) if world>1 else None
    if a.global_batch%(world*a.micro_batch):raise ValueError('global batch must divide world*microbatch')
    accumulation=a.global_batch//(world*a.micro_batch)
    if a.speed_gate:
        gate=json.loads(Path(a.speed_gate).read_text())
        if not gate['complete'] or not gate['candidates'].get(a.family,{}).get('passed'):raise ValueError('Video >=20% speed gate NOT passed')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    (out/'checkpoints').mkdir(exist_ok=True)
    torch.manual_seed(a.seed)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    source_step=int(ck['step'])
    base=DiTMultiRes_models['DiT-L/2'](num_classes=1000,base_size=16)
    base.load_state_dict(ck['model'],strict=True)
    teacher=CascadeDMD(copy.deepcopy(base).to(dev).float().eval().requires_grad_(False),'dense',False)
    student=CascadeDMD(base.to(dev).float().eval().requires_grad_(True),a.family,True)
    del base,ck
    generator=torch.Generator(device=dev).manual_seed(a.seed+rank*100003)
    optimizer=torch.optim.AdamW(student.parameters(),lr=a.lr,betas=(.9,.999),weight_decay=.01,fused=True)
    step=0
    if a.resume and (out/'checkpoints/latest.pt').exists():
        previous=torch.load(out/'checkpoints/latest.pt',map_location='cpu',weights_only=False,mmap=True)
        if previous['family']!=a.family:raise ValueError('Resume family mismatch')
        student.backbone.load_state_dict(previous['model'],strict=True);optimizer.load_state_dict(previous['optimizer'])
        step=int(previous['step']);generator.set_state(previous['generator_states'][rank]);del previous
    wrapper=DDP(student,device_ids=[local],broadcast_buffers=False,gradient_as_bucket_view=True) if world>1 else student
    stop={'requested':False}
    def request_stop(signum,frame):stop['requested']=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    amp=lambda:torch.autocast('cuda',dtype=torch.bfloat16)
    if step==0:
        # Verify the custom block traversal against the unchanged original DMD
        # model and its exact q_sample boundary on the REAL checkpoint.
        checkgen=torch.Generator(device=dev).manual_seed(31917)
        check_noises,check_y=coupled_noises(2,dev,checkgen)
        with torch.no_grad(),amp():
            wrapped=teacher(check_noises,check_y)['cleans']
            check_state=check_noises[0];errors=[]
            for s,diff in enumerate(teacher.diffusions):
                t=torch.full((2,),diff.num_timesteps-1,device=dev,dtype=torch.long)
                reference=diff.p_mean_variance(teacher.backbone,check_state,t,clip_denoised=False,model_kwargs={'y':check_y})['pred_xstart'].float()
                torch.testing.assert_close(reference,wrapped[s],atol=0,rtol=0)
                errors.append((reference-wrapped[s]).abs().max().item())
                if s<2:check_state=fresh_transition(reference,(16,32)[s],teacher.diffusions[s+1],noise=check_noises[s+1])
        if rank==0:atomic_json(out/'real_checkpoint_parity.json',{'max_abs_by_stage':errors,'exact':True,'nfe':3})
        del wrapped,reference,check_state,check_noises
    protocol={'family':description(a.family,24),'video_mapping':description(a.family,30),'args':vars(a),'world_size':world,
              'accumulation':accumulation,'architecture':'DiT-L/2','source_dmd_step':source_step,'inference_nfe':3,
              'training_type':'full-generator compression distillation; frozen original DMD generator teacher; no critic',
              'trainable_parameters':sum(p.numel() for p in student.parameters()),'new_parameters':0,
              'global_t':[int(d.timestep_map[-1]) for d in student.diffusions],
              'EMA':'none; compare fixed last checkpoints, original DMD uses current generator',
              'shared_gpu':True,'no_video_finetuning':True}
    if rank==0:atomic_json(out/'protocol.json',protocol);print(json.dumps(protocol),flush=True)

    def validate(at_step,export=True):
        # Fresh training seeds never overlap this fixed validation seed range.
        valgen=torch.Generator(device=dev).manual_seed(9100000+rank*100003)
        losses=torch.zeros(6,device=dev,dtype=torch.float64);saved=[]
        student.eval();torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for _ in range(a.validation_batches):
                noises,y=coupled_noises(a.validation_batch,dev,valgen)
                with amp():
                    target=teacher(noises,y);current=student(noises,y)
                    for s in range(3):losses[s]+=clean_mse(current['cleans'][s],target['cleans'][s])
                    final=current['cleans'][-1];ref=target['cleans'][-1]
                    losses[3]+=(final-ref).square().sum()/ref.square().sum().clamp_min(1e-9)
                    losses[4]+=final.std();losses[5]+=float(torch.isfinite(final).all())
                if export:saved.append({'labels':y.cpu(),'teacher':ref.float().cpu(),'student':final.float().cpu()})
        if export:atomic_torch(out/f'validation_{at_step:07d}_rank{rank}.pt',saved)
        if world>1:dist.all_reduce(losses)
        values=(losses/(world*a.validation_batches)).tolist()
        report={'step':at_step,'clean_mse_8_16_32':values[:3],'final_relative_l2':math.sqrt(values[3]),
                'student_latent_std':values[4],'finite_batch_fraction':values[5],
                'validation_images':world*a.validation_batches*a.validation_batch,
                'NOT_FID':True,'student_blend':1.,'training_seed_disjoint':True}
        if rank==0:atomic_json(out/f'validation_{at_step:07d}.json',report);print('VALIDATION '+json.dumps(report),flush=True)
        return report

    def save(at_step):
        state=generator.get_state();states=[None]*world if rank==0 else None
        if world>1:dist.gather_object(state,states,dst=0,group=control_group)
        else:states=[state]
        if rank==0:
            cpu_model={k:v.detach().cpu() for k,v in student.backbone.state_dict().items()}
            payload={'model':cpu_model,'optimizer':optimizer.state_dict(),'step':at_step,'family':a.family,
                     'generator_states':states,'protocol':protocol}
            atomic_torch(out/'checkpoints/latest.pt',payload)
            atomic_torch(out/f'checkpoints/step_{at_step:07d}.pt',{'model':cpu_model,'step':at_step,'family':a.family,'protocol':protocol})
        if world>1:dist.barrier(group=control_group)

    if step==0 and not a.smoke:validate(0)
    start=time.perf_counter();last=start;last_val=None
    for update in range(step+1,a.steps+1):
        optimizer.zero_grad(set_to_none=True)
        lr=a.lr*min(1.,update/a.lr_warmup)
        for group in optimizer.param_groups:group['lr']=lr
        blend=min(1.,.25+.75*update/a.input_curriculum)
        values=torch.zeros(7,device=dev)
        torch.cuda.reset_peak_memory_stats()
        for micro in range(accumulation):
            noises,y=coupled_noises(a.micro_batch,dev,generator)
            with torch.no_grad(),amp():target=teacher(noises,y)
            # Suppress DDP reductions on accumulated microbatches only.
            import contextlib
            sync=wrapper.no_sync() if world>1 and micro<accumulation-1 else contextlib.nullcontext()
            with sync,amp():
                current=wrapper(noises,y,target['inputs'],blend)
                pair=[clean_mse(p,q) for p,q in zip(current['cleans'],target['cleans'])]
                with torch.no_grad():
                    policy=[teacher.clean_at(x.detach(),y,s)[0] for s,x in enumerate(current['inputs'])]
                local_loss=sum(clean_mse(p,q) for p,q in zip(current['cleans'],policy))/3
                variance_loss=sum(clean_mse(p[:,4:],q[:,4:]) for p,q in zip(current['raws'],target['raws']))/3
                loss=pair[2]+.5*(pair[0]+pair[1])+.25*local_loss+.01*variance_loss
                if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss rank={rank} step={update}')
                (loss/accumulation).backward()
            values+=torch.stack([loss.detach(),*(v.detach() for v in pair),local_loss.detach(),variance_loss.detach(),torch.tensor(blend,device=dev)])/accumulation
            del target,current,policy,loss
        grad=torch.nn.utils.clip_grad_norm_(student.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        if a.smoke and update==1:
            # All blocks, including blocks skipped twice, must receive gradients.
            for i in range(24):
                g=student.backbone.blocks[i].attn.qkv.weight.grad
                if g is None or not torch.isfinite(g).all() or g.abs().sum()==0:raise AssertionError(f'No producer gradient block {i}')
        peak=torch.tensor(torch.cuda.max_memory_allocated()/2**30,device=dev)
        stop_tensor=torch.tensor(int(stop['requested']),device=dev)
        if world>1:
            dist.all_reduce(values);values/=world;dist.all_reduce(peak,op=dist.ReduceOp.MAX);dist.all_reduce(stop_tensor,op=dist.ReduceOp.MAX)
        if update%a.log_every==0 or update==1 or a.smoke:
            now=time.perf_counter()
            row={'step':update,'loss':values[0].item(),'pair_mse_8_16_32':values[1:4].tolist(),'onpolicy_mse':values[4].item(),
                 'variance_mse':values[5].item(),'student_blend':blend,'lr':lr,'grad_norm':float(grad),
                 'max_rank_peak_gib':peak.item(),'elapsed_s':now-start,'global_batch':a.global_batch}
            if rank==0:
                print(json.dumps(row),flush=True)
                with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                atomic_json(out/'progress.json',{**row,'complete':False,'state':'training'})
            last=now
        checkpoint_due=update in (1,200,800,2000) or update%a.save_every==0 or update==a.steps or bool(stop_tensor)
        if checkpoint_due and not a.smoke:
            save(update);last_val=validate(update)
        if bool(stop_tensor):break
    if rank==0:
        result={'complete':update==a.steps,'state':'smoke_passed' if a.smoke else ('completed' if update==a.steps else 'paused'),
                'step':update,'family':a.family,'elapsed_s':time.perf_counter()-start,'peak_gib':peak.item(),'validation':last_val}
        atomic_json(out/('smoke_result.json' if a.smoke else 'progress.json'),result);print(json.dumps(result),flush=True)
    if world>1:dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--family',choices=['high_heavy','balanced','balanced_plus'],required=True)
    p.add_argument('--speed-gate');p.add_argument('--steps',type=int,default=5000)
    p.add_argument('--micro-batch',type=int,default=64);p.add_argument('--global-batch',type=int,default=256)
    p.add_argument('--lr',type=float,default=1e-5);p.add_argument('--lr-warmup',type=int,default=100)
    p.add_argument('--input-curriculum',type=int,default=200);p.add_argument('--memory-fraction',type=float,default=.28)
    p.add_argument('--validation-batches',type=int,default=2);p.add_argument('--validation-batch',type=int,default=8)
    p.add_argument('--save-every',type=int,default=1000);p.add_argument('--log-every',type=int,default=10)
    p.add_argument('--seed',type=int,default=814017);p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true')
    main(p.parse_args())
