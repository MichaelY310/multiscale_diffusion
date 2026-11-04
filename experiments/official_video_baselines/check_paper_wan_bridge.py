"""Independent paper optimization loop versus the actual FastVideo Trainer.

The paper implementation's backbone/math have separate native-Wan/scalar
reference checks. This check isolates framework backward/update integration;
it cannot establish parity with unpublished author code.
"""
import argparse
import json
import os
from pathlib import Path
import time
import torch
import torch.nn.functional as F


def reference_step(student, teacher, rows, generator, per_stage, weight):
    from pyramidal_wan_paper_math import prepare, teacher_boundary, NATURAL_CLEAN, NATURAL_NOISY
    samples=student.prepare_batch(rows,generator=generator)
    prepared=[];texts=[];destinations=[]
    for n,sample in enumerate(samples):
        stage=n//per_stage
        clean=sample['pyramid'][0]
        epsilon=torch.randn(clean.shape,device=clean.device,dtype=clean.dtype,generator=generator)
        natural=torch.rand(1,device=clean.device,generator=generator)
        natural=natural*(NATURAL_NOISY[stage]-NATURAL_CLEAN[stage])+NATURAL_CLEAN[stage]
        prepared.append(prepare(sample['pyramid'],epsilon,natural,stage));texts.append(sample['text'])
        with torch.no_grad(),torch.autocast('cuda',enabled=False):
            original_input=(1-natural.reshape(1,1,1,1,1))*clean+natural.reshape(1,1,1,1,1)*epsilon
            velocity=teacher.transformer(original_input.to(torch.bfloat16),timestep=natural*1000,
                encoder_hidden_states=sample['text'],return_dict=False)[0].float()
            original_partial=original_input-(natural.reshape(1,1,1,1,1)-NATURAL_CLEAN[stage])*velocity
            destinations.append(teacher_boundary(original_partial,stage))
    with torch.autocast('cuda',dtype=torch.bfloat16):
        velocities=student.transformer([x.noisy for x in prepared],[x.natural for x in prepared],texts)
    # Eqs.7/9/10 expressed independently from the Method's aggregate loss.
    flow_terms=[];teacher_terms=[]
    for state,predicted,destination in zip(prepared,velocities,destinations):
        flow_terms.append(F.mse_loss(predicted.float(),state.target))
        student_partial=state.noisy-(state.sigma-state.cleaner_sigma)*predicted.float()
        teacher_terms.append(F.mse_loss(student_partial,destination))
    return torch.stack(flow_terms).mean()+weight*torch.stack(teacher_terms).mean()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--mode',choices=['reference','fastvideo'],required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if not os.environ.get('RDD_JOB_TOKEN') or time.time()>=float(os.environ.get('RDD_DEADLINE_UNIX','0')):
        raise RuntimeError('Independent GPU guard required')
    torch.use_deterministic_algorithms(True)
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
    from fastvideo.train.utils.config import load_run_config
    from fastvideo.train.utils.builder import build_from_config
    from full_model_trace import hashed
    cfg=load_run_config(args.config)
    if cfg.training.loop.max_train_steps!=3 or cfg.training.distributed.num_gpus!=1:
        raise ValueError('This full-model integration parity check requires three steps on one GPU')
    maybe_init_distributed_environment_and_model_parallel(1,1)
    _,method,loader,start=build_from_config(cfg)
    trace={'inputs':[],'predictions':[],'teacher_inputs':[],'teacher_predictions':[],
           'gradients':[],'updates':[],'losses':[]}
    handles=[]
    def before(module,arguments,kwargs):trace['inputs'].append(hashed(arguments))
    handles.append(method.student.transformer.register_forward_pre_hook(before,with_kwargs=True))
    handles.append(method.student.transformer.register_forward_hook(lambda m,a,y:trace['predictions'].append(hashed(y))))
    handles.append(method.teacher.transformer.register_forward_pre_hook(
        lambda m,a,k:trace['teacher_inputs'].append(hashed({'args':a,'kwargs':k})),with_kwargs=True))
    handles.append(method.teacher.transformer.register_forward_hook(lambda m,a,y:trace['teacher_predictions'].append(hashed(y))))
    optimizer=method.optimizer
    handles.append(optimizer.register_step_pre_hook(lambda *a:trace['gradients'].append(
        hashed({k:p.grad for k,p in method.student.transformer.named_parameters()}))))
    handles.append(optimizer.register_step_post_hook(lambda *a:trace['updates'].append({
        'weights':hashed(method.student.transformer.state_dict()),'adamw':hashed(optimizer.state_dict())})))
    teacher_initial=hashed(method.teacher.transformer.state_dict())
    try:
        if args.mode=='fastvideo':
            from fastvideo.train.trainer import Trainer
            trainer=Trainer(cfg.training,config=cfg.resolved_config(),callback_configs=cfg.callbacks)
            old_step=method.single_train_step
            def step(batch,iteration):
                value=old_step(batch,iteration)
                trace['losses'].append(hashed(value[0]['total_loss']))
                return value
            method.single_train_step=step
            trainer.run(method,dataloader=loader,max_steps=3)
        else:
            generator=torch.Generator(device=method.student.device).manual_seed(cfg.training.data.seed)
            method.student.transformer.train();method.teacher.transformer.eval()
            reader=iter(loader)
            for iteration in range(3):
                rows=next(reader,None)
                if rows is None:reader=iter(loader);rows=next(reader)
                optimizer.zero_grad(set_to_none=True)
                loss=reference_step(method.student,method.teacher,rows,generator,method.per_stage,method.teacher_weight)
                trace['losses'].append(hashed(loss));loss.backward()
                if method.clip_grad>0:
                    torch.nn.utils.clip_grad_norm_(method.student.transformer.parameters(),method.clip_grad)
                optimizer.step();method.lr_scheduler.step()
        teacher_final=hashed(method.teacher.transformer.state_dict())
        assert teacher_initial==teacher_final
        result={'status':'passed','mode':args.mode,'trace':trace,'frozen_teacher_unchanged':True,
                'full_pretrained_model':True,'real_data':True,'author_source_parity':False,
                'scope':'paper reference optimization loop versus FastVideo framework; independent native-backbone and scalar-equation checks are separate'}
        args.output.write_text(json.dumps(result))
    finally:
        for handle in handles:handle.remove()


if __name__=='__main__':main()
