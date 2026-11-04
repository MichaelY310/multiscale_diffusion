"""All90-time same-state errors on teacher AND student trajectories.

This is a bounded diagnosis of fixed prefix6 best1600, not a quality/speed
benchmark or proof that on-policy training works. No checkpoint changes.
"""
import argparse
import json
from pathlib import Path
import torch
from diagnose_csra_prefix_rollout_shift import StateProbe
from audit_csra_jump_head_sensitivity import audit
from sample_csra_prefix_quality_pilot import load_student, BASE
from sample_csra_prefix_preview import PrefixRollout
from sample_csra_adapter_preview import sample_rdd, relative_l2
from train_rdd_csra_adapter import load_frozen_teacher
from probe_csra_prefix_information import write_json

TEMPLATE='checkpoints/RDD-CSRA-prefix6-conditional1600-{arm}-L400K-20260913/best.pt'


def local_jump_error(row,gain):
    # Exact local linear response on orthogonal cell-mean/difference bands;
    # not a terminal-error decomposition (errors may cancel or amplify).
    return row['low_mse']*gain['low_gain']**2+row['high_mse']*gain['high_gain']**2


@torch.inference_mode()
def run(args, student_loader=load_student, checkpoint_template=TEMPLATE, checkpoint_step=1600):
    out=args.root/f'seed_{args.seed}_{args.arm}.json'
    if out.exists(): raise FileExistsError('Completed diagnostic must not be rerun')
    torch.set_num_threads(2)
    dev=torch.device('cuda:0')
    if torch.cuda.mem_get_info()[0]<20*2**30: raise RuntimeError('Need20GiB free; never stop other jobs')
    torch.cuda.set_per_process_memory_fraction(.18,dev)
    torch.backends.cuda.matmul.allow_tf32=False
    gain_audit=audit()
    gains={r['t']:r for r in gain_audit['rows']}
    assert len(gains)==90
    teacher=load_frozen_teacher(BASE,dev,'DiT-L/2','rdd')
    student,config=student_loader(args.arm,dev,checkpoint_template,6,checkpoint_step)
    probes={};latents={};checks={}
    for role in ('teacher','student'):
        plain=PrefixRollout(teacher,None if role=='teacher' else student)
        expected,_=sample_rdd(plain,args.batch,[80,80,90],dev,args.seed)
        probe=StateProbe(teacher,student,role,times=gains)
        actual,_=sample_rdd(probe,args.batch,[80,80,90],dev,args.seed)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        for r in (8,16):
            for a,k in zip(probe.endpoints[r],('input','output')):
                torch.testing.assert_close(a,plain.endpoints[str(r)][k],rtol=0,atol=0)
        if len(probe.rows)!=args.batch*90 or set(probe.states)!=set(gains):
            raise ValueError('Missing per-time observations')
        for r in probe.rows:r['local_jump_mse']=local_jump_error(r,gains[r['t']])
        probes[role]=probe;latents[role]=actual.cpu();checks[role]=0.
        del expected,actual,plain
        print(json.dumps(dict(done=role,seed=args.seed,arm=args.arm,rows=len(probe.rows))),flush=True)
    for r in (8,16):
        for a,b in zip(probes['teacher'].endpoints[r],probes['student'].endpoints[r]):
            torch.testing.assert_close(a,b,atol=0,rtol=0)
    rows=probes['teacher'].rows+probes['student'].rows
    report=dict(complete=True,seed=args.seed,arm=args.arm,batch=args.batch,config=config,
        steps=[80,80,90],cfg=1.,upsample='noise_filling',prefix_depth=6,
        rows=rows,gains=gain_audit['rows'],instrumentation_final_max_abs=checks,
        coarse_parity_max_abs=0.,final_relative_latent_l2=relative_l2(latents['student'],latents['teacher']),
        state_relative_latent_l2={str(t):relative_l2(probes['student'].states[t],probes['teacher'].states[t]) for t in gains},
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        warning='Two development seeds x16images per arm; teachers recomputed on EACH respective current state. Extra calls must preserve RNG, original coarse and final outputs bitwise. Low/high are2x2 cell means/differences, not ideal FFT bands. Sum of local jump MSE is a descriptive allocation diagnostic, NOT final trajectory error/FID; local0 update is identity. No training or decoded images.')
    write_json(out,report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--arm',choices=['aligned','zero'],required=True)
    p.add_argument('--batch',type=int,default=16)
    run(p.parse_args())
