#!/usr/bin/env python3
"""Do eight trained times conceal errors BETWEEN those times? Read-only probe.

All predictions share the ORIGINAL teacher trajectory. Extra student calls must
leave RNG, original coarse endpoints and final teacher latent exactly unchanged.
This is a development diagnostic, not evidence that a different loss works.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from csra_prefix_student import prefix_student_class, prefix_checkpoint_format, check_prefix_initializer
from diagnose_csra_trained_prefix import scores
from probe_csra_prefix_information import BASE, TIMES, write_json
from sample_csra_prefix_preview import PrefixRollout
from sample_csra_adapter_preview import sample_rdd, manual_forward
from sample_rdd_stagewide_block_reuse_preview import make_diffusions
from train_rdd_csra_adapter import load_frozen_teacher


def bracket_reference(t, anchors):
    """Linear interpolation of a measured scalar vs time, NOT a target model."""
    keys = sorted(anchors)
    if t in anchors: return float(anchors[t])
    if not keys or not keys[0] < t < keys[-1]: raise ValueError('No extrapolation allowed')
    lo = max(k for k in keys if k < t)
    hi = min(k for k in keys if k > t)
    return ((hi-t)*float(anchors[lo])+(t-lo)*float(anchors[hi]))/(hi-lo)


class AllTimeProbe:
    def __init__(self, teacher, student):
        self.teacher, self.student = teacher, student
        self.cache, self.rows, self.endpoints = None, [], {}

    def forward(self, x, t, y):
        r, tv = x.shape[-1], int(t[0])
        if not torch.all(t == tv): raise ValueError('Mixed times')
        capture = {13} if r == 16 and tv == 300 else set()
        target, trace = manual_forward(self.teacher, x, t, y, multires=True, capture_layers=capture)
        if capture: self.cache = {13:trace[13]['output']}
        if r == 32:
            if self.cache is None: raise ValueError('Missing coarse source')
            rng = torch.cuda.get_rng_state(x.device)
            prediction = self.student(dict(x=x, t=t, labels=y, coarse=self.cache))
            if not torch.equal(rng, torch.cuda.get_rng_state(x.device)):
                raise RuntimeError('Diagnostic changed sampling RNG')
            measured = scores(prediction, target)
            power = target[:,:4].float().square().mean((1,2,3))
            self.rows.extend(dict(t=tv,image=i,label=int(y[i]),trained_time=tv in TIMES,
                target_power=float(power[i]),
                **{k:float(v[i]) for k,v in measured.items()}) for i in range(len(x)))
        if (r,tv) in ((8,600),(16,300)):
            self.endpoints[str(r)] = dict(input=x.cpu(),output=target.cpu())
        return target


@torch.inference_mode()
def run(args):
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    if torch.cuda.mem_get_info(device)[0] < 12*2**30:
        raise RuntimeError('Need12GiB free; never stop another job')
    name = 'conditional' if args.budget == 1600 else 'conditional3200'
    cp = Path(f'checkpoints/RDD-CSRA-prefix8-{name}-{args.arm}-L400K-20260913/best.pt')
    destination = Path(args.output_root)/f'budget_{args.budget}_{args.arm}'/f'seed_{args.seed}.json'
    if destination.exists(): raise FileExistsError('Read existing result rather than rerun')
    ck = torch.load(cp,map_location='cpu',mmap=True,weights_only=True)
    kind = ck['args'].get('bridge_kind','linear')
    if (kind != 'conditional_attention' or ck['format'] != prefix_checkpoint_format(kind)
        or ck['step'] != args.budget or ck['prefix_depth'] != 8 or ck['source_layer'] != 13
        or ck['args']['coarse_source'] != args.arm or Path(ck['base_checkpoint']).resolve() != BASE.resolve()):
        raise ValueError('Wrong trained checkpoint')
    ip = Path(ck['args']['initializer'])
    if hashlib.sha256(ip.read_bytes()).hexdigest() != ck['initializer_sha256']:
        raise ValueError('Initializer SHA mismatch')
    init = torch.load(ip,map_location='cpu',weights_only=True)
    check_prefix_initializer(init,8)
    teacher = load_frozen_teacher(BASE,device,'DiT-L/2','rdd')
    student = prefix_student_class(kind)(load_frozen_teacher(BASE,device,'DiT-L/2','rdd'),
        init['fits']['fine_only'],args.arm,prefix_depth=8,checkpoint_blocks=False).to(device)
    student.load_state_dict(ck['student']); student.eval().requires_grad_(False)
    config = dict(checkpoint=str(cp),step=ck['step'],args=ck['args'],initializer_sha256=ck['initializer_sha256'])
    del ck,init
    torch.backends.cuda.matmul.allow_tf32 = False
    reference = PrefixRollout(teacher)
    baseline,_ = sample_rdd(reference,4,[80,80,90],device,args.seed)
    probe = AllTimeProbe(teacher,student)
    actual,_ = sample_rdd(probe,4,[80,80,90],device,args.seed)
    torch.testing.assert_close(actual,baseline,atol=0,rtol=0)
    for r in ('8','16'):
        for k in ('input','output'):
            torch.testing.assert_close(probe.endpoints[r][k],reference.endpoints[r][k],atol=0,rtol=0)
    times = list(make_diffusions([80,80,90])[0].timestep_map)
    if len(probe.rows)!=360 or sorted({r['t'] for r in probe.rows})!=sorted(times):
        raise ValueError('Missing or repeated fine times')
    if not set(TIMES).issubset(times): raise ValueError('Training anchors missing')
    for image in range(4):
        rows = [r for r in probe.rows if r['image']==image]
        anchors = {r['t']:r['main_mse'] for r in rows if r['trained_time']}
        for r in rows:
            r['nearest_trained_distance'] = min(abs(r['t']-t) for t in TIMES)
            value = bracket_reference(r['t'],anchors)
            r['interpolated_anchor_main_mse'] = value
            r['main_mse_over_anchor_interpolation'] = r['main_mse']/max(value,1e-20)
    write_json(destination,dict(complete=True,seed=args.seed,arm=args.arm,budget=args.budget,config=config,
        batch=4,steps=[80,80,90],cfg=1.,upsample='noise_filling',trained_times=TIMES,
        rows=probe.rows,teacher_final_parity_max_abs=0.,coarse_endpoint_parity_max_abs=0.,
        peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,
        warning='All90 points on ORIGINAL teacher trajectories; compares1600/3200 and matched trained-zero. Errors between anchors are not independently held-out images. Interpolated anchor error is only a descriptive reference: noise/task changes with time; this ratio does not prove interpolation causes poor images. Final local0 sample is identity. No training, output changes, FID or latency claims. Extra student calls consume no RNG, teacher final and coarse endpoints must be bitwise identical.'))
    print(json.dumps(dict(complete=True,path=str(destination),rows=len(probe.rows))),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--budget',type=int,choices=(1600,3200),required=True)
    p.add_argument('--arm',choices=('aligned','zero'),required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--output-root',default='eval_results/csra_prefix8_all_times_20260913')
    run(p.parse_args())
