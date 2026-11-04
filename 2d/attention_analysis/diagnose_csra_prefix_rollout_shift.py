#!/usr/bin/env python3
"""Paired teacher/student-state prediction error; no training or decoded images.

Same seed, labels, sampling noise, ORIGINAL8/16 path, and fixed prefix checkpoint.
Extra teacher predictions never alter the student trajectory or consume RNG.
"""
import argparse
import json
from pathlib import Path
import torch
from csra_prefix_student import prefix_student_class, prefix_checkpoint_format, check_prefix_initializer
from diagnose_csra_trained_prefix import scores
from probe_csra_prefix_information import BASE, TIMES, write_json
from sample_csra_adapter_preview import sample_rdd, manual_forward, relative_l2
from train_rdd_csra_adapter import load_frozen_teacher


class StateProbe:
    def __init__(self, teacher, student, role, times=None):
        if role not in ('teacher', 'student'): raise ValueError(role)
        self.teacher, self.student, self.role = teacher, student, role
        self.times = set(TIMES if times is None else times)
        self.cache = None
        self.rows, self.states, self.endpoints = [], {}, {}

    def forward(self, x, t, y):
        r, tv = x.shape[-1], int(t[0])
        if not torch.all(t == tv): raise ValueError('Mixed timestep')
        if r == 16 and tv == 300:
            out, trace = manual_forward(self.teacher, x, t, y, multires=True, capture_layers={13})
            self.cache = {13:trace[13]['output']}
        elif r != 32:
            out, _ = manual_forward(self.teacher, x, t, y, multires=True)
        else:
            if self.cache is None: raise RuntimeError('No original coarse state')
            selected = tv in self.times
            rng = torch.cuda.get_rng_state(x.device)
            pred = target = None
            if self.role == 'student' or selected:
                pred = self.student(dict(x=x, t=t, labels=y, coarse=self.cache))
            if self.role == 'teacher' or selected:
                target, _ = manual_forward(self.teacher, x, t, y, multires=True)
            if not torch.equal(rng, torch.cuda.get_rng_state(x.device)):
                raise RuntimeError('A diagnostic model call changed sampling RNG')
            out = target if self.role == 'teacher' else pred
            if selected:
                if tv in self.states: raise ValueError('Repeated recorded time')
                self.states[tv] = x.cpu()
                measured = scores(pred, target)
                power = target[:, :4].float().square().mean((1, 2, 3))
                self.rows.extend(dict(t=tv, image=i, label=int(y[i]), state_role=self.role,
                    target_power=float(power[i]), **{k:float(v[i]) for k,v in measured.items()})
                    for i in range(len(x)))
        if (r, tv) in ((8,600), (16,300)):
            self.endpoints[r] = (x.cpu(), out.cpu())
        return out


@torch.inference_mode()
def main(args):
    torch.set_num_threads(4)
    dev = torch.device('cuda:0')
    if torch.cuda.mem_get_info(dev)[0] < 12*2**30:
        raise RuntimeError('Need12GiB free; never stop other jobs')
    out = Path(args.output_root)/f'seed_{args.seed}_{args.arm}.json'
    if out.exists(): raise FileExistsError('Read completed result; do not overwrite')
    path = Path(f'checkpoints/RDD-CSRA-prefix8-conditional-{args.arm}-L400K-20260913/best.pt')
    ck = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
    kind = ck['args'].get('bridge_kind', 'linear')
    if (kind != 'conditional_attention' or ck['format'] != prefix_checkpoint_format(kind)
        or ck['prefix_depth'] != 8 or ck['source_layer'] != 13 or ck['args']['coarse_source'] != args.arm
        or Path(ck['base_checkpoint']).resolve() != BASE.resolve()):
        raise ValueError('Wrong checkpoint')
    init = torch.load(ck['args']['initializer'], map_location='cpu', weights_only=True)
    check_prefix_initializer(init, 8)
    teacher = load_frozen_teacher(BASE, dev, 'DiT-L/2', 'rdd')
    student = prefix_student_class(kind)(load_frozen_teacher(BASE, dev, 'DiT-L/2', 'rdd'),
        init['fits']['fine_only'], args.arm, prefix_depth=8, checkpoint_blocks=False).to(dev)
    student.load_state_dict(ck['student']); student.eval().requires_grad_(False)
    config = dict(checkpoint=str(path), step=ck['step'], args=ck['args'])
    del ck, init
    torch.backends.cuda.matmul.allow_tf32 = False
    probes, latents = {}, {}
    for role in ('teacher', 'student'):
        probe = StateProbe(teacher, student, role)
        latents[role], _ = sample_rdd(probe, 4, [80,80,90], dev, args.seed)
        if set(probe.states) != set(TIMES) or len(probe.rows) != 32:
            raise RuntimeError('Missing diagnostic times')
        probes[role] = probe
        print(json.dumps(dict(seed=args.seed, arm=args.arm, done=role)), flush=True)
    for r in (8,16):
        for a,b in zip(probes['teacher'].endpoints[r], probes['student'].endpoints[r]):
            torch.testing.assert_close(a,b,atol=0,rtol=0)
    deviation = relative_l2(latents['student'], latents['teacher'])
    prior = json.loads((Path('eval_results/csra_prefix8_conditional_pair_20260913') /
                       f'seed_{args.seed}'/'metrics.json').read_text())
    expected = prior['relative_latent_l2'][f'prefix_{args.arm}']
    if abs(deviation-expected) > 1e-6:
        raise RuntimeError(f'Instrumentation changed previous rollout: {deviation} vs {expected}')
    rows = probes['teacher'].rows + probes['student'].rows
    means = {role:{str(t):sum(r['main_mse'] for r in rows if r['state_role']==role and r['t']==t)/4
                   for t in TIMES} for role in probes}
    write_json(out, dict(complete=True, seed=args.seed, arm=args.arm, config=config, rows=rows,
        main_mse_by_time=means, final_relative_latent_l2=deviation,
        deviation_difference_vs_original_preview=deviation-expected, coarse_endpoint_parity_max_abs=0.,
        state_relative_latent_l2={str(t):relative_l2(probes['student'].states[t],probes['teacher'].states[t])
                                  for t in TIMES}, peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,
        warning='Small reused development seeds. Teacher target is recomputed at each respective trajectory state. Teacher-state vs student-state MSE tests prediction error shift, not quality/FID/speed, gradient variance, or proof of global stability. Extra model evaluations check CUDA RNG invariance, coarse endpoints match bitwise, final latent deviation must reproduce existing previews within1e-6. No new training or decoded images.'))
    print(json.dumps(dict(complete=True, main_mse_by_time=means)),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--arm',choices=('aligned','zero'),required=True)
    p.add_argument('--output-root',default='eval_results/csra_prefix8_conditional_rollout_shift_20260913')
    main(p.parse_args())
