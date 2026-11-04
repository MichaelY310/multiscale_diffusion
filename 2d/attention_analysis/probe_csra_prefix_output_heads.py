#!/usr/bin/env python3
"""Oracle head-isolation previews. Extra original teacher work is NOT deployable speed."""
import argparse
import json
from pathlib import Path
import torch
from csra_prefix_student import prefix_student_class, prefix_checkpoint_format, check_prefix_initializer
from probe_csra_prefix_information import BASE, write_json
from sample_csra_prefix_preview import PrefixRollout
from sample_csra_adapter_preview import (sample_rdd, manual_forward, relative_l2, decode,
                                        save_grid, to_pil, AutoencoderKL)
from train_rdd_csra_adapter import load_frozen_teacher


def mix_heads(student, teacher, mode):
    if student.shape != teacher.shape or student.ndim != 4 or student.shape[1] != 8:
        raise ValueError('Expected matching Bx8xHxW mean/variance outputs')
    if mode == 'student_main_teacher_var':
        return torch.cat((student[:,:4], teacher[:,4:]), 1)
    if mode == 'teacher_main_student_var':
        return torch.cat((teacher[:,:4], student[:,4:]), 1)
    raise ValueError(mode)


class HeadOracle(PrefixRollout):
    def __init__(self, teacher, student, mode):
        super().__init__(teacher, student)
        self.mode = mode

    def forward(self, x, t, y):
        out = super().forward(x,t,y)
        if x.shape[-1] == 32:
            rng = torch.cuda.get_rng_state(x.device)
            original, _ = manual_forward(self.teacher,x,t,y,multires=True)
            if not torch.equal(rng,torch.cuda.get_rng_state(x.device)):
                raise RuntimeError('Oracle teacher call consumed sampling RNG')
            out = mix_heads(out,original,self.mode)
        return out


@torch.inference_mode()
def main(args):
    torch.set_num_threads(4)
    out=Path(args.output_root)/f'seed_{args.seed}'
    if (out/'metrics.json').exists(): raise FileExistsError('Read completed result instead')
    out.mkdir(parents=True,exist_ok=True)
    dev=torch.device('cuda:0')
    if torch.cuda.mem_get_info(dev)[0] < 12*2**30: raise RuntimeError('Need12GiB free; never stop other jobs')
    path=Path('checkpoints/RDD-CSRA-prefix8-conditional-aligned-L400K-20260913/best.pt')
    ck=torch.load(path,map_location='cpu',mmap=True,weights_only=True)
    kind=ck['args'].get('bridge_kind','linear')
    if (kind!='conditional_attention' or ck['format']!=prefix_checkpoint_format(kind)
        or ck['prefix_depth']!=8 or ck['source_layer']!=13 or ck['args']['coarse_source']!='aligned'
        or Path(ck['base_checkpoint']).resolve()!=BASE.resolve()): raise ValueError('Wrong checkpoint')
    init=torch.load(ck['args']['initializer'],map_location='cpu',weights_only=True)
    check_prefix_initializer(init,8)
    teacher=load_frozen_teacher(BASE,dev,'DiT-L/2','rdd')
    student=prefix_student_class(kind)(load_frozen_teacher(BASE,dev,'DiT-L/2','rdd'),
        init['fits']['fine_only'],'aligned',prefix_depth=8,checkpoint_blocks=False).to(dev)
    student.load_state_dict(ck['student']); student.eval().requires_grad_(False)
    config=dict(checkpoint=str(path),step=ck['step'],args=ck['args'])
    del ck,init
    torch.backends.cuda.matmul.allow_tf32=False
    latents={};parity={}
    baseline=PrefixRollout(teacher)
    latents['original_dense'],_=sample_rdd(baseline,4,[80,80,90],dev,args.seed)
    for mode in ('student_both','student_main_teacher_var','teacher_main_student_var'):
        wrapper=PrefixRollout(teacher,student) if mode=='student_both' else HeadOracle(teacher,student,mode)
        latents[mode],_=sample_rdd(wrapper,4,[80,80,90],dev,args.seed)
        if wrapper.fine_calls!=90 or wrapper.reused_fine_calls!=90: raise RuntimeError('Wrong fine calls')
        parity[mode]=max(float((v.float()-baseline.endpoints[r][k].float()).abs().max())
                         for r,d in wrapper.endpoints.items() for k,v in d.items())
        if set(wrapper.endpoints)!={'8','16'} or parity[mode]!=0: raise RuntimeError('Coarse changed')
        print(json.dumps(dict(seed=args.seed,completed=mode)),flush=True)
    deviations={k:relative_l2(v,latents['original_dense']) for k,v in latents.items()}
    prior=json.loads((Path('eval_results/csra_prefix8_conditional_pair_20260913')/
                      f'seed_{args.seed}'/'metrics.json').read_text())
    expected=prior['relative_latent_l2']['prefix_aligned']
    if abs(deviations['student_both']-expected)>1e-6: raise RuntimeError('Student reference changed')
    vae=AutoencoderKL.from_pretrained('checkpoints/pretrained/stabilityai/sd-vae-ft-ema',
        local_files_only=True,torch_dtype=torch.bfloat16).to(dev).eval().requires_grad_(False)
    images={}
    for key,z in latents.items():
        images[key]=decode(vae,z)
        folder=out/key;folder.mkdir(exist_ok=True)
        for i,img in enumerate(images[key]): to_pil(img).save(folder/f'{i:03d}.png')
    save_grid(list(images.items()),out/'comparison_grid.png')
    write_json(out/'metrics.json',dict(complete=True,seed=args.seed,batch=4,steps=[80,80,90],cfg=1.,
        upsample='noise_filling',config=config,relative_latent_l2=deviations,
        coarse_endpoint_max_abs_vs_original=parity,
        peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,
        warning='ORACLE diagnostic, not an accelerated method/FID or fair latency benchmark. Only32stage heads mixed; original8/16 unchanged. Student main head predicts drift-absorbed epsilon, not pure standard epsilon or x0. Teacher head recomputed at each respective current state (not a cached dense trajectory). Same90fine calls but bothmodels evaluated on hybrid trajectories; all extra cost would need counting if deployed. At final localt0, variance has no noise contribution. Original and student reference image hashes should match prior paired previews.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True)
    p.add_argument('--output-root',default='eval_results/csra_prefix8_conditional_head_oracle_20260913')
    main(p.parse_args())
