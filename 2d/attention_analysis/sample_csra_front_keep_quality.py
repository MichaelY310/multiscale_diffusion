#!/usr/bin/env python3
"""Use the unchanged paired sampling/evaluation pipeline with opt-in front1 loader."""
import argparse
from pathlib import Path
import torch
from csra_front_keep_student import FORMAT,FrontKeepStudent,check_initializer
from probe_csra_front_keep import BASE,BASE_SHA,file_sha
from train_rdd_csra_adapter import load_frozen_teacher
from sample_csra_prefix_quality_pilot import main


def load_student(arm,device,checkpoint_template,prefix_depth,checkpoint_step):
    path=Path(checkpoint_template.format(arm=arm))
    ck=torch.load(path,map_location='cpu',mmap=True,weights_only=True)
    if (ck['format']!=FORMAT or ck['step']!=checkpoint_step or ck['prefix_depth']!=prefix_depth
            or prefix_depth!=6 or ck['front_keep']!=1 or ck['source_layer']!=13
            or ck['base_sha256']!=BASE_SHA or Path(ck['base_checkpoint']).resolve()!=BASE.resolve()
            or ck['args']['coarse_source']!=arm):
        raise ValueError('Wrong fixed front1 checkpoint')
    ip=Path(ck['args']['initializer'])
    if file_sha(ip)!=ck['initializer_sha256']:raise ValueError('Initializer SHA changed')
    init=torch.load(ip,map_location='cpu',weights_only=True);check_initializer(init)
    model=FrontKeepStudent(load_frozen_teacher(BASE,device,'DiT-L/2','rdd'),init['fits']['fine_only'],arm,
                          prefix_depth=6,source_layer=13,checkpoint_blocks=False).to(device)
    model.load_state_dict(ck['student']);model.eval().requires_grad_(False)
    config=dict(checkpoint=str(path),checkpoint_sha256=file_sha(path),step=ck['step'],args=ck['args'],
                initializer_sha256=ck['initializer_sha256'],format=FORMAT,front_keep=1,skipped_layers=list(range(1,7)))
    return model,config


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rank',type=int,required=True)
    p.add_argument('--prefix-depth',type=int,default=6)
    p.add_argument('--checkpoint-step',type=int,default=800)
    p.add_argument('--seed-base',type=int,default=20460913)
    p.add_argument('--memory-fraction',type=float,default=.18)
    p.add_argument('--checkpoint-template',default='checkpoints/RDD-CSRA-front1-skip6-{arm}-L400K-20260914/latest.pt')
    p.add_argument('--output-root',default='eval_results/csra_front1_skip6_quality256_20260914')
    main(p.parse_args(),student_loader=load_student)
