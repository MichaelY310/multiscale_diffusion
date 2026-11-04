"""Use unchanged paired256 pipeline with strict new fixed2400 neighbor loader."""
import argparse
from pathlib import Path
import torch
from csra_neighbor_protocol import check_new, source_paths, FORMAT, PROTOCOL
from csra_neighbor_student import NeighborFrontStudent
from csra_front_keep_student import check_initializer
from probe_csra_front_keep import BASE, BASE_SHA, file_sha
from train_rdd_csra_adapter import load_frozen_teacher
from sample_csra_prefix_quality_pilot import main


def checked_loader(arm,device,template,depth,step):
    if depth != 6 or step != 2400: raise ValueError('Only fixed2400, front1 skips6')
    path = Path(template.format(arm=arm))
    ck = torch.load(path,map_location='cpu',mmap=True,weights_only=True)
    args = ck['args']
    if args['coarse_source'] != arm or Path(ck['base_checkpoint']).resolve() != BASE.resolve():
        raise ValueError('Source/base mismatch')
    init_sha = file_sha(Path(args['initializer']))
    check_new(ck,args,{name:file_sha(Path(name)) for name in source_paths()},BASE_SHA,init_sha,require_final=True)
    init = torch.load(args['initializer'],map_location='cpu',weights_only=True)
    check_initializer(init)
    model = NeighborFrontStudent(load_frozen_teacher(BASE,device,'DiT-L/2','rdd'),
        init['fits']['fine_only'],arm,prefix_depth=6,source_layer=13,checkpoint_blocks=False,
        neighbor_enabled=args['neighbor']=='on',neighbor_rank=128).to(device)
    model.load_state_dict(ck['student'],strict=True);model.eval().requires_grad_(False)
    return model,dict(checkpoint=str(path),checkpoint_sha256=file_sha(path),args=args,step=ck['step'],
        format=FORMAT,training_protocol=PROTOCOL,parent=ck['parent'],code_sha256=ck['code_sha256'],
        optimizer_policy=ck['optimizer_policy'],initializer_sha256=init_sha,
        front_keep=1,skipped_layers=list(range(1,7)),neighbor_rank=128)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rank',type=int,required=True)
    p.add_argument('--checkpoint-template',required=True)
    p.add_argument('--output-root',required=True)
    p.add_argument('--prefix-depth',type=int,default=6)
    p.add_argument('--checkpoint-step',type=int,default=2400)
    p.add_argument('--seed-base',type=int,default=20460913)
    p.add_argument('--memory-fraction',type=float,default=.18)
    main(p.parse_args(),student_loader=checked_loader)
