"""Unchanged paired256 pipeline; require the explicit fixed1600 training protocol."""
import argparse
from pathlib import Path
import torch
from csra_staged_protocol import PROTOCOL
from sample_csra_front_keep_quality import load_student as front_loader
from sample_csra_prefix_quality_pilot import main


def checked_loader(arm, device, template, depth, step):
    path = Path(template.format(arm=arm))
    ck = torch.load(path,map_location='cpu',mmap=True,weights_only=True)
    if (ck.get('training_protocol') != PROTOCOL or ck['step'] != 1600
            or ck['args']['mode'] not in ('staged','direct_control')
            or ck['args']['mode'] == 'staged' and not ck['warmup_backbone_verified']):
        raise ValueError('Wrong fixed1600 staged/control checkpoint')
    protocol, mode = ck['training_protocol'], ck['args']['mode']
    del ck
    model, config = front_loader(arm,device,template,depth,step)
    config.update(training_protocol=protocol,training_mode=mode)
    return model,config


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rank',type=int,required=True)
    p.add_argument('--checkpoint-template',required=True)
    p.add_argument('--output-root',required=True)
    p.add_argument('--prefix-depth',type=int,default=6)
    p.add_argument('--checkpoint-step',type=int,default=1600)
    p.add_argument('--seed-base',type=int,default=20460913)
    p.add_argument('--memory-fraction',type=float,default=.18)
    main(p.parse_args(),student_loader=checked_loader)
