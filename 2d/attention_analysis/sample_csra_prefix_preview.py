#!/usr/bin/env python3
"""Original dense versus matched V7 source/zero students, with ORIGINAL8/16."""
import argparse
import json
from pathlib import Path
import torch
from csra_prefix_student import prefix_student_class,prefix_checkpoint_format,check_prefix_initializer
from probe_csra_prefix_information import write_json
from sample_csra_adapter_preview import sample_rdd,manual_forward,decode,relative_l2,save_grid,to_pil,AutoencoderKL
from train_rdd_csra_adapter import load_frozen_teacher


class PrefixRollout:
    def __init__(self,teacher,student=None,reuse_fine_steps=90):
        self.teacher,self.student=teacher,student
        if not 1 <= reuse_fine_steps <= 90: raise ValueError('Expected1–90 reused fine NFEs')
        self.reuse_fine_steps=reuse_fine_steps
        self.fine_calls,self.reused_fine_calls=0,0
        self.cache=None
        self.endpoints={}

    def forward(self,x,t,y):
        r,step=int(x.shape[-1]),int(t[0])
        if r==32: self.fine_calls+=1
        if r==16 and step==300 and self.student is not None and self.student.coarse_source=='aligned':
            out,trace=manual_forward(self.teacher,x,t,y,multires=True,capture_layers={13})
            self.cache={13:trace[13]['output']}
        elif r==32 and self.student is not None and self.fine_calls<=self.reuse_fine_steps:
            if self.student.coarse_source=='aligned' and self.cache is None: raise RuntimeError('Missing original16 hidden')
            out=self.student({'x':x,'t':t,'labels':y,'coarse':self.cache})
            self.reused_fine_calls+=1
        else:
            out,_=manual_forward(self.teacher,x,t,y,multires=True)
        if (r,step) in ((8,600),(16,300)):
            self.endpoints[str(r)]={'input':x.cpu(),'output':out.cpu()}
        return out


@torch.inference_mode()
def main(args):
    out=Path(args.output_dir)
    if (out/'metrics.json').exists(): return
    out.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda:0')
    if torch.cuda.mem_get_info(device)[0] < 12*2**30: raise RuntimeError('Need12GiB free; never stop other jobs')
    base=Path('checkpoints/RDD-Unified-L/0400000.pt')
    teacher=load_frozen_teacher(base,device,'DiT-L/2','rdd')
    torch.backends.cuda.matmul.allow_tf32=False
    original=PrefixRollout(teacher)
    latents={}
    latents['original_dense'],_=sample_rdd(original,4,[80,80,90],device,args.seed)
    config,parity={},{}
    for mode in ('aligned','zero'):
        path=Path(args.checkpoint_template.format(mode=mode))
        ck=torch.load(path,map_location='cpu',mmap=True,weights_only=True)
        kind=ck['args'].get('bridge_kind','linear')
        if (ck['format']!=prefix_checkpoint_format(kind) or ck['args']['coarse_source']!=mode
            or Path(ck['base_checkpoint']).resolve()!=base.resolve() or ck['prefix_depth']!=args.prefix_depth or ck['source_layer']!=13):
            raise ValueError('Wrong prefix checkpoint')
        init=torch.load(ck['args']['initializer'],map_location='cpu',weights_only=True)
        check_prefix_initializer(init,args.prefix_depth)
        model=prefix_student_class(kind)(load_frozen_teacher(base,device,'DiT-L/2','rdd'),init['fits']['fine_only'],mode,
                                prefix_depth=args.prefix_depth,checkpoint_blocks=False).to(device)
        model.load_state_dict(ck['student']); model.eval().requires_grad_(False)
        wrapper=PrefixRollout(teacher,model,args.reuse_fine_steps)
        key=f'prefix_{mode}'
        latents[key],_=sample_rdd(wrapper,4,[80,80,90],device,args.seed)
        if wrapper.fine_calls!=90 or wrapper.reused_fine_calls!=args.reuse_fine_steps:
            raise RuntimeError('Unexpected fine invocation/reuse count')
        if set(wrapper.endpoints)!={'8','16'}: raise RuntimeError('Missing coarse endpoints')
        parity[key]=max(float((value.float()-original.endpoints[r][name].float()).abs().max())
                        for r,record in wrapper.endpoints.items() for name,value in record.items())
        if parity[key]!=0: raise RuntimeError('Original coarse path changed')
        config[key]={'checkpoint':str(path),'step':ck['step'],'args':ck['args'],
                     'fine_calls':wrapper.fine_calls,'reused_fine_calls':wrapper.reused_fine_calls}
        if 'hidden_supervision' in ck:
            # Audit which objective actually produced the selected checkpoint;
            # this changes metadata only, never the sampling computation.
            config[key]['hidden_supervision']=ck['hidden_supervision']
        del ck,init,model,wrapper
        torch.cuda.empty_cache()
    vae=AutoencoderKL.from_pretrained('checkpoints/pretrained/stabilityai/sd-vae-ft-ema',
        local_files_only=True,torch_dtype=torch.bfloat16).to(device).eval().requires_grad_(False)
    images={}
    for key,z in latents.items():
        images[key]=decode(vae,z)
        folder=out/key;folder.mkdir(exist_ok=True)
        for i,img in enumerate(images[key]): to_pil(img).save(folder/f'{i:03d}.png')
    save_grid(list(images.items()),out/'comparison_grid.png')
    write_json(out/'metrics.json',{'complete':True,'seed':args.seed,'batch':4,'steps':[80,80,90],
        'cfg':1.0,'upsample':'noise_filling','prefix_depth':args.prefix_depth,'coarse_source_layer':13,
        'reuse_fine_steps':args.reuse_fine_steps,
        'relative_latent_l2':{k:relative_l2(v,latents['original_dense']) for k,v in latents.items()},
        'coarse_endpoint_max_abs_vs_original':parity,'configurations':config,
        'warning':f'Development seeds, not FID/latency. Original8/16 plus separate updatedfine weightset; trainedprefix{args.prefix_depth} replacement and{24-args.prefix_depth}originalarchitecturalfineblocks for the first{args.reuse_fine_steps}of90 fineNFEs; remaining fineNFEs run ORIGINAL full backbone. Initialfunctions matched, sourcebranch initiallyzero. This window is selected using reused development diagnostics; it is not a new trained model or validation of video speed.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--prefix-depth',type=int,default=14)
    p.add_argument('--reuse-fine-steps',type=int,default=90)
    p.add_argument('--checkpoint-template',default='checkpoints/RDD-CSRA-prefix_v7-{mode}-L400K-20260913/best.pt')
    main(p.parse_args())
