#!/usr/bin/env python3
"""256 new paired images per method, four shards; NOT FID or a speed benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50
from csra_prefix_student import prefix_student_class, prefix_checkpoint_format, check_prefix_initializer
from probe_csra_prefix_information import BASE, write_json
from sample_csra_prefix_preview import PrefixRollout
from sample_csra_adapter_preview import sample_rdd, decode, save_grid, to_pil, AutoencoderKL, relative_l2
from train_rdd_csra_adapter import load_frozen_teacher


class LabelledRollout(PrefixRollout):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        self.labels=None

    def forward(self,x,t,y):
        if self.labels is None: self.labels=y.detach().cpu().clone()
        return super().forward(x,t,y)


def load_student(arm, device, checkpoint_template='checkpoints/RDD-CSRA-prefix8-conditional3200-{arm}-L400K-20260913/best.pt',
                 prefix_depth=8, checkpoint_step=3200):
    path=Path(checkpoint_template.format(arm=arm))
    ck=torch.load(path,map_location='cpu',mmap=True,weights_only=True)
    kind=ck['args'].get('bridge_kind','linear')
    if (kind!='conditional_attention' or ck['format']!=prefix_checkpoint_format(kind)
        or ck['step']!=checkpoint_step or ck['prefix_depth']!=prefix_depth or ck['source_layer']!=13
        or ck['args']['coarse_source']!=arm or Path(ck['base_checkpoint']).resolve()!=BASE.resolve()):
        raise ValueError('Wrong student checkpoint')
    ip=Path(ck['args']['initializer'])
    if hashlib.sha256(ip.read_bytes()).hexdigest()!=ck['initializer_sha256']:
        raise ValueError('Initializer mismatch')
    init=torch.load(ip,map_location='cpu',weights_only=True);check_prefix_initializer(init,prefix_depth)
    model=prefix_student_class(kind)(load_frozen_teacher(BASE,device,'DiT-L/2','rdd'),
        init['fits']['fine_only'],arm,prefix_depth=prefix_depth,checkpoint_blocks=False).to(device)
    model.load_state_dict(ck['student']);model.eval().requires_grad_(False)
    config=dict(checkpoint=str(path),step=ck['step'],args=ck['args'],initializer_sha256=ck['initializer_sha256'])
    if 'jump_supervision' in ck:
        config['jump_supervision']=ck['jump_supervision']
    return model,config


@torch.inference_mode()
def main(args, student_loader=load_student):
    if args.rank not in range(4): raise ValueError('Expected four shards')
    torch.set_num_threads(2)
    seed=args.seed_base+args.rank*10000
    root=Path(args.output_root)/f'shard_{args.rank}'
    if (root/'metrics.json').exists(): raise FileExistsError('Read completed shard')
    root.mkdir(parents=True,exist_ok=True)
    for mp in Path('eval_results/csra_rollout_data4k_20260912').glob('shard_*/manifest.json'):
        manifest=json.loads(mp.read_text())
        if any(item['seed']==seed for item in manifest['files']):
            raise ValueError('Pilot seed overlaps saved training/development corpus')
    device=torch.device('cuda:0')
    if torch.cuda.mem_get_info(device)[0]<18*2**30:
        raise RuntimeError('Need18GiB free; never stop another job')
    if getattr(args,'memory_fraction',None) is not None:
        if not 0 < args.memory_fraction <= .24: raise ValueError('Memory cap must be in (0,.24]')
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction,device)
    torch.backends.cuda.matmul.allow_tf32=False
    teacher=load_frozen_teacher(BASE,device,'DiT-L/2','rdd')
    baseline=LabelledRollout(teacher)
    z,_=sample_rdd(baseline,64,[80,80,90],device,seed)
    latents={'original_dense':z.cpu()}
    labels=baseline.labels
    endpoints=baseline.endpoints
    config={};parity={}
    del baseline,z
    for arm in ('aligned','zero'):
        model,config[arm]=student_loader(arm,device,args.checkpoint_template,args.prefix_depth,args.checkpoint_step)
        wrapper=LabelledRollout(teacher,model)
        z,_=sample_rdd(wrapper,64,[80,80,90],device,seed)
        torch.testing.assert_close(wrapper.labels,labels,atol=0,rtol=0)
        if wrapper.fine_calls!=90 or wrapper.reused_fine_calls!=90: raise ValueError('Wrong fine calls')
        delta=max(float((v.float()-endpoints[r][k].float()).abs().max())
                  for r,d in wrapper.endpoints.items() for k,v in d.items())
        if delta!=0 or set(wrapper.endpoints)!={'8','16'}: raise ValueError('Original coarse changed')
        parity[arm]=delta;latents[arm]=z.cpu()
        print(json.dumps(dict(rank=args.rank,phase='sampled',arm=arm)),flush=True)
        del model,wrapper,z
        torch.cuda.empty_cache()
    del teacher
    vae=AutoencoderKL.from_pretrained('checkpoints/pretrained/stabilityai/sd-vae-ft-ema',
        local_files_only=True,torch_dtype=torch.bfloat16).to(device).eval().requires_grad_(False)
    pictures={}
    for name,z in latents.items():
        images=torch.cat([decode(vae,z[i:i+4].to(device)) for i in range(0,64,4)])
        folder=root/name;folder.mkdir(exist_ok=True)
        for i,img in enumerate(images): to_pil(img).save(folder/f'{i:03d}.png')
        pictures[name]=images
    # Fixed first8 examples per shard, chosen BEFORE viewing; all64 PNGs saved.
    save_grid([(key,value[:8]) for key,value in pictures.items()],root/'comparison_grid.png')
    peak=torch.cuda.max_memory_allocated()/2**30
    del vae,pictures
    torch.cuda.empty_cache()
    weights=ResNet50_Weights.IMAGENET1K_V2
    cp=Path(torch.hub.get_dir())/'checkpoints'/'resnet50-11ad3fa6.pth'
    if not cp.is_file(): raise FileNotFoundError('Need already cached ResNet50 V2; no download')
    classifier=resnet50(weights=None).eval()
    classifier.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True))
    transform=weights.transforms();names=weights.meta['categories'];rows=[]
    for variant in latents:
        for start in range(0,64,8):
            batch=[]
            for i in range(start,start+8):
                with Image.open(root/variant/f'{i:03d}.png') as im: batch.append(transform(im.convert('RGB')))
            probabilities=classifier(torch.stack(batch)).softmax(-1)
            for j,prob in enumerate(probabilities):
                i=start+j;target=int(labels[i]);top=prob.topk(5)
                file=root/variant/f'{i:03d}.png'
                rows.append(dict(image=args.rank*64+i,local_image=i,variant=variant,
                    target=target,target_name=names[target],target_probability=float(prob[target]),
                    top1_correct=int(top.indices[0])==target,top5_correct=target in top.indices.tolist(),
                    top5=[dict(index=int(k),name=names[int(k)],probability=float(v)) for k,v in zip(top.indices,top.values)],
                    image_sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    write_json(root/'metrics.json',dict(complete=True,rank=args.rank,seed=seed,images_per_variant=64,
        rows=rows,configs=config,base_checkpoint=str(BASE),steps=[80,80,90],cfg=1.,upsample='noise_filling',
        prefix_depth=args.prefix_depth,checkpoint_step=args.checkpoint_step,seed_base=args.seed_base,
        coarse_endpoint_max_abs=parity,labels=labels.tolist(),peak_memory_gib=peak,
        relative_latent_l2={k:relative_l2(z,latents['original_dense']) for k,z in latents.items()},
        classifier='torchvision ResNet50 ImageNet1K V2, CPU FP32, official transforms',
        classifier_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),
        warning=f'New-seed small quality pilot, not final held-out evaluation, FID or speed. All3variants matched labels, RNG, original8/16,80/80/90,CFG1,noise filling; trained models fixed at{args.checkpoint_step}, prefix{args.prefix_depth}. Classification does not measure realism and latent deviation is not quality percentage. Save64PNG pervariant and fixed first8-example grid; report failed cases too. Extra CPU copies and first-run overhead invalidate using these timings as a speed benchmark.'))
    print(json.dumps(dict(complete=True,rank=args.rank,peak_memory_gib=peak)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rank',type=int,required=True)
    p.add_argument('--prefix-depth',type=int,default=8)
    p.add_argument('--checkpoint-step',type=int,default=3200)
    p.add_argument('--seed-base',type=int,default=20360913)
    p.add_argument('--memory-fraction',type=float,default=None)
    p.add_argument('--checkpoint-template',default='checkpoints/RDD-CSRA-prefix8-conditional3200-{arm}-L400K-20260913/best.pt')
    p.add_argument('--output-root',default='eval_results/csra_prefix8_quality256_20260913')
    main(p.parse_args())
