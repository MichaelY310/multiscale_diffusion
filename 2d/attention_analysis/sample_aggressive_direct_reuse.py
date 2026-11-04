"""32 paired images,200 vs800-step repairs, ACTUAL sampling+VAE ABBA.

No FID claims. Same resident BF16 VAE for all arms, identical seeds/classes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import torch
from aggressive_direct_reuse import DirectReuseStudent, DirectReuseRollout, MASKS
from train_rdd_csra_adapter import load_frozen_teacher
from sample_csra_adapter_preview import sample_rdd,decode,to_pil,save_grid
from diffusers.models import AutoencoderKL
from torchvision.models import resnet50,ResNet50_Weights


def main(a):
    if os.environ.get('CUDA_VISIBLE_DEVICES','') not in ('0','1','2','3'):
        raise ValueError('Exactly one GPU0-3')
    dev=torch.device('cuda:0');torch.cuda.set_device(dev)
    if torch.cuda.mem_get_info()[0]<22*2**30: raise RuntimeError('Need22GiB free')
    torch.cuda.set_per_process_memory_fraction(.25,dev)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=Path(a.root);out=root/'image_preview'/f'shard_{a.rank}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.json').exists(): return
    original=load_frozen_teacher(Path('checkpoints/RDD-Unified-L/0400000.pt'),dev,'DiT-L/2','rdd')
    nets={'original':DirectReuseStudent(original,[],False)};parents={}
    for mask in ('reuse12','reuse18'):
        nets[f'raw_{mask}']=DirectReuseStudent(original,MASKS[mask],False)
        for step in (200,800):
            path=root/f'train_{mask}'/f'sample_{step:04d}.pt'
            check=torch.load(path,map_location='cpu',weights_only=True)
            if check['step']!=step or check['mask']!=mask: raise ValueError('Wrong snapshot')
            parents[f'{mask}_{step}']=check['protocol'];del check
            model=load_frozen_teacher(path,dev,'DiT-L/2','rdd')
            nets[f'{mask}_{step}']=DirectReuseStudent(model,MASKS[mask],False)
    vae=AutoencoderKL.from_pretrained('checkpoints/pretrained/stabilityai/sd-vae-ft-ema',local_files_only=True,
        torch_dtype=torch.bfloat16).to(dev).eval().requires_grad_(False)
    seed=91560915+a.rank*10000;pictures={};labels=None;latents={}
    for name,net in nets.items():
        w=DirectReuseRollout(net);z,_=sample_rdd(w,a.batch,[80,80,90],dev,seed)
        if labels is None: labels=w.labels
        torch.testing.assert_close(w.labels,labels,atol=0,rtol=0)
        if w.source_calls!=1 or w.replaced_blocks!=90*len(net.layers): raise AssertionError('Cache audit')
        latents[name]=z.float().cpu()
        pictures[name]=torch.cat([decode(vae,z[i:i+4]) for i in range(0,a.batch,4)]).cpu()
        (out/name).mkdir(exist_ok=True)
        for i,img in enumerate(pictures[name]): to_pil(img).save(out/name/f'{i:03d}.png')
        print(json.dumps(dict(rank=a.rank,sampled=name)),flush=True)
    benchmark={}
    if a.rank==0:
        def timed(name):
            torch.cuda.synchronize();begin=time.perf_counter()
            z,_=sample_rdd(DirectReuseRollout(nets[name]),8,[80,80,90],dev,9315)
            torch.cuda.synchronize();d=time.perf_counter()-begin
            decode(vae,z);torch.cuda.synchronize();total=time.perf_counter()-begin
            return dict(variant=name,denoise=d,total=total)
        for name in ('original','reuse12_800','reuse18_800'): timed(name)
        for name in ('reuse12_800','reuse18_800'):
            groups=[]
            for g in range(4):
                order=['original',name,name,'original'] if g%2==0 else [name,'original','original',name]
                rows=[timed(v) for v in order]
                reduction={k:100*(1-statistics.mean(r[k] for r in rows if r['variant']==name)/
                                     statistics.mean(r[k] for r in rows if r['variant']=='original')) for k in ('denoise','total')}
                groups.append(dict(rows=rows,reduction_percent=reduction))
                print(json.dumps(dict(benchmark=name,group=g,reduction=reduction)),flush=True)
            benchmark[name]=groups
    save_grid([(name,ims[:8]) for name,ims in pictures.items()],out/'comparison.png')
    del nets,original,model,net,w,vae,z;torch.cuda.empty_cache()
    cp=Path(torch.hub.get_dir())/'checkpoints/resnet50-11ad3fa6.pth'
    classifier=resnet50(weights=None).eval();classifier.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True))
    transform=ResNet50_Weights.IMAGENET1K_V2.transforms();rows=[]
    with torch.inference_mode():
        for name,images in pictures.items():
            for i,img in enumerate(images):
                prob=classifier(transform(to_pil(img))[None]).softmax(-1)[0];label=int(labels[i]);top=prob.topk(5).indices.tolist()
                path=out/name/f'{i:03d}.png'
                rows.append(dict(variant=name,image=a.rank*a.batch+i,target=label,top1=top[0]==label,top5=label in top,
                    target_probability=float(prob[label]),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    metrics=dict(complete=True,rank=a.rank,batch=a.batch,seed=seed,rows=rows,benchmark=benchmark,
        stages=[80,80,90],cfg=1.,boundary='noise_filling',parents=parents,
        precision='BF16 model and VAE,TF32off; SAME VAE each arm,real decode in paired timing',
        peak_gib=torch.cuda.max_memory_allocated()/2**30,
        warning='32 paired classifier diagnostics,NOT FID; GPU shared with other training',
        latent_relative_l2={n:((z-latents['original']).square().sum()/latents['original'].square().sum()).sqrt().item() for n,z in latents.items()})
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='/home/yuheng/rdd_experiments/aggressive_direct_reuse_20260915')
    p.add_argument('--rank',required=True,type=int);p.add_argument('--batch',default=16,type=int)
    main(p.parse_args())
