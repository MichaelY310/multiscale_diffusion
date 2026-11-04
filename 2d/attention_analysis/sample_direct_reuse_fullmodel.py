"""Paired real250-step image rollout + warm ABBA denoising timings.

No FID claim from64 images; export every example and use classifier only as a
diagnostic. Same BF16 weights/AMP execution in all inference arms. Whole-model
finetunes are reloaded to BF16 for deployment, like original RDD evaluation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import torch
from direct_reuse_fullmodel import DirectReuseStudent, DirectReuseRollout, MASKS
from train_rdd_csra_adapter import load_frozen_teacher
from sample_csra_adapter_preview import sample_rdd, decode, to_pil, save_grid
from diffusers.models import AutoencoderKL
from torchvision.models import resnet50, ResNet50_Weights


def main(a):
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if visible not in ('0','1','2','3'): raise ValueError('Exactly one GPU0-3')
    dev=torch.device('cuda:0');torch.cuda.set_device(dev)
    if torch.cuda.mem_get_info()[0]<18*2**30: raise RuntimeError('Need18GiB free')
    torch.cuda.set_per_process_memory_fraction(.20,dev)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=Path(a.root);out=root/'image_preview'/f'shard_{a.rank}';out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.json').exists(): return
    base=Path('checkpoints/RDD-Unified-L/0400000.pt')
    original=load_frozen_teacher(base,dev,'DiT-L/2','rdd')
    trained={}
    for key in ('reuse6','dense'):
        trained[key]=load_frozen_teacher(root/f'train_{key}/sample.pt',dev,'DiT-L/2','rdd')
    nets={'original':DirectReuseStudent(original,[],False),
          'raw_reuse6':DirectReuseStudent(original,MASKS['reuse6'],False),
          'fullmodel_reuse6':DirectReuseStudent(trained['reuse6'],MASKS['reuse6'],False),
          'fullmodel_dense_control':DirectReuseStudent(trained['dense'],[],False)}
    # Real400K no-op parity against original model.forward, identical AMP policy.
    x=torch.randn(2,4,32,32,device=dev);t=torch.tensor([299,0],device=dev);y=torch.tensor([0,1],device=dev)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        p=original(x,t,y);q=nets['original'].stage(x,t,y)[0]
    torch.testing.assert_close(p,q,atol=0,rtol=0);del x,p,q
    seed=91560915+a.rank*10000;latents={};timings={};labels=None
    for name,net in nets.items():
        w=DirectReuseRollout(net);z,s=sample_rdd(w,a.batch,[80,80,90],dev,seed)
        if labels is None: labels=w.labels
        torch.testing.assert_close(w.labels,labels,atol=0,rtol=0)
        if w.source_calls!=1 or w.replaced_blocks!=90*len(net.layers): raise AssertionError('Cache/skip count')
        latents[name]=z.cpu();timings[name]=s
        torch.save(z.cpu(),out/f'{name}_latents.pt')
        print(json.dumps(dict(rank=a.rank,sampled=name,seconds=s)),flush=True)
    # Small fixed-batch matched trajectory benchmark. Includes cache creation,
    # upsampling, sampler/noise_filling overhead; excludes loading and VAE.
    bench=[]
    if a.rank==0:
        for name in ('original','raw_reuse6','fullmodel_reuse6'):
            sample_rdd(DirectReuseRollout(nets[name]),8,[80,80,90],dev,9315)
        for g in range(4):
            order=['original','fullmodel_reuse6','fullmodel_reuse6','original'] if g%2==0 else ['fullmodel_reuse6','original','original','fullmodel_reuse6']
            rows=[]
            for name in order:
                _,s=sample_rdd(DirectReuseRollout(nets[name]),8,[80,80,90],dev,9315)
                rows.append(dict(variant=name,seconds=s))
            b=statistics.mean(r['seconds'] for r in rows if r['variant']=='original')
            c=statistics.mean(r['seconds'] for r in rows if r['variant']=='fullmodel_reuse6')
            bench.append(dict(group=g,rows=rows,latency_reduction_percent=100*(1-c/b),speedup=b/c))
            print(json.dumps(bench[-1]),flush=True)
    del nets,original,trained,net,w;torch.cuda.empty_cache()
    vae=AutoencoderKL.from_pretrained('checkpoints/pretrained/stabilityai/sd-vae-ft-ema',local_files_only=True,
        torch_dtype=torch.bfloat16).to(dev).eval().requires_grad_(False)
    pictures={};vae_times={}
    for name,z in latents.items():
        torch.cuda.synchronize();begin=time.perf_counter()
        pictures[name]=torch.cat([decode(vae,z[i:i+4].to(dev)) for i in range(0,a.batch,4)])
        torch.cuda.synchronize();vae_times[name]=time.perf_counter()-begin
        (out/name).mkdir(exist_ok=True)
        for i,img in enumerate(pictures[name]): to_pil(img).save(out/name/f'{i:03d}.png')
    save_grid([(n,ims[:8]) for n,ims in pictures.items()],out/'comparison.png')
    del vae;torch.cuda.empty_cache()
    cp=Path(torch.hub.get_dir())/'checkpoints/resnet50-11ad3fa6.pth'
    classifier=resnet50(weights=None).eval();classifier.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True))
    transform=ResNet50_Weights.IMAGENET1K_V2.transforms();rows=[]
    with torch.inference_mode():
        for name,images in pictures.items():
            for i,image in enumerate(images):
                prob=classifier(transform(to_pil(image))[None]).softmax(-1)[0];label=int(labels[i]);top=prob.topk(5).indices.tolist()
                path=out/name/f'{i:03d}.png'
                rows.append(dict(variant=name,image=a.rank*a.batch+i,target=label,top1_correct=top[0]==label,
                    top5_correct=label in top,target_probability=float(prob[label]),image_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    metrics=dict(complete=True,rank=a.rank,batch=a.batch,seed=seed,rows=rows,benchmark=bench,
        sampling_seconds=timings,vae_seconds=vae_times,no_op_max_abs=0.,steps=[80,80,90],cfg=1.,boundary='noise_filling',
        execution='BF16 original/fine-tuned model weights; BF16 AMP; TF32 off; same sampler,250NFE',
        warning='Small paired quality screen, not FID. GPU shared with existing training. No independent timing guarantee.',
        paired_latent_relative_l2={n:((z-latents['original']).square().flatten(1).sum(1)/latents['original'].square().flatten(1).sum(1)).sqrt().mean().item() for n,z in latents.items()},
        parent_protocol={k:json.loads((root/f'train_{k}/protocol.json').read_text()) for k in ('reuse6','dense')},
        peak_gib=torch.cuda.max_memory_allocated()/2**30)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='/home/yuheng/rdd_experiments/direct_reuse_fullmodel_20260915')
    p.add_argument('--rank',required=True,type=int);p.add_argument('--batch',type=int,default=16)
    main(p.parse_args())
