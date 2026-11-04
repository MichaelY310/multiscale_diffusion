"""Actual no-attention execution,768 PNGs, SAME256 as800 pilot. No training.

Keep spatially aligned coarse affine path and current-fine MLP unchanged.
Only multiply native-coarse attention read_out by1/.5/0 in both readers.
pointwise arm actually omits reader QKV/SDPA; this file still measures quality, NOT runtime.
"""
import argparse
import json
import os
from pathlib import Path
import torch
from torchvision.models import ResNet50_Weights,resnet50
from sample_csra_split_quality256 import load as load800
from sample_csra_split1600_quality256 import load as load1600
from sample_csra_adapter_preview import sample_rdd,decode,save_grid,to_pil,AutoencoderKL
from csra_skip_schedule import ScheduledPrefixRollout
from sample_csra_skip_schedules import sha,ROOT
from probe_csra_prefix_information import BASE,write_json
from train_rdd_csra_adapter import load_frozen_teacher
from csra_aligned_pointwise_reader import AlignedPointwiseReader


@torch.inference_mode()
def main(a):
    if a.rank not in range(4) or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(a.rank):raise ValueError('GPU0-3 exact shard')
    dev=torch.device('cuda:0');torch.set_num_threads(2)
    if torch.cuda.mem_get_info(dev)[0]<20*2**30:raise RuntimeError('Need20GiB free')
    torch.cuda.set_per_process_memory_fraction(.20,dev);torch.backends.cuda.matmul.allow_tf32=False
    step=800;seed=61560915+a.rank*10000;batch=64
    out=Path(a.root)/'aligned_pointwise256'/f'shard_{a.rank}';out.mkdir(parents=True,exist_ok=True)
    protocol=dict(version='csra_aligned_pointwise_quality256_v1',step=step,seed=seed,rank=a.rank,batch=batch,
        gates={'original':None,'full_reader':1.,'pointwise_reader':0.},script_sha256=sha(__file__),
        steps=[80,80,90],cfg=1.,boundary='noise_filling',
        warning='SAME256 development pairs as800; actual pointwise bypass removes reader attention; NOT a runtime benchmark or FID.',
        pointwise_source_sha256=sha(Path(__file__).with_name('csra_aligned_pointwise_reader.py')))
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())!=protocol:raise ValueError('Different existing protocol')
    if (out/'metrics.json').exists():return
    write_json(out/'protocol.json',protocol)
    teacher=load_frozen_teacher(BASE,dev,'DiT-L/2','rdd')
    student,config=(load800 if step==800 else load1600)(a.root,'split3_3','aligned',dev)
    latents={}
    full_readers=student.readers
    pointwise_readers=torch.nn.ModuleList([AlignedPointwiseReader(reader) for reader in full_readers])
    if a.rank==0:
        hooks=[reader.read_out.register_forward_hook(lambda module,args,out:torch.zeros_like(out)) for reader in full_readers]
        try:
            w=ScheduledPrefixRollout(teacher,student,range(90))
            gate_z,_=sample_rdd(w,4,[80,80,90],dev,915915)
        finally:
            for hook in hooks:hook.remove()
        student.readers=pointwise_readers
        w=ScheduledPrefixRollout(teacher,student,range(90))
        bypass_z,_=sample_rdd(w,4,[80,80,90],dev,915915)
        torch.testing.assert_close(gate_z,bypass_z,atol=0,rtol=0)
        write_json(out/'runtime_parity.json',dict(complete=True,gate0_vs_pointwise_max_abs=0.,samples=4,steps=[80,80,90]))
        student.readers=full_readers
        del w,gate_z,bypass_z
    for name,gate in protocol['gates'].items():
        handles=[]
        student.readers=pointwise_readers if gate==0. else full_readers
        try:
            wrapper=ScheduledPrefixRollout(teacher,None if gate is None else student,() if gate is None else range(90))
            z,_=sample_rdd(wrapper,batch,[80,80,90],dev,seed)
        finally:
            for handle in handles:handle.remove()
        if name=='original':
            endpoints,labels=wrapper.endpoints,wrapper.labels
            old=Path(a.root)/'split_quality256'/f'shard_{a.rank}/original_dense_latents.pt'
            torch.testing.assert_close(z.cpu(),torch.load(old,map_location='cpu',weights_only=True),atol=0,rtol=0)
        torch.testing.assert_close(wrapper.labels,labels,atol=0,rtol=0)
        for res,d in wrapper.endpoints.items():
            for key,value in d.items():torch.testing.assert_close(value,endpoints[res][key],atol=0,rtol=0)
        latents[name]=z.cpu();del z,wrapper
        print(json.dumps(dict(rank=a.rank,step=step,sampled=name)),flush=True)
    del teacher,student,full_readers,pointwise_readers;torch.cuda.empty_cache()
    vae=AutoencoderKL.from_pretrained('checkpoints/pretrained/stabilityai/sd-vae-ft-ema',local_files_only=True,
        torch_dtype=torch.bfloat16).to(dev).eval().requires_grad_(False)
    pictures={}
    for name,z in latents.items():
        pictures[name]=torch.cat([decode(vae,z[i:i+4].to(dev)) for i in range(0,batch,4)]).cpu()
        folder=out/name;folder.mkdir(exist_ok=True)
        for i,img in enumerate(pictures[name]):to_pil(img).save(folder/f'{i:03d}.png')
    save_grid([(name,images[:8]) for name,images in pictures.items()],out/'comparison.png')
    del vae;torch.cuda.empty_cache()
    weights=ResNet50_Weights.IMAGENET1K_V2;cp=Path(torch.hub.get_dir())/'checkpoints/resnet50-11ad3fa6.pth'
    classifier=resnet50(weights=None).eval();classifier.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True))
    transform=weights.transforms();rows=[]
    for name,images in pictures.items():
        for start in range(0,batch,8):
            probs=classifier(torch.stack([transform(to_pil(im)) for im in images[start:start+8]])).softmax(-1)
            for j,p in enumerate(probs):
                i=start+j;target=int(labels[i]);top=p.topk(5).indices.tolist()
                rows.append(dict(variant=name,image=a.rank*batch+i,local_image=i,target=target,
                    top1_correct=top[0]==target,top5_correct=target in top,target_probability=float(p[target]),
                    image_sha256=sha(out/name/f'{i:03d}.png')))
    write_json(out/'metrics.json',dict(complete=True,protocol=protocol,config=config,rows=rows,
        labels=labels.tolist(),coarse_endpoint_max_abs=0.,classifier_sha256=sha(cp),
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=ROOT);p.add_argument('--rank',type=int,required=True)
    main(p.parse_args())
