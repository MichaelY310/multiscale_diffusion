"""Decode saved fixed teacher/student latents; never loads a training model."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from diffusers import AutoencoderKL

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--step',type=int,required=True)
    p.add_argument('--vae',default='/scr/yuheng/DevOpsBench/multiscale_diffusion/checkpoints/pretrained/stabilityai/sd-vae-ft-ema')
    a=p.parse_args();root=Path(a.run);out=root/f'preview_{a.step:07d}';out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():raise SystemExit(0)
    torch.cuda.set_per_process_memory_fraction(.05);torch.set_num_threads(2)
    vae=AutoencoderKL.from_pretrained(a.vae,local_files_only=True).to('cuda').eval().requires_grad_(False)
    files=sorted(root.glob(f'validation_{a.step:07d}_rank*.pt'))
    assert len(files)==4,'Wait until all four ranks have saved validation latents'
    items=[]
    for file in files:items.extend(torch.load(file,map_location='cpu',weights_only=True))
    labels=torch.cat([b['labels'] for b in items]);teacher=torch.cat([b['teacher'] for b in items]);student=torch.cat([b['student'] for b in items])
    images=[]
    with torch.inference_mode():
        for side,latents in [('teacher',teacher),('student',student)]:
            rows=[]
            for start in range(0,len(labels),2):
                decoded=vae.decode(latents[start:start+2].cuda()/0.18215).sample.float().clamp(-1,1)
                arrays=((decoded+1)*127.5).round().byte().permute(0,2,3,1).cpu().numpy()
                for j,array in enumerate(arrays,start):
                    im=Image.fromarray(array);im.save(out/f'{side}_{j:03d}_class{int(labels[j])}.png');rows.append(im)
            images.append(rows)
    # Fixed first 16, no success selection. Every decoded image is also retained.
    grid=Image.new('RGB',(4*256,8*278),'white');draw=ImageDraw.Draw(grid)
    for i in range(min(16,len(labels))):
        row=i//2;col=(i%2)*2
        for side in range(2):
            x=(col+side)*256;y=row*278;grid.paste(images[side][i],(x,y+22))
            draw.text((x+4,y+4),f'{i} class={int(labels[i])} '+('DMD baseline' if side==0 else 'reuse student'),fill='black')
    grid.save(out/'comparison_first16.jpg',quality=93)
    vae.cpu();torch.cuda.empty_cache()
    audit={'complete':True,'step':a.step,'samples':len(labels),'selection':'all images saved; grid=fixed first16','NOT_FID':True}
    weights=Path.home()/'.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth'
    if weights.exists():
        from torchvision.models import resnet50,ResNet50_Weights
        classifier=resnet50(weights=None).eval().requires_grad_(False)
        classifier.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
        transform=ResNet50_Weights.IMAGENET1K_V2.transforms()
        scores={}
        with torch.inference_mode():
            for side,rows in zip(('teacher','student'),images):
                logits=torch.cat([classifier(torch.stack([transform(im) for im in rows[i:i+8]])) for i in range(0,len(rows),8)])
                top=logits.topk(5,dim=1).indices
                scores[side]={'top1_correct':int((top[:,0]==labels).sum()),'top5_correct':int((top==labels[:,None]).any(1).sum()),
                              'target_probability':float(logits.softmax(1).gather(1,labels[:,None]).mean())}
        audit['class_consistency']=scores
        audit['class_consistency_warning']=f'{len(labels)} fixed development images; not FID or proof of perceptual quality equivalence'
    else:audit['class_consistency']='cached classifier unavailable; no download attempted'
    (out/'complete.json').write_text(json.dumps(audit,indent=2))
    print(f'PREVIEW COMPLETE {out}',flush=True)
