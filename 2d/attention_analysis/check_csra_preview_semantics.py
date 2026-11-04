#!/usr/bin/env python3
"""CPU-only ImageNet-label sanity for EXISTING previews; not a quality metric."""
import hashlib
import json
from pathlib import Path
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50
from probe_csra_prefix_information import write_json


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    out=Path('eval_results/csra_prefix8_preview_semantics_20260913/metrics.json')
    if out.exists(): raise FileExistsError('Read previous results; do not rerun')
    weights=ResNet50_Weights.IMAGENET1K_V2
    path=Path(torch.hub.get_dir())/'checkpoints'/'resnet50-11ad3fa6.pth'
    if not path.is_file(): raise FileNotFoundError('Offline audit requires existing classifier cache')
    classifier=resnet50(weights=None).eval()
    classifier.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
    transform=weights.transforms()
    names=weights.meta['categories']
    rows=[]
    for seed in (20260921,20275004):
        probe=Path(f'eval_results/csra_prefix8_all_times_20260913/budget_3200_aligned/seed_{seed}.json')
        observed=json.loads(probe.read_text())
        labels=sorted({(r['image'],r['label']) for r in observed['rows']})
        if len(labels)!=4: raise ValueError('Expected four recorded actual CUDA labels')
        variants=[('original_dense','conditional','original_dense')]
        variants += [(f'{budget}_{arm}','conditional' if budget==1600 else 'conditional3200',f'prefix_{arm}')
                     for budget in (1600,3200) for arm in ('aligned','zero')]
        for variant,version,folder in variants:
            files=[Path(f'eval_results/csra_prefix8_{version}_pair_20260913/seed_{seed}/{folder}/{i:03d}.png') for i,_ in labels]
            pixels=[]
            for file in files:
                with Image.open(file) as image: pixels.append(transform(image.convert('RGB')))
            probs=classifier(torch.stack(pixels)).softmax(-1)
            for index,(image_id,target) in enumerate(labels):
                top=probs[index].topk(5)
                file=files[index]
                rows.append(dict(seed=seed,image=image_id,variant=variant,
                    file=str(file),image_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                    target=target,target_name=names[target],target_probability=float(probs[index,target]),
                    top1_correct=int(top.indices[0])==target,top5_correct=target in top.indices.tolist(),
                    top5=[dict(index=int(k),name=names[int(k)],probability=float(v))
                          for k,v in zip(top.indices,top.values)]))
    summary={}
    for variant in sorted({r['variant'] for r in rows}):
        selected=[r for r in rows if r['variant']==variant]
        summary[variant]=dict(images=len(selected),top1=sum(r['top1_correct'] for r in selected),
            top5=sum(r['top5_correct'] for r in selected),
            mean_target_probability=sum(r['target_probability'] for r in selected)/len(selected))
    write_json(out,dict(complete=True,rows=rows,summary=summary,
        classifier='torchvision ResNet50 ImageNet1K V2; CPU FP32; official weights.transforms()',
        classifier_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        warning='Only8 reused development images per variant, not independent accuracy/FID/human-quality validation. ImageNet target may be an object rather than a visible person. Composition change is not automatically class failure; classifier agreement also does not certify realistic geometry or visual quality. No generation, GPU work or model updates. Labels taken from actual recorded trajectories, not guessed from filenames or CPU RNG.'))
    print(json.dumps(summary),flush=True)


if __name__=='__main__': main()
