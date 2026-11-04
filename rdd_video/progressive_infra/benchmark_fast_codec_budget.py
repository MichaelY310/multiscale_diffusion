"""Strong fast-codec and untrained renderer budget screen, no generation win."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time
import torch

from progressive_renderer import TemporalInnovationRenderer


def main(a):
    root=Path(a.root)
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.json').exists():
        raise FileExistsError(out)
    if torch.cuda.mem_get_info()[0]<40*2**30:
        raise RuntimeError('At least40GiB free; never evict jobs')
    torch.cuda.set_per_process_memory_fraction(.5)
    torch.set_num_threads(2)
    sys.path.insert(0,str(root/'Wan2.1'))
    from wan.modules.vae import WanVAE
    native=WanVAE(z_dim=16,vae_pth=str(root/'pretrained_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth'),device='cuda',dtype=torch.bfloat16)
    code=root/'third_party/Flash-VAED-ref/models/wan/model_hybrid_aggressive.py'
    spec=importlib.util.spec_from_file_location('flash_student_budget',code)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    flash=mod.WanVAE(vae_pth=str(code.parent/'Flash_VAED_Wan.pth'),device='cuda',dtype=torch.bfloat16)
    z=torch.load(a.latent,map_location='cpu',weights_only=True).to('cuda',dtype=torch.bfloat16)
    if z.ndim==4:
        z=z.unsqueeze(0)
    if z.shape[2]%4:
        raise ValueError('Uniform temporal screen only')
    parent=(z[:,:,0::2]+z[:,:,1::2])/2
    detail=(z[:,:,0::2]-z[:,:,1::2])/2
    root_parent=parent.reshape(*parent.shape[:2],parent.shape[2]//2,2,*parent.shape[3:]).mean(3)
    renderer=TemporalInnovationRenderer(width=a.width,depth=a.depth).to('cuda',dtype=torch.bfloat16).eval().requires_grad_(False)
    cases={
        'native_full':lambda:native.decode([z[0].float()])[0],
        'flash_full':lambda:flash.decode(z)[0],
        'flash_parent_half':lambda:flash.decode(parent)[0],
        'flash_parent_quarter':lambda:flash.decode(root_parent)[0],
    }
    outputs={}
    timing={}
    with torch.inference_mode():
        for name,fn in cases.items():
            print('Warming '+name,flush=True)
            for _ in range(2):
                result=fn()
            torch.cuda.synchronize()
            records=[]
            for _ in range(8):
                start=time.perf_counter()
                result=fn()
                torch.cuda.synchronize()
                records.append(time.perf_counter()-start)
            outputs[name]=result.detach().cpu()
            timing[name]=dict(median_seconds=statistics.median(records),records=records)
            print(json.dumps({name:timing[name]}),flush=True)
        rgb=outputs['flash_parent_half'].unsqueeze(0).to('cuda')
        for name,fn in [('renderer_prepare',lambda:renderer.prepare_detail(parent,detail)),
                        ('renderer_full',lambda:renderer(parent,detail,rgb))]:
            for _ in range(2):
                result=fn()
            torch.cuda.synchronize()
            records=[]
            for _ in range(8):
                start=time.perf_counter()
                result=fn()
                torch.cuda.synchronize()
                records.append(time.perf_counter()-start)
            timing[name]=dict(median_seconds=statistics.median(records),records=records)
            print(json.dumps({name:timing[name]}),flush=True)
        output=renderer(parent,detail,rgb)
        no_h=renderer(parent,torch.zeros_like(detail),rgb)
        h_response=float((output-no_h).square().mean().sqrt())
        prepared=renderer.prepare_detail(parent,detail)
        split=renderer.finish(rgb,prepared)
        torch.testing.assert_close(output,split,rtol=0,atol=0)
        from PIL import Image
        for name,value in [('native',outputs['native_full']),('flash',outputs['flash_full']),('untrained_renderer',output[0].cpu())]:
            frame=value[:,value.shape[1]//2].float().clamp(-1,1)
            Image.fromarray(((frame.permute(1,2,0)+1)*127.5).round().byte().numpy()).save(out/(name+'.png'))
    mse=float((outputs['native_full']-outputs['flash_full']).square().mean())
    report=dict(complete=True,mode='Single-device codec budget, NOT an early pipeline or trained quality result',
        shape=list(z.shape),latent_source=str(a.latent),gpu=torch.cuda.get_device_name(),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        timings=timing,renderer=dict(width=a.width,depth=a.depth,parameters=sum(p.numel() for p in renderer.parameters()),
            trained=False,zero_h_response_rmse=h_response,split_execution_exact=True),
        flash_native_mse=mse,flash_native_psnr_db=10*__import__('math').log10(4/max(mse,1e-12)),
        warning='Parent pooling is a fixed-latent hardware screen. Ordinary post-final can run h-only prepare while parent decoder runs. Current RDD does NOT guarantee immutable parent. First-frame time basis requires training/validation. No quality recovery promise.')
    (out/'metrics.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='/scr/yuheng/DevOpsBench/multiscale_diffusion')
    p.add_argument('--output',required=True)
    p.add_argument('--latent',required=True)
    p.add_argument('--width',type=int,default=64)
    p.add_argument('--depth',type=int,default=4)
    main(p.parse_args())
