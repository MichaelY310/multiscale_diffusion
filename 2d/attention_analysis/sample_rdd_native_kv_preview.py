#!/usr/bin/env python3
"""Small paired generation diagnostic; full QKV retained, NO speed claims."""
import argparse
import json
from pathlib import Path
import torch
from probe_rdd_native_coarse_kv import LAYERS,project,parent_center_rope,block_with_memory,apply_rotary_emb
from sample_csra_adapter_preview import manual_forward,sample_rdd,decode,relative_l2,save_grid,to_pil,AutoencoderKL
from train_rdd_csra_adapter import load_frozen_teacher


class KVPreview:
    def __init__(self,model,mode=None,active_calls=90):
        self.model,self.mode,self.active_calls=model,mode,active_calls
        self.fine_calls=0
        self.replaced=0
        self.cache={}
        self.endpoints={}

    def forward(self,x,t,y):
        model=self.model
        res,step=int(x.shape[-1]),int(t[0])
        if res==16 and step==300 and self.mode is not None:
            result,trace=manual_forward(model,x,t,y,multires=True,capture_layers=set(LAYERS))
            source_rope=model.get_rope(8,8,x.device)
            centers=parent_center_rope(model.head_dim,16,model.base_size,x.device)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                c=model.t_embedder(t)+model.y_embedder(y,False)
                for l in LAYERS:
                    _,k,v,_,norm=project(model.blocks[l],trace[l]['input'],c)
                    self.cache[l]={'raw_k':k,'v':v,'normalized':norm,
                        'rotated_k':apply_rotary_emb(k,*source_rope),'center_rope':centers}
        elif res==32 and self.mode is not None and self.fine_calls<self.active_calls:
            if set(self.cache)!=set(LAYERS):raise RuntimeError('Missing original source stage')
            rope=model.get_rope(16,16,x.device)
            callbacks={l:(lambda h,c,l=l:block_with_memory(model.blocks[l],h,c,rope,self.cache[l],self.mode)) for l in LAYERS}
            result,_=manual_forward(model,x,t,y,multires=True,replacements=callbacks,
                conditioned_replacements=True,replacement_mode='hidden')
            self.replaced+=len(LAYERS)
        else:
            result,_=manual_forward(model,x,t,y,multires=True)
        if res==32:self.fine_calls+=1
        if (res,step) in ((8,600),(16,300)):
            self.endpoints[str(res)]={'input':x.cpu().clone(),'output':result.cpu().clone()}
        return result


@torch.inference_mode()
def main(args):
    out=Path(args.output_dir)
    if (out/'metrics.json').exists():return
    device=torch.device('cuda:0')
    if torch.cuda.mem_get_info(device)[0]<10*2**30:raise RuntimeError('Need10GiB free, no other process affected')
    model=load_frozen_teacher(Path('checkpoints/RDD-Unified-L/0400000.pt'),device,'DiT-L/2','rdd')
    settings={'original_dense':(None,0),'old_coarse_kv_all90':('cached_native',90),
        'old_coarse_kv_first20':('cached_native',20),'current_parent_all90':('current_parent',90),
        'current_parent_first20':('current_parent',20)}
    latents,parity,counts={},{},{}
    reference=None
    for name,(mode,window) in settings.items():
        wrapper=KVPreview(model,mode,window)
        latents[name],_=sample_rdd(wrapper,4,[80,80,90],device,args.seed)
        assert wrapper.fine_calls==90 and set(wrapper.endpoints)=={'8','16'}
        if reference is None:reference=wrapper.endpoints
        parity[name]=max(float((value.float()-reference[r][key].float()).abs().max())
            for r,fields in wrapper.endpoints.items() for key,value in fields.items())
        assert parity[name]==0
        counts[name]=wrapper.replaced
        assert counts[name]==(window*len(LAYERS) if mode else 0)
        del wrapper
    del model
    torch.cuda.empty_cache()
    vae=AutoencoderKL.from_pretrained('checkpoints/pretrained/stabilityai/sd-vae-ft-ema',
        local_files_only=True,torch_dtype=torch.bfloat16).to(device).eval().requires_grad_(False)
    decoded={name:decode(vae,z) for name,z in latents.items()}
    out.mkdir(parents=True,exist_ok=True)
    for name,images in decoded.items():
        (out/name).mkdir(exist_ok=True)
        for i,im in enumerate(images):to_pil(im).save(out/name/f'{i:03d}.png')
    save_grid(list(decoded.items()),out/'comparison_grid.png')
    payload={'complete':True,'seed':args.seed,'batch':4,'layers':LAYERS,'steps':[80,80,90],
        'cfg':1.,'upsample':'noise_filling','settings':settings,'replaced_attention_calls':counts,
        'coarse_endpoint_max_abs_vs_original':parity,
        'relative_latent_l2':{name:relative_l2(z,latents['original_dense']) for name,z in latents.items()},
        'warning':'Three old development seeds. No training, no latency/FID. Native coarse KV is cached at16t300, current_parent is same-time pooledfine KV NOTVSA. CURRENT fineMLP/gates retained. FullQKV deliberately computed for exactQ parity. first20 returns to original dense attention after20 of90 fineNFEs, smaller theoretical savings.'}
    temp=out/'metrics.json.tmp'
    temp.write_text(json.dumps(payload,indent=2,allow_nan=False))
    temp.replace(out/'metrics.json')
    print(json.dumps(payload),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--output-dir',required=True)
    main(p.parse_args())
