"""Independent CPU comparison of every native VAE weight with source safetensors."""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU only')
    import torch
    from safetensors import safe_open
    from diffusers import AutoencoderKLWan
    from encode_small_real_data import WAN_MODEL
    path=WAN_MODEL/'vae/diffusion_pytorch_model.safetensors'
    model=AutoencoderKLWan.from_pretrained(str(WAN_MODEL),subfolder='vae',torch_dtype=torch.float32,
        local_files_only=True)
    values=model.state_dict();checks=[]
    with safe_open(str(path),framework='pt',device='cpu') as source:
        assert set(values)==set(source.keys())
        for name,value in values.items():
            original=source.get_tensor(name)
            checks.append({'name':name,'source_dtype':str(original.dtype),'loaded_dtype':str(value.dtype),
                'bitwise_equal':torch.equal(value,original)})
    assert all(c['bitwise_equal'] and c['loaded_dtype']=='torch.float32' for c in checks)
    code=Path(inspect.getfile(AutoencoderKLWan))
    result={'status':'passed','weight_tensors':len(checks),'checks':checks,
        'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'diffusers_distribution_version':importlib.metadata.version('diffusers'),
        'source_file':str(code),'source_sha256':hashlib.sha256(code.read_bytes()).hexdigest(),
        'cuda_context_initialized':torch.cuda.is_initialized(),
        'old_cache_caveat':'wan_v1 loaded VAE in BF16 then upcast; v2 loads original FP32 directly'}
    assert not result['cuda_context_initialized'];args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({'status':result['status'],'weight_tensors':len(checks)}))


if __name__=='__main__':main()
