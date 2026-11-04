#!/usr/bin/env python3
"""Import trusted local legacy RDD weights into an inference-only umdiff envelope.

No downloads, no optimizer/resume conversion. Model keys and tensor shapes must
match the selected legacy DiT architecture exactly. Sampling uses EMA by default.
"""
import argparse, copy, hashlib, json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'_deps')]
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import torch
from umdiff import build
from umdiff.cli import make_method


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read_config(path,vae=None):
    config=json.loads(Path(path).read_text())
    if config.get('format')=='umdiff_experiment_v1':config=config['method']
    config=copy.deepcopy(config)
    if config.get('model',{}).get('name')!='native_rdd_unified':
        raise ValueError('This importer supports unchanged legacy DiTMultiRes via native_rdd_unified only')
    if config.get('task')!='rdd_fm_native' and config.get('sampler',{}).get('name')!='rdd_native_stages':
        raise ValueError('Select a checkpoint-compatible native RDD DDPM or FM configuration')
    if vae:
        vae=Path(vae).resolve()
        if not (vae/'config.json').is_file():raise FileNotFoundError('Provide an existing local Diffusers VAE folder')
        def replace(x):
            if isinstance(x,str) and x in ('@sd_vae_ema@','@vae@'):return str(vae)
            if isinstance(x,dict):return {k:replace(v) for k,v in x.items()}
            if isinstance(x,list):return [replace(v) for v in x]
            return x
        config=replace(config)
    make_method(config)
    return config


def import_checkpoint(config_path,checkpoint,output,weights='ema',vae=None,strip_module_prefix=False):
    source=Path(checkpoint).resolve();output=Path(output).resolve()
    if not source.is_file():raise FileNotFoundError(source)
    if output.exists():raise FileExistsError(output)
    config=read_config(config_path,vae)
    # Original checkpoints may contain argparse.Namespace. Only import local
    # trusted project artifacts; this command does not retrieve remote files.
    checkpoint=torch.load(source,map_location='cpu',weights_only=False)
    if not isinstance(checkpoint,dict) or weights not in checkpoint:
        raise ValueError(f'Legacy checkpoint is missing requested {weights!r} weights; no fallback was used')
    state=checkpoint[weights]
    if not isinstance(state,dict) or not state or not all(isinstance(k,str) and torch.is_tensor(v) for k,v in state.items()):
        raise ValueError('Selected field must be a nonempty tensor state_dict')
    if strip_module_prefix:
        if not all(k.startswith('module.') for k in state):raise ValueError('Explicit prefix stripping requires every key to start with module.')
        state={k[len('module.'):]:v for k,v in state.items()}
    # Meta construction validates full XL/L layouts without allocating random
    # duplicate multi-GB weights. assign=True only adopts the existing tensors.
    with torch.device('meta'):model=build('model',config['model'])
    expected=model.state_dict()
    if expected.keys()!=state.keys():
        raise ValueError(f'Strict key mismatch: missing={sorted(expected.keys()-state.keys())[:12]}, unexpected={sorted(state.keys()-expected.keys())[:12]}')
    mismatches=[(k,list(expected[k].shape),list(state[k].shape)) for k in expected if expected[k].shape!=state[k].shape]
    if mismatches:raise ValueError(f'Strict shape mismatch: {mismatches[:12]}')
    model.load_state_dict(state,strict=True,assign=True)
    resolutions=config.get('process',{}).get('resolutions',config.get('sampler',{}).get('args',{}).get('resolutions'))
    if checkpoint.get('resolutions') is not None and list(checkpoint['resolutions'])!=list(resolutions):
        raise ValueError('Checkpoint resolution metadata differs from the selected configuration')
    if config.get('task')=='rdd_fm_native':
        bounds=config['process']['bounds']
        if checkpoint.get('s_boundaries') is not None and list(checkpoint['s_boundaries'])!=bounds:
            raise ValueError('Checkpoint FM boundary metadata differs; legacy-v2 and corrected clocks are not interchangeable')
        if checkpoint.get('stage_lengths') is not None:
            lengths=list(checkpoint['stage_lengths']);total=sum(lengths)
            recovered=[0.];current=0
            for n in lengths:current+=n;recovered.append(current/total)
            if recovered!=bounds:raise ValueError('Checkpoint FM stage-length metadata differs from the requested clock')
    elif checkpoint.get('stage_lengths') is not None and list(checkpoint['stage_lengths'])!=config['sampler']['args']['stage_lengths']:
        raise ValueError('Checkpoint DDPM stage-length metadata differs from requested clock')
    config.setdefault('training',{})['inference_only']=True
    metadata=dict(source_checkpoint=str(source),source_sha256=sha256(source),selected_weights=weights,
        source_step=checkpoint.get('step'),strict_keys_and_shapes=True,state_dict_entries=len(state),
        stripped_module_prefix=strip_module_prefix,optimizer_imported=False,resume_supported=False,
        statement='Inference-only envelope. Both model and ema refer to the selected original weights; no new EMA or optimizer is invented.',
        code_sha256=sha256(__file__))
    envelope=dict(format='umdiff_legacy_rdd_inference_v1',config=config,model=state,ema=state,legacy_import=metadata)
    output.parent.mkdir(parents=True,exist_ok=True);temporary=output.with_name(output.name+f'.{os.getpid()}.tmp')
    try:torch.save(envelope,temporary);temporary.replace(output)
    finally:
        if temporary.exists():temporary.unlink()
    report=dict(**metadata,output=str(output),output_sha256=sha256(output),output_bytes=output.stat().st_size)
    sidecar=output.with_suffix(output.suffix+'.import.json');sidecar.write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--weights',choices=['ema','model'],default='ema');p.add_argument('--vae')
    p.add_argument('--strip-module-prefix',action='store_true')
    a=p.parse_args();report=import_checkpoint(a.config,a.checkpoint,a.output,a.weights,a.vae,a.strip_module_prefix)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
