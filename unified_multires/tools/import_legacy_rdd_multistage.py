#!/usr/bin/env python3
"""Bundle explicitly mapped trusted legacy stage weights for umdiff sampling.

No downloads or training-state conversion. A 400K/600K mixed stage bundle is
allowed only through an explicit resolution -> local path mapping.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'_deps')]
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import torch
from umdiff import build
from umdiff.cli import make_method
from umdiff import rdd_multistage


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()


def resolution_value(value):
    if isinstance(value, str) and 'x' in value:
        parts=value.split('x')
        if len(parts)!=2 or parts[0]!=parts[1] or not parts[0].isdigit():
            raise ValueError('Checkpoint resolution must be a square grid')
        return int(parts[0])
    return int(value)


def stage_lengths(value):
    return [int(x) for x in value.split(',')] if isinstance(value,str) else list(value)


def convert(config_file,checkpoints,output,weights='ema'):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    config=json.loads(Path(config_file).read_text())
    if config.get('format')=='umdiff_experiment_v1':config=config['method']
    config=copy.deepcopy(config)
    fm=config.get('task')=='rdd_fm_native'
    if config.get('model',{}).get('name')!='native_rdd_multistage' or (not fm and config.get('sampler',{}).get('name')!='rdd_native_stages'):
        raise ValueError('Choose native_rdd_multistage with native RDD DDPM or FM')
    make_method(config)
    settings=config['process'] if fm else config['sampler']['args']
    ordered_resolutions=[str(int(r)) for r in settings['resolutions']]
    resolutions=set(ordered_resolutions)
    paths={str(int(r)):Path(path).resolve() for r,path in checkpoints.items()}
    if set(paths)!=resolutions:raise ValueError('One explicit checkpoint is required per configured stage resolution')
    if any(not p.is_file() for p in paths.values()):raise FileNotFoundError('At least one local checkpoint path does not exist')
    with torch.device('meta'):model=build('model',config['model'])
    records=[]
    for r,path in paths.items():
        value=torch.load(path,map_location='cpu',weights_only=False)
        if not isinstance(value,dict) or weights not in value:raise ValueError(f'{path}: missing requested {weights} weights; no fallback')
        state=value[weights]
        if not isinstance(state,dict) or not state or not all(isinstance(k,str) and torch.is_tensor(v) for k,v in state.items()):
            raise ValueError('Selected weights must be a nonempty tensor state dict')
        net=model.for_resolution(int(r));expected=net.state_dict()
        if expected.keys()!=state.keys():raise ValueError(f'Checkpoint keys differ for resolution {r}')
        if any(expected[k].shape!=state[k].shape for k in expected):raise ValueError(f'Checkpoint tensor shapes differ for resolution {r}')
        if 'resolution' in value and resolution_value(value['resolution'])!=int(r):raise ValueError(f'Checkpoint resolution metadata disagrees with mapping {r}')
        args=value.get('args',{})
        args=vars(args) if hasattr(args,'__dict__') else args
        if not isinstance(args,dict):args={}
        length_fields=[v for v in (value.get('stage_lengths'),args.get('stage_lengths')) if v is not None]
        for field in length_fields:
            lengths=stage_lengths(field)
            if not lengths or min(lengths)<=0:raise ValueError('Invalid checkpoint stage lengths')
            if fm:
                current=0;bounds=[0.]
                for length in lengths:current+=length;bounds.append(current/sum(lengths))
                if bounds!=settings['bounds']:raise ValueError('Checkpoint FM stage-length metadata differs from requested clock')
            elif lengths!=settings['stage_lengths']:raise ValueError('Checkpoint stage-length metadata differs from requested recipe')
        if fm and value.get('s_boundaries') is not None and list(value['s_boundaries'])!=settings['bounds']:
            raise ValueError('Checkpoint FM boundary metadata differs from requested clock')
        if not fm:
            i=ordered_resolutions.index(r);lengths=settings['stage_lengths']
            for key,expected in [('T_c',sum(lengths[:i])),('T_c_plus_1',sum(lengths[:i+1]))]:
                if value.get(key) is not None and int(value[key])!=expected:
                    raise ValueError(f'Checkpoint {key} metadata differs from requested stage interval')
        net.load_state_dict(state,strict=True,assign=True)
        records.append({'resolution':int(r),'path':str(path),'sha256':sha256(path),
                        'weight_key':weights,'source_step':value.get('step',value.get('train_steps')),
                        'tensor_count':len(state),'strict_keys_shapes':True})
    config.setdefault('training',{})['inference_only']=True
    state=model.state_dict();meta={'source_stages':records,'optimizer_imported':False,'resume_supported':False,
        'statement':'Inference-only envelope; model and ema reference the explicitly selected source weights.',
        'importer_sha256':sha256(__file__)}
    envelope={'format':'umdiff_legacy_rdd_multistage_inference_v1','config':config,'model':state,'ema':state,'legacy_import':meta}
    output.parent.mkdir(parents=True,exist_ok=True);temporary=output.with_name(output.name+f'.{os.getpid()}.tmp')
    try:torch.save(envelope,temporary);temporary.replace(output)
    finally:
        if temporary.exists():temporary.unlink()
    result={**meta,'output':str(output),'output_bytes':output.stat().st_size,'output_sha256':sha256(output)}
    output.with_suffix(output.suffix+'.import.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--checkpoint',action='append',required=True,metavar='RESOLUTION=LOCAL_PATH')
    parser.add_argument('--weights',choices=['ema','model'],default='ema');parser.add_argument('--output',required=True)
    args=parser.parse_args();mapping={}
    for item in args.checkpoint:
        resolution,path=item.split('=',1);resolution=str(int(resolution))
        if resolution in mapping:parser.error(f'Duplicate resolution {resolution}')
        mapping[resolution]=path
    print(json.dumps(convert(args.config,mapping,args.output,args.weights),indent=2))


if __name__=='__main__':main()
