"""Real local RGB -> random VAE -> configured RDD -> exact resume -> sample."""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from umdiff.experiment import load, resolve


ROOT=Path(__file__).resolve().parents[1]


def test_experiment_rgb_resume_and_sample(tmp_path):
    from diffusers import AutoencoderKL
    vae=AutoencoderKL(in_channels=3,out_channels=3,
                     down_block_types=('DownEncoderBlock2D',)*4,
                     up_block_types=('UpDecoderBlock2D',)*4,
                     block_out_channels=(8,8,8,8),layers_per_block=1,
                     latent_channels=4,norm_num_groups=4,sample_size=256)
    vae.save_pretrained(tmp_path/'vae',safe_serialization=True)
    del vae
    rng=np.random.default_rng(29)
    for i in range(4):
        p=tmp_path/'images'/f'n{i%2:08d}';p.mkdir(parents=True,exist_ok=True)
        Image.fromarray(rng.integers(0,256,(270,260,3),dtype=np.uint8)).save(p/f'{i}.png')
    spec=load(ROOT/'configs/comparison_imagenet256/main/rdd_ddpm_xl_legacy.json')
    spec['method']['model']['args'].update(hidden_size=32,depth=2,num_heads=4)
    spec['run'].update(global_batch=2,micro_batch=1,updates=2,save_every=1,log_every=1)
    config=tmp_path/'experiment.json'
    config.write_text(json.dumps(spec))
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'}
    def run(module,*args):
        result=subprocess.run([sys.executable,'-m',module,*map(str,args)],env=env,text=True,capture_output=True,timeout=120)
        assert result.returncode==0,result.stdout+'\n'+result.stderr
        return result.stdout
    plan=run('umdiff.experiment','plan','--config',config)
    assert 'no network or data loaded' in plan
    args=['--config',config,'--data',tmp_path/'images','--vae',tmp_path/'vae','--threads',2]
    run('umdiff.experiment','train',*args,'--output',tmp_path/'full')
    spec['run']['updates']=1;config.write_text(json.dumps(spec))
    run('umdiff.experiment','train',*args,'--output',tmp_path/'split')
    spec['run']['updates']=2;config.write_text(json.dumps(spec))
    run('umdiff.experiment','train',*args,'--output',tmp_path/'split','--resume',tmp_path/'split/checkpoint.pt')
    a=torch.load(tmp_path/'full/checkpoint.pt',weights_only=False)
    b=torch.load(tmp_path/'split/checkpoint.pt',weights_only=False)
    def exact(x,y):
        if torch.is_tensor(x):assert torch.equal(x,y)
        elif isinstance(x,dict):
            assert x.keys()==y.keys()
            for k in x:exact(x[k],y[k])
        elif isinstance(x,(list,tuple)):
            assert len(x)==len(y)
            for v,w in zip(x,y):exact(v,w)
        else:assert x==y
    for key in ('model','ema','optimizer','step','images_seen','stream'):
        exact(a[key],b[key])
    assert a['step']==2
    for name in ('full','split'):
        run('umdiff.experiment','sample','--config',config,
            '--checkpoint',tmp_path/name/'checkpoint.pt','--count',2,'--labels',0,1,
            '--sample-seed',39,'--output',tmp_path/(name+'.pt'))
    assert torch.equal(torch.load(tmp_path/'full.pt',weights_only=True),torch.load(tmp_path/'split.pt',weights_only=True))


def test_sample_frontend_uses_configured_generic_budget(tmp_path,monkeypatch):
    from umdiff import cli, experiment
    spec=load(ROOT/'configs/comparison_imagenet256/main/flow_xl_fixed.json')
    path=tmp_path/'flow.json';path.write_text(json.dumps(spec))
    ckpt=tmp_path/'checkpoint.pt';torch.save({'config':resolve(spec,'/local/vae')},ckpt)
    calls=[]
    monkeypatch.setattr(cli,'sample',lambda args:calls.append(args))
    monkeypatch.setattr(sys,'argv',['experiment','sample','--config',str(path),
                                  '--checkpoint',str(ckpt),'--labels','2',
                                  '--output',str(tmp_path/'sample.pt')])
    experiment.main()
    assert len(calls)==1 and calls[0].steps==250 and calls[0].raw_weights is False
    spec['method']['model']['args']['time_scale']=1.
    path.write_text(json.dumps(spec))
    import pytest
    with pytest.raises(ValueError,match='Checkpoint method differs'):
        experiment.main()


def test_anchor_fields_and_controlled_protocol():
    source=json.loads((ROOT/'configs/paired_experiments/rdd_unified_xl_300_300_400.json').read_text())
    rdd=load(ROOT/'configs/comparison_imagenet256/main/rdd_ddpm_xl_legacy.json')
    for key in ('model','training_stages','training_stage_reduction'):
        assert rdd['method'][key]==source[key]
    cfg=resolve(rdd)
    assert cfg['sampler']['args']['solver']=='ddpm'
    assert cfg['sampler']['args']['steps']==[90,80,80]
    assert cfg['training']['optimizer']['accumulation_steps']==8
    assert rdd['run']['global_batch']==256 and rdd['run']['updates']==400000
    controlled=load(ROOT/'configs/comparison_imagenet256/controlled/rdd_ddpm_xl_legacy.json')
    assert controlled['method']['training_stages']==rdd['method']['training_stages']
