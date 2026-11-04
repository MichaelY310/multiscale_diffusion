"""Real CPU entry-point tests with random tiny weights/codecs; no downloads."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from PIL import Image

from umdiff import build

BUNDLE = Path(__file__).resolve().parents[1]/'experiments/imagenet256_fm_pf'
sys.path.insert(0, str(BUNDLE))
import workflow
import sampling


@pytest.fixture(scope='module')
def codec(tmp_path_factory):
    from diffusers import AutoencoderKL
    path = tmp_path_factory.mktemp('tiny_codec')
    torch.manual_seed(481)
    model = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4,
        down_block_types=('DownEncoderBlock2D',)*4,
        up_block_types=('UpDecoderBlock2D',)*4, block_out_channels=(8,8,8,8),
        layers_per_block=1, norm_num_groups=4, sample_size=256)
    model.save_pretrained(path)
    return path


def test_asset_integrity_and_fair_config(tmp_path):
    path = tmp_path/'source'
    path.write_bytes(b'pinned asset')
    entry = {'size':len(path.read_bytes()), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    assert workflow.verify_asset(path, entry)['sha256'] == entry['sha256']
    path.write_bytes(b'changed-data')
    with pytest.raises(ValueError):workflow.verify_asset(path, entry)
    fm, pf = workflow.read(BUNDLE/'rdd_fm_fresh.json'), workflow.read(BUNDLE/'pyramid_flow_train.json')
    assert fm['process']['bounds'] == pf['method']['process']['args']['stage_range'] == [0,.3,.6,1]
    assert fm['process']['boundary'] == 'fresh'
    assert [n+int(i<2) for i,n in enumerate(fm['process']['steps'])] == pf['method']['process']['args']['steps']
    assert pf['run']['global_batch'] == 256 and pf['run']['updates'] == 400000
    assert pf['method']['process']['args']['stage_weights'] == [1.,1.,1.]
    assert pf['method']['model']['args']['architecture'] == {**fm['model']['args'], 'num_classes':1000}


def test_fm_sampling_pixels_resume_and_corruption(tmp_path, codec):
    from umdiff.cli import make_method
    from umdiff.randomness import RandomStream
    from umdiff.input_transforms import local_autoencoder
    torch.set_num_threads(2)
    config = workflow.read(BUNDLE/'rdd_fm_fresh.json')
    config['model']['args'].update(hidden_size=32, depth=1, num_heads=4)
    torch.manual_seed(82)
    net = build('model', config['model']).eval()
    # Nonzero outputs so a zero-head initialization cannot hide drift/CFG bugs.
    with torch.no_grad():net.final_layer.linear.weight.normal_(0,.002)
    ckpt = tmp_path/'ema.pt'
    torch.save({'config': config, 'model':net.state_dict(), 'ema':net.state_dict()}, ckpt)
    output = tmp_path/'samples'
    options = dict(count=2, batch_size=1, decode_batch=1, seed=9, cfg=1.5,
                   kind='fm', device='cpu', threads=2)
    result = sampling.sample_directory(ckpt, codec, output, **options)
    assert result['nfe'] == 252
    mtimes = [p.stat().st_mtime_ns for p in sorted((output/'png').glob('*.png'))]
    again = sampling.sample_directory(ckpt, codec, output, **options)
    assert again['new_samples_this_invocation'] == 0
    assert mtimes == [p.stat().st_mtime_ns for p in sorted((output/'png').glob('*.png'))]
    # Independent direct native-method execution, same initial noise/RNG/EMA.
    config['process']['cfg_scale'] = 1.5
    method = make_method(config)
    stream = RandomStream(9, 'cpu')
    initial = stream.normal(torch.empty(1,*method.initial_shape()))
    with torch.no_grad():
        latent = method.sample(net, initial, stream=stream, model_kwargs={'y':torch.tensor([0])})
        decoded = local_autoencoder(codec).decode(latent/.18215, return_dict=False)[0]
    reference = ((decoded.clamp(-1,1)+1)/2*255).byte().permute(0,2,3,1)[0].numpy()
    import numpy as np
    assert np.array_equal(reference, np.asarray(Image.open(output/'png/000000.png')))
    with pytest.raises(ValueError, match='settings/assets/environment'):
        sampling.sample_directory(ckpt, codec, output, **{**options,'batch_size':2})
    (output/'png/000000.png').write_bytes(b'corruption')
    with pytest.raises(ValueError, match='missing/corrupt'):
        sampling.sample_directory(ckpt, codec, output, **options)


def test_pf_real_codec_training_resume_and_sampling(tmp_path, codec):
    torch.set_num_threads(2)
    spec = workflow.read(BUNDLE/'pyramid_flow_train.json')
    spec['method']['model']['args']['architecture'].update(hidden_size=64, depth=1, num_heads=2)
    spec['run'].update(global_batch=2, micro_batch=1, updates=3, save_every=1, log_every=1)
    from umdiff.experiment import resolve
    config = resolve(spec, str(codec))
    data = tmp_path/'data'
    for c in range(2):
        folder = data/f'n{c:08d}'
        folder.mkdir(parents=True)
        for j in range(2):
            Image.new('RGB',(280+j,270+j),(30+20*c,60+10*j,120)).save(folder/f'{j}.png')
    split, full = tmp_path/'split', tmp_path/'full'
    for dest in (split, full):
        workflow.write(dest/'experiment.json', spec)
        workflow.write(dest/'resolved_config.json', config)
    child_env = dict(os.environ, CUDA_VISIBLE_DEVICES='', UMDIFF_CPU_ONLY='1',
                     HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    def run(dest, target):
        result = subprocess.run([sys.executable, str(BUNDLE/'workflow.py'), '_train-worker',
            '--output', str(dest), '--data', str(data), '--target-step', str(target),
            '--device','cpu','--threads','2'], env=child_env, text=True, capture_output=True, timeout=120)
        assert result.returncode == 0, result.stdout+result.stderr
    run(split,1);run(split,3);run(full,3)
    a,b = workflow.load_checkpoint(split/'checkpoint.pt'), workflow.load_checkpoint(full/'checkpoint.pt')
    assert a['step'] == b['step'] == 3
    for key in ('model','ema'):
        assert all(torch.equal(a[key][k],b[key][k]) for k in a[key])
    for key, slots in a['optimizer']['state'].items():
        for name, value in slots.items():
            assert torch.equal(value,b['optimizer']['state'][key][name])
    assert any(p.abs().sum()>0 for name,p in a['model'].items() if name.endswith('final_layer.linear.weight'))
    result = sampling.sample_directory(split/'checkpoint.pt',codec,tmp_path/'pf_samples',
        count=1,batch_size=1,decode_batch=1,seed=9,cfg=1.,kind='pf',device='cpu',threads=2)
    assert result['nfe'] == 252 and result['samples'] == 1
    with pytest.raises(ValueError, match='CFG1'):
        sampling.sample_directory(split/'checkpoint.pt',codec,tmp_path/'unsupported',
            count=1,batch_size=1,decode_batch=1,seed=9,cfg=1.5,kind='pf',device='cpu')
    with pytest.raises(ValueError, match='1000'):
        workflow.inspect_data(data)


def test_imagenet_export_does_not_reencode_or_overwrite(tmp_path):
    import prepare_imagenet
    path = tmp_path/'class/image.JPEG'
    body = b'original encoded image fixture'
    prepare_imagenet.save_bytes(path,body)
    prepare_imagenet.save_bytes(path,body)
    assert path.read_bytes() == body
    with pytest.raises(ValueError):prepare_imagenet.save_bytes(path,b'changed')
