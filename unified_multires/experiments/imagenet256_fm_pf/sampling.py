"""Batch sampling with atomic completion records and deterministic restart.

One model/VAE load per process. Only PNGs and compact JSON records are retained.
The sampling protocol intentionally fixes batch size: changing it changes RNG.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import time

from workflow import environment, load_checkpoint, read, sha256, write


def validate_completed(marker, folder, start, count):
    if not marker.exists():
        return False
    record = read(marker)
    if record['start'] != start or len(record['files']) != count:
        raise ValueError('Batch completion record does not match this sampling plan')
    for name, digest in record['files'].items():
        path = folder/name
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f'Completed sample is missing/corrupt: {path}; review before rerunning')
    return True


def sample_directory(checkpoint, vae, output, *, count, batch_size, decode_batch,
                     seed, cfg, kind, device, threads=4):
    import torch
    import numpy as np
    from PIL import Image, ImageDraw
    from umdiff.cli import choose_device, make_method
    from umdiff import build
    from umdiff.randomness import RandomStream
    from umdiff.input_transforms import local_autoencoder
    if min(count, batch_size, decode_batch)<1 or not math.isfinite(cfg) or cfg<0:
        raise ValueError('Positive batch/count and finite nonnegative CFG required')
    if kind == 'pf' and cfg != 1.:
        raise ValueError('PF entry currently supports CFG1 only; metadata alone does not enable CFG')
    torch.set_num_threads(threads)
    device = choose_device(device)
    checkpoint, vae, output = Path(checkpoint).resolve(), Path(vae).resolve(), Path(output).resolve()
    envelope = load_checkpoint(checkpoint)
    config = copy.deepcopy(envelope['config'])
    if kind == 'fm':
        if config.get('task') != 'rdd_fm_native' or config['process']['bounds'] != [0., .3, .6, 1.]:
            raise ValueError('This experiment requires the legacy FM v2 0/.3/.6/1 clock')
        config['process'].update(boundary='fresh', cfg_scale=cfg, solver='euler', steps=[80,80,90])
    elif config.get('task') != 'multires' or config['process']['name'] != 'pyramid_flow':
        raise ValueError('Expected the PF training checkpoint')
    state = envelope['ema']
    del envelope
    # Keep exact saved model shape and PE. No author weights are loaded here.
    model = build('model', config['model'])
    model.load_state_dict(state, strict=True)
    del state
    model = model.eval().requires_grad_(False).to(device)
    decoder = local_autoencoder(vae).eval().requires_grad_(False).to(device)
    method = make_method(config)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    vae_files = [vae/'config.json', vae/'diffusion_pytorch_model.safetensors']
    if not vae_files[1].exists():
        vae_files[1] = vae/'diffusion_pytorch_model.bin'
    contract = {'format': 'fm_pf_sampling_v1', 'kind': kind, 'checkpoint_sha256': sha256(checkpoint),
        'method': config, 'vae_sha256': {p.name:sha256(p) for p in vae_files},
        'latent_scale': .18215, 'cfg': cfg, 'seed': seed, 'batch_size': batch_size,
        'decode_batch': decode_batch, 'count': count, 'labels': 'global_image_index %1000',
        'rng': 'RandomStream(seed+batch_start); initial and boundary draws share the stream',
        'quantization': '((clamp(x,-1,1)+1)/2*255).uint8 (legacy evaluator truncation)', 'precision': 'float32_tf32_allowed',
        'device_type': device.type, 'environment': environment(),
        'gpu_name': torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        'torch_cuda': torch.version.cuda,
        'gpu_capability': list(torch.cuda.get_device_capability(device)) if device.type=='cuda' else None}
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (output/'.writer.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = output/'run.json'
        if manifest.exists() and read(manifest) != contract:
            raise ValueError('Sampling settings/assets/environment changed: choose a new output directory')
        write(manifest, contract)
        pngs = output/'png'
        markers = output/'batches'
        pngs.mkdir(exist_ok=True); markers.mkdir(exist_ok=True)
        began = time.monotonic()
        new_samples = 0
        with torch.no_grad():
            for start in range(0, count, batch_size):
                n = min(batch_size, count-start)
                marker = markers/f'{start:06d}.json'
                if validate_completed(marker, pngs, start, n):
                    continue
                labels = torch.arange(start, start+n, device=device, dtype=torch.long)%1000
                stream = RandomStream(seed+start, device)
                initial = stream.normal(torch.empty(n, *method.initial_shape(), device=device))
                before = time.monotonic()
                latents = method.sample(model, initial, stream=stream, model_kwargs={'y': labels})
                if not torch.isfinite(latents).all():
                    raise FloatingPointError('Non-finite sampled latents')
                hashes = {}
                for offset in range(0, n, decode_batch):
                    pixels = decoder.decode(latents[offset:offset+decode_batch]/.18215, return_dict=False)[0]
                    if not torch.isfinite(pixels).all():
                        raise FloatingPointError('Non-finite decoded pixels')
                    pixels = ((pixels.clamp(-1,1)+1)/2*255).to(torch.uint8).permute(0,2,3,1).cpu().numpy()
                    for index, pixel in enumerate(pixels):
                        path = pngs/f'{start+offset+index:06d}.png'
                        temporary = path.with_suffix('.tmp')
                        Image.fromarray(pixel).save(temporary, format='PNG')
                        temporary.replace(path)
                        hashes[path.name] = sha256(path)
                elapsed = time.monotonic()-before
                metrics = getattr(method, 'last_sample_metrics', None)
                if kind == 'fm':
                    if metrics['stage_nfe'] != [81,81,90]:
                        raise ValueError('Unexpected FM model-call allocation')
                    nfe = metrics['nfe']
                else:
                    nfe = sum(method.process.steps)
                record = {'start': start, 'labels': labels.cpu().tolist(), 'files': hashes,
                          'nfe': nfe, 'seconds_with_decode_io': elapsed}
                write(marker, record)
                new_samples += n
                print(json.dumps({'completed': start+n, 'total': count, 'nfe': nfe,
                                  'seconds': round(elapsed, 3)}), flush=True)
                del latents
        # A compact contact sheet is useful before launching formal training.
        preview_count = min(count, 64)
        tile = 128; columns = 8
        sheet = Image.new('RGB', (columns*tile, math.ceil(preview_count/columns)*(tile+18)), 'white')
        draw = ImageDraw.Draw(sheet)
        for index in range(preview_count):
            x, y = index%columns*tile, index//columns*(tile+18)
            with Image.open(pngs/f'{index:06d}.png') as image:
                sheet.paste(image.resize((tile,tile)), (x,y))
            draw.text((x+2,y+tile), f'id={index} label={index%1000}', fill='black')
        sheet.save(output/'preview.png')
        summary = {'complete': True, 'samples': count, 'new_samples_this_invocation': new_samples,
            'seconds_this_invocation': time.monotonic()-began, 'png_directory': str(pngs),
            'cfg': cfg, 'nfe': 252 if kind=='fm' else sum(method.process.steps),
            'FID': None, 'note': 'PNG generation completed. Formal FID/IS require separate ADM evaluation.'}
        write(output/'SUMMARY.json', summary)
        print(json.dumps(summary), flush=True)
        return summary
