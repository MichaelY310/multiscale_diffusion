"""Guarded native PF or Wan encoding of a bounded, captioned real fixture."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import torch

WORK = Path(__file__).resolve().parents[3]
PF_MODEL = WORK / 'assets_20261006/pyramid-flow-miniflux'
WAN_MODEL = Path('/scr/yuheng/bugswarm/FastVideo/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b')
RAW = WORK / 'assets_20261006/small_mixkit'


def items():
    manifest = json.loads((RAW / 'download_manifest.json').read_text())
    entries = []
    for n, row in enumerate(manifest['selected_videos']):
        if not row.get('caption', '').strip():
            raise ValueError('Every real video requires its original caption')
        entries.append({**row, 'video': str(RAW / row['path']),
            'id': row['path'].removeprefix('videos/'),
            'split': 'validation' if n >= len(manifest['selected_videos']) - 2 else 'train'})
    return entries


def guarded():
    if not os.environ.get('RDD_JOB_TOKEN') or float(os.environ.get('RDD_DEADLINE_UNIX', '0')) <= time.time():
        raise RuntimeError('GPU preprocessing requires the independent deadline guard')
    from launch_smoke import bounded_deadline
    import datetime as dt
    bounded_deadline(dt.datetime.now(dt.timezone.utc), 60)
    torch.cuda.set_device(0)
    torch.manual_seed(20261006)


def save(path, value):
    if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
        raise FloatingPointError('Nonfinite cache rejected')
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)


def encode_pf(args):
    from pyramid_dit import FluxTextEncoderWithMask
    from video_vae import CausalVideoVAELossWrapper
    from tools.extract_video_vae_latents import load_video_and_transform
    entries = items()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    model = FluxTextEncoderWithMask(str(PF_MODEL), torch_dtype=torch.bfloat16).eval().cuda()
    for n, caption in enumerate([''] + [x['caption'] for x in entries]):
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            embedded, mask, pooled = model([caption], torch.device('cuda'))
        value = {'prompt_embed': embedded.cpu(), 'prompt_attention_mask': mask.cpu(),
                 'pooled_prompt_embed': pooled.cpu()}
        path = out / ('empty_prompt.pt' if n == 0 else f'text/{n - 1:03}.pt')
        save(path, value)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    vae = CausalVideoVAELossWrapper(str(PF_MODEL / 'causal_video_vae'),
        model_dtype='bf16', interpolate=False, add_discriminator=False).eval().cuda()
    vae.vae.enable_tiling()
    annotations = {'train': [], 'validation': []}
    for n, row in enumerate(entries):
        # Exact official loader, transform, posterior sample and chunk options.
        pixels = load_video_and_transform(row['video'], list(range(81)), 81,
            new_width=640, new_height=384, resize=True)
        if pixels is None or pixels.shape[1] != 81:
            raise ValueError(f"Real clip lacks 81 frames: {row['id']}")
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            latent = vae.encode_latent(pixels[None].cuda(), sample=True,
                window_size=16, temporal_chunk=True, tile_sample_min_size=256)
        latent = latent.to(torch.bfloat16).cpu()
        path = out / f'latent/{n:03}.pt'
        save(path, latent)
        annotations[row['split']].append({'video': row['video'], 'text': row['caption'],
            'latent': str(path), 'text_fea': str(out / f'text/{n:03}.pt'), 'id': row['id']})
        print('PF encoded', n, list(latent.shape), flush=True)
    for split, rows in annotations.items():
        (out / f'{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (out / 'encoder_provenance.json').write_text(json.dumps({
        'synthetic': False, 'dataset': 'FastVideo/mixkit_filtered_6k_wan1.3_t2v',
        'dataset_revision': '2232687a98e56bd590919f18949333615aa14dcd',
        'model_revision': 'c0ea661ae8fc64ddf57e2caa4717daea085dca91',
        'source_commit': 'a012faa1dc4d71301a7a153c7f9554c081947ea2',
        'posterior': 'official sample=True, raw latents, native runner normalizes first/tail separately',
        'shape': [81, 384, 640], 'actual_original_caption': True,
        'train_ids': [x['id'] for x in annotations['train']],
        'validation_ids': [x['id'] for x in annotations['validation']]}, indent=2))


def encode_wan(args):
    from diffusers import WanPipeline, AutoencoderKLWan
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from rdd.comparison.data import read_clip, tensor_fields, pyramid_schema
    from rdd.comparison.paths import spatial_resize
    from pyramidal_wan_paper_math import pixel_pyramid
    import pyarrow as pa
    import pyarrow.parquet as pq
    entries = items()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    vae=AutoencoderKLWan.from_pretrained(str(WAN_MODEL),subfolder='vae',
        torch_dtype=torch.float32,local_files_only=True)
    pipe = WanPipeline.from_pretrained(str(WAN_MODEL), transformer=None, vae=vae,
        torch_dtype=torch.bfloat16, local_files_only=True).to('cuda')
    assert all(p.dtype==torch.float32 for p in pipe.vae.parameters())
    pipe.vae.to(dtype=torch.float32)
    pipe.vae.enable_tiling()
    prompts = [x['caption'] for x in entries] + ['',
        'Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards']
    contexts = []
    for prompt in prompts:
        with torch.inference_mode():
            text, _ = pipe.encode_prompt(prompt=prompt, negative_prompt='', device='cuda',
                do_classifier_free_guidance=False, max_sequence_length=512)
        encoded = pipe.tokenizer(prompt_clean(prompt), padding='max_length', max_length=512,
            truncation=True, add_special_tokens=True, return_tensors='pt')
        length = int(encoded.attention_mask.sum())
        contexts.append((text.cpu(), length))
    save(out / 'empty_prompt.pt', contexts[-2][0])
    save(out / 'negative_prompt.pt', contexts[-1][0])
    del pipe.text_encoder
    pipe.text_encoder = None
    gc.collect()
    torch.cuda.empty_cache()
    audit_rows = []
    for n, row in enumerate(entries):
        cached = {'id': row['id'], 'caption': row['caption'], 'split': row['split'],
                  'text_embedding': contexts[n][0], 'synthetic': False}
        # Full published dimensions, plus a shorter/lower-resolution smoke.
        for label, frames, height, width in [('paper', 81, 448, 832), ('small', 33, 128, 256)]:
            pixels = read_clip(row['video'], frames, 16, height, width).cuda()
            pyramid = pixel_pyramid(pixels)
            raw_latents = []
            with torch.inference_mode():
                for rgb in pyramid:
                    raw_latents.append(pipe.vae.encode(rgb.float()).latent_dist.mode().cpu())
            cached[f'{label}_raw_pixel_pyramid'] = raw_latents
            if label == 'small':
                # Controlled spatial-only baseline remains clearly separate.
                spatial = [raw_latents[0]]
                with torch.inference_mode():
                    for factor in (2, 4):
                        rgb = spatial_resize(pixels, (height // factor, width // factor), 'bilinear')
                        spatial.append(pipe.vae.encode(rgb.float()).latent_dist.mode().cpu())
                base = {'id': row['id'], 'caption': row['caption'], 'file_name': row['video'],
                    'media_type': 'video', 'width': width, 'height': height, 'num_frames': frames,
                    'duration_sec': (frames - 1) / 16, 'fps': 16}
                base.update(tensor_fields('text_embedding', contexts[n][0][0, :contexts[n][1]]))
                for name, latent in zip(('vae_latent', 'rgb_latent_2', 'rgb_latent_4'), spatial):
                    base.update(tensor_fields(name, latent[0]))
                folder = out / 'controlled_wan' / row['split']
                folder.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.Table.from_pylist([base], schema=pyramid_schema()),
                    folder / f'shard_{n:03}.parquet', compression='zstd', row_group_size=1)
        save(out / f'paper_samples/{n:03}.pt', cached)
        audit_rows.append({'id': row['id'], 'split': row['split'],
            'sample': str(out / f'paper_samples/{n:03}.pt'),
            'paper_shapes': [list(x.shape) for x in cached['paper_raw_pixel_pyramid']],
            'small_shapes': [list(x.shape) for x in cached['small_raw_pixel_pyramid']]})
        print('Wan encoded', n, audit_rows[-1]['paper_shapes'], flush=True)
    manifest = {'synthetic': False, 'dataset': 'FastVideo/mixkit_filtered_6k_wan1.3_t2v',
        'dataset_revision': '2232687a98e56bd590919f18949333615aa14dcd',
        'model_revision': '0fad780a534b6463e45facd96134c9f345acfa5b',
        'posterior': 'mode; unpublished paper posterior choice explicitly recorded',
        'vae_load_dtype':'FP32 directly from the original checkpoint; no BF16 load/upcast',
        'rgb_resize': 'first frame spatial and tail spatiotemporal trilinear align_corners=False',
        'paper_dataset_difference': 'real Mixkit integration fixture; paper used 80K Wan2.1-14B synthetic videos',
        'rows': audit_rows}
    (out / 'paper_manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--family', choices=['pf', 'wan'], required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    guarded()
    encode_pf(args) if args.family == 'pf' else encode_wan(args)
