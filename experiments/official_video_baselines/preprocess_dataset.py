"""Streaming, resumable native preprocessing of a local captioned JSONL dataset.

No dataset downloads or artificial video padding. GPU workers commit each clip;
CPU merge exports only committed clips and rejects incomplete formal datasets.
"""
from typing import Any
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import errno
import shutil


def link_or_copy(source: Any, destination: Any) -> Any:
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        shutil.copy2(source, destination)


def sha(path: Any) -> Any:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rows(path: Any) -> Any:
    result = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if not result or len({r['id'] for r in result}) != len(result):
        raise ValueError('Empty manifest or duplicate IDs')
    for r in result:
        if r['split'] not in ('train', 'validation') or not r['caption'].strip() or not Path(r['video']).is_file():
            raise ValueError('Every row requires a real video, caption and explicit train/validation split')
    return result


def encode(args: Any) -> Any:
    from launch_smoke import active_policy, bounded_deadline
    import datetime as dt
    active_policy()
    bounded_deadline(dt.datetime.now(dt.timezone.utc), 60)
    if not os.environ.get('RDD_JOB_TOKEN'):
        raise RuntimeError('Use the independent GPU guard')
    import torch
    from encode_small_real_data import PF_MODEL, WAN_MODEL
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    entries = rows(args.manifest)
    output = args.output / f'rank{rank}'
    output.mkdir(parents=True, exist_ok=args.resume)
    contract = {
        'schema': 1,
        'family': args.family,
        'input_sha256': sha(args.manifest),
        'frames': 81,
        'height': 448 if args.family == 'wan' else 384,
        'width': 832 if args.family == 'wan' else 640,
        'fps': 16,
        'seed': args.seed,
        'world_size': world,
        'rank': rank,
        'synthetic': False,
        'native_source': 'a012faa1dc4d71301a7a153c7f9554c081947ea2' if args.family == 'pf' else
        'diffusers Wan native classes, pinned checkpoint 0fad780a534b6463e45facd96134c9f345acfa5b',
        'posterior': 'sample' if args.family == 'pf' else 'mode'
    }
    contract['vae_weight_precision'] = 'original FP32 load' if args.family == 'wan' else 'native BF16 load'
    cp = output / 'encoding_contract.json'
    if cp.exists() and json.loads(cp.read_text()) != contract:
        raise ValueError('Resume encoding contract changed')
    cp.write_text(json.dumps(contract, indent=2))
    if args.family == 'pf':
        from pyramid_dit import FluxTextEncoderWithMask
        from video_vae import CausalVideoVAELossWrapper
        from tools.extract_video_vae_latents import load_video_and_transform
        encoder = FluxTextEncoderWithMask(str(PF_MODEL), torch_dtype=torch.bfloat16).eval().to(device)
        vae = CausalVideoVAELossWrapper(str(PF_MODEL / 'causal_video_vae'),
                                        model_dtype='bf16',
                                        interpolate=False,
                                        add_discriminator=False).eval().to(device)
        vae.vae.enable_tiling()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            emb, mask, pool = encoder([''], device)
        torch.save(dict(prompt_embed=emb.cpu(), prompt_attention_mask=mask.cpu(), pooled_prompt_embed=pool.cpu()),
                   output / 'empty_prompt.pt')
    else:
        from diffusers import WanPipeline, AutoencoderKLWan
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean
        from rdd.comparison.data import read_clip, tensor_fields, pyramid_schema
        from rdd.comparison.paths import spatial_resize
        from pyramidal_wan_paper_math import pixel_pyramid
        import pyarrow as pa
        import pyarrow.parquet as pq
        original_vae = AutoencoderKLWan.from_pretrained(str(WAN_MODEL),
                                                        subfolder='vae',
                                                        torch_dtype=torch.float32,
                                                        local_files_only=True)
        pipeline = WanPipeline.from_pretrained(str(WAN_MODEL),
                                               transformer=None,
                                               vae=original_vae,
                                               torch_dtype=torch.bfloat16,
                                               local_files_only=True).to(device)
        pipeline.vae.to(dtype=torch.float32)
        pipeline.vae.enable_tiling()
        assert all(p.dtype == torch.float32 for p in pipeline.vae.parameters())
        with torch.inference_mode():
            empty, _ = pipeline.encode_prompt('',
                                              device=device,
                                              do_classifier_free_guidance=False,
                                              max_sequence_length=512)
        torch.save(empty.cpu(), output / 'empty_prompt.pt')
    for index, item in enumerate(entries):
        if index % world != rank:
            continue
        folder = output / 'clips' / hashlib.sha256(item['id'].encode()).hexdigest()
        marker = folder / 'complete.json'
        identity = {**item, 'video_sha256': sha(item['video'])}
        if marker.exists():
            old = json.loads(marker.read_text())
            if old['identity'] != identity or any(sha(p) != h for p, h in old['file_sha256'].items()):
                raise ValueError('Committed clip changed: ' + item['id'])
            continue
        if time.time() + 90 >= float(os.environ['RDD_DEADLINE_UNIX']):
            raise RuntimeError('Stop preprocessing before another clip; completed clips can resume')
        folder.mkdir(parents=True, exist_ok=args.resume)
        torch.manual_seed(args.seed + int(hashlib.sha256(item['id'].encode()).hexdigest()[:8], 16))
        if args.family == 'pf':
            pixels = load_video_and_transform(item['video'],
                                              list(range(81)),
                                              81,
                                              new_width=640,
                                              new_height=384,
                                              resize=True)
            if pixels is None or pixels.shape[1] != 81:
                raise ValueError('Too short; no repeated-frame padding')
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                emb, mask, pool = encoder([item['caption']], device)
                latent = vae.encode_latent(pixels[None].to(device),
                                           sample=True,
                                           window_size=16,
                                           temporal_chunk=True,
                                           tile_sample_min_size=256)
            torch.save(latent.to(torch.bfloat16).cpu(), folder / 'latent.pt')
            torch.save(dict(prompt_embed=emb.cpu(), prompt_attention_mask=mask.cpu(), pooled_prompt_embed=pool.cpu()),
                       folder / 'text.pt')
            record = {
                'id': item['id'],
                'video': item['video'],
                'text': item['caption'],
                'latent': str(folder / 'latent.pt'),
                'text_fea': str(folder / 'text.pt'),
                'split': item['split']
            }
        else:
            pixels = read_clip(item['video'], 81, 16, 448, 832).to(device)
            with torch.inference_mode():
                text, _ = pipeline.encode_prompt(item['caption'],
                                                 device=device,
                                                 do_classifier_free_guidance=False,
                                                 max_sequence_length=512)
                temporal = [pipeline.vae.encode(rgb.float()).latent_dist.mode().cpu() for rgb in pixel_pyramid(pixels)]
                spatial = [temporal[0]] + [
                    pipeline.vae.encode(spatial_resize(pixels, (448 // factor, 832 // factor),
                                                       'bilinear').float()).latent_dist.mode().cpu()
                    for factor in (2, 4)
                ]
            cached = {
                'id': item['id'],
                'caption': item['caption'],
                'split': item['split'],
                'synthetic': False,
                'text_embedding': text.cpu(),
                'paper_raw_pixel_pyramid': temporal
            }
            torch.save(cached, folder / 'paper.pt')
            length = int(
                pipeline.tokenizer(prompt_clean(item['caption']),
                                   padding='max_length',
                                   max_length=512,
                                   truncation=True,
                                   add_special_tokens=True,
                                   return_tensors='pt').attention_mask.sum())
            row = {
                'id': item['id'],
                'caption': item['caption'],
                'file_name': item['video'],
                'media_type': 'video',
                'width': 832,
                'height': 448,
                'num_frames': 81,
                'duration_sec': 5.,
                'fps': 16.
            }
            row.update(tensor_fields('text_embedding', text[0, :length]))
            for name, value in zip(('vae_latent', 'rgb_latent_2', 'rgb_latent_4'), spatial, strict=False):
                row.update(tensor_fields(name, value[0]))
            pq.write_table(pa.Table.from_pylist([row], schema=pyramid_schema()),
                           folder / 'controlled.parquet',
                           compression='zstd',
                           row_group_size=1)
            record = {
                'id': item['id'],
                'split': item['split'],
                'sample': str(folder / 'paper.pt'),
                'paper_shapes': [list(x.shape) for x in temporal],
                'parquet': str(folder / 'controlled.parquet')
            }
        files = {str(p): sha(p) for p in folder.iterdir() if p.is_file() and p.suffix in ('.pt', '.parquet')}
        temporary = marker.with_suffix('.pending')
        temporary.write_text(json.dumps({'identity': identity, 'record': record, 'file_sha256': files}, indent=2))
        temporary.replace(marker)
        print(json.dumps({'completed': item['id'], 'rank': rank}), flush=True)


def merge(args: Any) -> Any:
    expected = rows(args.manifest)
    records = {}
    contracts = []
    identities = {r['id']: {**r, 'video_sha256': sha(r['video'])} for r in expected}
    for rank in sorted(args.input.glob('rank*')):
        c = json.loads((rank / 'encoding_contract.json').read_text())
        contracts.append(c)
        if c['input_sha256'] != sha(args.manifest) or c['family'] != args.family:
            raise ValueError('Source mismatch')
        precision = 'original FP32 load' if args.family == 'wan' else 'native BF16 load'
        if c.get('vae_weight_precision') != precision:
            raise ValueError('Unverified original VAE weight precision')
        if len(contracts) > 1 and {
                k: v
                for k, v in c.items() if k != 'rank'
        } != {
                k: v
                for k, v in contracts[0].items() if k != 'rank'
        }:
            raise ValueError('Worker encoding contracts disagree')
        for marker in rank.glob('clips/*/complete.json'):
            row = json.loads(marker.read_text())
            identifier = row['identity']['id']
            if identifier in records:
                raise ValueError('Overlapping worker IDs')
            if identifier not in identities or row['identity'] != identities[identifier]:
                raise ValueError('Caption, split or source video changed')
            if any(sha(p) != h for p, h in row['file_sha256'].items()):
                raise ValueError('Cache file changed')
            records[identifier] = row
    missing = [r['id'] for r in expected if r['id'] not in records]
    if missing and not args.allow_partial:
        raise ValueError('Unfinished videos; resume encode first: ' + str(missing[:8]))
    args.output.mkdir(parents=True, exist_ok=False)
    selected = [records[r['id']]['record'] for r in expected if r['id'] in records]
    if args.family == 'pf':
        for split in ('train', 'validation'):
            (args.output / (split + '.jsonl')).write_text(''.join(
                json.dumps(r) + '\n' for r in selected if r['split'] == split))
        link_or_copy(args.input / 'rank0/empty_prompt.pt', args.output / 'empty_prompt.pt')
    else:
        for index, row in enumerate(selected):
            folder = args.output / 'controlled_wan' / row['split']
            folder.mkdir(parents=True, exist_ok=True)
            link_or_copy(row['parquet'], folder / f'shard_{index:08}.parquet')
        (args.output / 'paper_manifest.json').write_text(
            json.dumps(
                {
                    'synthetic': False,
                    'posterior': 'mode',
                    'dataset': 'user-provided captioned local manifest',
                    'source_manifest_sha256': sha(args.manifest),
                    'paper_dataset_difference': 'Not the authors private80K synthetic Wan14B corpus',
                    'rows': selected
                },
                indent=2))
    (args.output / 'cache_audit.json').write_text(
        json.dumps(
            {
                'status': 'passed',
                'family': args.family,
                'complete_manifest_processed': not missing,
                'input_sha256': sha(args.manifest),
                'counts': {
                    s: sum(r['split'] == s for r in selected)
                    for s in ('train', 'validation')
                },
                'missing': missing,
                'contracts': contracts,
                'committed_clips': list(records.values())
            },
            indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['encode', 'merge'])
    p.add_argument('--family', choices=['pf', 'wan'], required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--input', type=Path)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--allow-partial', action='store_true')
    p.add_argument('--seed', type=int, default=20261006)
    args = p.parse_args()
    encode(args) if args.command == 'encode' else merge(args)
