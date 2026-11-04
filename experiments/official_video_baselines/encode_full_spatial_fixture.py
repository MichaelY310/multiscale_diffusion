"""Add original-VAE, spatial-only full-size controlled caches to real fixtures.

Reuse audited full fine latents and original UMT5 contexts. Encode the two
RGB-resized 81-frame videos independently; temporal-paper caches are different.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source',type=Path)
    args = p.parse_args()
    from encode_small_real_data import guarded, items, WAN_MODEL, WORK
    guarded()
    import torch
    import pyarrow as pa
    import pyarrow.parquet as pq
    from diffusers import AutoencoderKLWan
    from rdd.comparison.data import read_clip, tensor_fields, pyramid_schema
    from rdd.comparison.paths import spatial_resize
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.source or WORK/'fair_data_20261006/wan_v2'
    vae = AutoencoderKLWan.from_pretrained(str(WAN_MODEL), subfolder='vae',
        torch_dtype=torch.float32, local_files_only=True).cuda().eval()
    vae.enable_tiling()
    records = []
    for n, row in enumerate(items()):
        import os
        if time.time() + 90 >= float(os.environ['RDD_DEADLINE_UNIX']):
            raise RuntimeError('Insufficient time to finish another real-video cache')
        old = pq.read_table(source/'controlled_wan'/row['split']/f'shard_{n:03}.parquet').to_pylist()[0]
        cached = torch.load(source/f'paper_samples/{n:03}.pt', map_location='cpu', weights_only=True)
        pixels = read_clip(row['video'], 81, 16, 448, 832).cuda()
        values = [cached['paper_raw_pixel_pyramid'][0]]
        with torch.inference_mode():
            for factor in (2, 4):
                rgb = spatial_resize(pixels, (448//factor, 832//factor), 'bilinear')
                values.append(vae.encode(rgb.float()).latent_dist.mode().cpu())
        outrow = {k:v for k,v in old.items() if k.startswith('text_embedding')}
        outrow.update(id=row['id'], caption=row['caption'], file_name=row['video'], media_type='video',
            width=832, height=448, num_frames=81, duration_sec=5.0, fps=16.0)
        for name, value in zip(('vae_latent','rgb_latent_2','rgb_latent_4'), values):
            outrow.update(tensor_fields(name, value[0]))
        folder = args.output/row['split']; folder.mkdir(exist_ok=True)
        path = folder/f'shard_{n:03}.parquet'
        pq.write_table(pa.Table.from_pylist([outrow], schema=pyramid_schema()), path,
            compression='zstd', row_group_size=1)
        records.append({'id':row['id'], 'split':row['split'], 'path':str(path),
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'shapes':[list(v.shape) for v in values], 'synthetic':False})
        print(json.dumps(records[-1]), flush=True)
    (args.output/'cache_audit.json').write_text(json.dumps({'status':'passed',
        'source_manifest_sha256':hashlib.sha256((source/'paper_manifest.json').read_bytes()).hexdigest(),
        'model_revision':WAN_MODEL.name, 'frames':81, 'height':448, 'width':832,
        'posterior':'mode', 'rgb_pyramid':'spatial bilinear before original VAE; no temporal resize',
        'fine_latents_reused_bitwise':True, 'rows':records}, indent=2))


if __name__ == '__main__':
    main()
