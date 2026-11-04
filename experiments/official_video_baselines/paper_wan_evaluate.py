"""Full paper Wan weights, actual conditional sampler and native Wan VAE export."""
import argparse
import json
import os
from pathlib import Path
import time
import torch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', default='base')
    p.add_argument('--shape', choices=['small', 'paper'], default='paper')
    p.add_argument('--counts', default='20,20,10')
    p.add_argument('--solver', choices=['euler', 'ab2', 'heun'], default='euler')
    p.add_argument('--seeds', default='42')
    p.add_argument('--max-prompts', type=int, default=2)
    p.add_argument('--prompt-start', type=int, default=0)
    p.add_argument('--quality-only', action='store_true')
    p.add_argument('--negative-cache', type=Path, required=True)
    p.add_argument('--prompts', type=Path, help='Actual shared Wan UMT5 official-prompt parquet cache')
    p.add_argument('--height', type=int, default=448)
    p.add_argument('--width', type=int, default=832)
    p.add_argument('--warmup', type=int, default=0)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--cfg', type=float, default=5.)
    args = p.parse_args()
    if args.quality_only:
        args.warmup, args.repeats = 0, 1
    if args.prompt_start < 0:
        p.error('Nonnegative prompt offset required')
    counts = tuple(map(int, args.counts.split(',')))
    if len(counts) != 3 or min(counts) < 1 or args.warmup < 0 or args.repeats < 1 or args.max_prompts < 1:
        p.error('Three positive stage counts, positive repeats/prompts, nonnegative warmup required')
    if args.height % 64 or args.width % 64:
        p.error('Paper spatial pyramid requires resolution divisible by64')
    if not os.environ.get('RDD_JOB_TOKEN') or time.time() >= float(os.environ.get('RDD_DEADLINE_UNIX', '0')):
        raise RuntimeError('Independent GPU guard required')
    if args.output.exists():
        raise FileExistsError(args.output)
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
    from fastvideo.train.utils.config import load_run_config
    from fastvideo.train.utils.instantiate import instantiate
    from diffusers import AutoencoderKLWan
    from diffusers.utils import export_to_video
    from pyramidal_wan_sampling import sample
    from full_model_trace import hashed
    cfg = load_run_config(args.config)
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    model = instantiate(cfg.models['student'], training_config=cfg.training)
    model.transformer.eval().requires_grad_(False)
    model.transformer.gradient_checkpointing = False
    if args.checkpoint != 'base':
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict, StateDictOptions
        state = {'roles.student.transformer': get_model_state_dict(model.transformer)}
        reader = dcp.FileSystemReader(str(Path(args.checkpoint) / 'dcp'))
        prefix = 'roles.student.transformer.'
        stored = {k[len(prefix):] for k in reader.read_metadata().state_dict_metadata if k.startswith(prefix)}
        if stored != set(state['roles.student.transformer']):
            raise ValueError('Incomplete paper model checkpoint')
        dcp.load(state, storage_reader=reader)
        set_model_state_dict(model.transformer,
                             state['roles.student.transformer'],
                             options=StateDictOptions(strict=True))
    vae = AutoencoderKLWan.from_pretrained(model.init_from,
                                           subfolder='vae',
                                           torch_dtype=torch.float32,
                                           local_files_only=True).to(model.device).eval().requires_grad_(False)
    manifest = json.loads(Path(model.manifest).read_text())
    rows = [r for r in manifest['rows'] if r['split'] == 'validation']
    if args.prompts:
        import pyarrow.parquet as pq
        import numpy as np
        rows = []
        for file in sorted(args.prompts.glob('*.parquet')):
            for group in pq.ParquetFile(file).iter_batches(batch_size=1):
                row = group.to_pylist()[0]
                value = np.frombuffer(row['text_embedding_bytes'],
                                      dtype=row['text_embedding_dtype']).reshape(row['text_embedding_shape']).copy()
                context = torch.zeros(1, 512, 4096, dtype=torch.bfloat16)
                context[0, :value.shape[0]] = torch.from_numpy(value).to(torch.bfloat16)
                rows.append({'id': row['id'], 'caption': row['caption'], 'text_embedding': context})
        if not rows:
            raise ValueError('No actual prompt embeddings')
    negative = torch.load(args.negative_cache, map_location='cpu', weights_only=True).to(model.device,
                                                                                         dtype=torch.bfloat16)
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for n, row in enumerate(rows[args.prompt_start:args.prompt_start + args.max_prompts], start=args.prompt_start):
        cached = row if args.prompts else torch.load(row['sample'], map_location='cpu', weights_only=True)
        shape = (1, 16, 21, args.height // 8,
                 args.width // 8) if args.prompts else cached[args.shape + '_raw_pixel_pyramid'][0].shape
        text = cached['text_embedding'].to(model.device, dtype=torch.bfloat16)
        for seed in map(int, args.seeds.split(',')):
            timings = []
            for repetition in range(-args.warmup, args.repeats):
                torch.cuda.synchronize()
                start = time.perf_counter()
                latent, accounting = sample(model.transformer,
                                            shape,
                                            text,
                                            negative,
                                            counts=counts,
                                            seed=seed,
                                            solver=args.solver,
                                            cfg=args.cfg)
                torch.cuda.synchronize()
                denoise = time.perf_counter() - start
                start = time.perf_counter()
                with torch.no_grad():
                    pixels = (vae.decode(latent / (1.0 / model.std) + model.mean, return_dict=False)[0] / 2 + .5).clamp(
                        0, 1)
                torch.cuda.synchronize()
                decode = time.perf_counter() - start
                if repetition >= 0:
                    timings.append({'denoise': denoise, 'vae': decode, 'total': denoise + decode})
            if not torch.isfinite(pixels).all():
                raise FloatingPointError('Nonfinite actual native VAE decode')
            name = f'prompt_{n:04d}_seed_{seed}'
            frames = pixels[0].permute(1, 2, 3, 0).float().cpu().numpy()
            export_to_video(list(frames), str(args.output / (name + '.mp4')), fps=16)
            results.append({
                'name': name,
                'id': row['id'],
                'caption': cached['caption'],
                'seed': seed,
                'shape': list(latent.shape),
                'latent': hashed(latent),
                'pixels': hashed((pixels * 255).round().to(torch.uint8)),
                'frames': len(frames),
                'resolution': list(pixels.shape[-2:]),
                **accounting, 'denoise_seconds': denoise,
                'vae_seconds': decode
            })
            import statistics
            results[-1].update(timings=timings,
                               denoise_median=statistics.median(t['denoise'] for t in timings),
                               total_median=statistics.median(t['total'] for t in timings))
            (args.output / 'metrics.json').write_text(
                json.dumps(
                    {
                        'status': 'passed',
                        'method': 'paper_pyramidal_wan',
                        'latency_benchmark': not args.quality_only,
                        'prompt_start': args.prompt_start,
                        'checkpoint': args.checkpoint,
                        'author_source_parity': False,
                        'quality_benchmark': False,
                        'synthetic': False,
                        'cfg': args.cfg,
                        'fps': 16,
                        'warmup': args.warmup,
                        'repeats': args.repeats,
                        'gpu': torch.cuda.get_device_name(),
                        'gpu_uuid': str(getattr(torch.cuda.get_device_properties(), 'uuid', 'unavailable')),
                        'torch': torch.__version__,
                        'timing_scope':
                        'cached_text_gpu_denoise_plus_vae_decode; excludes text encoding, model loading, transfer and MP4 writing',
                        'solver': args.solver,
                        'stage_counts': counts,
                        'scope':
                        'strict paper implementation; 3D temporal+spatial stages; native HF Wan BF16 autocast with FP32 masters + original FP32 Wan VAE',
                        'samples': results
                    },
                    indent=2))
    if not results:
        raise ValueError('Prompt offset selected no rows')
    (args.output / 'complete.json').write_text(json.dumps({'completed': True, 'samples': len(results)}))
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
