"""Original Wan pipeline/UniPC sampler with native modules through FastVideo API."""
from typing import Any
import argparse
import json
import os
from pathlib import Path
import statistics
import time


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--prompts', type=Path, required=True)
    p.add_argument('--negative-cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', default='base')
    p.add_argument('--height', type=int, default=448)
    p.add_argument('--width', type=int, default=832)
    p.add_argument('--max-prompts', type=int, default=2)
    p.add_argument('--prompt-start', type=int, default=0)
    p.add_argument('--quality-only', action='store_true')
    p.add_argument('--seeds', default='42')
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--repeats', type=int, default=10)
    p.add_argument('--steps', type=int, default=50)
    args = p.parse_args()
    if args.quality_only:
        args.warmup, args.repeats = 0, 1
    if args.prompt_start < 0 or args.max_prompts < 1 or args.repeats < 1 or (args.warmup < 1 and not args.quality_only):
        p.error('Positive prompt/benchmark limits and nonnegative offset required')
    if args.checkpoint != 'base':
        p.error('Native pretrained baseline uses the original base checkpoint')
    if args.output.exists():
        raise FileExistsError(args.output)
    if not os.environ.get('RDD_JOB_TOKEN') or time.time() >= float(os.environ.get('RDD_DEADLINE_UNIX', '0')):
        raise RuntimeError('Independent GPU guard required')
    import torch
    import numpy as np
    import pyarrow.parquet as pq
    from diffusers import WanPipeline
    from diffusers.utils import export_to_video
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
    from fastvideo.train.utils.config import load_run_config
    from fastvideo.train.utils.instantiate import instantiate
    from full_model_trace import hashed
    from types import SimpleNamespace
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    cfg = load_run_config(args.config)
    model = instantiate(dict(cfg.models['student'], trainable=False), training_config=cfg.training)
    model.transformer.eval().requires_grad_(False)
    counts = [0]

    class Interface(torch.nn.Module):

        def __init__(self: Any) -> None:
            super().__init__()
            self.original = model.transformer
            self.config = self.original.config

        @property
        def dtype(self: Any) -> Any:
            return next(self.original.parameters()).dtype

        @property
        def device(self: Any) -> Any:
            return next(self.original.parameters()).device

        def cache_context(self: Any, name: Any) -> Any:
            return self.original.cache_context(name)

        def forward(self: Any,
                    hidden_states: Any,
                    timestep: Any,
                    encoder_hidden_states: Any,
                    return_dict: Any = False,
                    **kwargs: Any) -> Any:
            counts[0] += 1
            batch = SimpleNamespace(conditional_dict={'encoder_hidden_states': encoder_hidden_states})
            return (model.predict_noise(hidden_states.permute(0, 2, 1, 3, 4), timestep, batch,
                                        conditional=True).permute(0, 2, 1, 3, 4), )

    model.vae = model.load_original_vae(cfg.models['student']['init_from'])
    pipeline = WanPipeline.from_pretrained(cfg.models['student']['init_from'],
                                           transformer=model.transformer,
                                           vae=model.vae,
                                           text_encoder=None,
                                           tokenizer=None,
                                           torch_dtype=torch.bfloat16,
                                           local_files_only=True).to('cuda')
    pipeline.transformer = Interface()
    negative = torch.load(args.negative_cache, weights_only=True).cuda().to(torch.bfloat16)
    args.output.mkdir(parents=True)
    result = dict(
        method='native_wan',
        latency_benchmark=not args.quality_only,
        prompt_start=args.prompt_start,
        native_scheduler=dict(pipeline.scheduler.config),
        cfg=5.,
        fps=16,
        warmup=args.warmup,
        repeats=args.repeats,
        gpu=torch.cuda.get_device_name(),
        gpu_uuid=str(getattr(torch.cuda.get_device_properties(), 'uuid', 'unavailable')),
        torch=torch.__version__,
        timing_scope=
        'cached_text_gpu_denoise_plus_vae_decode; excludes text encoding, model loading, transfer and MP4 writing',
        scope=
        'native original Wan UniPC flow_shift3 + original BF16 Wan modules through FastVideo API + original FP32 VAE',
        checkpoint='base',
        synthetic=False,
        quality_benchmark=False,
        samples=[])
    n = 0
    for file in sorted(args.prompts.glob('*.parquet')):
        for batch in pq.ParquetFile(file).iter_batches(batch_size=1):
            if n >= args.prompt_start + args.max_prompts:
                break
            if n < args.prompt_start:
                n += 1
                continue
            row = batch.to_pylist()[0]
            value = np.frombuffer(row['text_embedding_bytes'],
                                  dtype=row['text_embedding_dtype']).reshape(row['text_embedding_shape']).copy()
            text = torch.zeros(1, 512, 4096, dtype=torch.bfloat16, device='cuda')
            text[0, :len(value)] = torch.from_numpy(value).cuda().to(torch.bfloat16)
            for seed in map(int, args.seeds.split(',')):
                timings = []
                for repeat in range(-args.warmup, args.repeats):
                    before = counts[0]
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    with torch.no_grad():
                        latent = pipeline(prompt_embeds=text,
                                          negative_prompt_embeds=negative,
                                          height=args.height,
                                          width=args.width,
                                          num_frames=81,
                                          num_inference_steps=args.steps,
                                          guidance_scale=5.,
                                          output_type='latent',
                                          generator=torch.Generator(device='cuda').manual_seed(seed)).frames
                    torch.cuda.synchronize()
                    denoise = time.perf_counter() - start
                    start = time.perf_counter()
                    with torch.no_grad():
                        pixels = model.decode_latents(latent.permute(0, 2, 1, 3, 4))
                    torch.cuda.synchronize()
                    vae = time.perf_counter() - start
                    calls = counts[0] - before
                    if calls != 2 * args.steps or not torch.isfinite(pixels).all():
                        raise AssertionError('Native count or decode invalid')
                    if repeat >= 0:
                        timings.append(dict(denoise=denoise, vae=vae, total=denoise + vae))
                name = f'prompt_{n:04d}_seed_{seed}'
                frames = pixels[0].permute(1, 2, 3, 0).float().cpu().numpy()
                export_to_video(list(frames), str(args.output / (name + '.mp4')), fps=16)
                result['samples'].append(
                    dict(name=name,
                         id=row['id'],
                         caption=row['caption'],
                         seed=seed,
                         shape=list(latent.shape),
                         frames=81,
                         resolution=[args.height, args.width],
                         actual_nfe=args.steps,
                         branch_forward_calls=calls,
                         latent=hashed(latent),
                         pixels=hashed((pixels * 255).round().to(torch.uint8)),
                         timings=timings,
                         denoise_median=statistics.median(t['denoise'] for t in timings),
                         total_median=statistics.median(t['total'] for t in timings)))
                (args.output / 'metrics.json').write_text(json.dumps(result, indent=2))
            n += 1
        if n >= args.prompt_start + args.max_prompts:
            break
    if not result['samples']:
        raise ValueError('Prompt offset selected no rows')
    if n == 0:
        raise ValueError('No actual prompt cache')
    (args.output / 'complete.json').write_text(json.dumps(dict(completed=True, prompts=n)))
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
