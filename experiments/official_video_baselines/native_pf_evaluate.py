"""Original Pyramid Flow prompt encoding, AR sampling and native VAE evaluation.

The original native sampler is Euler and autoregressive. At 81 decoded frames
its three positive stage counts require >=33 guided evaluations; the matched
50-call recipe is first unit4/4/2 plus ten later units1/2/1. No Wan transplant.
"""
from typing import Any
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

from native_pf_decoded_check import CachedNativeEncoder, MODEL, SOURCE, AUGMENT


def encode(args: Any) -> Any:
    import torch
    from pyramid_dit import FluxTextEncoderWithMask
    entries = json.loads(args.full_info.read_text())
    prompts = list(dict.fromkeys(r['prompt_en'] for r in entries))[:args.max_prompts]
    args.output.mkdir(parents=True, exist_ok=False)
    encoder = FluxTextEncoderWithMask(str(MODEL), torch_dtype=torch.bfloat16).eval().cuda()
    files = {}
    rows = []
    for n, prompt in enumerate([''] + [p + AUGMENT for p in prompts]):
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            context, mask, pooled = encoder([prompt], torch.device('cuda'))
        file = f'prompt_{n:05}.pt'
        torch.save(dict(prompt_embed=context.cpu(), prompt_attention_mask=mask.cpu(), pooled_prompt_embed=pooled.cpu()),
                   args.output / file)
        files[prompt] = file
    for prompt in prompts:
        rows.append(dict(id='vbench-' + hashlib.sha256(prompt.encode()).hexdigest(), text=prompt))
    (args.output / 'manifest.json').write_text(
        json.dumps(
            {
                'prompts': files,
                'rows': rows,
                'training_data': False,
                'synthetic': False,
                'native_prompt_augmentation': AUGMENT,
                'negative_prompt': '',
                'source_commit': 'a012faa1dc4d71301a7a153c7f9554c081947ea2',
                'full_info_sha256': hashlib.sha256(args.full_info.read_bytes()).hexdigest()
            },
            indent=2))


def evaluate(args: Any) -> Any:
    import torch
    from pyramid_dit import PyramidDiTForVideoGeneration
    from video_vae import CausalVideoVAE
    from pyramid_compat import sample_psd_block_noise
    from full_model_trace import hashed
    if args.native_reference:
        if args.checkpoint != 'base':
            raise ValueError('Reference profile takes original pretrained base weights')
        runner = PyramidDiTForVideoGeneration(str(MODEL),
                                              model_dtype='bf16',
                                              model_name='pyramid_flux',
                                              model_variant='diffusion_transformer_384p',
                                              load_text_encoder=False,
                                              load_vae=False,
                                              use_flash_attn=False,
                                              max_temporal_length=11,
                                              frame_per_unit=1,
                                              video_sync_group=1)
    else:
        from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
        from fastvideo.train.utils.config import load_run_config
        from fastvideo.train.utils.instantiate import instantiate
        maybe_init_distributed_environment_and_model_parallel(1, 1)
        cfg = load_run_config(args.config)
        # The original training plugin deliberately requires all DiT parameters
        # trainable at construction. Freeze immediately after strict loading;
        # evaluation never builds an optimizer or training Method.
        model_config = dict(cfg.models['student'], trainable=True)
        model_config['recipe'] = dict(model_config['recipe'], max_temporal_length=11)
        plugin = instantiate(model_config, training_config=cfg.training)
        runner = plugin.runner
        if args.checkpoint != 'base':
            import torch.distributed.checkpoint as dcp
            from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict, StateDictOptions
            state = {'roles.student.transformer': get_model_state_dict(runner.dit)}
            reader = dcp.FileSystemReader(str(Path(args.checkpoint) / 'dcp'))
            prefix = 'roles.student.transformer.'
            stored = {k[len(prefix):] for k in reader.read_metadata().state_dict_metadata if k.startswith(prefix)}
            if stored != set(state['roles.student.transformer']):
                raise ValueError('Incomplete PF student checkpoint')
            dcp.load(state, storage_reader=reader)
            set_model_state_dict(runner.dit, state['roles.student.transformer'], options=StateDictOptions(strict=True))
    runner.dit.to('cuda', dtype=torch.bfloat16).eval().requires_grad_(False)
    runner.vae = CausalVideoVAE.from_pretrained(str(MODEL / 'causal_video_vae'),
                                                torch_dtype=torch.bfloat16,
                                                interpolate=False).to('cuda').eval().requires_grad_(False)
    runner.vae.enable_tiling()
    runner.text_encoder = CachedNativeEncoder(args.prompts)
    runner.sample_block_noise = lambda *pos: sample_psd_block_noise(runner.scheduler.config.gamma, *pos)
    calls = []
    hook = runner.dit.register_forward_pre_hook(lambda m, a, k: calls.append(k['sample'][0][-1].shape[0]),
                                                with_kwargs=True)
    rows = json.loads((args.prompts / 'manifest.json').read_text())['rows']
    args.output.mkdir(parents=True, exist_ok=False)
    results = dict(
        method='native_pyramid_flow',
        latency_benchmark=not args.quality_only,
        prompt_start=args.prompt_start,
        checkpoint=args.checkpoint,
        source_commit='a012faa1dc4d71301a7a153c7f9554c081947ea2',
        cfg=args.cfg,
        fps=16,
        warmup=args.warmup,
        repeats=args.repeats,
        gpu=torch.cuda.get_device_name(),
        gpu_uuid=str(getattr(torch.cuda.get_device_properties(), 'uuid', 'unavailable')),
        torch=torch.__version__,
        timing_scope=
        'cached_text_gpu_denoise_plus_vae_decode; excludes text encoding, model loading, transfer and MP4 writing',
        solver='original_euler',
        nfe_budget=50,
        scope='native original PF Flux/T5/CLIP/CausalVideoVAE AR profile; BF16; no offload',
        native_positive_suffix=AUGMENT,
        negative_prompt='',
        native_reference=args.native_reference,
        psd_compatibility=
        'exact Gaussian covariance at singular gamma=1/3; different RNG mapping from unavailable Cholesky',
        quality_benchmark=False,
        samples=[])
    for n, row in enumerate(rows[args.prompt_start:args.prompt_start + args.max_prompts], start=args.prompt_start):
        for seed in map(int, args.seeds.split(',')):
            timings = []
            for repeat in range(-args.warmup, args.repeats):
                torch.manual_seed(seed)
                generator = torch.Generator(device='cuda').manual_seed(seed)
                before = len(calls)
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    latent = runner.generate(prompt=row['text'],
                                             negative_prompt='',
                                             height=384,
                                             width=640,
                                             temp=11,
                                             num_inference_steps=[4, 4, 2],
                                             video_num_inference_steps=[1, 2, 1],
                                             guidance_scale=args.cfg,
                                             video_guidance_scale=args.cfg,
                                             output_type='latent',
                                             generator=generator)
                torch.cuda.synchronize()
                denoise = time.perf_counter() - start
                start = time.perf_counter()
                if not torch.isfinite(latent).all():
                    raise FloatingPointError('Nonfinite native PF latent')
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    images = runner.decode_latent(latent.clone(), save_memory=True)
                torch.cuda.synchronize()
                vae = time.perf_counter() - start
                used = len(calls) - before
                branches = sum(calls[before:])
                if used != 50 or branches != 100:
                    raise AssertionError((used, branches))
                if repeat >= 0:
                    timings.append(dict(denoise=denoise, vae=vae, total=denoise + vae))
            import numpy as np
            from diffusers.utils import export_to_video
            pixels = torch.from_numpy(np.stack([np.asarray(x) for x in images]))
            if len(images) != 81 or tuple(pixels.shape[1:3]) != (384, 640):
                raise AssertionError('Native decoded shape changed')
            name = f'prompt_{n:04d}_seed_{seed}'
            export_to_video(images, str(args.output / (name + '.mp4')), fps=16)
            results['samples'].append(
                dict(name=name,
                     id=row['id'],
                     caption=row['text'],
                     seed=seed,
                     shape=list(latent.shape),
                     frames=81,
                     resolution=[384, 640],
                     actual_nfe=used,
                     branch_forward_calls=branches,
                     latent=hashed(latent),
                     pixels=hashed(pixels),
                     timings=timings,
                     denoise_median=statistics.median(t['denoise'] for t in timings),
                     total_median=statistics.median(t['total'] for t in timings)))
            (args.output / 'metrics.json').write_text(json.dumps(results, indent=2))
    hook.remove()
    if not results['samples']:
        raise ValueError('Prompt offset selected no rows')
    (args.output / 'complete.json').write_text(json.dumps(dict(completed=True, samples=len(results['samples']))))
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['encode', 'sample'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--full-info', type=Path)
    p.add_argument('--prompts', type=Path)
    p.add_argument('--config')
    p.add_argument('--checkpoint', default='base')
    p.add_argument('--native-reference', action='store_true')
    p.add_argument('--seeds', default='0,1,2,3,4')
    p.add_argument('--cfg', type=float, default=5.)
    p.add_argument('--max-prompts', type=int, default=100000)
    p.add_argument('--prompt-start', type=int, default=0)
    p.add_argument('--quality-only', action='store_true')
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--repeats', type=int, default=10)
    args = p.parse_args()
    if args.quality_only:
        args.warmup, args.repeats = 0, 1
    if not os.environ.get('RDD_JOB_TOKEN') or time.time() >= float(os.environ.get('RDD_DEADLINE_UNIX', '0')):
        raise RuntimeError('Independent GPU guard required')
    if args.max_prompts < 1 or (args.warmup < 1 and not args.quality_only) or args.repeats < 1 or args.prompt_start < 0:
        p.error('Positive repeats/prompts/warmup required')
    if args.action == 'encode' and not args.full_info:
        p.error('--full-info required')
    if args.action == 'sample' and (not args.prompts or not args.config):
        p.error('--prompts and --config required')
    from pyramid_smoke import check_source
    check_source(SOURCE)
    encode(args) if args.action == 'encode' else evaluate(args)
    check_source(SOURCE)


if __name__ == '__main__':
    main()
