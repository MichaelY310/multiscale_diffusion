"""Full original Wan pipeline versus original modules through FastVideo adapter.

Same cached real UMT5, negative prompt, latents, CFG arithmetic, original
scheduler and original FP32 VAE. This is native pipeline parity, not the
controlled RDD/PF/PW custom flow or a quality benchmark.
"""
from typing import Any
import argparse
import json
import os
from pathlib import Path
import time


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--negative', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get('RDD_JOB_TOKEN') or time.time() >= float(os.environ.get('RDD_DEADLINE_UNIX', '0')):
        raise RuntimeError('Independent guard required')
    import torch
    from diffusers import WanPipeline, AutoencoderKLWan, WanTransformer3DModel
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
    from fastvideo.train.utils.config import load_run_config
    from fastvideo.train.utils.instantiate import instantiate
    from full_model_trace import hashed
    from types import SimpleNamespace
    torch.use_deterministic_algorithms(True)
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    cfg = load_run_config(args.config)
    model = instantiate(dict(cfg.models['student'], trainable=False), training_config=cfg.training)
    checkpoint = cfg.models['student']['init_from']
    original = WanTransformer3DModel.from_pretrained(checkpoint,
                                                     subfolder='transformer',
                                                     torch_dtype=torch.bfloat16,
                                                     local_files_only=True).cuda().eval().requires_grad_(False)
    vae = AutoencoderKLWan.from_pretrained(checkpoint,
                                           subfolder='vae',
                                           torch_dtype=torch.float32,
                                           local_files_only=True).cuda().eval().requires_grad_(False)
    pipeline = WanPipeline.from_pretrained(checkpoint,
                                           transformer=original,
                                           vae=vae,
                                           text_encoder=None,
                                           tokenizer=None,
                                           torch_dtype=torch.bfloat16,
                                           local_files_only=True).to('cuda')
    sample = torch.load(args.cache, map_location='cpu', weights_only=True)
    text = sample['text_embedding'].cuda().to(torch.bfloat16)
    negative = torch.load(args.negative, weights_only=True).cuda().to(torch.bfloat16)
    initial = torch.randn((1, 16, 9, 16, 32),
                          device='cuda',
                          dtype=torch.float32,
                          generator=torch.Generator(device='cuda').manual_seed(42))
    counts = [0, 0]
    handle = original.register_forward_hook(lambda *unused: counts.__setitem__(0, counts[0] + 1))
    kwargs = dict(prompt_embeds=text,
                  negative_prompt_embeds=negative,
                  height=128,
                  width=256,
                  num_frames=33,
                  num_inference_steps=6,
                  guidance_scale=5.,
                  output_type='latent')
    with torch.no_grad():
        reference = pipeline(**kwargs, latents=initial.clone()).frames
    handle.remove()

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
                    **extra: Any) -> Any:
            counts[1] += 1
            batch = SimpleNamespace(conditional_dict={'encoder_hidden_states': encoder_hidden_states})
            prediction = model.predict_noise(hidden_states.permute(0, 2, 1, 3, 4), timestep, batch,
                                             conditional=True).permute(0, 2, 1, 3, 4)
            return (prediction, )

    pipeline.transformer = Interface()
    with torch.no_grad():
        actual = pipeline(**kwargs, latents=initial.clone()).frames
    mean = torch.tensor(vae.config.latents_mean, device='cuda').view(1, 16, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device='cuda').view(1, 16, 1, 1, 1)
    with torch.no_grad():
        expected_pixels = (vae.decode(reference / (1.0 / std) + mean, return_dict=False)[0] / 2 + .5).clamp(0, 1)
        model.vae = vae
        actual_pixels = model.decode_latents(actual.permute(0, 2, 1, 3, 4))
    checks = dict(native_scheduler_and_cfg_latent=torch.equal(reference, actual),
                  original_vae_decoded_pixels=torch.equal(expected_pixels, actual_pixels),
                  branch_calls=counts == [12, 12])
    args.output.write_text(
        json.dumps(dict(
            status='passed' if all(checks.values()) else 'failed',
            checks=checks,
            native_scheduler=dict(pipeline.scheduler.config),
            original_forward_calls=counts,
            reference_latent=hashed(reference),
            adapter_latent=hashed(actual),
            pixels=hashed(actual_pixels),
            actual_original_T5=True,
            original_FP32_VAE=True,
            quality_benchmark=False,
            scope=
            'Original native Wan full pipeline through current FastVideo original-module interface; not custom controlled solver parity'
        ),
                   indent=2))
    if not all(checks.values()):
        raise AssertionError(checks)
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
