"""Matched real-FastVideo runtime pilot. Untrained bridge: NO video quality claim.

Counts complete 3-NFE noise-filling trajectory, original middle capture, actual
conditional bridge, native VSA suffix and separate fine weights. No VAE timing.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from sample_fastwan_native_vsa import add_fastvideo_paths, sample_three_call_dmd


def aggregate(values):
    ordered = sorted(values)
    return dict(median=statistics.median(values), min=min(values), max=max(values),
                p25=ordered[len(ordered)//4], p75=ordered[3*len(ordered)//4])


def group_bootstrap_ci(values):
    """Resample whole ABBA groups, not correlated individual trajectories."""
    rng = random.Random(88103)
    draws = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(4000))
    return dict(mean=statistics.mean(values), lower=draws[100], upper=draws[3899],
                unit='ABBA/BAAB group', groups=len(values), bootstrap_draws=4000)


def main(args):
    if args.memory_refine_depth and (args.neighbor_rank or args.front_keep != 1):
        raise ValueError('Memory-refiner pilot requires front1, neighbors off')
    root, out = Path(args.root).resolve(), Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'metrics.json').exists():
        raise FileExistsError('Completed run exists; use a new output directory')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if visible not in ('0', '1', '2', '3'):
        raise ValueError('Set exactly one authorized physical GPU 0–3')
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info(device)
    if free < args.min_free_gib * 2**30:
        raise RuntimeError(f'Not enough headroom: {free / 2**30:.2f}GiB')
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20560913)
    torch.cuda.manual_seed_all(20560913)
    fv = root / 'third_party/FastVideo-ref'
    add_fastvideo_paths(fv, fv / 'fastvideo-kernel')
    from attention_analysis.fastwan_native_attention import install_projected_cache_class_patch
    from attention_analysis.csra_prefix_video_bridge import VideoConditionalPrefixBridge, VideoPrefixSamplerProxy
    from fastvideo.configs.models.dits import WanVideoConfig
    from fastvideo.configs.pipelines.wan import FastWan2_1_T2V_480P_Config
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.models.dits.wanvideo import WanTransformerBlock_VSA
    from fastvideo.models.loader.component_loader import TransformerLoader
    from wan.modules.t5 import T5EncoderModel

    # Match the existing FastWan evaluation text encoding, before model loads.
    wan = root / 'pretrained_models/Wan2.1-T2V-1.3B'
    text = T5EncoderModel(text_len=512, dtype=torch.float32, device=torch.device('cpu'),
                         checkpoint_path=str(wan / 'models_t5_umt5-xxl-enc-bf16.pth'),
                         tokenizer_path=str(wan / 'google/umt5-xxl'))
    text.model.to(device)
    with torch.inference_mode():
        unpadded = text([args.prompt], device)[0]
        if len(unpadded) > 512: raise ValueError('Prompt too long')
        context = torch.zeros((1, 512, unpadded.shape[-1]), device=device, dtype=torch.bfloat16)
        context[0, :len(unpadded)] = unpadded.to(torch.bfloat16)
    text.model.cpu()
    del text, unpadded
    torch.cuda.empty_cache()
    install_projected_cache_class_patch(WanTransformerBlock_VSA)
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(args.port),
                      RANK='0', WORLD_SIZE='1', LOCAL_RANK='0')
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    weights = root / 'pretrained_models/FastWan2.1-T2V-1.3B-Diffusers/transformer'
    fv_args = FastVideoArgs(model_path=str(weights), num_gpus=1,
        dit_cpu_offload=False, dit_layerwise_offload=False,
        pipeline_config=FastWan2_1_T2V_480P_Config(dit_config=WanVideoConfig(), dit_precision='bf16'))
    fv_args.device = device
    original = TransformerLoader().load(str(weights), fv_args).to(dtype=torch.bfloat16).eval()
    fine = TransformerLoader().load(str(weights), fv_args).to(dtype=torch.bfloat16).eval()
    for m in (original, fine):
        assert all(type(b).__name__ == 'WanTransformerBlock_VSA' and hasattr(b, 'to_gate_compress') for b in m.blocks)
        m.requires_grad_(False)
    hidden, heads = original.blocks[0].hidden_dim, original.blocks[0].num_attention_heads
    if args.memory_refine_depth:
        from attention_analysis.csra_refined_video_bridge import VideoRefinedPrefixBridge
        bridge = VideoRefinedPrefixBridge(hidden, heads, args.memory_refine_depth)
    elif args.neighbor_rank:
        from attention_analysis.csra_neighbor_video_bridge import VideoNeighborPrefixBridge
        bridge = VideoNeighborPrefixBridge(hidden, heads, args.neighbor_rank)
    else:
        bridge = VideoConditionalPrefixBridge(hidden, heads)
    bridge = bridge.to(device).eval().requires_grad_(False)
    model_bytes = [sum(p.numel()*p.element_size() for p in m.parameters()) for m in (original, fine)]
    bridge_bytes = sum(p.numel()*p.element_size() for p in bridge.parameters())
    protocol = dict(version=('csra_video_front1_neighbor_feasibility_v2' if args.neighbor_rank or args.front_keep
                             else 'csra_video_prefix_feasibility_v1'), complete=False,
        warning='UNTRAINED video bridge; feasibility only. No trained-video quality, FID or VAE claim.',
        gpu=visible, shared_gpu=True, free_gib_at_start=free/2**30,
        memory_fraction=args.memory_fraction, model=str(weights), hidden=hidden, heads=heads,
        layers=len(original.blocks), model_parameter_bytes=model_bytes, bridge_parameter_bytes=bridge_bytes,
        two_weight_sets_resident_for_all_paired_runs=True, prompt=args.prompt,
        context_sha256=hashlib.sha256(context.cpu().view(torch.uint16).numpy().tobytes()).hexdigest(),
        frames=29, NFE=3, timesteps=[1000,757,522], shift=8, transition='noise_filling',
        boundary_executor='native', vsa_sparsity=.8, gates_and_coarse_output='native in all executed blocks',
        offload=False, layerwise_offload=False, tf32=False, bridge_sdpa='FLASH_ATTENTION only',
        bridge_affine_precision='FP32', bridge_attention_ffn_precision='BF16 AMP',
        bridge_condition='actual Wan unmodulated current timestep embedding',
        source='middle NFE, full hidden after original block13 (zero-based)',
        init='random untrained bridge, original FastWan weights in both backbones',
        front_keep=args.front_keep, neighbor_rank=args.neighbor_rank,
        neighbor='LN -> H/rank -> depthwise spatial3x3 -> SiLU -> rank/H, zero-output init, branch always executes',
        placement='fine executes0, bridge replaces1..6, executes7..29' if args.front_keep == 1 else 'legacy front0',
        warmups=args.warmups, abba_groups=args.groups, entries=[])
    for path in (Path(__file__), root/'rdd_video/attention_analysis/csra_prefix_video_bridge.py',
                 fv/'fastvideo/models/dits/wanvideo.py'):
        protocol.setdefault('source_sha256', {})[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if args.neighbor_rank:
        for name in ('csra_fine_neighbor.py', 'csra_neighbor_video_bridge.py'):
            path = root/'rdd_video/attention_analysis'/name
            protocol['source_sha256'][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if args.memory_refine_depth:
        protocol.update(version='csra_video_current_coarse_memory_cost_v1',
            memory_refine_depth=args.memory_refine_depth,
            memory_refiner='pool current fine h to native coarse; two full-H conditioned coarse SDPA/FFN blocks; '
                           'zero-init full-H output correction; complete branch always executes; no new RoPE',
            memory_refiner_training='UNTRAINED; current-state refinement recomputed at each fine call, not cached')
        for name in ('csra_memory_refiner.py', 'csra_refined_video_bridge.py'):
            path = root/'rdd_video/attention_analysis'/name
            protocol['source_sha256'][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    cases = [('832x448_spatial421', (1,16,8,56,104), [(4,1),(2,1),(1,1)]),
             ('832x448_st421', (1,16,8,56,104), [(4,4),(2,2),(1,1)]),
             ('1280x768_spatial421', (1,16,8,96,160), [(4,1),(2,1),(1,1)]),
             ('1280x768_st421', (1,16,8,96,160), [(4,4),(2,2),(1,1)])]
    if args.smoke: cases = cases[:1]
    def sample(model, shape, factors, controller=None):
        # Exclude verbose print/I/O from algorithm wall time, but retain the
        # scheduler, transitions, RNG, hooks, bridge and synchronization work.
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()), torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            result = sample_three_call_dmd(model, context, shape, factors, [1000,757,522],
                20660913, .8, 'noise_filling', device, attention_controller=controller,
                boundary_executor='native', return_breakdown=True)
        torch.cuda.synchronize(device)
        return result, time.perf_counter()-start

    # Loader-copy and no-op wrapper parity on the full three-call trajectory.
    _, small_shape, small_factors = cases[0]
    ref, _ = sample(original, small_shape, small_factors)
    duplicate, _ = sample(fine, small_shape, small_factors)
    protocol['duplicate_model_max_abs'] = (ref[0].float()-duplicate[0].float()).abs().max().item()
    del duplicate
    proxy = VideoPrefixSamplerProxy(original, fine, bridge, front_keep=args.front_keep)
    proxy.reset(0)
    wrapped, _ = sample(proxy, small_shape, small_factors, proxy)
    protocol['baseline_wrapper_max_abs'] = (ref[0].float()-wrapped[0].float()).abs().max().item()
    proxy.audit()
    if protocol['duplicate_model_max_abs'] != 0 or protocol['baseline_wrapper_max_abs'] != 0:
        raise AssertionError('Native baseline or duplicate model lost exact parity')
    del ref, wrapped
    print('PASS native / duplicate / no-op wrapper exact parity', flush=True)
    if args.neighbor_rank:
        bridge.neighbor_enabled = False
        proxy.reset(6)
        no_neighbor, _ = sample(proxy, small_shape, small_factors, proxy)
        proxy.audit()
        bridge.neighbor_enabled = True
        proxy.reset(6)
        with_neighbor, _ = sample(proxy, small_shape, small_factors, proxy)
        proxy.audit()
        protocol['zero_init_neighbor_final_latent_max_abs'] = (
            no_neighbor[0].float()-with_neighbor[0].float()).abs().max().item()
        if protocol['zero_init_neighbor_final_latent_max_abs'] != 0:
            raise AssertionError('Zero-initialized new branch changed the full real-model function')
        del no_neighbor, with_neighbor
        print('PASS actual front1 zero-init neighbor trajectory exact parity', flush=True)

    if args.memory_refine_depth:
        bridge.refiner_enabled = False
        proxy.reset(6)
        before, _ = sample(proxy, small_shape, small_factors, proxy)
        proxy.audit()
        bridge.refiner_enabled = True
        proxy.reset(6)
        after, _ = sample(proxy, small_shape, small_factors, proxy)
        proxy.audit()
        protocol['zero_init_refiner_final_latent_max_abs'] = (before[0].float()-after[0].float()).abs().max().item()
        if protocol['zero_init_refiner_final_latent_max_abs'] != 0:
            raise AssertionError('Zero-initialized memory refiner changed the real-model trajectory')
        del before, after
        print('PASS actual memory-refiner zero-init trajectory exact parity', flush=True)

    def run(depth, shape, factors):
        proxy.reset(depth)
        torch.cuda.reset_peak_memory_stats(device)
        result, elapsed = sample(proxy, shape, factors, proxy)
        audit = proxy.audit()
        if not torch.isfinite(result[0]).all().item():
            raise AssertionError('Nonfinite latent even for computational sanity')
        record = dict(depth=depth, front_keep=args.front_keep,
                      memory_refiner_executed=bool(depth and args.memory_refine_depth and bridge.refiner_enabled),
                      neighbor_executed=bool(depth and args.neighbor_rank and bridge.neighbor_enabled),
                      wall_seconds=elapsed, stage_seconds=result[2],
                      breakdown=result[4], audit=audit,
                      peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30)
        del result
        return record

    variants = ['reader_only', 'reader_neighbor'] if args.neighbor_rank else ['reader_only']
    if args.memory_refine_depth:
        variants = ['reader_only', 'reader_refined_memory']
    work = [(name, shape, factors, variant, depth) for name, shape, factors in cases
            for variant in variants for depth in ([6] if args.smoke or args.neighbor_rank or args.front_keep else [6,8])]
    for name, shape, factors, variant, depth in work:
            if args.neighbor_rank: bridge.neighbor_enabled = variant == 'reader_neighbor'
            if args.memory_refine_depth: bridge.refiner_enabled = variant == 'reader_refined_memory'
            for _ in range(args.warmups):
                run(0, shape, factors); run(depth, shape, factors)
            groups, baseline, candidate, ratios = [], [], [], []
            for group in range(args.groups):
                order = [0,depth,depth,0] if group % 2 == 0 else [depth,0,0,depth]
                rows = [run(d, shape, factors) for d in order]
                b = statistics.mean(r['wall_seconds'] for r in rows if r['depth']==0)
                c = statistics.mean(r['wall_seconds'] for r in rows if r['depth']==depth)
                ratios.append(b/c)
                groups.append(dict(index=group, order=order, rows=rows, paired_speedup=b/c))
                baseline += [r for r in rows if r['depth']==0]
                candidate += [r for r in rows if r['depth']==depth]
                print(f'{name} {variant} front{args.front_keep} skip{depth} group{group+1}/{args.groups} paired_speedup={b/c:.4f}', flush=True)
            entry = dict(case=name, variant=variant, front_keep=args.front_keep, full_shape=shape,
                stage_factors=factors, prefix_depth=depth,
                baseline_wall=aggregate([r['wall_seconds'] for r in baseline]),
                candidate_wall=aggregate([r['wall_seconds'] for r in candidate]),
                paired_speedup=aggregate(ratios),
                latency_reduction_percent=aggregate([100*(1-1/x) for x in ratios]),
                paired_latency_ci95_percent=group_bootstrap_ci([100*(1-1/x) for x in ratios]),
                bridge_seconds=aggregate([r['audit']['bridge_seconds'] for r in candidate]),
                groups=groups)
            protocol['entries'].append(entry)
            (out/'progress.json').write_text(json.dumps(protocol, indent=2)+'\n')
    protocol['complete'] = True
    protocol['smoke'] = args.smoke
    (out/'metrics.json').write_text(json.dumps(protocol, indent=2)+'\n')
    print(f'COMPLETE {out}/metrics.json (feasibility only)', flush=True)
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='/scr/yuheng/DevOpsBench/multiscale_diffusion')
    p.add_argument('--output', required=True)
    p.add_argument('--port', type=int, default=29693)
    p.add_argument('--min-free-gib', type=float, default=32.)
    p.add_argument('--memory-fraction', type=float, default=.24)
    p.add_argument('--warmups', type=int, default=2)
    p.add_argument('--groups', type=int, default=12)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--front-keep', type=int, choices=(0,1), default=0)
    p.add_argument('--neighbor-rank', type=int, choices=(0,128), default=0)
    p.add_argument('--memory-refine-depth', type=int, choices=(0,1,2), default=0)
    p.add_argument('--prompt', default='A red fox walking through a snowy forest, cinematic natural light.')
    main(p.parse_args())
