"""Hardware feasibility of committed temporal bands with ancestor KV memory.

NOT a trained RDD sampler: the released FastWan weights do not implement the
partially noised, committed-band conditional distribution. This experiment
measures a new architecture's complete runtime, not quality-equivalent speedup.
Every transformer layer, text attention, FFN and VSA coarse gate stays live.
An ordinary larger-temporal-patch model is a mandatory unresolved countertest.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import types
import subprocess
import faulthandler

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sample_fastwan_native_vsa import add_fastvideo_paths, sample_three_call_dmd, scheduler_sigma
from benchmark_csra_prefix_feasibility import aggregate, group_bootstrap_ci


def haar_synthesis(dc, detail):
    """Orthonormal temporal pair, not nearest-neighbor output copying."""
    if dc.shape != detail.shape:
        raise ValueError("Each committed DC has exactly one new detail coefficient")
    return torch.stack(((dc + detail) / math.sqrt(2),
                        (dc - detail) / math.sqrt(2)), dim=3).flatten(2, 3)


class TemporalPatch2:
    """Untrained ordinary-model control: pack, do not average, two frames.

    The 128 input coordinates map injectively into the unchanged 1536-wide
    token. Both output frames are predicted freshly. All 30 blocks remain.
    This is a hardware counterexample, not a quality-equivalent checkpoint.
    """
    def __init__(self, model):
        self.model = model
        self.original = (model.patch_embedding.proj, model.proj_out,
                         model.patch_size, model.patch_embedding.patch_size)
        proj, head, patch, _ = self.original
        if tuple(patch) != (1, 2, 2):
            raise ValueError('Expected the original temporal patch1 Wan')
        dev, dtype = proj.weight.device, proj.weight.dtype
        w = proj.weight.flatten(1).float()
        q = torch.linalg.qr(w, mode='reduced').Q
        generator = torch.Generator(device=dev).manual_seed(87121)
        detail = torch.randn(w.shape, device=dev, generator=generator)
        detail = detail - q @ (q.T @ detail)
        detail *= w.norm() / detail.norm()
        packed = torch.cat((((w+detail)/math.sqrt(2)).reshape(proj.out_channels,proj.in_channels,1,2,2),
                            ((w-detail)/math.sqrt(2)).reshape(proj.out_channels,proj.in_channels,1,2,2)),dim=2)
        self.proj = torch.nn.Conv3d(proj.in_channels, proj.out_channels,
            kernel_size=(2,2,2), stride=(2,2,2), device=dev, dtype=dtype)
        self.proj.weight.data.copy_(packed.to(dtype))
        self.proj.bias.data.copy_(proj.bias)
        self.head = torch.nn.Linear(head.in_features, 2*head.out_features,
            device=dev, dtype=dtype)
        # Frame-major [pt,ph,pw,c] ordering used by Wan's unpatchification.
        noise = torch.randn(head.weight.shape, device=dev, generator=generator)
        noise *= .1 * head.weight.float().norm() / noise.norm()
        self.head.weight.data.copy_(torch.cat((head.weight.float()+noise,
                                              head.weight.float()-noise)).to(dtype))
        self.head.bias.data.copy_(torch.cat((head.bias,head.bias)))
        self.proj.requires_grad_(False)
        self.head.requires_grad_(False)
        singular = torch.linalg.svdvals(self.proj.weight.flatten(1).float())
        self.audit = dict(input_coordinates=128, input_projection_rank=int((singular>1e-5).sum()),
            minimum_input_singular=float(singular.min()), token_width=proj.out_channels,
            patch=[2,2,2], prediction_channels_per_token=2*head.out_features,
            warning='New untrained embedding/head; timing-only ordinary packing control')
        if self.audit['input_projection_rank'] != 128:
            raise AssertionError('Packing must retain all 128 input coordinates')

    def enable(self, enabled):
        model = self.model
        proj, head, patch, embed_patch = self.original
        model.patch_embedding.proj = self.proj if enabled else proj
        model.proj_out = self.head if enabled else head
        model.patch_size = (2,2,2) if enabled else patch
        model.patch_embedding.patch_size = (2,2,2) if enabled else embed_patch


def physical_rope(freqs, shape, full_t):
    """Use support-center time positions; spatial positions remain native."""
    cos, sin = (x.clone() for x in freqs)
    t, h, w = shape
    scale = full_t / t
    pos = (torch.arange(t, device=cos.device, dtype=torch.float64) + .5) * scale - .5
    dim = 44  # Wan-1.3B head128: [44,42,42] temporal/spatial RoPE.
    inv = 10000. ** (-torch.arange(0, dim, 2, device=cos.device, dtype=torch.float64) / dim)
    angles = pos[:, None] * inv[None, :]
    c = angles.cos().repeat_interleave(2, -1).repeat_interleave((h//2)*(w//2), 0)
    s = angles.sin().repeat_interleave(2, -1).repeat_interleave((h//2)*(w//2), 0)
    cos[..., :dim] = c.to(cos.dtype)
    sin[..., :dim] = s.to(sin.dtype)
    return cos, sin


class AncestorKV:
    """SP=1 feasibility wrapper, with fresh rectangular VSA on detail queries.

    Caches the *generating* earlier-stage K/V, not a hypothetical clean-parent
    encoding. Cache capture/write/cat costs are inside the timed trajectory.
    Root KV + middle-detail KV each has N/4 tokens; final memory totals N/2.
    """
    def __init__(self, model):
        from fastvideo.forward_context import get_forward_context
        from fastvideo.attention.layer import _apply_rotary_emb
        from fastvideo_kernel.ops import video_sparse_attn
        self.get_context = get_forward_context
        self.rope = _apply_rotary_emb
        self.vsa = video_sparse_attn
        self.originals = []
        self.cache = {}
        self.enabled = False
        self.stage = 0
        self.native_positions = False
        self.audit = []
        for index, block in enumerate(model.blocks):
            layer = block.attn1
            original = layer.forward
            self.originals.append((layer, original))
            owner = self
            def forward(layer, q, k, v, original_seq_len, replicated_q=None,
                        replicated_k=None, replicated_v=None, gate_compress=None,
                        freqs_cis=None, _index=index, _original=original):
                if not owner.enabled:
                    return _original(q, k, v, original_seq_len,
                        replicated_q=replicated_q, replicated_k=replicated_k,
                        replicated_v=replicated_v, gate_compress=gate_compress,
                        freqs_cis=freqs_cis)
                if replicated_q is not None or gate_compress is None:
                    raise ValueError("This pilot requires the native learned VSA coarse gate, SP1")
                metadata = owner.get_context().attn_metadata
                b = q.shape[0]
                stacked = torch.cat((q, k, v, gate_compress), 0)[:, :original_seq_len]
                if not owner.native_positions:
                    freqs_cis = physical_rope(freqs_cis, owner.shape, owner.full_t)
                stacked[:2*b] = owner.rope(stacked[:2*b], *freqs_cis, is_neox_style=False)
                tiled = layer.attn_impl.preprocess_qkv(stacked, metadata)
                q, k, v, gate = (x.transpose(1, 2).contiguous() for x in tiled.chunk(4, 0))
                fresh_k, fresh_v = k, v
                key_sizes = metadata.variable_block_sizes
                if owner.stage > 0:
                    old_k, old_v, old_sizes = owner.cache[_index]
                    k = torch.cat((old_k, k), dim=2)
                    v = torch.cat((old_v, v), dim=2)
                    key_sizes = torch.cat((old_sizes, key_sizes))
                topk = max(1, math.ceil(.2 * key_sizes.numel()))
                result = owner.vsa(q, k, v, key_sizes,
                    metadata.variable_block_sizes, topk, block_size=(4, 4, 4),
                    compress_attn_weight=gate).transpose(1, 2)
                if owner.stage < 2:
                    # Contiguous BHSD copies own their lifetime; never retain
                    # an alias to the metadata's reused scatter scratch buffer.
                    owner.cache[_index] = (k.clone(), v.clone(), key_sizes.clone())
                if _index == 0:
                    owner.audit.append(dict(stage=owner.stage, q_tokens=q.shape[2],
                        k_tokens=k.shape[2], fresh_tokens=fresh_k.shape[2],
                        inherited_tokens=k.shape[2]-fresh_k.shape[2], topk=topk,
                        key_blocks=key_sizes.numel()))
                return layer.attn_impl.postprocess_output(result, metadata), None
            layer.forward = types.MethodType(forward, layer)

    def begin(self, stage, shape, full_t):
        self.stage, self.shape, self.full_t = stage, shape, full_t
        if stage == 0:
            self.cache.clear()
            self.audit = []

    def restore(self):
        for layer, original in self.originals:
            layer.forward = original
        self.cache.clear()


@torch.inference_mode()
def sample_bands(model, context, shape, seed, dev, memory):
    from fastvideo.attention.backends.video_sparse_attn import VideoSparseAttentionMetadataBuilder
    from fastvideo.forward_context import set_forward_context
    from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
    b, c, full_t, h, w = shape
    generator = torch.Generator(device=dev).manual_seed(seed)
    scheduler = FlowMatchEulerDiscreteScheduler(shift=8.)
    builder = VideoSparseAttentionMetadataBuilder()
    forward_batch = ForwardBatch(data_type="dummy")
    clean = None
    events = []
    dc_audit = []
    root_dc = None
    for stage, (t, timestep) in enumerate(zip((full_t//4, full_t//4, full_t//2), (1000,757,522))):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        sigma = scheduler_sigma(scheduler, timestep, dev).to(torch.bfloat16)
        state = torch.randn((b, c, t, h, w), generator=generator,
            dtype=torch.bfloat16, device=dev) * sigma
        memory.begin(stage, (t,h,w), full_t)
        metadata = builder.build(current_timestep=stage, raw_latent_shape=(t,h,w),
            patch_size=model.patch_size, VSA_sparsity=.8, device=dev)
        with set_forward_context(current_timestep=stage, attn_metadata=metadata,
                                 forward_batch=forward_batch):
            prediction = model(hidden_states=state, encoder_hidden_states=context,
                timestep=torch.full((b,), float(timestep), device=dev, dtype=torch.float32))
        band = state - sigma * prediction
        parent = clean
        clean = band if stage == 0 else haar_synthesis(clean, band)
        if stage == 0:
            root_dc = clean
        else:
            recovered = (clean[:,:,0::2].float()+clean[:,:,1::2].float())/math.sqrt(2)
            # Return GPU scalars, synchronize/convert only AFTER timing below.
            dc_audit.append(((recovered-parent.float()).abs().max(),
                (recovered-parent.float()).norm()/parent.float().norm()))
        end.record()
        events.append((start,end))
    torch.cuda.synchronize()
    audit = list(memory.audit)
    cache_gib = sum(t.numel()*t.element_size() for pair in memory.cache.values() for t in pair[:2])/2**30
    memory.cache.clear()
    return clean, dict(stages_seconds=[s.elapsed_time(e)/1000 for s,e in events],
        attention=audit, cache_gib=cache_gib,
        dc_roundoff=[dict(max_abs=float(x),relative_l2=float(y)) for x,y in dc_audit])


def validate_rectangular_vsa(dev):
    """Independent FP32 reference includes routing, softmax, AV, coarse gate."""
    from fastvideo_kernel.ops import video_sparse_attn
    # Small exact full blocks; rectangular Q128/K256, head128, two heads.
    generator = torch.Generator(device=dev).manual_seed(8811)
    q = torch.randn((1,2,128,128), device=dev, dtype=torch.bfloat16, generator=generator)
    k = torch.randn((1,2,256,128), device=dev, dtype=torch.bfloat16, generator=generator)
    v = torch.randn(k.shape, device=dev, dtype=torch.bfloat16, generator=generator)
    gate = torch.randn(q.shape, device=dev, dtype=torch.bfloat16, generator=generator)
    qs = torch.full((2,),64,device=dev,dtype=torch.int32)
    ks = torch.full((4,),64,device=dev,dtype=torch.int32)
    qc,kc,vc = [x.float().reshape(1,2,-1,64,128).mean(3).to(torch.bfloat16) for x in (q,k,v)]
    scores = qc@kc.transpose(-2,-1)/math.sqrt(128)
    coarse = (scores.softmax(-1)@vc).repeat_interleave(64,2)
    selected = scores.topk(2,-1).indices
    block_mask = torch.zeros_like(scores,dtype=torch.bool).scatter_(-1,selected,True)
    mask = block_mask.repeat_interleave(64,2).repeat_interleave(64,3)
    fine = (q.float()@k.float().transpose(-2,-1)/math.sqrt(128)).masked_fill(~mask,float('-inf')).softmax(-1)@v.float()
    ref = coarse*gate+fine.to(torch.bfloat16)
    actual = video_sparse_attn(q,k,v,ks,qs,2,(4,4,4),gate)
    error = float((actual.float()-ref.float()).abs().max())
    relative = float((actual.float()-ref.float()).norm()/ref.float().norm())
    if relative > .015:
        raise AssertionError(f"Rectangular VSA mismatch: {relative}")
    return dict(q=128,k=256,max_abs=error,relative_l2=relative,reference="FP32 softmax/AV plus BF16 native coarse branch")


def route_union_audit(route_capture, shape):
    """CPU audit; spatial ownership persists across frames, same for both methods."""
    def union_rows(mask):
        entries=[]
        n=mask.shape[-1]
        hw=(shape[3]//8)*(shape[4]//8)
        width=shape[4]//8
        ids=torch.arange(n)
        for ranks in (2,4):
            ownership={
                'contiguous_sequence':(ids*ranks//n).clamp_max(ranks-1),
                'persistent_spatial':((ids%hw)*ranks//hw).clamp_max(ranks-1),
                'persistent_columns':((ids%width)*ranks//width).clamp_max(ranks-1),
                'periodic_columns':ids%width%ranks,
            }
            for layout,owner in ownership.items():
                values=[]
                remote=[]
                for rank in range(ranks):
                    query_ids=(owner==rank).nonzero().flatten()
                    used=mask[:,:,query_ids,:].any(2)
                    values.extend(used.float().mean(-1).flatten().tolist())
                    remote_ids=owner!=rank
                    remote.extend(used[...,remote_ids].float().mean(-1).flatten().tolist())
                entries.append(dict(ranks=ranks,layout=layout,
                    global_key_union_mean=statistics.mean(values),global_key_union_max=max(values),
                    remote_key_union_mean=statistics.mean(remote),remote_key_union_max=max(remote)))
        return dict(blocks=n,density=float(mask.float().mean()),query_sharded=entries,
            head_sharded_global_union=float(mask.any(2).float().mean()))
    route_report=[]
    root_routes={}
    counts={0:0,1:0,2:0}
    for item in route_capture:
        stage,mask=item['stage'],item['mask']
        layer=counts[stage]
        counts[stage]+=1
        route_report.append(dict(stage=stage,layer=layer,kind='online_vsa',**union_rows(mask)))
        if stage==0:
            root_routes[layer]=mask
        else:
            root_mask=root_routes[layer]
            root_t=shape[2]//16
            hw=(shape[3]//8)*(shape[4]//8)
            ratio=2**stage
            fine_t=root_t*ratio
            ids=torch.arange(fine_t*hw)
            parent=(ids//hw//ratio)*hw+ids%hw
            inherited=root_mask[:,:,parent][:,:,:,parent]
            route_report.append(dict(stage=stage,layer=layer,kind='root_lifted_c2f_math_control',**union_rows(inherited)))
    return route_report,counts


def main(a):
    faulthandler.enable()
    faulthandler.dump_traceback_later(240,repeat=True)
    root,out = Path(a.root),Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.json').exists():
        raise FileExistsError("Refuse overwriting completed benchmark")
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in tuple(str(i) for i in range(8)):
        raise ValueError("Exactly one physical GPU required; never evict other jobs")
    dev = torch.device('cuda:0')
    torch.cuda.set_device(dev)
    if torch.cuda.mem_get_info()[0] < a.minimum_free_gib*2**30:
        raise RuntimeError(f"Need {a.minimum_free_gib}GiB free")
    torch.cuda.set_per_process_memory_fraction(a.memory_fraction,dev)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    fv=root/'third_party/FastVideo-ref'
    add_fastvideo_paths(fv,fv/'fastvideo-kernel')
    from fastvideo.configs.models.dits import WanVideoConfig
    from fastvideo.configs.pipelines.wan import FastWan2_1_T2V_480P_Config
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.models.loader.component_loader import TransformerLoader
    from wan.modules.t5 import T5EncoderModel
    from wan.modules.vae import WanVAE
    from sample_rdd_wan import save_video
    wan=root/'pretrained_models/Wan2.1-T2V-1.3B'
    text=T5EncoderModel(text_len=512,dtype=torch.float32,device=torch.device('cpu'),
        checkpoint_path=str(wan/'models_t5_umt5-xxl-enc-bf16.pth'),tokenizer_path=str(wan/'google/umt5-xxl'))
    text.model.to(dev)
    with torch.inference_mode():
        tokens=text([a.prompt],dev)[0]
        context=torch.zeros((1,512,tokens.shape[-1]),device=dev,dtype=torch.bfloat16)
        context[0,:len(tokens)]=tokens.to(torch.bfloat16)
    text.model.cpu()
    del text,tokens
    torch.cuda.empty_cache()
    os.environ.update(MASTER_ADDR='127.0.0.1',MASTER_PORT=str(a.port),RANK='0',WORLD_SIZE='1',LOCAL_RANK='0')
    maybe_init_distributed_environment_and_model_parallel(1,1)
    weights=root/'pretrained_models/FastWan2.1-T2V-1.3B-Diffusers/transformer'
    fa=FastVideoArgs(model_path=str(weights),num_gpus=1,dit_cpu_offload=False,
        dit_layerwise_offload=False,pipeline_config=FastWan2_1_T2V_480P_Config(dit_config=WanVideoConfig(),dit_precision='bf16'))
    fa.device=dev
    model=TransformerLoader().load(str(weights),fa).to(dtype=torch.bfloat16).eval().requires_grad_(False)
    vae=WanVAE(z_dim=16,vae_pth=str(wan/'Wan2.1_VAE.pth'),dtype=torch.bfloat16,device=dev)
    shape=(1,16,(a.frames-1)//4+1,a.height//8,a.width//8)
    if shape[2]%16 or a.height%64 or a.width%64:
        raise ValueError("Use temporal grids divisible16 and spatial dimensions divisible64")
    checks=validate_rectangular_vsa(dev)
    print('Rectangular VSA validated; installing controllers',flush=True)
    memory=AncestorKV(model)
    c2f=None
    packing = TemporalPatch2(model) if a.counter_patch2 else None
    candidate = 'ordinary_patch2' if packing else 'bands'
    route_capture=[]
    capture_routes=False
    import fastvideo_kernel.ops as kernel_ops
    original_mask=kernel_ops.fused_topk_mask
    def mask_hook(scores,topk):
        mask=original_mask(scores,topk)
        if capture_routes:
            route_capture.append(dict(stage=memory.stage,mask=mask.detach().cpu().bool()))
        return mask
    kernel_ops.fused_topk_mask=mask_hook
    class StageNotice:
        attention_mode='vsa'
        def begin_model_forward(self,stage,shape):
            memory.stage=stage
            if c2f is not None:
                c2f.begin_model_forward(stage,shape)
    notice=StageNotice()
    def sample(v):
        memory.enabled=v=='bands'
        if c2f is not None:
            c2f.enabled=v!='online_vsa'
            c2f.executor='current_grouped' if v=='current_grouped' else ('siblings' if v==candidate else 'native')
        if packing:
            packing.enable(v==candidate)
        with contextlib.redirect_stdout(io.StringIO()),torch.inference_mode(),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            if v=='bands':
                return sample_bands(model,context,shape,a.seed,dev,memory)
            native_result=sample_three_call_dmd(model,context,shape,[(1,4),(1,2),(1,1)],
                [1000,757,522],a.seed,.8,'noise_filling',dev,return_breakdown=True,attention_controller=notice)
            return native_result[0],native_result[-1]
    def run(v,export=False):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start=time.perf_counter()
        z,audit=sample(v)
        torch.cuda.synchronize()
        denoise=time.perf_counter()-start
        with torch.inference_mode():
            video=vae.decode([z[0].float()])[0]
        torch.cuda.synchronize()
        total=time.perf_counter()-start
        if not torch.isfinite(z).all() or not torch.isfinite(video).all():
            raise AssertionError('Nonfinite result')
        if export:
            torch.save(z.detach().cpu(),out/f'{v}_latent.pt')
            save_video(video,str(out/f'{v}.mp4'),fps=16)
            from PIL import Image
            frame=video[:,video.shape[1]//2].float().cpu().clamp(-1,1)
            Image.fromarray(((frame.permute(1,2,0)+1)*127.5).round().byte().numpy()).save(out/f'{v}.png')
        return dict(variant=v,denoise_seconds=denoise,total_seconds=total,
            vae_seconds=total-denoise,peak_gib=torch.cuda.max_memory_allocated()/2**30,audit=audit)
    report=dict(complete=False,warning='Untrained new generative process: timing feasibility ONLY, not quality-equivalent RDD acceleration',
        ordinary_packing_countertest='Unresolved: larger temporal patch packs both coefficients in one token; strict RDD-exclusive gate NOT passed',
        training_requirements='Train conditional clean-DC/noised-detail transport; train band-type embeddings and parent-memory conditioning. This timing pilot has neither.',
        model=str(weights),full_shape=shape,dtype='bf16',NFE=3,timesteps=[1000,757,522],
        VSA_keep=.2,VSA_coarse_branch=True,all_30_layers_live=True,hidden_dimension=1536,
        candidate_live_temporal=[shape[2]//4,shape[2]//4,shape[2]//2],
        baseline_live_temporal=[shape[2]//4,shape[2]//2,shape[2]],
        memory_contract='Readonly earlier-generating per-layer KV, NOT current clean-parent KV',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rectangular_checks=checks,
        physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],gpu=torch.cuda.get_device_name(),seed=a.seed,prompt=a.prompt,
        paired_groups=[],reductions={})
    if packing:
        report.update(warning='Untrained temporal patch2 hardware counter-screen; NOT a quality-equivalent acceleration',
            ordinary_packing_countertest='Measured without parent reuse: all input coordinates retained, both outputs freshly predicted',
            packing=packing.audit, candidate_live_temporal=[shape[2]//8,shape[2]//4,shape[2]//2],
            candidate= candidate, memory_contract='No cache, fresh QKV at every stage',
            training_requirements='New temporal embedding/head must be trained; quality potential not established')
    from fastvideo.attention.backends.video_sparse_attn import VideoSparseAttentionMetadataBuilder
    from fastvideo.forward_context import set_forward_context
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
    test_shape=(shape[2]//4,shape[3],shape[4])
    test_input=torch.randn((1,16,*test_shape),device=dev,dtype=torch.bfloat16)
    metadata=VideoSparseAttentionMetadataBuilder().build(current_timestep=0,
        raw_latent_shape=test_shape,patch_size=model.patch_size,VSA_sparsity=.8,device=dev)
    results=[]
    for enabled in (False,True):
        memory.enabled=enabled
        memory.native_positions=True
        memory.begin(0,test_shape,shape[2])
        with torch.inference_mode(),set_forward_context(current_timestep=0,
                attn_metadata=metadata,forward_batch=ForwardBatch(data_type='dummy')):
            results.append(model(hidden_states=test_input,encoder_hidden_states=context,
                timestep=torch.full((1,),1000.,device=dev,dtype=torch.float32)))
    torch.testing.assert_close(results[0],results[1],rtol=0,atol=0)
    report['root_native_wrapper_max_abs']=0.
    print('Root wrapper exactness validated; starting warmups',flush=True)
    del results,test_input
    memory.native_positions=False
    memory.cache.clear()
    if a.c2f_siblings:
        from ancestry_sparse_kernel import SharedRouteController,validate_uniform_lift
        report['uniform_lift_reference_checks']=validate_uniform_lift(dev)
        report['kernel_source_sha256']=hashlib.sha256((Path(__file__).parent/'ancestry_sparse_kernel.py').read_bytes()).hexdigest()
        c2f=SharedRouteController(model)
        candidate='c2f_siblings'
        report.update(warning='Actual C2F executor ablation: SAME masks/current coarse gates; no trained-RDD quality claim',
            ordinary_packing_countertest='Shared-route execution can also serve ordinary grouped routers; strict exclusivity unresolved',
            baseline_definition='RDD+C2F native64; separate online_vsa uses ordinary native RDD+VSA',
            candidate_live_temporal=report['baseline_live_temporal'],memory_contract='No stale hidden KV or output cache',
            training_requirements='Executor adds no approximation beyond BF16 rounding; route-guide quality requires separate validation')
    if a.routing_only:
        capture_routes=True
        run('baseline')
        capture_routes=False
        route_report,counts=route_union_audit(route_capture,shape)
        import numpy as np
        payload={f"s{x['stage']}_l{i%30}":np.packbits(x['mask'].numpy().reshape(-1)) for i,x in enumerate(route_capture)}
        np.savez_compressed(out/'packed_masks.npz',**payload)
        (out/'route_union.json').write_text(json.dumps(route_report,indent=2))
        (out/'metrics.json').write_text(json.dumps(dict(complete=True,mode='routing_only',
            shape=shape,counts=counts,warning='CPU byte-union diagnostic, no measured SP speed. C2F slightly denser due ceil inherited child budget.'),indent=2))
        memory.restore()
        kernel_ops.fused_topk_mask=original_mask
        torch.distributed.destroy_process_group()
        print('ROUTING COMPLETE',flush=True)
        return
    # Synthesis orthogonality audit is independent of untrained model outputs.
    x=torch.randn((1,16,4,8,8),device=dev)
    h=torch.randn_like(x)
    y=haar_synthesis(x,h)
    dc=(y[:,:,0::2]+y[:,:,1::2])/math.sqrt(2)
    detail=(y[:,:,0::2]-y[:,:,1::2])/math.sqrt(2)
    torch.testing.assert_close(dc,x,rtol=2e-5,atol=1e-6)
    torch.testing.assert_close(detail,h,rtol=2e-5,atol=1e-6)
    report['haar_fp32_max_abs']=float((dc-x).abs().max())
    for v in ('baseline',candidate):
        for j in range(a.warmups):
            capture_routes=v=='baseline' and j==a.warmups-1
            row=run(v,export=j==a.warmups-1)
            capture_routes=False
        print(json.dumps(dict(warmup=v,row=row)),flush=True)
    if c2f is not None:
        for j in range(a.warmups):
            row=run('online_vsa',export=j==a.warmups-1)
        print(json.dumps(dict(warmup='online_vsa',row=row)),flush=True)
        if a.current_router_control:
            for j in range(a.warmups):
                row=run('current_grouped',export=j==a.warmups-1)
            print(json.dumps(dict(warmup='current_grouped',row=row)),flush=True)
    # Snapshot only owner identities, not another user's command arguments.
    report['gpu_process_snapshot']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader'],text=True)
    report['timing_environment']='Shared server; other GPU processes may compete. Screening timings, not isolated final confidence.'
    reductions={k:[] for k in ('denoise','total')}
    for g in range(a.groups):
        order=['baseline',candidate,candidate,'baseline'] if g%2==0 else [candidate,'baseline','baseline',candidate]
        if c2f is not None:
            order=['online_vsa','baseline',candidate,candidate,'baseline','online_vsa'] if g%2==0 else [candidate,'baseline','online_vsa','online_vsa','baseline',candidate]
            if a.current_router_control:
                order=['online_vsa','baseline',candidate,'current_grouped','current_grouped',candidate,'baseline','online_vsa'] if g%2==0 else ['current_grouped',candidate,'baseline','online_vsa','online_vsa','baseline',candidate,'current_grouped']
        rows=[run(v) for v in order]
        report['paired_groups'].append(rows)
        for k in reductions:
            base=statistics.mean(r[k+'_seconds'] for r in rows if r['variant']=='baseline')
            cand=statistics.mean(r[k+'_seconds'] for r in rows if r['variant']==candidate)
            reductions[k].append(100*(1-cand/base))
        print(json.dumps(dict(group=g,reduction={k:x[-1] for k,x in reductions.items()})),flush=True)
        (out/'progress.json').write_text(json.dumps(report,indent=2))
    report['reductions']={k:group_bootstrap_ci(v) for k,v in reductions.items()}
    report['baseline']={k:aggregate([r[k+'_seconds'] for rows in report['paired_groups'] for r in rows if r['variant']=='baseline']) for k in reductions}
    report['candidate']={k:aggregate([r[k+'_seconds'] for rows in report['paired_groups'] for r in rows if r['variant']==candidate]) for k in reductions}
    report['complete']=True
    if c2f is not None:
        report['online_vsa']={k:aggregate([r[k+'_seconds'] for rows in report['paired_groups'] for r in rows if r['variant']=='online_vsa']) for k in reductions}
        report['vs_rdd_vsa']={k:group_bootstrap_ci([100*(1-statistics.mean(r[k+'_seconds'] for r in rows if r['variant']==candidate)/statistics.mean(r[k+'_seconds'] for r in rows if r['variant']=='online_vsa')) for rows in report['paired_groups']]) for k in reductions}
        if a.current_router_control:
            report['current_grouped']={k:aggregate([r[k+'_seconds'] for rows in report['paired_groups'] for r in rows if r['variant']=='current_grouped']) for k in reductions}
            report['ancestral_vs_current_grouped']={k:group_bootstrap_ci([100*(1-statistics.mean(r[k+'_seconds'] for r in rows if r['variant']==candidate)/statistics.mean(r[k+'_seconds'] for r in rows if r['variant']=='current_grouped')) for rows in report['paired_groups']]) for k in reductions}
            report['ordinary_shared_router_counter']='Current Q64/K64 means are pooled across temporal siblings; fresh Top20% on the smaller graph is lifted. Native CURRENT64 coarse output/gate, all fine QKV/FFN and all30blocks remain. No ancestry KV/mask needed on a fixed-resolution model.'
        c2f.restore()
    # Untimed diagnostic: key union for query-sharded Ring/SP ranks. A head-
    # sharded Ulysses rank owns ALL queries, so use global union separately.
    route_report,counts=route_union_audit(route_capture,shape)
    report['route_union_counts']=counts
    (out/'route_union.json').write_text(json.dumps(route_report,indent=2))
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    memory.restore()
    kernel_ops.fused_topk_mask=original_mask
    torch.distributed.destroy_process_group()
    print('COMPLETE',flush=True)
    faulthandler.cancel_dump_traceback_later()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='/scr/yuheng/DevOpsBench/multiscale_diffusion')
    p.add_argument('--output',required=True)
    p.add_argument('--width',type=int,default=1280)
    p.add_argument('--height',type=int,default=768)
    p.add_argument('--frames',type=int,default=61)
    p.add_argument('--groups',type=int,default=4)
    p.add_argument('--warmups',type=int,default=2)
    p.add_argument('--port',type=int,default=29903)
    p.add_argument('--seed',type=int,default=20261003)
    p.add_argument('--minimum-free-gib',type=float,default=65)
    p.add_argument('--memory-fraction',type=float,default=.65)
    p.add_argument('--routing-only',action='store_true')
    p.add_argument('--counter-patch2',action='store_true')
    p.add_argument('--c2f-siblings',action='store_true')
    p.add_argument('--current-router-control',action='store_true')
    p.add_argument('--prompt',default='A red fox walking through a snowy forest, cinematic natural light.')
    a=p.parse_args()
    if a.groups<4 or a.warmups<2:
        p.error('At least four paired groups and two warmups')
    if a.current_router_control and not a.c2f_siblings:
        p.error('--current-router-control requires --c2f-siblings')
    main(a)
