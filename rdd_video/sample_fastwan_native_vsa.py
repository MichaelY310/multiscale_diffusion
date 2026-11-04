"""Run FastWan with FastVideo's native Wan/VSA model implementation.

This file deliberately separates two questions:

* ``fullres`` is the released FastWan three-call DMD trajectory.  Every call
  uses one fixed latent grid.
* ``faux-rdd`` is a counterfactual in which the same three calls are assigned
  to nested coarse/middle/fine grids.  It tests plumbing only: the released
  checkpoint was not trained for the RDD vector field.

Unlike ``sample_fastwan_faux_rdd.py``, this loader instantiates
``WanTransformerBlock_VSA`` and therefore consumes the checkpoint's learned
``to_gate_compress`` parameters instead of silently dropping them.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Wan2.1"))


def parse_pairs(text: str) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for item in text.split(";"):
        spatial, temporal = (int(value) for value in item.split(","))
        result.append((spatial, temporal))
    return result


def upsample_nested(value: torch.Tensor, target: tuple[int, int, int]) -> torch.Tensor:
    target_t, target_h, target_w = target
    ratio_t = math.ceil(target_t / value.shape[2])
    ratio_h = math.ceil(target_h / value.shape[3])
    ratio_w = math.ceil(target_w / value.shape[4])
    value = value.repeat_interleave(ratio_t, 2)
    value = value.repeat_interleave(ratio_h, 3).repeat_interleave(ratio_w, 4)
    return value[:, :, :target_t, :target_h, :target_w]


def upsample_nested_torch_nearest(
    value: torch.Tensor, target: tuple[int, int, int]
) -> torch.Tensor:
    """Single-kernel exact nested upsample for integer RDD stage ratios."""

    import torch.nn.functional as functional

    source = value.shape[2:]
    if any(target_dim % source_dim for source_dim, target_dim in zip(source, target)):
        raise ValueError(
            "torch_nearest boundary executor requires exact nested grids, got "
            f"{tuple(source)} -> {target}"
        )
    # CUDA autocast may choose fp32 for interpolate even when the input state is
    # bf16.  The historical repeat_interleave path preserves the input dtype,
    # and FastVideo asserts that latent/context dtypes match at the next call.
    return functional.interpolate(value, size=target, mode="nearest").to(value.dtype)


def lift_noise_with_nullspace(
    coarse_noise: torch.Tensor,
    target: tuple[int, int, int],
    generator: torch.Generator,
) -> torch.Tensor:
    """Isometrically lift coarse Gaussian noise and fill new fine dimensions.

    For an exact ``rt x rh x rw`` nested grid, nearest-neighbour repetition
    divided by ``sqrt(rt*rh*rw)`` is an isometry.  Adding the component of a
    fresh fine Gaussian orthogonal to that repeated subspace makes the result
    a standard Gaussian on the fine grid while preserving the coarse
    innovation direction.  This is the 3-D analogue of RDD's 2-D
    noise-filling transition.
    """
    target_t, target_h, target_w = target
    source_t, source_h, source_w = coarse_noise.shape[2:]
    if target_t % source_t or target_h % source_h or target_w % source_w:
        raise ValueError(
            "noise-filling requires exact nested grids, got "
            f"{(source_t, source_h, source_w)} -> {target}"
        )
    ratio_t = target_t // source_t
    ratio_h = target_h // source_h
    ratio_w = target_w // source_w
    block_volume = ratio_t * ratio_h * ratio_w

    lifted = coarse_noise.repeat_interleave(ratio_t, 2)
    lifted = lifted.repeat_interleave(ratio_h, 3).repeat_interleave(ratio_w, 4)
    lifted = lifted / math.sqrt(float(block_volume))

    fresh = torch.randn(
        lifted.shape,
        generator=generator,
        device=lifted.device,
        dtype=lifted.dtype,
    )
    batch, channels = fresh.shape[:2]
    pooled = fresh.reshape(
        batch,
        channels,
        source_t,
        ratio_t,
        source_h,
        ratio_h,
        source_w,
        ratio_w,
    ).mean(dim=(3, 5, 7))
    projected = pooled.repeat_interleave(ratio_t, 2)
    projected = projected.repeat_interleave(ratio_h, 3).repeat_interleave(ratio_w, 4)
    return lifted + (fresh - projected)


def scheduler_sigma(scheduler, timestep: int, device: torch.device) -> torch.Tensor:
    """Return the exact shifted FM sigma used by FastVideo for ``timestep``."""
    timesteps = scheduler.timesteps.to(device=device, dtype=torch.float64)
    sigmas = scheduler.sigmas.to(device=device, dtype=torch.float64)
    index = torch.argmin(torch.abs(timesteps - float(timestep)))
    return sigmas[index]


def add_fastvideo_paths(fastvideo_root: Path, kernel_root: Path) -> None:
    # FastVideo chooses the block class while importing wanvideo.py, so this
    # environment variable must be set before the first fastvideo import.
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN"
    sys.path.insert(0, str(fastvideo_root))
    # The source checkout keeps the importable package under
    # fastvideo-kernel/python/fastvideo_kernel.  Also accept an already
    # installable/importable ``python`` directory for packaged environments.
    kernel_python = kernel_root / "python"
    bundled_kernel_python = fastvideo_root / "fastvideo-kernel" / "python"
    if (kernel_python / "fastvideo_kernel" / "__init__.py").is_file():
        kernel_root = kernel_python
    elif (bundled_kernel_python / "fastvideo_kernel" / "__init__.py").is_file():
        # Prefer FastVideo's matching vendored kernel over an incomplete
        # standalone checkout containing only raw Triton source files.
        kernel_root = bundled_kernel_python
    sys.path.insert(0, str(kernel_root))


@torch.inference_mode()
def sample_three_call_dmd(
    model,
    context: torch.Tensor,
    full_shape: tuple[int, int, int, int, int],
    stage_factors: list[tuple[int, int]],
    timesteps: list[int],
    seed: int,
    vsa_sparsity: float,
    transition_mode: str,
    device: torch.device,
    attention_controller=None,
    boundary_executor: str = "native",
    return_breakdown: bool = False,
    stage_output_callback=None,
):
    from fastvideo.attention.backends.video_sparse_attn import VideoSparseAttentionMetadataBuilder
    from fastvideo.forward_context import set_forward_context
    from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    from fastvideo.models.utils import pred_noise_to_pred_video
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

    if len(stage_factors) != len(timesteps):
        raise ValueError("one stage factor is required for every DMD timestep")
    generator = torch.Generator(device=device).manual_seed(seed)
    batch, channels, full_t, full_h, full_w = full_shape
    clean = None
    previous_state = None
    previous_sigma = None
    stage_shapes: list[tuple[int, int, int]] = []
    stage_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    prepare_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    reconstruct_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    prepare_cpu_seconds: list[float] = []
    reconstruct_cpu_seconds: list[float] = []
    stage_log_rows: list[tuple[int, int, int, tuple[int, int, int], float]] = []
    builder = VideoSparseAttentionMetadataBuilder()
    forward_batch = ForwardBatch(data_type="dummy")
    # DmdDenoisingStage constructs this exact scheduler independently of the
    # pipeline's ordinary inference schedule.
    scheduler = FlowMatchEulerDiscreteScheduler(shift=8.0)
    start = time.perf_counter()

    for stage, ((spatial, temporal), timestep) in enumerate(zip(stage_factors, timesteps)):
        prepare_cpu_start = time.perf_counter()
        prepare_start_event = torch.cuda.Event(enable_timing=True)
        prepare_end_event = torch.cuda.Event(enable_timing=True)
        prepare_start_event.record()
        shape = (
            math.ceil(full_t / temporal),
            math.ceil(full_h / spatial),
            math.ceil(full_w / spatial),
        )
        # Wan patch size is (1,2,2), therefore spatial latent dimensions must
        # remain even at every nested stage.
        if shape[1] % 2 or shape[2] % 2:
            raise ValueError(f"stage latent H/W must be even, got {shape}")
        stage_shapes.append(shape)
        if clean is None:
            state = torch.randn(
                (batch, channels, *shape),
                generator=generator,
                device=device,
                dtype=torch.bfloat16,
            )
        else:
            sigma_next = scheduler_sigma(scheduler, timestep, device).to(clean.dtype)
            # Canonical fresh: upsample clean then draw a new full-space Gaussian.
            # Old video noise_filling labels are migrated to fresh explicitly.
            if transition_mode in {"fresh", "refresh", "redraw", "legacy_video_noise_filling"}:
                if boundary_executor == "triton_fused":
                    from attention_analysis.rdd_noise_filling_triton import (
                        fused_noise_filling_transition,
                    )

                    state = fused_noise_filling_transition(
                        clean, shape, sigma_next, generator
                    )
                elif boundary_executor == "native":
                    projected_clean = upsample_nested(clean, shape)
                    noise = torch.randn(
                        projected_clean.shape,
                        generator=generator,
                        device=device,
                        dtype=projected_clean.dtype,
                    )
                    # FastVideo's shifted FM convention is
                    # x_sigma=(1-sigma)*x_clean+sigma*epsilon.
                    state = (1.0 - sigma_next) * projected_clean + sigma_next * noise
                elif boundary_executor == "torch_nearest":
                    projected_clean = upsample_nested_torch_nearest(clean, shape)
                    noise = torch.randn(
                        projected_clean.shape,
                        generator=generator,
                        device=device,
                        dtype=projected_clean.dtype,
                    )
                    state = (1.0 - sigma_next) * projected_clean + sigma_next * noise
                else:
                    raise ValueError(f"unknown boundary executor: {boundary_executor}")
            elif transition_mode in {"noise_filling", "innovation_preserving"}:
                projected_clean = upsample_nested(clean, shape)
                if previous_state is None or previous_sigma is None:
                    raise RuntimeError(
                        "missing previous state for innovation-preserving transition"
                    )
                # The current model call defines clean = state - sigma*v.
                # Therefore this recovers the innovation paired with that
                # clean estimate exactly, even when the estimate itself is
                # imperfect.
                coarse_noise = (
                    previous_state - (1.0 - previous_sigma) * clean
                ) / previous_sigma
                noise = lift_noise_with_nullspace(coarse_noise, shape, generator)
                state = (1.0 - sigma_next) * projected_clean + sigma_next * noise
            else:
                raise ValueError(f"unknown transition mode: {transition_mode}")

        metadata_sparsity = vsa_sparsity
        if (
            attention_controller is not None
            and attention_controller.attention_mode.startswith("c2f_route")
        ):
            # Stage 0 seeds C2F with the native VSA router at the requested
            # route density. Later stages replace that online router.
            metadata_sparsity = 1.0 - attention_controller.route_density
        metadata = builder.build(
            current_timestep=stage,
            raw_latent_shape=shape,
            patch_size=tuple(int(value) for value in model.patch_size),
            VSA_sparsity=metadata_sparsity,
            device=device,
        )
        time_input = torch.full((batch,), float(timestep), device=device, dtype=torch.float32)
        if attention_controller is not None:
            attention_controller.begin_model_forward(stage, shape)
        prepare_end_event.record()
        prepare_cpu_seconds.append(time.perf_counter() - prepare_cpu_start)
        stage_start_event = torch.cuda.Event(enable_timing=True)
        stage_end_event = torch.cuda.Event(enable_timing=True)
        stage_start_event.record()
        with set_forward_context(
            current_timestep=stage,
            attn_metadata=metadata,
            forward_batch=forward_batch,
        ):
            prediction = model(
                hidden_states=state,
                encoder_hidden_states=context,
                timestep=time_input,
            )
        stage_end_event.record()
        reconstruct_cpu_start = time.perf_counter()
        reconstruct_start_event = torch.cuda.Event(enable_timing=True)
        reconstruct_end_event = torch.cuda.Event(enable_timing=True)
        reconstruct_start_event.record()
        state_btchw = state.permute(0, 2, 1, 3, 4)
        prediction_btchw = prediction.permute(0, 2, 1, 3, 4)
        clean_btchw = pred_noise_to_pred_video(
            pred_noise=prediction_btchw.flatten(0, 1),
            noise_input_latent=state_btchw.flatten(0, 1),
            timestep=time_input.to(torch.long),
            scheduler=scheduler,
        ).unflatten(0, prediction_btchw.shape[:2])
        clean = clean_btchw.permute(0, 2, 1, 3, 4)
        # Optional async consumers may read a completed stage prediction.
        # Ordinary RDD does NOT promise that its pooled final output equals
        # this prediction; a committed-parent process requires new training.
        if stage_output_callback is not None:
            stage_output_callback(stage, clean)
        previous_state = state
        previous_sigma = scheduler_sigma(scheduler, timestep, device).to(state.dtype)
        reconstruct_end_event.record()
        reconstruct_cpu_seconds.append(time.perf_counter() - reconstruct_cpu_start)
        prepare_events.append((prepare_start_event, prepare_end_event))
        stage_events.append((stage_start_event, stage_end_event))
        reconstruct_events.append((reconstruct_start_event, reconstruct_end_event))
        stage_log_rows.append((stage, spatial, temporal, shape, metadata_sparsity))

    assert clean is not None
    # A single end-of-trajectory synchronization is sufficient: all stage,
    # transition and x0-reconstruction kernels run on the same CUDA stream.
    # The old six host barriers (before/after every stage) were measurement
    # overhead, not part of the RDD algorithm, and especially hurt tiny stages.
    torch.cuda.synchronize(device)
    stage_seconds = [
        float(begin.elapsed_time(end)) / 1000.0 for begin, end in stage_events
    ]
    breakdown = {
        "stage_prepare_cuda_seconds": [
            float(begin.elapsed_time(end)) / 1000.0 for begin, end in prepare_events
        ],
        "stage_model_cuda_seconds": stage_seconds,
        "stage_reconstruct_cuda_seconds": [
            float(begin.elapsed_time(end)) / 1000.0
            for begin, end in reconstruct_events
        ],
        "stage_prepare_cpu_enqueue_seconds": prepare_cpu_seconds,
        "stage_reconstruct_cpu_enqueue_seconds": reconstruct_cpu_seconds,
    }
    for row, elapsed in zip(stage_log_rows, stage_seconds):
        stage, spatial, temporal, shape, metadata_sparsity = row
        print(
            f"stage={stage} factor=({spatial},{temporal}) timestep={timesteps[stage]} "
            f"shape={shape} transition={transition_mode} "
            f"attention={getattr(attention_controller, 'attention_mode', 'vsa')} "
            f"VSA_sparsity={metadata_sparsity:.3f} seconds={elapsed:.3f}",
            flush=True,
        )
    result = (clean, stage_shapes, stage_seconds, time.perf_counter() - start)
    return (*result, breakdown) if return_breakdown else result


def main(args: argparse.Namespace) -> None:
    fastvideo_root = Path(args.fastvideo_root).resolve()
    kernel_root = Path(args.fastvideo_kernel_root).resolve()
    add_fastvideo_paths(fastvideo_root, kernel_root)

    from fastvideo.configs.models.dits import WanVideoConfig
    from fastvideo.configs.pipelines.wan import FastWan2_1_T2V_480P_Config
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.models.loader.component_loader import TransformerLoader
    from fastvideo.models.dits.wanvideo import WanTransformerBlock_VSA
    from sample_rdd_wan import save_video
    from wan.configs.wan_t2v_1_3B import t2v_1_3B as wan_config
    from wan.modules.t5 import T5EncoderModel
    from wan.modules.vae import WanVAE
    from attention_analysis.fastwan_native_attention import (
        FastWanNativeAttentionController,
        install_projected_cache_class_patch,
    )

    # Must happen before TransformerLoader constructs/manages the blocks; see
    # the helper's parameter-materialization contract.
    install_projected_cache_class_patch(WanTransformerBlock_VSA)

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29617")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    seed = args.seed if args.seed >= 0 else random.randint(0, 2**32 - 1)
    width, height = (int(value) for value in args.size.split("x"))
    latent_t = (args.frame_num - 1) // 4 + 1
    if args.frame_num != (latent_t - 1) * 4 + 1:
        raise ValueError("frame_num must equal 4*k+1 for the Wan VAE")
    full_shape = (1, 16, latent_t, height // 8, width // 8)
    stage_factors = parse_pairs(args.stage_factors)
    timesteps = [int(value) for value in args.dmd_timesteps.split(",")]

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats(device)

    original = Path(args.original_wan_dir)
    text_encoder = T5EncoderModel(
        text_len=512,
        # FastVideo's Wan pipeline keeps T5 in fp32 and only casts the
        # postprocessed embeddings before the transformer call.
        dtype=torch.float32,
        device=torch.device("cpu"),
        checkpoint_path=str(original / "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=str(original / "google/umt5-xxl"),
    )
    text_encoder.model.to(device)
    context_unpadded = text_encoder([args.prompt], device)[0]
    # FastVideo's t5_postprocess_text always pads each prompt to 512 tokens.
    # The custom Wan cross-attention has no context_lens mask, so passing only
    # the unpadded tokens is not equivalent and breaks the distilled model.
    if context_unpadded.shape[0] > 512:
        raise ValueError(f"prompt produced {context_unpadded.shape[0]} tokens, expected <=512")
    context = torch.zeros(
        (1, 512, context_unpadded.shape[-1]),
        device=device,
        dtype=torch.bfloat16,
    )
    context[0, : context_unpadded.shape[0]] = context_unpadded.to(torch.bfloat16)
    text_encoder.model.cpu()
    del text_encoder
    torch.cuda.empty_cache()

    maybe_init_distributed_environment_and_model_parallel(1, 1)
    transformer_path = str(Path(args.fastwan_dir).resolve() / "transformer")
    fastvideo_args = FastVideoArgs(
        model_path=transformer_path,
        num_gpus=1,
        dit_cpu_offload=False,
        # This is separate from dit_cpu_offload and defaults to True inside
        # FastVideo.  Avoid layer-by-layer H2D copies when the model fits.
        dit_layerwise_offload=args.dit_layerwise_offload,
        pipeline_config=FastWan2_1_T2V_480P_Config(dit_config=WanVideoConfig(), dit_precision="bf16"),
    )
    fastvideo_args.device = device
    print(f"Loading native FastVideo Wan/VSA transformer from {transformer_path}", flush=True)
    model = TransformerLoader().load(transformer_path, fastvideo_args).to(dtype=torch.bfloat16).eval()
    first_block = model.blocks[0]
    if first_block.__class__.__name__ != "WanTransformerBlock_VSA":
        raise RuntimeError(f"expected WanTransformerBlock_VSA, got {first_block.__class__.__name__}")
    if not hasattr(first_block, "to_gate_compress"):
        raise RuntimeError("native VSA gate weights were not instantiated")
    print("Verified native WanTransformerBlock_VSA and to_gate_compress weights", flush=True)

    attention_controller = FastWanNativeAttentionController(model)
    attention_controller.configure(
        attention_mode=args.attention_mode,
        route_density=args.route_density,
        reuse_every=args.reuse_every,
        reuse_scope=args.reuse_scope,
        small_stage_executor=args.small_stage_executor,
        small_stage_max_padded_tokens=args.small_stage_max_padded_tokens,
    )
    print(
        f"Attention mode={args.attention_mode}; route_density={args.route_density}; "
        f"StepReuse-Out(k)={args.reuse_every}; reuse_scope={args.reuse_scope}; "
        f"small_stage_executor={args.small_stage_executor}; "
        f"small_stage_max_padded_tokens={args.small_stage_max_padded_tokens}",
        flush=True,
    )

    def run_sampler_once():
        # Keep compiled CUDA/Triton kernels warm, but never leak routes or
        # attention outputs from one generated video into the next one.
        attention_controller.reset()
        start = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result = sample_three_call_dmd(
                model=model,
                context=context,
                full_shape=full_shape,
                stage_factors=stage_factors,
                timesteps=timesteps,
                seed=seed,
                vsa_sparsity=args.vsa_sparsity,
                transition_mode=args.transition_mode,
                device=device,
                attention_controller=attention_controller,
                boundary_executor=args.boundary_executor,
            )
        return result, time.perf_counter() - start

    for warmup_id in range(args.warmup_runs):
        print(f"Warm-up trajectory {warmup_id + 1}/{args.warmup_runs}", flush=True)
        run_sampler_once()

    timed_records = []
    for timed_id in range(args.timed_runs):
        print(f"Timed trajectory {timed_id + 1}/{args.timed_runs}", flush=True)
        result, elapsed = run_sampler_once()
        latent, stage_shapes, stage_seconds, denoise_seconds = result
        timed_records.append(
            {
                "denoise_and_boundary_seconds": float(denoise_seconds),
                "sample_wrapper_seconds": float(elapsed),
                "stage_forward_seconds": [float(value) for value in stage_seconds],
            }
        )

    denoise_seconds = statistics.median(
        record["denoise_and_boundary_seconds"] for record in timed_records
    )
    wrapper_seconds = statistics.median(
        record["sample_wrapper_seconds"] for record in timed_records
    )
    attention_summary = attention_controller.summary()
    peak_before_decode = int(torch.cuda.max_memory_allocated(device))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    latent_path = Path(args.latent_output or output.with_suffix(".latent.pt"))
    torch.save(latent.detach().cpu(), latent_path)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    vae = WanVAE(
        z_dim=16,
        vae_pth=str(original / "Wan2.1_VAE.pth"),
        dtype=torch.float32,
        device=device,
    )
    torch.cuda.synchronize(device)
    decode_start = time.perf_counter()
    video = vae.decode([latent[0].float()])[0]
    torch.cuda.synchronize(device)
    decode_seconds = time.perf_counter() - decode_start
    save_video(video, str(output), fps=args.fps)

    metrics = {
        "warning": "FastWan is DMD/VSA-trained, not RDD-trained; faux-RDD is a plumbing sanity test.",
        "loader": "FastVideo TransformerLoader / WanTransformerBlock_VSA",
        "dit_cpu_offload": False,
        "dit_layerwise_offload": bool(args.dit_layerwise_offload),
        "resident_dit": not bool(args.dit_layerwise_offload),
        "gate_weights_consumed": True,
        "output": str(output.resolve()),
        "latent_output": str(latent_path.resolve()),
        "size": args.size,
        "frame_num": args.frame_num,
        "prompt": args.prompt,
        "seed": seed,
        "dmd_timesteps": timesteps,
        "stage_factors_spatial_temporal": stage_factors,
        "stage_latent_shapes": stage_shapes,
        "transition_mode": args.transition_mode,
        "boundary_executor": args.boundary_executor,
        "VSA_sparsity": args.vsa_sparsity,
        "attention_pair_keep_fraction_nominal": (
            args.route_density
            if args.attention_mode.startswith("c2f_route")
            else 1.0 - args.vsa_sparsity
        ),
        "attention": attention_summary,
        "stage_forward_seconds": stage_seconds,
        "denoise_and_boundary_seconds": denoise_seconds,
        "sample_wrapper_seconds": wrapper_seconds,
        "steady_state_benchmark": {
            "warmup_runs": args.warmup_runs,
            "timed_runs": args.timed_runs,
            "aggregation": "median",
            "records": timed_records,
        },
        "vae_decode_seconds": decode_seconds,
        "peak_cuda_memory_before_decode_bytes": peak_before_decode,
    }
    metrics_path = Path(args.metrics_json or output.with_suffix(".metrics.json"))
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fastvideo-root", required=True)
    parser.add_argument("--fastvideo-kernel-root", required=True)
    parser.add_argument("--fastwan-dir", required=True)
    parser.add_argument("--original-wan-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--size", default="832x448")
    parser.add_argument("--frame-num", type=int, default=29)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Complete unmeasured trajectories used to warm shape-specific kernels",
    )
    parser.add_argument(
        "--timed-runs",
        type=int,
        default=1,
        help="Complete trajectories to time; report their median latency",
    )
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--dmd-timesteps", default="1000,757,522")
    parser.add_argument("--stage-factors", default="1,1;1,1;1,1")
    parser.add_argument("--vsa-sparsity", type=float, default=0.8)
    parser.add_argument(
        "--attention-mode",
        choices=(
            "vsa",
            "c2f_route",
            "c2f_route_only",
            "c2f_route_reuse_coarse",
        ),
        default="vsa",
        help=(
            "vsa uses FastVideo's online router; c2f_route keeps its coarse "
            "output but transfers routes; c2f_route_only executes only the "
            "transferred sparse branch after the first RDD stage; "
            "c2f_route_reuse_coarse also reuses the seed-stage block "
            "probabilities with current fine-stage values"
        ),
    )
    parser.add_argument(
        "--route-density",
        type=float,
        default=0.2,
        help="Nominal block-pair fraction retained by C2F-Route",
    )
    parser.add_argument(
        "--reuse-every",
        type=int,
        default=1,
        help="StepReuse-Out(k): recompute self-attention output once every k calls at the same shape",
    )
    parser.add_argument(
        "--reuse-scope",
        choices=("backend", "projected"),
        default="backend",
        help=(
            "backend preserves old experiments; projected caches the post-to_out "
            "self-attention tensor and skips norm/QKV/gate/RoPE/tiling/attention/"
            "to_out on reuse hits"
        ),
    )
    parser.add_argument(
        "--transition-mode",
        choices=("fresh", "refresh", "noise_filling", "innovation_preserving", "redraw", "legacy_video_noise_filling"),
        default="fresh",
        help=(
            "fresh/refresh (default) discards the old residual: "
            "fresh full-space Gaussian after clean upsampling; "
            "noise_filling (alias innovation_preserving) recovers old noise and fills only "
            "the new null space; redraw is an alias for fresh"
        ),
    )
    parser.add_argument(
        "--boundary-executor",
        choices=("native", "torch_nearest", "triton_fused"),
        default="native",
        help=(
            "native preserves the historical repeat_interleave implementation; "
            "torch_nearest replaces three repeat_interleave launches with one exact "
            "5-D nearest kernel; triton_fused is experimental because arithmetic "
            "fusion may change BF16 rounding"
        ),
    )
    parser.add_argument(
        "--small-stage-executor",
        choices=(
            "native",
            "masked_sdpa",
            "dense_sdpa",
            "triton_q16",
            "triton_q32",
        ),
        default="native",
        help=(
            "executor at or below the padded-token threshold; masked_sdpa exactly "
            "preserves the sparse route, dense_sdpa is a training-free dense "
            "coarse-stage fastpath, and triton_q16/q32 preserve the exact 64-token "
            "VSA route while exposing more query work tiles to the GPU"
        ),
    )
    parser.add_argument(
        "--small-stage-max-padded-tokens",
        type=int,
        default=0,
        help="0 disables the small-stage override",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dit-layerwise-offload",
        action="store_true",
        help="opt into FastVideo layerwise CPU offload (resident GPU is default)",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--latent-output")
    parser.add_argument("--metrics-json")
    parsed = parser.parse_args()
    if parsed.warmup_runs < 0:
        parser.error("--warmup-runs must be >= 0")
    if parsed.timed_runs < 1:
        parser.error("--timed-runs must be >= 1")
    main(parsed)
