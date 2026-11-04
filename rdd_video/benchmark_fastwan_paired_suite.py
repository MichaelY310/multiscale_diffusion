"""Run multiple FastWan/RDD configurations in one loaded process.

The old worker started one Python process per method.  That is useful for
standalone videos, but it mixes method latency with shape compilation, model
reload order, and GPU drift.  This runner loads T5/DiT once, warms every shape,
then alternates the method order on successive rounds.  It is the preferred
source for paper-facing latency comparisons.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

from sample_fastwan_native_vsa import (
    add_fastvideo_paths,
    parse_pairs,
    sample_three_call_dmd,
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    location = fraction * (len(ordered) - 1)
    low = int(location)
    high = min(len(ordered) - 1, low + 1)
    weight = location - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def aggregate(values: list[float]) -> dict[str, float]:
    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    return {
        "median": median,
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "p25": percentile(values, 0.25),
        "p75": percentile(values, 0.75),
        "mad": statistics.median(deviations),
        "spread_fraction": (max(values) - min(values)) / median,
    }


def normalize_config(raw: dict[str, Any]) -> dict[str, Any]:
    result = {
        "name": str(raw["name"]),
        "stage_factors": str(raw["stage_factors"]),
        "attention_mode": str(raw.get("attention_mode", "vsa")),
        "vsa_sparsity": float(raw.get("vsa_sparsity", 0.8)),
        "route_density": float(raw.get("route_density", 0.2)),
        "reuse_every": int(raw.get("reuse_every", 1)),
        "reuse_scope": str(raw.get("reuse_scope", "backend")),
        "small_stage_executor": str(raw.get("small_stage_executor", "native")),
        "small_stage_max_padded_tokens": int(
            raw.get("small_stage_max_padded_tokens", 0)
        ),
        "cross_stage_reuse_layers": [
            int(value) for value in raw.get("cross_stage_reuse_layers", [])
        ],
        "cross_stage_reuse_layers_by_stage": {
            str(stage): [int(value) for value in values]
            for stage, values in raw.get(
                "cross_stage_reuse_layers_by_stage", {}
            ).items()
        },
        "cross_stage_reuse_stages": [
            int(value) for value in raw.get("cross_stage_reuse_stages", [])
        ],
        "cross_stage_reuse_scale": float(raw.get("cross_stage_reuse_scale", 1.0)),
        "cross_stage_reuse_scale_by_stage": {
            str(stage): float(value)
            for stage, value in raw.get(
                "cross_stage_reuse_scale_by_stage", {}
            ).items()
        },
        "cross_stage_reuse_layer_coefficients": {
            str(layer): [float(value) for value in values]
            for layer, values in raw.get(
                "cross_stage_reuse_layer_coefficients", {}
            ).items()
        },
        "cross_stage_reuse_precompose": bool(
            raw.get("cross_stage_reuse_precompose", False)
        ),
        "cross_stage_reuse_layer_coefficients_by_stage": {
            str(stage): {
                str(layer): [float(value) for value in values]
                for layer, values in coefficients.items()
            }
            for stage, coefficients in raw.get(
                "cross_stage_reuse_layer_coefficients_by_stage", {}
            ).items()
        },
        "cross_stage_reuse_resize": str(
            raw.get("cross_stage_reuse_resize", "nearest")
        ),
        "cross_stage_branch_reuse": str(
            raw.get("cross_stage_branch_reuse", "none")
        ),
        "cross_stage_branch_reuse_layers": [
            int(value) for value in raw.get("cross_stage_branch_reuse_layers", [])
        ],
        "cross_stage_branch_reuse_stages": [
            int(value) for value in raw.get("cross_stage_branch_reuse_stages", [])
        ],
        "cross_stage_branch_reuse_scale": float(
            raw.get("cross_stage_branch_reuse_scale", 1.0)
        ),
        "boundary_executor": str(raw.get("boundary_executor", "native")),
    }
    return result


def main(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = [
        normalize_config(item)
        for item in json.loads(Path(args.configs_json).read_text(encoding="utf-8"))
    ]
    if len({item["name"] for item in configs}) != len(configs):
        raise ValueError("config names must be unique")
    decode_names = {item for item in args.decode_names.split(",") if item}
    unknown_decode = decode_names - {item["name"] for item in configs}
    if unknown_decode:
        raise ValueError(f"unknown decode names: {sorted(unknown_decode)}")
    latent_only_names = {
        item for item in args.latent_only_names.split(",") if item
    }
    unknown_latent = latent_only_names - {item["name"] for item in configs}
    if unknown_latent:
        raise ValueError(f"unknown latent-only names: {sorted(unknown_latent)}")
    retain_latent_names = decode_names | latent_only_names

    fastvideo_root = (root / args.fastvideo_root).resolve()
    kernel_root = (root / args.fastvideo_kernel_root).resolve()
    add_fastvideo_paths(fastvideo_root, kernel_root)
    from attention_analysis.fastwan_native_attention import (
        FastWanNativeAttentionController,
        install_projected_cache_class_patch,
    )
    from fastvideo.configs.models.dits import WanVideoConfig
    from fastvideo.configs.pipelines.wan import FastWan2_1_T2V_480P_Config
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.models.dits.wanvideo import WanTransformerBlock_VSA
    from fastvideo.models.loader.component_loader import TransformerLoader
    from sample_rdd_wan import save_video
    from wan.configs.wan_t2v_1_3B import t2v_1_3B as wan_config
    from wan.modules.t5 import T5EncoderModel
    from wan.modules.vae import WanVAE

    install_projected_cache_class_patch(WanTransformerBlock_VSA)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(args.master_port))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    width, height = (int(value) for value in args.size.split("x"))
    latent_t = (args.frame_num - 1) // 4 + 1
    if args.frame_num != (latent_t - 1) * 4 + 1:
        raise ValueError("frame_num must equal 4*k+1")
    full_shape = (1, 16, latent_t, height // 8, width // 8)
    timesteps = [int(value) for value in args.dmd_timesteps.split(",")]
    seed = args.seed if args.seed >= 0 else random.randint(0, 2**32 - 1)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    original = (root / args.original_wan_dir).resolve()
    text_encoder = T5EncoderModel(
        text_len=512,
        dtype=torch.float32,
        device=torch.device("cpu"),
        checkpoint_path=str(original / "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=str(original / "google/umt5-xxl"),
    )
    text_encoder.model.to(device)
    context_unpadded = text_encoder([args.prompt], device)[0]
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
    transformer_path = str((root / args.fastwan_dir / "transformer").resolve())
    fastvideo_args = FastVideoArgs(
        model_path=transformer_path,
        num_gpus=1,
        dit_cpu_offload=False,
        # FastVideo has a second, independent offload switch whose default is
        # True.  Keeping it enabled copies every transformer block from pinned
        # host memory for every NFE and masks the compute saved by RDD stages.
        dit_layerwise_offload=args.dit_layerwise_offload,
        pipeline_config=FastWan2_1_T2V_480P_Config(
            dit_config=WanVideoConfig(), dit_precision="bf16"
        ),
    )
    fastvideo_args.device = device
    print(f"Loading one shared transformer: {transformer_path}", flush=True)
    model = TransformerLoader().load(transformer_path, fastvideo_args).to(
        dtype=torch.bfloat16
    ).eval()
    controller = FastWanNativeAttentionController(model)

    records: dict[str, list[dict[str, Any]]] = {
        item["name"]: [] for item in configs
    }
    final_latents: dict[str, torch.Tensor] = {}
    final_attention: dict[str, dict[str, Any]] = {}
    final_shapes: dict[str, list[tuple[int, int, int]]] = {}
    peak_memory: dict[str, int] = {item["name"]: 0 for item in configs}

    def run_one(config: dict[str, Any], keep_latent: bool) -> None:
        controller.configure(
            attention_mode=config["attention_mode"],
            route_density=config["route_density"],
            reuse_every=config["reuse_every"],
            reuse_scope=config["reuse_scope"],
            small_stage_executor=config["small_stage_executor"],
            small_stage_max_padded_tokens=config[
                "small_stage_max_padded_tokens"
            ],
            cross_stage_reuse_layers=config["cross_stage_reuse_layers"],
            cross_stage_reuse_layers_by_stage=config[
                "cross_stage_reuse_layers_by_stage"
            ],
            cross_stage_reuse_stages=config["cross_stage_reuse_stages"],
            cross_stage_reuse_scale=config["cross_stage_reuse_scale"],
            cross_stage_reuse_scale_by_stage=config[
                "cross_stage_reuse_scale_by_stage"
            ],
            cross_stage_reuse_layer_coefficients=config[
                "cross_stage_reuse_layer_coefficients"
            ],
            cross_stage_reuse_precompose=config[
                "cross_stage_reuse_precompose"
            ],
            cross_stage_reuse_layer_coefficients_by_stage=config[
                "cross_stage_reuse_layer_coefficients_by_stage"
            ],
            cross_stage_reuse_resize=config["cross_stage_reuse_resize"],
            cross_stage_branch_reuse=config["cross_stage_branch_reuse"],
            cross_stage_branch_reuse_layers=config[
                "cross_stage_branch_reuse_layers"
            ],
            cross_stage_branch_reuse_stages=config[
                "cross_stage_branch_reuse_stages"
            ],
            cross_stage_branch_reuse_scale=config[
                "cross_stage_branch_reuse_scale"
            ],
        )
        torch.cuda.reset_peak_memory_stats(device)
        wrapper_start = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            (
                latent,
                stage_shapes,
                stage_seconds,
                denoise_seconds,
                breakdown,
            ) = sample_three_call_dmd(
                model=model,
                context=context,
                full_shape=full_shape,
                stage_factors=parse_pairs(config["stage_factors"]),
                timesteps=timesteps,
                seed=seed,
                vsa_sparsity=config["vsa_sparsity"],
                transition_mode=args.transition_mode,
                device=device,
                attention_controller=controller,
                boundary_executor=config["boundary_executor"],
                return_breakdown=True,
            )
        wrapper_seconds = time.perf_counter() - wrapper_start
        peak_memory[config["name"]] = max(
            peak_memory[config["name"]], int(torch.cuda.max_memory_allocated(device))
        )
        records[config["name"]].append(
            {
                "denoise_and_boundary_seconds": float(denoise_seconds),
                "wrapper_seconds": float(wrapper_seconds),
                "stage_forward_seconds": [float(value) for value in stage_seconds],
                "breakdown": breakdown,
            }
        )
        final_attention[config["name"]] = controller.summary()
        final_shapes[config["name"]] = stage_shapes
        if keep_latent:
            final_latents[config["name"]] = latent.detach().cpu()

    # Every config/shape is warmed before any measurements are retained.
    for warmup_round in range(args.warmup_rounds):
        order = configs if warmup_round % 2 == 0 else list(reversed(configs))
        for config in order:
            print(
                f"warmup={warmup_round + 1}/{args.warmup_rounds} "
                f"config={config['name']}",
                flush=True,
            )
            run_one(config, keep_latent=False)
            records[config["name"]].clear()

    for timed_round in range(args.timed_rounds):
        order = configs if timed_round % 2 == 0 else list(reversed(configs))
        for config in order:
            print(
                f"timed={timed_round + 1}/{args.timed_rounds} "
                f"config={config['name']}",
                flush=True,
            )
            run_one(
                config,
                keep_latent=(
                    config["name"] in retain_latent_names
                    and timed_round == args.timed_rounds - 1
                ),
            )

    summary: dict[str, Any] = {}
    for config in configs:
        name = config["name"]
        denoise = [item["denoise_and_boundary_seconds"] for item in records[name]]
        wrapper = [item["wrapper_seconds"] for item in records[name]]
        stage_columns = list(zip(*(item["stage_forward_seconds"] for item in records[name])))
        breakdown_summary = {}
        if records[name]:
            for component in records[name][0]["breakdown"]:
                columns = list(
                    zip(*(item["breakdown"][component] for item in records[name]))
                )
                breakdown_summary[component] = [
                    aggregate(list(column)) for column in columns
                ]
        summary[name] = {
            "config": config,
            "stage_latent_shapes": final_shapes[name],
            "attention": final_attention[name],
            "denoise_and_boundary_seconds": aggregate(denoise),
            "wrapper_seconds": aggregate(wrapper),
            "stage_forward_seconds": [aggregate(list(column)) for column in stage_columns],
            "stage_breakdown": breakdown_summary,
            "peak_cuda_memory_bytes": peak_memory[name],
            "records": records[name],
        }

    baseline = summary[args.baseline_name]["denoise_and_boundary_seconds"]["median"]
    for row in summary.values():
        row["denoise_speedup_vs_baseline"] = (
            baseline / row["denoise_and_boundary_seconds"]["median"]
        )

    profile_names = {item for item in args.profile_names.split(",") if item}
    unknown_profiles = profile_names - {item["name"] for item in configs}
    if unknown_profiles:
        raise ValueError(f"unknown profile names: {sorted(unknown_profiles)}")
    for name in sorted(profile_names):
        config = next(item for item in configs if item["name"] == name)
        print(f"profiling config={name}", flush=True)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as profile:
            run_one(config, keep_latent=False)
        # The profiled invocation is intentionally excluded from timing stats.
        records[name].pop()
        (output_dir / f"{name}.profiler.txt").write_text(
            profile.key_averages(group_by_input_shape=True).table(
                sort_by="self_cuda_time_total", row_limit=120
            ),
            encoding="utf-8",
        )
        profile.export_chrome_trace(str(output_dir / f"{name}.trace.json"))

    # Decode only requested visual representatives.  Decode time is reported
    # separately and never included in the denoiser number above.
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if decode_names:
        vae = WanVAE(
            z_dim=16,
            vae_pth=str(original / "Wan2.1_VAE.pth"),
            dtype=torch.float32,
            device=device,
        )
        # Warm the VAE at this exact output shape once.
        first_latent = final_latents[next(iter(sorted(decode_names)))]
        vae.decode([first_latent[0].float().to(device)])
        torch.cuda.synchronize(device)
        for name in sorted(decode_names):
            latent = final_latents[name]
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            video = vae.decode([latent[0].float().to(device)])[0]
            torch.cuda.synchronize(device)
            vae_seconds = time.perf_counter() - start
            video_path = output_dir / f"{name}.mp4"
            save_video(video, str(video_path), fps=args.fps)
            summary[name]["video"] = str(video_path)
            summary[name]["vae_decode_seconds"] = vae_seconds
            summary[name]["sample_plus_vae_seconds"] = (
                summary[name]["denoise_and_boundary_seconds"]["median"]
                + vae_seconds
            )

    baseline_total = summary[args.baseline_name].get("sample_plus_vae_seconds")
    if baseline_total is not None:
        for row in summary.values():
            total = row.get("sample_plus_vae_seconds")
            if total is not None:
                row["sample_plus_vae_speedup_vs_baseline"] = baseline_total / total

    # Compare every decoded candidate with the decoded baseline.  This is not a
    # perceptual quality metric, but it is a cheap, deterministic guard against
    # a training-free approximation silently producing an unrelated latent.
    latent_comparisons: dict[str, dict[str, float]] = {}
    baseline_latent = final_latents.get(args.baseline_name)
    if baseline_latent is not None:
        baseline_float = baseline_latent.float()
        baseline_flat = baseline_float.reshape(-1)
        baseline_rms = torch.sqrt(torch.mean(baseline_flat.square())).item()
        for candidate_name, candidate_latent in final_latents.items():
            if candidate_name == args.baseline_name:
                continue
            candidate_float = candidate_latent.float()
            difference = candidate_float - baseline_float
            rmse = torch.sqrt(torch.mean(difference.square())).item()
            cosine = torch.nn.functional.cosine_similarity(
                baseline_flat.unsqueeze(0),
                candidate_float.reshape(1, -1),
                dim=1,
            ).item()
            latent_comparisons[
                f"{candidate_name}_vs_{args.baseline_name}"
            ] = {
                "max_abs": difference.abs().max().item(),
                "mean_abs": difference.abs().mean().item(),
                "rmse": rmse,
                "relative_rmse": rmse / max(baseline_rms, 1e-12),
                "cosine": cosine,
            }


    # Keep the old exact-kernel twin check for backward compatibility.
    for optimized_name, optimized_latent in final_latents.items():
        candidates = []
        if "_q32" in optimized_name:
            candidates.append(optimized_name.replace("_q32", ""))
        if optimized_name.endswith("_fused"):
            candidates.append(optimized_name.removesuffix("_fused"))
        if optimized_name.endswith("_nearest"):
            candidates.append(optimized_name.removesuffix("_nearest"))
        for native_name in candidates:
            native_latent = final_latents.get(native_name)
            if native_latent is None:
                continue
            difference = (optimized_latent.float() - native_latent.float()).abs()
            latent_comparisons[f"{optimized_name}_vs_{native_name}"] = {
                "max_abs": float(difference.max().item()),
                "mean_abs": float(difference.mean().item()),
                "rmse": float(difference.square().mean().sqrt().item()),
            }

    payload = {
        "protocol": {
            "single_loaded_process": True,
            "alternating_order": True,
            "warmup_rounds": args.warmup_rounds,
            "timed_rounds": args.timed_rounds,
            "aggregation": "median with raw records, IQR and MAD",
            "device": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__,
            "dit_cpu_offload": False,
            "dit_layerwise_offload": bool(args.dit_layerwise_offload),
            "resident_dit": not bool(args.dit_layerwise_offload),
        },
        "prompt": args.prompt,
        "seed": seed,
        "size": args.size,
        "frame_num": args.frame_num,
        "dmd_timesteps": timesteps,
        "transition_mode": args.transition_mode,
        "baseline_name": args.baseline_name,
        "latent_comparisons": latent_comparisons,
        "results": summary,
    }
    output = output_dir / "paired_suite_metrics.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--fastvideo-root", default="third_party/FastVideo-ref")
    parser.add_argument(
        "--fastvideo-kernel-root", default="third_party/FastVideo-ref/fastvideo-kernel"
    )
    parser.add_argument(
        "--fastwan-dir", default="pretrained_models/FastWan2.1-T2V-1.3B-Diffusers"
    )
    parser.add_argument("--original-wan-dir", default="pretrained_models/Wan2.1-T2V-1.3B")
    parser.add_argument("--configs-json", required=True)
    parser.add_argument("--baseline-name", default="fixed_vsa20")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--size", default="832x448")
    parser.add_argument("--frame-num", type=int, default=29)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--dmd-timesteps", default="1000,757,522")
    parser.add_argument("--transition-mode", default="fresh")
    parser.add_argument("--warmup-rounds", type=int, default=3)
    parser.add_argument("--timed-rounds", type=int, default=20)
    parser.add_argument("--decode-names", default="")
    parser.add_argument(
        "--latent-only-names",
        default="",
        help="comma-separated configs retained for latent comparisons but not VAE decode",
    )
    parser.add_argument(
        "--profile-names",
        default="",
        help="comma-separated configs to profile once after clean timing",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--master-port", type=int, default=29721)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dit-layerwise-offload",
        action="store_true",
        help=(
            "opt into FastVideo's per-layer CPU offload; the paper-facing "
            "default keeps the 1.3B DiT resident on GPU"
        ),
    )
    main(parser.parse_args())
