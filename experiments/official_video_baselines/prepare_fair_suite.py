"""Prepare separate native/controlled configs; do not train or download assets."""
import argparse
import json
from pathlib import Path
import sys

import yaml

ROOT = Path("/scr/yuheng/bugswarm/official_video_baselines_20261004")
WORKSPACE = Path(__file__).resolve().parents[3]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wan-model", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    from pyramid_smoke import PIN, check_source
    check_source(ROOT / "reference/Pyramid-Flow")
    rdd = WORKSPACE / "FastVideo-RDD-minimal"
    sys.path.append(str(rdd))
    from rdd.comparison.configure import configurations, fairness_signature
    args.output.mkdir(parents=True, exist_ok=False)
    configs = configurations(str(args.wan_model.resolve()), str(args.data_root / "controlled_wan"),
        str(ROOT / "runs/unified_fair_smoke/controlled"), gpus=1, global_batch=1,
        steps=3, frames=9, height=128, width=256)
    experiments = {}
    for name, cfg in configs.items():
        cfg["method"].update(smoke_all_stages=True, deadline_grace_seconds=5.0)
        cfg["training"]["checkpoint"].update(training_state_checkpointing_steps=0, checkpoints_total_limit=0)
        cfg["training"]["optimizer"]["lr_warmup_steps"] = 0
        filename = f"controlled_{name}.yaml"
        (args.output / filename).write_text(yaml.safe_dump(cfg, sort_keys=False))
        experiments[f"controlled_{name}"] = {"scope": "controlled_spatial_method_on_Wan",
            "config": filename, "native_reproduction": False,
            "author_implementation": False, "full_model_gpu_bridge_passed": False}
    signatures = [fairness_signature(cfg) for cfg in configs.values()]
    assert all(item == signatures[0] for item in signatures)
    native = {"models": {"student": {"_target_": "fastvideo_native_pyramid.NativePyramidModel",
        "source": str(ROOT / "reference/Pyramid-Flow"),
        "init_from": str(ROOT / "assets/pyramid-flow-miniflux"), "trainable": True,
        "annotation": str(args.data_root / "native_pf/train.jsonl"),
        "empty_prompt": str(args.data_root / "native_pf/empty_prompt.pt"),
        "cache_fingerprint": "REPLACE_WITH_AUDITED_NATIVE_CACHE_FINGERPRINT",
        "recipe": {"model_name": "pyramid_flux", "model_variant": "diffusion_transformer_384p",
            "model_dtype": "bf16", "use_mixed_training": True,
            "load_vae": False, "load_text_encoder": False, "sample_ratios": [1, 2, 1],
            "stages": [1, 2, 4], "stage_range": [0, 1/3, 2/3, 1], "scheduler_gamma": 1/3,
            "max_temporal_length": 8, "frame_per_unit": 1, "video_sync_group": 1,
            "use_temporal_causal": True, "use_flash_attn": False,
            "use_gradient_checkpointing": True, "gradient_checkpointing_ratio": 1.0}}},
        "method": {"_target_": "fastvideo_native_pyramid.NativePyramidMethod",
                   "use_temporal_pyramid": True, "native_lr_schedule": "constant_with_warmup", "clip_grad": 1.0},
        "training": {"distributed": {"num_gpus": 1, "sp_size": 1, "tp_size": 1},
            "data": {"data_path": str(args.data_root / "native_pf"), "train_batch_size": 4,
                     "dataloader_num_workers": 0, "seed": 42},
            "optimizer": {"learning_rate": 5e-5, "betas": [0.9, 0.999], "weight_decay": 1e-4,
                          "lr_scheduler": "constant_with_warmup", "lr_warmup_steps": 1},
            "loop": {"max_train_steps": 3, "gradient_accumulation_steps": 1},
            "checkpoint": {"output_dir": str(ROOT / "runs/unified_fair_smoke/native_pf"),
                           "training_state_checkpointing_steps": 0, "checkpoints_total_limit": 0},
            "tracker": {"trackers": ["none"]}, "dit_precision": "bf16"},
        "callbacks": {}}
    (args.output / "native_pyramid_flow.yaml").write_text(yaml.safe_dump(native, sort_keys=False))
    experiments["native_pyramid_flow"] = {"scope": "native_author_code",
        "source_commit": PIN, "config": "native_pyramid_flow.yaml", "native_reproduction": True,
        "cached_features_required": "native VAE/T5/CLIP and actual empty prompt; synthetic is not training readiness",
        "full_model_gpu_bridge_passed": False}
    reference = yaml.safe_load(yaml.safe_dump(native))
    reference["training"]["checkpoint"]["output_dir"] = str(ROOT / "runs/unified_fair_smoke/reference_pf")
    (args.output / "reference_pyramid_flow.yaml").write_text(yaml.safe_dump(reference, sort_keys=False))
    experiments["reference_pyramid_flow"] = {"scope": "native_author_code", "engine": "author_native_main",
        "config": "reference_pyramid_flow.yaml", "source_commit": PIN, "native_reproduction": True,
        "full_model_gpu_bridge_passed": False}
    native_rdd = yaml.safe_load(yaml.safe_dump(configs["rdd"]))
    native_rdd["models"]["student"]["_target_"] = "rdd.finetune.RDDWanModel"
    native_rdd["models"]["student"].pop("comparison")
    native_rdd["method"] = {"_target_": "rdd.finetune.RDDFineTuneMethod",
                            "smoke_all_stages": True, "preview_every": 0, "deadline_grace_seconds": 5.0}
    native_rdd["training"]["checkpoint"]["output_dir"] = str(ROOT / "runs/unified_fair_smoke/native_rdd")
    (args.output / "native_rdd.yaml").write_text(yaml.safe_dump(native_rdd, sort_keys=False))
    experiments["native_rdd"] = {"scope": "original_project_RDD_on_Wan", "config": "native_rdd.yaml",
        "native_reproduction": True, "source": "migrated rdd/finetune.py + unchanged rdd/flow.py",
        "full_model_gpu_bridge_passed": False,
        "controlled_difference": "Original stage-shaped independent noise vs paired full-noise projections; same law, different RNG traces."}
    experiments["native_pyramidal_wan"] = {"scope": "native_author_code", "config": None,
        "status": "blocked_no_author_training_source", "native_reproduction": False}
    manifest = {"schema": 1, "smoke_only": True, "max_updates": 3,
        "fastvideo_pin": "37dd31d5eadfa726e102702ec1ba1cf7c977eaa1", "experiments": experiments,
        "controlled_fairness_signature": signatures[0],
        "native_vs_controlled": "Separate tables. Native PF Flux and controlled Wan have different backbones, caches and recipes.",
        "evaluation_contract": {"paired_raw_video_ids_and_captions": True, "fixed_split": True,
            "prompts_and_seeds": "identical recorded lists", "report_output_frames_resolution_fps": True,
            "actual_nfe_includes_boundaries": True, "count_cfg_branches": True,
            "teacher_cost_in_gpu_hours": True, "quality_requires_decoded_native_video": True,
            "equal_updates_and_equal_measured_gpu_hours": "report both, separately"},
        "ready_for_formal_training": False}
    (args.output / "suite.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"prepared": str(args.output), "ready_for_formal_training": False}))


if __name__ == "__main__":
    main()
