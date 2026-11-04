"""Official AR sampler plumbing with synthetic text; latent output, no quality claim."""
import argparse
import json
import sys
import time
from pathlib import Path
from pyramid_smoke import PIN, check_source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--psd-noise-compat", action="store_true")
    a = p.parse_args()
    source, model, output = a.source.resolve(), a.model.resolve(), a.output.resolve()
    hashes = check_source(source)
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(source))
    import torch
    from pyramid_dit import PyramidDiTForVideoGeneration
    torch.manual_seed(42)
    metrics = {"source_commit": PIN, "source_hashes": hashes,
               "conditioning": "synthetic cached text and zero negative embedding",
               "quality_evaluation": False, "output_type": "latent", "vae_loaded": False,
               "precision": "BF16 weights and autocast", "offload": False,
               "sp": 1, "tp": 1, "frames_latent": 4, "height": 384, "width": 640,
               "steps_first_unit": [2, 2, 2], "steps_video_unit": [2, 2, 2],
               "guidance_scale": 7.0, "video_guidance_scale": 7.0,
               "psd_noise_compat": a.psd_noise_compat,
               "dit_calls": [], "boundary_noise_calls": []}
    start = time.monotonic()
    try:
        runner = PyramidDiTForVideoGeneration(str(model), model_dtype="bf16", model_name="pyramid_flux",
                    model_variant="diffusion_transformer_384p", load_text_encoder=False, load_vae=False,
                    use_flash_attn=False, max_temporal_length=4, frame_per_unit=1)
        runner.dit.to("cuda").eval()
        cache = torch.load(output.parent / "fixtures/text_0.pt", map_location="cpu")
        def synthetic_encoder(prompt, device):
            negative = prompt == ""
            return (cache["prompt_embed"].to(device, torch.bfloat16) * (0 if negative else 1),
                    cache["prompt_attention_mask"].to(device),
                    cache["pooled_prompt_embed"].to(device, torch.bfloat16) * (0 if negative else 1))
        runner.text_encoder = synthetic_encoder
        def count(module, pos, kw):
            sample = kw["sample"][0]
            metrics["dit_calls"].append({"history_and_current_shapes": [list(t.shape) for t in sample],
                                         "branch_batch": sample[-1].shape[0]})
        runner.dit.register_forward_pre_hook(count, with_kwargs=True)
        noise = runner.sample_block_noise
        if a.psd_noise_compat:
            from pyramid_compat import sample_psd_block_noise
            noise = lambda *pos: sample_psd_block_noise(runner.scheduler.config.gamma, *pos)
            metrics["compatibility_changes"] = ["exact PSD covariance factor; same Gaussian law, different RNG mapping"]
        def boundary(*pos, **kw):
            result = noise(*pos, **kw)
            metrics["boundary_noise_calls"].append(list(result.shape))
            return result
        runner.sample_block_noise = boundary
        torch.cuda.synchronize()
        sample_start = time.monotonic()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            latent = runner.generate(prompt="synthetic plumbing fixture", negative_prompt="",
                       height=384, width=640, temp=4, num_inference_steps=[2, 2, 2],
                       video_num_inference_steps=[2, 2, 2], output_type="latent",
                       generator=torch.Generator(device="cuda").manual_seed(42))
        torch.cuda.synchronize()
        metrics["sampling_seconds"] = time.monotonic() - sample_start
        metrics["actual_nfe_including_boundary"] = len(metrics["dit_calls"])
        metrics["actual_branch_forwards"] = sum(c["branch_batch"] for c in metrics["dit_calls"])
        metrics["boundary_dit_calls"] = 0  # official transition uses corrective noise, no model call
        metrics["latent_shape"] = list(latent.shape)
        metrics["finite"] = bool(torch.isfinite(latent).all())
        assert metrics["actual_nfe_including_boundary"] == 24
        assert metrics["actual_branch_forwards"] == 48
        assert len(metrics["boundary_noise_calls"]) == 8
        assert metrics["finite"] and list(latent.shape) == [1, 16, 4, 48, 80]
        torch.save(latent.cpu(), output / "synthetic_sample_latent.pt")
        metrics["status"] = "passed"
    except BaseException as exc:
        metrics.update(status="failed", error=repr(exc))
        raise
    finally:
        metrics["wall_seconds"] = time.monotonic() - start
        metrics["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        metrics["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
        check_source(source)


if __name__ == "__main__":
    main()
