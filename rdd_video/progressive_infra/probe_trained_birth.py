"""Measure noise-adjusted prediction variability of real RDD-Wan teachers.

This is a conditional diagnostic, NOT a speed/quality benchmark. A base Wan
teacher is a separate control, not a trained ordinary-DMD comparator.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import time


def main(a):
    os.environ.update(RANK="0", LOCAL_RANK="0", WORLD_SIZE="1", MASTER_ADDR="127.0.0.1",
                      MASTER_PORT=str(a.port), FASTVIDEO_ATTENTION_BACKEND="FLASH_ATTN")
    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict, StateDictOptions
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel, get_sp_group, get_world_group
    from fastvideo.train.utils.config import load_run_config
    from rdd.finetune import RDDWanModel
    from fastvideo.dataset.utils import collate_rows_from_parquet_schema
    from fastvideo.dataset.dataloader.schema import pyarrow_schema_t2v
    import pyarrow.parquet as pq

    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    if torch.cuda.mem_get_info()[0] < 40*2**30:
        raise RuntimeError("Need 40 GiB free, never evict an existing process")
    torch.cuda.set_per_process_memory_fraction(.7)
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    assets = Path(a.assets)
    cfg = load_run_config(a.config, overrides=["--training.distributed.num_gpus", "1",
        "--training.distributed.hsdp_shard_dim", "1"])
    model_args = dict(cfg.models["student"])
    model_args.pop("_target_")
    model_args.update(trainable=False, init_from=str(assets/"Wan2.1-T2V-1.3B-Diffusers"))
    model = RDDWanModel(training_config=cfg.training, **model_args)
    model.transformer.eval()
    # Only exact normalization constants are needed; no decoder weights or GPU VAE.
    from types import SimpleNamespace
    vae_config = json.loads((assets/"Wan2.1-T2V-1.3B-Diffusers/vae/config.json").read_text())
    model.vae = SimpleNamespace(latents_mean=vae_config["latents_mean"], latents_std=vae_config["latents_std"])
    if a.checkpoint:
        state = {"roles.student.transformer": get_model_state_dict(model.transformer)}
        dcp.load(state, storage_reader=dcp.FileSystemReader(str(Path(a.checkpoint)/"dcp")))
        set_model_state_dict(model.transformer, model_state_dict=state["roles.student.transformer"],
                             options=StateDictOptions(strict=True))
        del state
    model.world_group, model.sp_group = get_world_group(), get_sp_group()
    model._init_timestep_mechanics()
    model.ensure_negative_conditioning()
    rows = [row for f in sorted((assets/"mixkit/validation").glob("*.parquet"))
            for row in pq.read_table(f).to_pylist()]
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    result = dict(checkpoint=a.checkpoint or "original_non_distilled_Wan", seed=a.seed,
                  draws=a.draws, cfg=5 if a.contract_only else "conditional only; not a sample", records=[],
                  mode="Euler50_parent_contract" if a.contract_only else "conditional_velocity_perturbation",
                  warning="Noise-adjusted variability is not generation quality, sufficiency, or speed evidence.")
    generator = torch.Generator(device=model.device).manual_seed(a.seed)
    with torch.no_grad():
        for row_index in range(min(a.rows, len(rows))):
            raw = collate_rows_from_parquet_schema([rows[row_index]], pyarrow_schema_t2v,
                                                   text_padding_length=512, cfg_rate=0.)
            model.rdd_stage = 2
            prepared = model.prepare_batch(raw, generator=generator, latents_source="data")
            clean = prepared.latents.permute(0, 2, 1, 3, 4).float()
            path = model.rdd_path
            if a.contract_only:
                from rdd.finetune import sample_continuous
                parents = {}
                class TracePath:
                    def __getattr__(self, name):
                        return getattr(path, name)
                    def transition(self, c, full_shape, stage, gen):
                        parents[stage] = c.detach().clone()
                        return path.transition(c, full_shape, stage, gen)
                final, calls = sample_continuous(model, prepared, TracePath(), seed=a.seed)
                parent_final = path.pool(final, 2, 1)
                parent_committed = parents[2]
                record = dict(row=row_index, calls=calls,
                    parent_revision_rmse=float((parent_final-parent_committed).square().mean().sqrt()),
                    parent_revision_relative_norm=float((parent_final-parent_committed).norm()/parent_committed.norm()),
                    parent_reference_rms=float(parent_committed.square().mean().sqrt()),
                    meaning="Current RDD does not enforce an immutable parent; locking it requires new training.")
                result["records"].append(record)
                torch.save(dict(parent=parent_committed.cpu(), final=final.cpu()), out/f"contract_row{row_index}.pt")
                (out/"metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                print(json.dumps(record), flush=True)
                continue
            eps0 = torch.randn(clean.shape, device=clean.device, generator=generator)
            parent_noise = path.project(eps0, 2)
            etas = [torch.randn(clean.shape, device=clean.device, generator=generator) for _ in range(a.draws)]
            for family in ("rdd", "ordinary_input_control"):
                for t in (.7, .71, .75):
                    adjusted, parent_predictions, cast_parents, quantization_floors, errors = [], [], [], [], []
                    t0 = time.perf_counter()
                    for eta in etas:
                        noise = parent_noise + eta-path.project(eta, 2)
                        times = torch.full((1,), (1-t)*1000, device=clean.device)
                        if family == "rdd":
                            x, target = path.noisy_and_target(clean, noise, torch.tensor([t], device=clean.device), 2)
                        else:
                            x, target = t*clean+(1-t)*noise, noise-clean
                        batch = copy.copy(prepared)
                        batch.timesteps, batch.raw_latent_shape = times, x.shape
                        model._build_attention_metadata(batch)
                        batch.attn_metadata_vsa = batch.attn_metadata
                        pred = model.predict_noise(x.permute(0, 2, 1, 3, 4), times, batch,
                            conditional=True, attn_kind="dense").permute(0, 2, 1, 3, 4).float()
                        analytic = (x-path.project(x, 2))/(1-t)
                        adjusted.append((pred-path.project(pred, 2)-analytic).cpu())
                        parent_predictions.append(path.project(pred, 2).cpu())
                        cast = x.to(torch.bfloat16).float()
                        cast_parents.append(path.project(cast, 2).cpu())
                        roundoff = cast-x
                        quantization_floors.append(float(((roundoff-path.project(roundoff, 2))/(1-t)).square().mean().sqrt()))
                        errors.append(float((pred-target).square().mean()))
                    stack = torch.stack(adjusted)
                    mean = stack.mean(0)
                    def std_rms(values):
                        return float(torch.stack(values).var(0, unbiased=True).mean().sqrt())
                    record = dict(row=row_index, family=family, data_time=t,
                        adjusted_prediction_std_rms=std_rms(adjusted),
                        parent_prediction_std_rms=std_rms(parent_predictions),
                        bf16_parent_input_std_rms=std_rms(cast_parents),
                        analytic_noise_bf16_roundoff_rms=sum(quantization_floors)/len(quantization_floors),
                        adjusted_prediction_mean_rms=float(mean.square().mean().sqrt()),
                        velocity_mse_mean=sum(errors)/len(errors), seconds=time.perf_counter()-t0)
                    result["records"].append(record)
                    (out/"metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                    print(json.dumps(record), flush=True)
    result["complete"] = True
    (out/"metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="rdd/wan_finetune_300300400.yaml")
    p.add_argument("--assets", required=True)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--output", required=True)
    p.add_argument("--rows", type=int, default=2)
    p.add_argument("--draws", type=int, default=4)
    p.add_argument("--seed", type=int, default=20261003)
    p.add_argument("--port", type=int, default=29941)
    p.add_argument("--contract-only", action="store_true")
    main(p.parse_args())
