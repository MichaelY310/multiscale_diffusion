#!/usr/bin/env python3
"""Read-only, teacher-state diagnostics: which layer fails, and is cache used?

No training, no sampling, no speed claims. Uses ONLY existing validation states.
Restoring a dense block is a diagnostic oracle, not a proposed acceleration.
Cache zero/permutation controls are distribution shifts, not quality estimates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from csra_runtime import predict_adapter
from probe_rdd_vs_dit_block_reuse import manual_forward, low_high_tokens, upsample_tokens
from sample_csra_adapter_preview import RDDRollout, load_adapter
from train_rdd_csra_adapter import load_frozen_teacher


def replacement_callbacks(adapter, coarse, t, *, restore=None, cache_mode="paired"):
    if cache_mode not in {"paired", "zero", "permuted"}:
        raise ValueError("Unknown cache control")
    if restore is not None and restore not in adapter.layers:
        raise ValueError("Restore must name a replaced layer")
    if set(coarse) != set(adapter.layers):
        raise ValueError("Cache mask mismatch")
    result = {}
    for layer in adapter.layers:
        if layer == restore:
            continue
        source = coarse[layer]
        if cache_mode == "zero":
            source = torch.zeros_like(source)
        elif cache_mode == "permuted":
            if source.shape[0] < 2:
                raise ValueError("Permutation requires at least two different samples")
            source = source.roll(1, 0)
        result[layer] = lambda hidden, condition, layer=layer, source=source: predict_adapter(
            adapter.adapters[str(layer)], source, hidden, t, "fp32", condition=condition)
    return result


def prediction_metrics(candidate, target):
    if candidate.shape != target.shape or candidate.shape[1] != 8:
        raise ValueError("Expected matching main4+variance4 predictions")
    candidate, target = candidate.float(), target.float()
    main = (candidate[:, :4] - target[:, :4]).square().flatten(1).mean(1)
    power = target[:, :4].square().flatten(1).mean(1)
    variance = (candidate[:, 4:] - target[:, 4:]).square().flatten(1).mean(1)
    if not all(torch.isfinite(value).all() for value in (main, power, variance)):
        raise ValueError("Nonfinite diagnostic")
    return {"main_mse": float(main.mean()), "variance_mse": float(variance.mean()),
            "main_mse_per_image": main.tolist(), "main_power_per_image": power.tolist()}


def residual_metrics(candidate, target):
    candidate, target = candidate.float(), target.float()
    low_pred, high_pred = low_high_tokens(candidate)
    low_target, high_target = low_high_tokens(target)
    return {"mse": float((candidate-target).square().mean()),
            "target_power": float(target.square().mean()),
            "low_mse": float((low_pred-low_target).square().mean()),
            "low_target_power": float(low_target.square().mean()),
            "high_mse": float((high_pred-high_target).square().mean()),
            "high_target_power": float(high_target.square().mean())}


@torch.inference_mode()
def run(args):
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Never overwrite a diagnostic result")
    root = Path(args.data_root) / f"shard_{args.shard}"
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest["complete"] or not manifest["parity_checked"] or manifest["parity_max_abs"] != 0:
        raise ValueError("Require completed, parity-checked teacher data")
    if Path(manifest["base_checkpoint"]).resolve() != Path(args.base_ckpt).resolve():
        raise ValueError("Wrong frozen teacher")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    teacher = load_frozen_teacher(Path(args.base_ckpt), device, "DiT-L/2", "rdd")
    adapter, info = load_adapter(Path(args.adapter_ckpt), device)
    if adapter.layers != manifest["layers"] or Path(info["base_checkpoint"]).resolve() != Path(args.base_ckpt).resolve():
        raise ValueError("Wrong adapter")
    items = [x for x in manifest["files"] if x["split"] == "validation"]
    if not items:
        raise ValueError("No validation data")
    output.mkdir(parents=True, exist_ok=True)
    rows, parity_checked, images = [], False, 0
    for item in items:
        if Path(item["name"]).name != item["name"]:
            raise ValueError("Unsafe dataset filename")
        reference = torch.load(root / item["name"], map_location="cpu", weights_only=True)
        if reference["split"] != "validation" or reference["trajectory"] != "dense_teacher":
            raise ValueError("This probe cannot use training/student states")
        if reference["layers"] != adapter.layers or reference["seed"] != item["seed"]:
            raise ValueError("Metadata mismatch")
        if not torch.all(reference["source_timestep"] == 300):
            raise ValueError("Wrong cache time")
        y = reference["labels"].to(device)
        coarse = {k: upsample_tokens(v.to(device), 16) for k,v in reference["coarse"].items()}
        images += len(y)
        for index, nfe in enumerate(reference["nfe_indices"].tolist()):
            x, t, target = (reference[k][index].to(device) for k in ("x", "t", "target"))
            if not torch.all(t == t[0]):
                raise ValueError("Each observation must have one timestep")
            dense, traces = manual_forward(teacher, x, t, y, multires=True,
                                           capture_layers=set(adapter.layers))
            if not torch.equal(dense, target):
                raise RuntimeError("Saved dense target failed exact replay")
            variants = [("adapter_all4", None, "paired")]
            variants += [(f"restore_{k}", k, "paired") for k in adapter.layers]
            variants += [("cache_zero", None, "zero"), ("cache_permuted", None, "permuted")]
            results = {}
            for name, restore, mode in variants:
                replacements = replacement_callbacks(adapter, coarse, t, restore=restore, cache_mode=mode)
                candidate, _ = manual_forward(teacher, x, t, y, multires=True,
                                              replacements=replacements, conditioned_replacements=True)
                results[name] = prediction_metrics(candidate, target)
                if name == "adapter_all4" and not parity_checked:
                    ordinary = RDDRollout(teacher, adapter, adapter.layers, "adapter", "fp32")
                    ordinary.coarse = coarse
                    if not torch.equal(candidate, ordinary.forward(x, t, y)):
                        raise RuntimeError("Diagnostic baseline differs from ordinary adapter inference")
                    parity_checked = True
            with torch.autocast(device.type, dtype=torch.bfloat16):
                condition = teacher.t_embedder(t) + teacher.y_embedder(y, False)
            local = {}
            for layer, trace in traces.items():
                predicted = predict_adapter(adapter.adapters[str(layer)], coarse[layer], trace["input"],
                                            t, "fp32", condition=condition)
                # Match the BF16 residual conversion used by real sampling.
                predicted = predicted.to(trace["input"].dtype)
                local[str(layer)] = residual_metrics(predicted, trace["residual"])
            row = {"seed": item["seed"], "images": len(y), "nfe": nfe, "t": int(t[0]),
                   "predictions": results, "teacher_input_local_residuals": local}
            rows.append(row)
            print(json.dumps({"seed":item["seed"], "nfe":nfe, "t":int(t[0]),
                              "main_mse": {k:v["main_mse"] for k,v in results.items()}}), flush=True)
    if not parity_checked or any(p.grad is not None for p in teacher.parameters()):
        raise RuntimeError("Diagnostic parity/frozen-model failure")
    result = {"complete": True, "config": vars(args), "adapter_architecture": adapter.architecture,
              "adapter_train_step": info["step"], "layers": adapter.layers, "images": images,
              "states": sum(x["images"] for x in rows), "baseline_parity_max_abs":0,
              "dense_replay_max_abs":0, "split":"validation", "rows": rows,
              "max_memory_gib":torch.cuda.max_memory_allocated()/2**30}
    temporary = output / "metrics.json.tmp"
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(output / "metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ckpt", required=True)
    parser.add_argument("--adapter-ckpt", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard", type=int, choices=range(4), required=True)
    run(parser.parse_args())
