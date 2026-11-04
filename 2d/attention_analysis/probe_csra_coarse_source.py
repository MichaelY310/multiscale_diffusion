#!/usr/bin/env python3
"""Does the trained replacement actually use its sample-aligned coarse source?

Read-only interventions on validation states, NOT a retrained no-coarse control,
FID, deployable variant, or speed benchmark. Original checkpoints are untouched.
"""
import argparse
import json
from pathlib import Path

import torch

from probe_csra_preserved_shift import load_student


def intervene(cache, mode):
    if mode == "aligned":
        return cache
    if mode == "zero":
        return {k: torch.zeros_like(v) for k, v in cache.items()}
    if mode == "other_image":
        if any(v.shape[0] < 2 for v in cache.values()):
            raise ValueError("Need at least two images to replace the source")
        return {k: v.roll(1, 0) for k, v in cache.items()}
    if mode == "shift_parent":
        # Same source image, but wrong parent-cell alignment; preserves values.
        return {k: v.roll(1, 1) for k, v in cache.items()}
    raise ValueError(mode)


@torch.inference_mode()
def main(args):
    out = Path(args.output_dir)
    if (out / "metrics.json").exists():
        return
    device = torch.device("cuda:0")
    if torch.cuda.mem_get_info(device)[0] < 8 * 2**30:
        raise RuntimeError("Need at least 8 GiB free; no other process will be stopped")
    base = Path("checkpoints/RDD-Unified-L/0400000.pt")
    modes = ("aligned", "shift_parent", "other_image", "zero")
    arms = {}
    for arm in ("preserved_frozen", "preserved_joint"):
        student, ck = load_student(base, arm, device)
        rows = []
        for shard in range(4):
            root = Path(args.data_root) / f"shard_{shard}"
            manifest = json.loads((root / "manifest.json").read_text())
            if not manifest["complete"] or manifest["parity_max_abs"] != 0:
                raise ValueError("Incomplete/parity-failed corpus")
            if Path(manifest["base_checkpoint"]).resolve() != base.resolve():
                raise ValueError("Wrong teacher corpus")
            items = [i for i in manifest["files"] if i["split"] == "validation"][:1]
            if len(items) != 1:
                raise ValueError("Missing validation batch")
            item = items[0]
            if Path(item["name"]).name != item["name"]:
                raise ValueError("Unsafe dataset filename")
            ref = torch.load(root / item["name"], map_location="cpu", weights_only=True)
            if ref["split"] != "validation" or ref["layers"] != student.adapter.layers:
                raise ValueError("Wrong validation metadata")
            labels = ref["labels"].to(device)
            cache = {l: v.to(device) for l, v in ref["coarse"].items()}
            for i in range(len(ref["t"])):
                x, t, target = (ref[k][i].to(device) for k in ("x", "t", "target"))
                aligned = None
                rng = torch.cuda.get_rng_state(device)
                for mode in modes:
                    pred = student.stage(x, t, labels, cache=intervene(cache, mode))[0][:, :4].float()
                    if mode == "aligned":
                        aligned = pred
                    delta = pred - target[:, :4].float()
                    rows.append({"shard": shard, "seed": item["seed"], "t": int(t[0]),
                        "mode": mode, "images": len(labels), "elements": delta.numel(),
                        "squared_error": float(delta.square().sum()),
                        "target_energy": float(target[:, :4].float().square().sum()),
                        "change_from_aligned": float((pred-aligned).square().sum()),
                        "aligned_energy": float(aligned.square().sum())})
                if not torch.equal(rng, torch.cuda.get_rng_state(device)):
                    raise RuntimeError("Intervention consumed CUDA RNG")
        summary = {}
        for mode in modes:
            selected = [r for r in rows if r["mode"] == mode]
            err = sum(r["squared_error"] for r in selected)
            power = sum(r["target_energy"] for r in selected)
            summary[mode] = {"states": sum(r["images"] for r in selected),
                "main_mse": err/sum(r["elements"] for r in selected),
                "relative_l2_to_teacher": (err/power)**.5,
                "relative_output_change": (sum(r["change_from_aligned"] for r in selected)
                    /sum(r["aligned_energy"] for r in selected))**.5}
        arms[arm] = {"checkpoint": ck, "summary": summary, "rows": rows}
        print(json.dumps({"arm": arm, "summary": summary}), flush=True)
        del student
        torch.cuda.empty_cache()
    out.mkdir(parents=True, exist_ok=True)
    payload = {"complete": True, "config": vars(args), "arms": arms,
        "max_memory_gib": torch.cuda.max_memory_allocated()/2**30,
        "warning": "32 validation trajectories already used for checkpoint selection, 256 states/arm. Evaluation-time source perturbations only. Cannot establish a trained no-coarse model is worse or claim RDD exclusivity, image quality, or speed."}
    temp = out / "metrics.json.tmp"
    temp.write_text(json.dumps(payload, indent=2))
    temp.replace(out / "metrics.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="eval_results/csra_rollout_data4k_20260912")
    p.add_argument("--output-dir", required=True)
    main(p.parse_args())
