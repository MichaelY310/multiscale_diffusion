#!/usr/bin/env python3
"""Reproducible metadata analysis; no torch/model loading or new experiments."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean


def summarize(root):
    arms = {arm:root / f"eval_results/csra_condition_{arm}_20260912" for arm in ("v2","v3")}
    seeds = [20260921,20260922,20260923,20275001,20275002,20275003,20275004]
    rows, assets, matched_hashes = [], [], 0
    for seed in seeds:
        pair = {}
        for arm,path in arms.items():
            directory = path / f"seed_{seed}"
            metric = json.loads((directory / "metrics.json").read_text())
            assert metric["adapter_train_step"] == 800 and metric["layers"] == [13,15,17,20]
            assert metric["stage_steps"] == [80,80,90] and metric["batch_size"] == 4
            assert metric["cfg_scale"] == 1 and metric["upsample"] == "noise_filling"
            assert metric["adapter_precision"] == "fp32"
            pair[arm] = metric["settings"]["csra_all6"]["latent_relative_l2_vs_dense"]
            for setting in metric["settings"]:
                assets.extend(directory / setting / f"{i:03d}.png" for i in range(4))
            assets.append(directory / "comparison_grid.png")
        rows.append({"seed":seed,**pair})
        for i in range(4):
            files = [path/f"seed_{seed}"/"dense"/f"{i:03d}.png" for path in arms.values()]
            if all(f.exists() and f.stat().st_size for f in files):
                assert hashlib.sha256(files[0].read_bytes()).digest() == hashlib.sha256(files[1].read_bytes()).digest()
                matched_hashes += 1
    files = sorted((root / "eval_results/csra_layer_repair_20260913").glob("shard_*/metrics.json"))
    assert len(files) == 4
    probes = [json.loads(p.read_text()) for p in files]
    assert all(p["complete"] and p["baseline_parity_max_abs"] == 0 and p["dense_replay_max_abs"] == 0 for p in probes)
    observations = [r for p in probes for r in p["rows"]]
    assert sum(p["images"] for p in probes) == 128
    def means(items):
        # Every record contains the same eight images; mean is also state-weighted.
        assert len({r["images"] for r in items}) == 1
        values = {k:mean(r["predictions"][k]["main_mse"] for r in items)
                  for k in items[0]["predictions"]}
        return {k:{"main_mse":v,"relative_to_all4":v/values["adapter_all4"]} for k,v in values.items()}
    return {"condition_rows":rows,
            "condition_means":{arm:{"original3":mean(r[arm] for r in rows[:3]),
                                    "development4":mean(r[arm] for r in rows[3:]),
                                    "all7":mean(r[arm] for r in rows)} for arm in arms},
            "matched_local_dense_hashes":matched_hashes,
            "missing_local_assets":[str(p.relative_to(root)) for p in assets if not p.exists() or not p.stat().st_size],
            "repair_images":sum(p["images"] for p in probes),
            "repair_states":sum(p["states"] for p in probes),
            "repair_overall":means(observations),
            "repair_by_t":{str(t):means([r for r in observations if r["t"]==t])
                           for t in sorted({r["t"] for r in observations},reverse=True)},
            "caveat":"Read-only teacher-state counterfactuals, not end-to-end quality/speed or causal proof."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    result = summarize(args.root)
    path = args.root / "eval_results/csra_layer_repair_20260913/summary.json"
    path.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("repair_by_t","missing_local_assets")},indent=2))
    print("Missing local assets:",len(result["missing_local_assets"]))
