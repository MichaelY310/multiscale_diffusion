#!/usr/bin/env python3
"""Pool four paired-validation probe shards and draw diagnostic curves."""
import argparse
import json
import math
from pathlib import Path


def summarize(root):
    records = []
    seeds = set()
    steps = set()
    for shard in range(4):
        meta = json.loads((root / f"shard_{shard}" / "metrics.json").read_text())
        if not meta["complete"] or meta["shard"] != shard:
            raise ValueError("Missing or incorrect completed shard")
        steps.add(meta["adapter_train_step"])
        for batch in meta["results"]:
            if batch["probe_parity_max_abs"] != 0 or batch["coarse_cache_max_abs"] != 0:
                raise ValueError("Probe/coarse parity failed")
            if batch["seed"] in seeds:
                raise ValueError("Duplicate trajectory seed")
            seeds.add(batch["seed"])
            records.extend(batch["rows"])
    if len(steps) != 1:
        raise ValueError("Mixed checkpoints")
    rows = []
    for nfe in sorted({r["nfe"] for r in records}):
        group = [r for r in records if r["nfe"] == nfe]
        count = sum(r["images"] for r in group)
        if len(group) != len(seeds) or len({r["t"] for r in group}) != 1:
            raise ValueError("Missing/mismatched time points")
        row = {"nfe": nfe, "t": group[0]["t"], "images": count}
        for name in ["teacher_state_main", "student_state_main", "teacher_state_variance",
                     "student_state_variance", "trajectory_state"]:
            values = {key: sum(r[name][key] * r["images"] for r in group) / count
                      for key in ("mse", "target_power")}
            values["relative_l2"] = math.sqrt(values["mse"] / max(values["target_power"], 1e-24))
            row[name] = values
        for name in ["student_main_low_error_power", "student_main_high_error_power"]:
            row[name] = sum(r[name] * r["images"] for r in group) / count
        row["high_error_energy_fraction"] = row["student_main_high_error_power"] / max(row["student_state_main"]["mse"], 1e-24)
        # Low/high spaces have 1/4 and 3/4 of coordinates respectively.
        row["high_vs_low_error_per_dimension"] = (row["student_main_high_error_power"] / .75) / max(row["student_main_low_error_power"] / .25, 1e-24)
        rows.append(row)
    if len(rows) != 8:
        raise ValueError("Expected eight captured times")
    return {"complete": True, "adapter_train_step": steps.pop(), "trajectory_seeds": sorted(seeds),
            "images": rows[0]["images"], "rows": rows,
            "interpretation": "Pooled squared norms, not mean of batch-relative-L2; no FID or speed claim."}


def plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = summary["rows"]
    x = [r["nfe"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    ax = axes[0, 0]
    for key, label in [("teacher_state_main", "Teacher-trajectory inputs"),
                       ("student_state_main", "Student-trajectory inputs")]:
        ax.plot(x, [100*r[key]["relative_l2"] for r in rows], "o-", label=label)
    ax.set_title("Single-call denoising prediction error")
    ax.set_ylabel("Relative L2 (%)"); ax.legend(fontsize=8)
    ax = axes[0, 1]
    ax.plot(x, [100*r["trajectory_state"]["relative_l2"] for r in rows], "o-", color="tab:red")
    ax.set_title("Paired student vs teacher latent drift")
    ax.set_ylabel("Relative L2 (%)")
    ax = axes[1, 0]
    for key, label in [("teacher_state_main", "Teacher inputs"), ("student_state_main", "Student inputs")]:
        ax.plot(x, [r[key]["mse"] for r in rows], "o-", label=label)
    ax.set_title("Absolute error (different target powers)")
    ax.set_ylabel("Prediction MSE"); ax.legend(fontsize=8)
    ax = axes[1, 1]
    ax.plot(x, [r["high_vs_low_error_per_dimension"] for r in rows], "o-", color="tab:purple")
    ax.axhline(1, ls="--", color="grey", label="Equal error per dimension")
    ax.set_title("High / low prediction error per dimension")
    ax.set_ylabel("Power ratio"); ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel("Fine-stage NFE index (0-89)")
        ax.grid(alpha=.2)
    fig.suptitle(f"RDD-L CSRA, actual four layers, checkpoint {summary['adapter_train_step']}\n"
                 f"{summary['images']} held-out paired trajectories; DDPM 80/80/90, CFG 1, noise filling", fontsize=12)
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    summary = summarize(args.root)
    (args.root / "summary.json").write_text(json.dumps(summary, indent=2))
    plot(summary, args.root / "rollout_shift.png")
    print(json.dumps(summary, indent=2))
