"""Resolution-labelled experiment metadata. No torch import or numerical changes.

Legacy CLI conventions differ: DDPM lengths are fine-first, FM lengths are
coarse-first, and sampling steps are always coarse-first. Never parse a compact
filename (908080, 808090, 300300400) as a configuration.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

SCHEMA = "rdd_resolution_identity_v1"


def integers(value):
    if isinstance(value, str):
        value = re.split(r"[,/]", value)
    return [int(v) for v in value]


def describe(family, resolutions_fine_first, lengths, steps, solver):
    """Return explicit grid-to-clock and grid-to-sampling mappings."""
    resolutions = integers(resolutions_fine_first)
    lengths, steps = integers(lengths), integers(steps)
    if not (len(resolutions) == len(lengths) == len(steps)):
        raise ValueError("One length and one sampling budget per resolution required")
    if resolutions != sorted(set(resolutions), reverse=True):
        raise ValueError("Legacy resolutions must be unique and fine-to-coarse")
    if min(resolutions + lengths + steps) <= 0:
        raise ValueError("Resolutions, lengths and budgets must be positive")
    family = family.lower()
    if family not in ("fm", "ddpm"):
        raise ValueError(f"Unknown clock convention: {family}")
    coarse = list(reversed(resolutions))
    durations = lengths if family == "fm" else list(reversed(lengths))
    total = sum(lengths)
    train = []
    cursor = 0
    solver = solver.lower()
    if family == "fm" and solver == "dpm2":
        solver = "midpoint"  # historical FM alias, not DPM-Solver
    for res, duration in zip(coarse, durations):
        start, end = ((cursor / total, (cursor + duration) / total)
                      if family == "fm" else
                      (total - cursor - 1, total - cursor - duration))
        train.append(dict(latent_resolution=res, clock="fm_noise_to_data" if family == "fm" else "ddpm_t",
                          start=start, end=end, duration=duration))
        cursor += duration
    sample = []
    for i, (res, n) in enumerate(zip(coarse, steps)):
        if family == "fm" and solver in ("euler", "midpoint"):
            nfe = n * (2 if solver == "midpoint" else 1) + (i < len(coarse) - 1)
        elif family == "ddpm" and solver == "dpm2" and n >= 2:
            nfe = 2 * n - 2
        elif family == "ddpm" and solver in ("ddpm", "ddim", "dpm", "dpm1"):
            nfe = n
        else:
            nfe = None
        sample.append(dict(latent_resolution=res, steps=n,
                           unit="ode_updates" if family == "fm" else "respaced_timesteps",
                           expected_native_nfe=nfe))
    return dict(schema=SCHEMA, family=family, solver=solver, training_stages=train,
                sampling_stages=sample,
                expected_native_nfe=sum(x["expected_native_nfe"] for x in sample)
                if all(x["expected_native_nfe"] is not None for x in sample) else None)


def display(stages, field="steps"):
    return " · ".join(f"{s['latent_resolution']}×{s['latent_resolution']}: {s[field]}" for s in stages)


def tag(stages, field="steps", prefix="sample"):
    return prefix + "_" + "_".join(f"r{s['latent_resolution']}-{s[field]}" for s in stages)


def legacy_identity(config):
    family = "fm" if config["model_type"] == "fm_unified" else "ddpm"
    result = describe(family, config["resolutions"], config["stage_lengths"],
                      config["num_steps"], config.get("sampler", "ddpm"))
    result["boundary_label"] = config.get("upsample_mode", "noise_filling" if family == "fm" else "fresh")
    result["cfg"] = config.get("cfg_scale", 1.)
    declared = config.get("resolution_identity")
    if declared is not None and declared != result:
        raise ValueError("resolution_identity disagrees with the actual sampler config; regenerate it")
    return result


def write_identity(directory, identity):
    """Prevent silently resuming a directory under a different grid allocation."""
    path = Path(directory) / "resolution_identity.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(identity, ensure_ascii=False, indent=2) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8") as out:
        temporary = Path(out.name)
        out.write(payload)
    try:
        try:
            os.link(temporary, path)  # publish complete bytes atomically across ranks
        except FileExistsError:
            if json.loads(path.read_text()) != identity:
                raise ValueError(f"Refusing mixed resolution configurations in {directory}")
    finally:
        temporary.unlink()
