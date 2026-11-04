#!/usr/bin/env python3
"""Replay existing teacher seeds to recover source inputs, without rewriting data.

Exact middle-cache equality is checked for EVERY file, not just the first.
This is still an offline teacher-trajectory corpus, not on-policy training.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from probe_rdd_vs_dit_block_reuse import manual_forward
from sample_csra_adapter_preview import sample_rdd
from train_rdd_csra_adapter import atomic_save, load_frozen_teacher


class SourceReached(Exception):
    pass


class Collector:
    def __init__(self, model, layers):
        self.model, self.layers = model, layers
        self.sources = {}
        self.cache = None

    def forward(self, x, t, y):
        source = (x.shape[-1] == 8 and bool(torch.all(t == 600))) or (
            x.shape[-1] == 16 and bool(torch.all(t == 300)))
        middle = x.shape[-1] == 16 and source
        output, trace = manual_forward(self.model, x, t, y, multires=True,
                                      capture_layers=set(self.layers) if middle else None)
        if source:
            self.sources[str(x.shape[-1])] = {
                "x": x.cpu().clone(), "t": t.cpu().clone(),
                "labels": y.cpu().clone(), "target": output.cpu().clone()}
        if middle:
            self.cache = {l: trace[l]["residual"].cpu().clone() for l in self.layers}
            raise SourceReached()
        return output


def main(args):
    src = Path(args.data_root) / f"shard_{args.shard}"
    manifest_path = src / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest["complete"] or manifest["parity_max_abs"] != 0:
        raise ValueError("Original corpus incomplete or failed parity")
    out = Path(args.output_root) / src.name
    out.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    completed = out / "manifest.json"
    if completed.exists():
        if json.loads(completed.read_text())["source_manifest_sha256"] != digest:
            raise ValueError("Sidecar does not match source corpus")
        print(f"Already complete: {out}", flush=True)
        return
    device = torch.device("cuda:0")
    model = load_frozen_teacher(Path(manifest["base_checkpoint"]), device, "DiT-L/2", "rdd")
    records = []
    start = time.perf_counter()
    for item in manifest["files"]:
        name = item["name"]
        if Path(name).name != name:
            raise ValueError("Unsafe corpus filename")
        dest = out / name
        if dest.exists():
            data = torch.load(dest, map_location="cpu", weights_only=True)
            if data["seed"] != item["seed"] or data["cache_parity_max_abs"] != 0:
                raise ValueError("Invalid partial sidecar")
        else:
            original = torch.load(src / name, map_location="cpu", weights_only=True)
            collector = Collector(model, manifest["layers"])
            try:
                sample_rdd(collector, item["images"], [80, 80, 90], device, item["seed"])
            except SourceReached:
                pass
            if set(collector.sources) != {"8", "16"}:
                raise RuntimeError("Missing source stage")
            parity = max(float((collector.cache[l] - original["coarse"][l]).abs().max())
                         for l in manifest["layers"])
            if parity != 0 or not torch.equal(collector.sources["16"]["labels"], original["labels"]):
                raise RuntimeError(f"Seed replay mismatch: {name}: {parity}")
            data = {"seed": item["seed"], "sources": collector.sources,
                    "cache_parity_max_abs": parity, "source_manifest_sha256": digest}
            atomic_save(data, dest)
        records.append({**item, "cache_parity_max_abs": 0})
        print(json.dumps({"file": name, "completed": len(records), "total": len(manifest["files"]),
                          "elapsed_s": time.perf_counter()-start,
                          "peak_gib": torch.cuda.max_memory_allocated()/2**30}), flush=True)
    temporary = completed.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"complete": True, "source_manifest_sha256": digest,
                                    "files": records}, indent=2))
    temporary.replace(completed)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--shard", type=int, required=True)
    main(p.parse_args())
