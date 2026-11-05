#!/usr/bin/env python3
"""Canonical names from recorded configurations, never from bare step digits.

The input CSV is an immutable copy of the historical ledger, including its
known column-direction errors. The output separates that provenance from the
corrected per-resolution fields. Does not download or modify model tensors.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rdd_identity import describe, display, integers, tag

OUT = ROOT / "docs/experiment_registry"
EVIDENCE = ROOT / "docs/validation/20261104"
REPO_ID = "yang335/search_agent_hl_to_i18n_transport"
REVISION = "848d8a1eccaab8c1807ab792c4ec403dde25288d"

# (old model label, old HF checkpoint group, explicit family/topology, fine-first
# grids, native CLI lengths). Corrected FM-L labels were fine-first, but the
# actual FM CLI lengths are coarse-first: these are deliberately different.
MODELS = [
 ("Original DiT-XL/2", "DiT-XL-2", "dit-ddpm-xl2", [32], [1000]),
 ("Unified XL 300/300/400", "RDD-Unified-XL", "rdd-ddpm-shared-xl2", [32,16,8], [300,300,400]),
 ("Unified L 300/300/400", "RDD-Unified-L", "rdd-ddpm-shared-l2", [32,16,8], [300,300,400]),
 ("Unified L 400/300/300", "RDD-Unified-L-400-300-300", "rdd-ddpm-shared-l2", [32,16,8], [400,300,300]),
 ("Unified L 100/300/600", "RDD-Unified-L-100-300-600", "rdd-ddpm-shared-l2", [32,16,8], [100,300,600]),
 ("Unified L 600/300/100", "RDD-Unified-L-600-300-100", "rdd-ddpm-shared-l2", [32,16,8], [600,300,100]),
 ("Unified L 16->32 500/500", "RDD-Unified-L-2Stage-500-500", "rdd-ddpm-shared-l2", [32,16], [500,500]),
 ("Unified L 8->32 500/500", "RDD-Unified-L-2Stage-8to32-500-500", "rdd-ddpm-shared-l2", [32,8], [500,500]),
 ("Multistage XL+L+B 300/300/400", "RDD-Multistage-300-300-400", "rdd-ddpm-separate-r32-xl2-r16-l2-r8-b2", [32,16,8], [300,300,400]),
 ("Multistage XL+L+B 100/300/600", "RDD-Multistage-100-300-600", "rdd-ddpm-separate-r32-xl2-r16-l2-r8-b2", [32,16,8], [100,300,600]),
 ("FM Unified XL v2 legacy", "RDD-FM-Unified-XL-v2", "rdd-fm-shared-xl2-v2", [32,16,8], [300,300,400]),
 ("FM Unified L corrected 300/300/400", "RDD-FM-Unified-L-corrected-300-300-400", "rdd-fm-shared-l2-corrected", [32,16,8], [400,300,300]),
 ("FM Unified L corrected 400/300/300", "RDD-FM-Unified-L-corrected-400-300-300", "rdd-fm-shared-l2-corrected", [32,16,8], [300,300,400]),
]

def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")

def writecsv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as out:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(out, keys); w.writeheader(); w.writerows(rows)

def clean(value):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(value)).strip("-")

def main():
    OUT.mkdir(exist_ok=True)
    models = {}
    for old, group, short, grids, lengths in MODELS:
        family = "fm" if "-fm-" in short else "ddpm"
        config = describe(family, grids, lengths, [1]*len(grids), "euler" if family == "fm" else "ddpm")
        name = short + "__" + tag(config["training_stages"], "duration", "train")
        models[old] = dict(legacy_model=old, legacy_hf_group=group, canonical_model=name,
                           family=family, resolutions_fine_first=grids,
                           native_stage_lengths=lengths,
                           training_stages=config["training_stages"],
                           training_display=display(config["training_stages"], "duration"))
    raw_path = OUT / "legacy_formal_results.csv"
    source = list(csv.DictReader(raw_path.open(encoding="utf-8-sig")))
    results, flat = [], []
    for old in source:
        m = models[old["model"]]
        values = integers(old["sampling_steps_fine_middle_coarse"])
        # Exactly these source formats mislabeled coarse-first values as fine-first.
        low_first = old["model"] == "FM Unified XL v2 legacy" or "interval_proportional" in old["source_repo_path"]
        steps = values if low_first else values[::-1]
        solver = "midpoint" if old["sampler"].startswith("Midpoint") else old["sampler"].lower()
        ident = describe(m["family"], m["resolutions_fine_first"], m["native_stage_lengths"], steps, solver)
        assert ident["expected_native_nfe"] == int(old["actual_nfe"]), old["record_id"]
        # Inference-clock ablations must not be mislabeled as the training clock.
        infer = re.search(r"split(\d{3})(\d{3})(\d{3})", old["source_repo_path"])
        ident["inference_stages"] = describe(m["family"], m["resolutions_fine_first"],
            list(map(int,infer.groups())) if infer else m["native_stage_lengths"], steps, solver)["training_stages"]
        boundary = old["boundary_definition_20261104"]
        variant = "__clean-bilinear" if old["transition"] == "bilinear" else ""
        if infer: variant += "__" + tag(ident["inference_stages"], "duration", "infer")
        canonical = (m["canonical_model"] + "__ckpt-" + clean(old["train_step"]) + "__" + solver
            + "__" + tag(ident["sampling_stages"]) + "__cfg-" + clean(old["cfg"])
            + "__" + boundary + variant)
        result = dict(record_id=old["record_id"], canonical_name=canonical, canonical_model=m["canonical_model"],
                      legacy_model=old["model"], legacy_hf_group=m["legacy_hf_group"],
                      train_step=old["train_step"], cfg=old["cfg"], boundary=boundary,
                      clean_upsample="bilinear" if old["transition"]=="bilinear" else "nearest" if m["family"]=="ddpm" and len(steps)>1 else None,
                      identity=ident, sampling_display=display(ident["sampling_stages"]),
                      training_display=m["training_display"],
                      metrics={k:old[k] for k in ("fid","sfid","inception_score","precision","recall")},
                      num_samples=old["num_samples"], source_path=old["source_repo_path"],
                      status="historical_result_not_rerun", source_direction="coarse_to_fine" if low_first else "fine_to_coarse",
                      evidence="Archived structured result + corresponding native config/launcher. CPU stage-loop call audit for main allocations.",
                      original_ledger_row=old)
        results.append(result)
        row={k:result[k] for k in ("record_id","canonical_name","legacy_model","train_step","cfg","boundary","source_path","status")}
        row.update(solver=solver, actual_nfe=ident["expected_native_nfe"], **result["metrics"])
        for t in ident["training_stages"]:
            grid=t["latent_resolution"];row[f"train_r{grid}_length"]=t["duration"];row[f"train_r{grid}_clock"]=f"{t['start']} -> {t['end']}"
        for t in ident["sampling_stages"]:row[f"sample_r{t['latent_resolution']}_steps"]=t["steps"]
        flat.append(row)
    assert len(results)==116 and len({r['record_id'] for r in results})==116
    write(OUT/"models.json", list(models.values()))
    write(OUT/"results.json", dict(schema="rdd_result_registry_v1", source_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(), results=results))
    writecsv(OUT/"results.csv",flat)
    # Server-side moves retain object hashes and old revision download URLs.
    inventory=json.loads((EVIDENCE/"HF_NAMING_INVENTORY.json").read_text())
    groups={m["legacy_hf_group"]:m["canonical_model"] for m in models.values()}
    groups["RDD-Multistage-100-300-600-8x8-continued-from400k"] = groups["RDD-Multistage-100-300-600"] + "__r8-continued-from-400k"
    moves=[]
    for a in inventory["files"]:
        parts=a["path"].split("/")
        if len(parts)>2 and parts[0]=="checkpoints" and parts[1] in groups:
            new="/".join(["checkpoints",groups[parts[1]],*parts[2:]])
            moves.append(dict(old_path=a["path"],new_path=new,size=a["size"],sha256=a["sha256"],blob_id=a["blob_id"],kind="checkpoint"))
    # Standalone one-run metric logs can be renamed exactly; aggregate tables
    # remain raw provenance because they contain several different allocations.
    by_path={}
    for r in results:by_path.setdefault(r["source_path"],[]).append(r)
    assets={a["path"]:a for a in inventory["files"]}
    for path,rows in by_path.items():
        if len(rows)!=1 or not path.endswith(".txt") or path not in assets:continue
        r=rows[0];a=assets[path]
        new="results/"+r["record_id"]+"/"+r["canonical_name"]+".txt"
        moves.append(dict(old_path=path,new_path=new,size=a["size"],sha256=a["sha256"],blob_id=a["blob_id"],kind="metric"))
    for r in results:
        if r['legacy_model']!='FM Unified XL v2 legacy':continue
        leaf='404045_midpoint_cfg15' if r['identity']['solver']=='midpoint' else '808090_euler_cfg15' if float(r['cfg'])==1.5 else '808090_euler_cfg1'
        path=f'eval_results/rddfm_unixl_v2_400k_{leaf}_noisefilling.txt'
        a=assets[path]
        moves.append(dict(old_path=path,new_path='results/'+r['record_id']+'/'+r['canonical_name']+'.txt',size=a['size'],sha256=a['sha256'],blob_id=a['blob_id'],kind='metric'))
    aliasfile=OUT/'repository_aliases.json'
    if aliasfile.exists():
        for row in json.loads(aliasfile.read_text()):
            if row.get('hf_source_kept_as_archive'):continue
            path=row['old_path']
            if path not in assets:continue
            a=assets[path]
            moves.append(dict(old_path=path,new_path='source/legacy/'+row['canonical_path'],size=a['size'],sha256=a['sha256'],blob_id=a['blob_id'],kind='archived_source'))
    assert len({r['new_path'] for r in moves})==len(moves)
    assert all(len(Path(r['new_path']).name.encode())<256 for r in moves)
    planned=dict(repo_id=REPO_ID,source_revision=REVISION,status="prepared_not_applied",operations=moves,
        policy="Move only identified checkpoint groups, single-result logs and identified source entry points. Original revision remains immutable; grouped raw sources retain legacy names. Unidentified records are not assigned guessed configurations.")
    receipt=EVIDENCE/'HF_NAMING_MIGRATION.json'
    if receipt.exists():
        previous=json.loads((OUT/'hf_rename_plan.json').read_text())
        assert previous['operations']==moves, 'Published plan differs; review instead of overwriting'
        planned.update(status=previous['status'],applied_revision=previous['applied_revision'])
    write(OUT/"hf_rename_plan.json",planned)
    print(json.dumps(dict(formal_results=len(results),model_groups=len(models),hf_moves=len(moves))))

if __name__=="__main__":main()
