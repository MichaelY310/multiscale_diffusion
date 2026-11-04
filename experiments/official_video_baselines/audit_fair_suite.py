"""CPU/read-only audit of separate native and controlled experiment readiness."""
import argparse
import datetime as dt
import json
from pathlib import Path
import shutil
import socket
from zoneinfo import ZoneInfo

import yaml

from prepare_fair_suite import ROOT
from pyramid_smoke import check_source
from launch_smoke import active_policy, bounded_deadline


def audit(suite_path):
    suite_path = Path(suite_path).resolve()
    suite = json.loads(suite_path.read_text())
    if suite["schema"] != 1 or not suite["smoke_only"]:
        raise ValueError("Only the prepared <=3-update smoke suite is authorized")
    now = dt.datetime.now(dt.timezone.utc)
    la = now.astimezone(ZoneInfo("America/Los_Angeles"))
    free = shutil.disk_usage(ROOT).free
    common = []
    if socket.gethostname().split(".")[0] != "surfi1":
        common.append("tests are restricted to surfi1")
    policy = active_policy()
    try:
        bounded_deadline(now, 60)
    except RuntimeError as error:
        common.append(str(error))
    if free < 8 * 2**30:
        common.append("/scr has less than the task's 8GiB minimum margin")
    hashes = check_source(ROOT / "reference/Pyramid-Flow")
    experiments = {}
    for key, item in suite["experiments"].items():
        reasons = list(common)
        if not item["config"]:
            reasons.append("no author-provided PyramidalWan training code/weights identified")
            experiments[key] = {"scope": item["scope"], "can_launch_now": False, "blockers": reasons}
            continue
        cfg = yaml.safe_load((suite_path.parent / item["config"]).read_text())
        tc = cfg["training"]
        if not 1 <= tc["distributed"]["num_gpus"] <= policy["max_simultaneous_gpus"] or tc["loop"]["max_train_steps"] > 3:
            raise ValueError("Suite config violates GPU/update limits")
        if tc["checkpoint"]["checkpoints_total_limit"] != 0:
            raise ValueError("Suite must not delete old checkpoints")
        if tc["distributed"]["sp_size"] != 1 or tc["distributed"]["tp_size"] != 1:
            raise ValueError("SP/TP parity has not passed")
        model = Path(cfg["models"]["student"]["init_from"])
        native_pf = key in ("native_pyramid_flow", "reference_pyramid_flow")
        components = ["diffusion_transformer_384p"] if native_pf else ["transformer", "vae", "text_encoder"]
        component_files = {}
        for name in components:
            files = sorted(str(p) for p in (model / name).glob("*.safetensors"))
            present = (model / name / "config.json").is_file() and bool(files)
            component_files[name] = {"present": present, "safetensor_files": files}
            if not present:
                reasons.append(f"missing {name} model/config")
        if native_pf:
            student = cfg["models"]["student"]
            for name in ("annotation", "empty_prompt"):
                if not Path(student[name]).is_file():
                    reasons.append(f"missing audited native {name}")
            if student["cache_fingerprint"].startswith("REPLACE_"):
                reasons.append("native cache fingerprint not supplied")
            data_audit = Path(tc["data"]["data_path"]) / "small_data_audit.json"
        else:
            if not list(Path(tc["data"]["data_path"]).glob("*.parquet")):
                reasons.append("missing small native-Wan VAE/text parquet caches")
            data_audit = Path(tc["data"]["data_path"]).parent / "small_data_audit.json"
        if not data_audit.is_file():
            reasons.append("small real-data audit is missing")
        else:
            data = json.loads(data_audit.read_text())
            if data.get("status") != "passed" or data.get("synthetic") is not False:
                reasons.append("small_data_audit is not a passed real-data check")
            if native_pf and data.get("cache_fingerprint") != cfg["models"]["student"]["cache_fingerprint"]:
                reasons.append("native cache fingerprint differs from small-data audit")
        if tc["checkpoint"]["training_state_checkpointing_steps"]:
            minimum = 1972059200 * 4 * 3 if native_pf else 1300000000 * 4 * 3
            if free < 8 * 2**30 + 2 * minimum:
                reasons.append("insufficient space for two complete AdamW checkpoints plus reserve")
        experiments[key] = {"scope": item["scope"], "config": item["config"],
            "components": component_files, "can_launch_now": not reasons, "blockers": reasons,
            "full_model_gpu_bridge_passed": item["full_model_gpu_bridge_passed"]}
    return {"utc": now.isoformat(), "la": la.isoformat(), "host": socket.gethostname(),
        "disk_free_gib": free / 2**30, "gpu_jobs_launched": 0,
        "source_hashes": hashes, "experiments": experiments, "ready_for_formal_training": False,
        "formal_pending": ["all real-data full-model updates", "full native checkpoint/fresh resume",
                           "native full-model FastVideo parity", "decoded sampling", "author PW source"],
        "native_assets_missing_for_encode_decode": [name for name in
            ("causal_video_vae", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2")
            if not (ROOT / "assets/pyramid-flow-miniflux" / name).is_dir()]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = audit(args.suite)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"ready_for_formal_training": False, "disk_free_gib": report["disk_free_gib"],
                      "blocked_experiments": list(report["experiments"])}))


if __name__ == "__main__":
    main()
