"""CPU config/author-CLI parsing and actual blocked-launcher checks."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU entrypoint check requires CUDA_VISIBLE_DEVICES empty")
    from prepare_fair_suite import ROOT
    from audit_fair_suite import audit
    from native_pyramid_reference_smoke import native_argv
    suite = json.loads(args.suite.read_text())
    report = audit(args.suite)
    assert not any(v["can_launch_now"] for v in report["experiments"].values())
    status = ROOT / "runs" / f"blocked_cpu_preflight_{uuid.uuid4().hex}.json"
    checks = {}
    launcher = Path(__file__).with_name("launch_fair_smoke.py")
    for name in suite["experiments"]:
        cmd = [sys.executable, str(launcher), "--suite", str(args.suite), "--experiment", name,
               "--seconds", "290", "--gpus", "1", "--guard-status", str(status), "--execute"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        assert result.returncode == 2 and not status.exists(), (name, result.stderr)
        checks[name] = {"execution_refused": True, "returncode": result.returncode,
                       "blockers": json.loads(result.stdout)["blockers"]}
    for option, value in [("--gpus", "1,2,3"), ("--seconds", "291")]:
        cmd = [sys.executable, str(launcher), "--suite", str(args.suite), "--experiment", "native_pyramid_flow",
               "--guard-status", str(status), option, value]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        assert result.returncode == 2 and not status.exists()
        checks[f"reject_{option}"] = True
    cfg = yaml.safe_load((args.suite.parent / "reference_pyramid_flow.yaml").read_text())
    sys.path.insert(0, str(ROOT / "reference/Pyramid-Flow"))
    from train import train_pyramid_flow as official
    sys.argv = native_argv(cfg)
    parsed = official.get_args()
    assert parsed.iters_per_epoch == 3 and parsed.gradient_accumulation_steps == 1
    assert parsed.opt_beta2 == .999 and parsed.lr == 5e-5 and parsed.weight_decay == 1e-4
    assert parsed.use_temporal_pyramid and not parsed.use_flash_attn and parsed.video_sync_group == 1
    import torch
    assert not torch.cuda.is_initialized()
    args.output.write_text(json.dumps({"status": "passed", "checks": checks,
        "original_author_cli_parse": True, "gpu_used": False, "cuda_initialized": False,
        "full_model_entrypoint_executed": False, "guard_file_created": False}, indent=2) + "\n")
    print(json.dumps({"status": "passed", "blocked_launches": len(suite["experiments"])}))


if __name__ == "__main__":
    main()
