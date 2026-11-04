"""One guarded launcher for native PF/RDD and controlled FastVideo smokes."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys

import yaml

from audit_fair_suite import audit
from launch_smoke import bounded_deadline, active_policy
from prepare_fair_suite import ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--experiment", required=True)
    p.add_argument("--gpus", default="4")
    p.add_argument("--seconds", type=int, default=240)
    p.add_argument("--guard-status", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    policy = active_policy()
    selected = args.gpus.split(",")
    if not all(x.isdigit() for x in selected) or not 1 <= len(set(selected)) == len(selected) <= policy["max_simultaneous_gpus"]:
        p.error("distinct authorized physical GPU indices required")
    if not set(map(int, selected)).issubset(policy["allowed_physical_gpus"]):
        p.error("only physical GPUs 4,5,6,7 are authorized")
    if args.seconds < 15:
        p.error("deadline must be at least 15 seconds")
    report = audit(args.suite)
    item = report["experiments"][args.experiment]
    print(json.dumps(item, indent=2), flush=True)
    if not item["can_launch_now"]:
        raise SystemExit(2)
    cfg = yaml.safe_load((args.suite.parent / item["config"]).read_text())
    if cfg["training"]["distributed"]["num_gpus"] != len(selected):
        p.error("GPU count differs from config; regenerate the matched suite")
    if args.experiment == "native_pyramid_flow" and len(selected) != 1:
        p.error("native FastVideo multi-rank bridge parity has not passed")
    bounded_deadline(dt.datetime.now(dt.timezone.utc), args.seconds)
    if args.execute and not args.guard_status.resolve().is_relative_to(ROOT):
        p.error("GPU guard/status output must be inside the official test runtime directory")
    helpers = Path(__file__).resolve().parent
    config = (args.suite.parent / item["config"]).resolve()
    worker = "native_pyramid_reference_smoke.py" if args.experiment == "reference_pyramid_flow" else "train_fair_smoke.py"
    command = ["bash", str(helpers / "run_unified_fastvideo_env.sh"), "-m", "torch.distributed.run",
               "--standalone", f"--nproc_per_node={len(selected)}", str(helpers / worker),
               "--config", str(config)]
    from launch_smoke import main as guarded_main
    sys.argv = ["launch_smoke", "--gpus", args.gpus, "--seconds", str(args.seconds),
                "--output", str(args.guard_status.resolve())]
    if args.execute:
        sys.argv.append("--execute")
        os.chdir(ROOT)
    sys.argv += ["--", *command]
    guarded_main()  # independent exact-token/same-UID/pidfd deadline guard


if __name__ == "__main__":
    main()
