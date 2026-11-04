"""Selected-idle-GPU preflight and exact-token, same-UID deadline supervisor."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import socket
from zoneinfo import ZoneInfo


def bounded_deadline(now, seconds):
    """Never launch at/after 08:00 LA; cap each job at today's cutoff."""
    explicit = os.environ.get("RDD_GPU_CUTOFF_UTC")
    if explicit:
        cutoff = dt.datetime.fromisoformat(explicit)
        if cutoff.tzinfo is None:
            raise ValueError("Absolute GPU cutoff must have a timezone")
        if now >= cutoff:
            raise RuntimeError("GPU use prohibited after the user-authorized absolute cutoff")
        end = min(now + dt.timedelta(seconds=seconds), cutoff)
        if (end - now).total_seconds() < 15:
            raise RuntimeError("Too close to the authorized GPU cutoff")
        return end
    local = now.astimezone(ZoneInfo("America/Los_Angeles"))
    cutoff = local.replace(hour=8, minute=0, second=0, microsecond=0)
    if local >= cutoff:
        raise RuntimeError("GPU use is prohibited at/after 08:00 America/Los_Angeles")
    end = min(now + dt.timedelta(seconds=seconds), cutoff.astimezone(dt.timezone.utc))
    if (end - now).total_seconds() < 15:
        raise RuntimeError("Too close to the 08:00 LA GPU cutoff")
    return end


def active_policy():
    """Do not roll an expired permission forward to another morning."""
    path = Path(__file__).resolve().parents[3] / "CURRENT_RUN_POLICY_20261006.json"
    policy = json.loads(path.read_text())
    if socket.gethostname().split(".")[0] != policy["host"]:
        raise RuntimeError("GPU work is restricted to the authorized host")
    os.environ["RDD_GPU_CUTOFF_UTC"] = policy["gpu_cutoff_utc"]
    return policy


def inventory():
    result = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                                      "--format=csv,noheader,nounits"], text=True)
    return {int(parts[0]): {"uuid": parts[1], "memory_mib": int(parts[2]), "utilization": int(parts[3])}
            for line in result.splitlines() if (parts := [p.strip() for p in line.split(",")])}


def occupants(selected):
    rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                   "--format=csv,noheader,nounits"], text=True)
    found = []
    for row in rows.splitlines():
        gpu, pid = [x.strip() for x in row.split(",")]
        if gpu in selected and pid.isdigit():
            try:
                found.append({"pid": int(pid), "uid": Path(f"/proc/{pid}").stat().st_uid})
            except FileNotFoundError:
                pass
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="4")
    parser.add_argument("--seconds", type=int, default=240)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--min-free-gib", type=float, default=8.0)
    parser.add_argument("--expected-write-gib", type=float, default=0.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    policy = active_policy()
    gpus = [int(x) for x in args.gpus.split(",")]
    if not 1 <= len(set(gpus)) == len(gpus) <= policy["max_simultaneous_gpus"] or args.seconds < 15:
        parser.error("Use distinct authorized GPUs and a deadline of at least 15 seconds")
    if not set(gpus).issubset(policy["allowed_physical_gpus"]):
        parser.error("Only physical GPUs 4,5,6,7 are authorized")
    if args.min_free_gib < 1 or args.expected_write_gib < 0:
        parser.error("Invalid disk budget")
    import shutil
    if shutil.disk_usage("/scr/yuheng/bugswarm").free < (args.min_free_gib + args.expected_write_gib) * 2**30:
        raise RuntimeError("Insufficient disk space for this task's estimated writes and margin")
    end = bounded_deadline(dt.datetime.now(dt.timezone.utc), args.seconds)
    cards = inventory()
    selected = {cards[g]["uuid"] for g in gpus}
    busy = occupants(selected)
    if busy or any(cards[g]["memory_mib"] > 1000 or cards[g]["utilization"] > 5 for g in gpus):
        raise RuntimeError(f"Selected GPUs are occupied; refusing launch: {busy}")
    print(json.dumps({"selected": {g: cards[g] for g in gpus}, "execute": args.execute}), flush=True)
    if not args.execute:
        return
    import deadline_guard as guard
    # Only selected devices matter; other users on unrelated cards are untouched.
    guard.foreign_gpu_users = lambda: [r["pid"] for r in occupants(selected) if r["uid"] != os.getuid()]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    end = min(end, bounded_deadline(dt.datetime.now(dt.timezone.utc), args.seconds))
    token = "official-pyramid-smoke-" + uuid.uuid4().hex
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        parser.error("Specify a command after --")
    sys.argv = ["guard", "--deadline", end.isoformat(), "--token", token,
                "--status", str(args.output), "--disk-root", "/scr/yuheng/bugswarm",
                "--min-free-gib", str(args.min_free_gib), "--grace-seconds", "5", "--check-foreign-gpus", "--", *command]
    guard.main()
    status = json.loads(args.output.read_text())
    if status["remaining_pids"] or status["exit_code"] != 0 or status["reason"] != "completed":
        raise SystemExit(f"Smoke failed or timed out: {status}")


if __name__ == "__main__":
    main()
