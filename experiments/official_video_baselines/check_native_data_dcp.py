"""Native bucketeer/worker state through the same DCP protocol as FastVideo."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile


def worker(args):
    import torch
    import numpy as np
    import torch.distributed.checkpoint as dcp
    from pyramid_smoke import check_source
    check_source(args.source)
    sys.path.insert(0, str(args.source))
    from dataset import LengthGroupedVideoTextDataset
    from pyramid_data_state import build_stateful_native_loader
    from check_native_data_resume import ProbeDataset, seed
    from check_fastvideo_native_pyramid import hashed
    fixture = args.fixture
    manifest = json.loads((fixture / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((fixture / name).read_bytes()).hexdigest() == expected
    seed(42)
    loader = build_stateful_native_loader(
        ProbeDataset(LengthGroupedVideoTextDataset(str(fixture / "annotation.jsonl"), max_frames=5)),
        batch_size=4, max_frames=5, num_workers=args.workers, rank=0, world_size=1,
        epoch=42, dataset_fingerprint=manifest["fingerprint"])
    if not args.raw:
        from pyramid_dcp_state import NativeDCPDataLoader
        loader = NativeDCPDataLoader(loader)
    try:
        if args.mode == "save":
            for _ in range(2):
                next(loader)
            state = loader.loader.state_dict() if not args.raw else loader.state_dict()
            dcp.save({"data": loader}, checkpoint_id=str(args.checkpoint))
            rng = np.random.get_state()
            torch.save({"python": random.getstate(), "numpy": (rng[0], rng[1].tolist(), *rng[2:]),
                        "torch": torch.get_rng_state()}, args.checkpoint.parent / "rng.pt")
            bucket_sizes = {key: len(value) for key, value in state["buckets"].items()}
        else:
            dcp.load({"data": loader}, checkpoint_id=str(args.checkpoint))
            state = loader.loader.state_dict() if not args.raw else loader.state_dict()
            bucket_sizes = {key: len(value) for key, value in state["buckets"].items()}
            rng = torch.load(args.checkpoint.parent / "rng.pt", weights_only=True)
            random.setstate(rng["python"])
            n = rng["numpy"]
            np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
            torch.set_rng_state(rng["torch"])
        trace = []
        for _ in range(8):
            batch = next(loader)
            trace.append({"batch": hashed(batch), "epoch": loader._epoch,
                          "sampler_epoch": loader._dataloader.sampler.epoch,
                          "python": random.random(), "numpy": float(np.random.rand()),
                          "torch": float(torch.rand(())), "cuda": torch.cuda.is_initialized()})
        assert not torch.cuda.is_initialized()
        args.result.write_text(json.dumps({"trace": trace, "bucket_sizes": bucket_sizes,
                                           "workers": args.workers, "pid": os.getpid()}, indent=2) + "\n")
    finally:
        loader.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["save", "restore"])
    p.add_argument("--workers", type=int, choices=[0, 2], default=0)
    p.add_argument("--raw", action="store_true", help="Negative control: expose dynamic buckets directly to DCP")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--result", type=Path)
    args = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU audit requires CUDA_VISIBLE_DEVICES empty")
    if args.mode:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    scratch = Path(tempfile.mkdtemp(prefix="rdd-native-data-dcp-"))
    reports = {}
    for mode in ("save", "restore"):
        cmd = [sys.executable, str(Path(__file__).resolve()), "--source", str(args.source),
               "--fixture", str(args.fixture), "--output", str(args.output), "--mode", mode,
               "--workers", str(args.workers), "--checkpoint", str(scratch / "dcp"),
               "--result", str(args.output / f"{mode}.json")]
        if args.raw:
            cmd.append("--raw")
        with (args.output / f"{mode}.log").open("w") as log:
            done = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=90)
        if done.returncode:
            raise RuntimeError(f"DCP data worker failed: {mode}")
        reports[mode] = json.loads((args.output / f"{mode}.json").read_text())
    checks = {k: reports["save"][k] == reports["restore"][k] for k in ("trace", "bucket_sizes")}
    summary = {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
               "workers": args.workers, "fixture": str(args.fixture), "scratch": str(scratch),
               "gpu_used": False, "model_optimizer_present": False}
    summary["dcp_stable_envelope"] = not args.raw
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    assert all(checks.values()), checks
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
