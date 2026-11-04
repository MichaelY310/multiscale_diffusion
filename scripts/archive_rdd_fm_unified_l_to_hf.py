#!/usr/bin/env python3
"""Archive the two corrected Unified-L RDD-FM runs, verify, then clean.

The deletion allowlist is intentionally narrow:

* checkpoints/RDD-FM-Unified-L-corrected-300-300-400
* checkpoints/RDD-FM-Unified-L-corrected-400-300-300
* sample images/temporary NPZ files in their matching evaluation roots

DMD, LPIPS, reuse experiments, and every unknown directory are out of scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download


CHECKPOINT_RE = re.compile(r"^(\d{7})\.pt$")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def is_below(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def collect_files(root: Path) -> tuple[list[Path], list[Path], list[Path]]:
    checkpoint_roots = [
        root
        / "checkpoints/RDD-FM-Unified-L-corrected-300-300-400/imagenet256-fm-unified/checkpoints",
        root
        / "checkpoints/RDD-FM-Unified-L-corrected-400-300-300/imagenet256-fm-unified/checkpoints",
    ]
    eval_roots = [
        root / "eval_results/rdd_fm_unified_l_corrected_300300400_convergence",
        root / "eval_results/rdd_fm_unified_l_corrected_400300300_convergence",
    ]

    checkpoints: list[Path] = []
    for checkpoint_root in checkpoint_roots:
        if not checkpoint_root.is_dir():
            raise RuntimeError(f"missing checkpoint directory: {checkpoint_root}")
        for path in sorted(checkpoint_root.glob("*.pt")):
            match = CHECKPOINT_RE.fullmatch(path.name)
            if match and int(match.group(1)) % 50_000 == 0:
                checkpoints.append(path.resolve())

    expected_names = {f"{step:07d}.pt" for step in range(50_000, 400_001, 50_000)}
    for checkpoint_root in checkpoint_roots:
        found = {path.name for path in checkpoints if path.parent == checkpoint_root.resolve()}
        if found != expected_names:
            raise RuntimeError(
                f"checkpoint set mismatch in {checkpoint_root}: "
                f"missing={sorted(expected_names - found)}, extra={sorted(found - expected_names)}"
            )
    if len(checkpoints) != 16:
        raise RuntimeError(f"expected exactly 16 checkpoint files, found {len(checkpoints)}")

    auxiliary: set[Path] = set()

    # Internal trainer logs and timing records, but never launcher locks.
    for run_root in [root / "checkpoints/RDD-FM-Unified-L-corrected-300-300-400",
                     root / "checkpoints/RDD-FM-Unified-L-corrected-400-300-300"]:
        for name in ("log.txt", "timing_log.jsonl"):
            for path in run_root.rglob(name):
                if path.is_file():
                    auxiliary.add(path.resolve())

    # Exact top-level logs associated with these two corrected FM runs.
    log_root = root / "logs"
    log_patterns = (
        "*rdd_fm_unified_l*",
        "*rdd_fm_l*",
    )
    dynamic_runtime_logs = {"archive_rdd_fm_unified_l_to_hf.log"}
    for pattern in log_patterns:
        for path in log_root.glob(pattern):
            if path.is_file() and path.name not in dynamic_runtime_logs:
                auxiliary.add(path.resolve())

    # Evaluation logs/configs/metric text. Exclude concrete generated images,
    # temporary NPZ files, and any Python cache.
    for eval_root in eval_roots:
        if not eval_root.exists():
            continue
        for path in eval_root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(eval_root)
            if "samples" in relative.parts or "npz_tmp" in relative.parts:
                continue
            if "__pycache__" in relative.parts or path.suffix.lower() in IMAGE_SUFFIXES | {".npz"}:
                continue
            auxiliary.add(path.resolve())

    delete_images: list[Path] = []
    delete_npz: list[Path] = []
    for eval_root in eval_roots:
        samples_root = eval_root / "samples"
        npz_root = eval_root / "npz_tmp"
        if samples_root.exists():
            delete_images.extend(
                path.resolve()
                for path in samples_root.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
        if npz_root.exists():
            delete_npz.extend(
                path.resolve()
                for path in npz_root.rglob("*.npz")
                if path.is_file()
            )

    upload_files = sorted(set(checkpoints) | auxiliary)
    return upload_files, checkpoints, sorted(set(delete_images + delete_npz))


def lfs_sha(remote: Any) -> str | None:
    lfs = getattr(remote, "lfs", None)
    if lfs is None:
        return None
    if isinstance(lfs, dict):
        return lfs.get("sha256") or lfs.get("oid")
    return getattr(lfs, "sha256", None) or getattr(lfs, "oid", None)


def remote_path(remote: Any) -> str:
    return str(getattr(remote, "path", None) or getattr(remote, "rfilename", ""))


def verify_remote(
    api: HfApi,
    repo_id: str,
    root: Path,
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    paths = [entry["repo_path"] for entry in entries]
    remote_by_path: dict[str, Any] = {}
    for start in range(0, len(paths), 50):
        batch = paths[start : start + 50]
        infos = api.get_paths_info(repo_id, paths=batch, repo_type="model", revision="main")
        remote_by_path.update({remote_path(info): info for info in infos})

    verification: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="rdd_hf_verify_") as temp_dir:
        for entry in entries:
            repo_path = entry["repo_path"]
            info = remote_by_path.get(repo_path)
            if info is None:
                raise RuntimeError(f"remote file missing after upload: {repo_path}")
            remote_size = int(getattr(info, "size", -1))
            if remote_size != entry["size"]:
                raise RuntimeError(
                    f"remote size mismatch for {repo_path}: {remote_size} != {entry['size']}"
                )
            remote_sha = lfs_sha(info)
            if remote_sha:
                if remote_sha.lower() != entry["sha256"].lower():
                    raise RuntimeError(
                        f"remote LFS SHA-256 mismatch for {repo_path}: "
                        f"{remote_sha} != {entry['sha256']}"
                    )
                method = "remote_lfs_sha256_and_size"
            else:
                # Small Git-backed logs have no LFS metadata. Download those
                # files and compare their full SHA-256 before allowing delete.
                downloaded = Path(
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=repo_path,
                        repo_type="model",
                        revision="main",
                        local_dir=temp_dir,
                        force_download=True,
                    )
                )
                downloaded_sha = sha256_file(downloaded)
                if downloaded_sha != entry["sha256"]:
                    raise RuntimeError(
                        f"downloaded SHA-256 mismatch for {repo_path}: "
                        f"{downloaded_sha} != {entry['sha256']}"
                    )
                method = "redownload_sha256_and_size"
            verification.append(
                {
                    "repo_path": repo_path,
                    "size": remote_size,
                    "sha256": entry["sha256"],
                    "method": method,
                    "verified": True,
                }
            )
    return verification


def remove_empty_children(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        try:
            path.rmdir()
        except OSError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/scr/yuheng/DevOpsBench/multiscale_diffusion"),
    )
    parser.add_argument(
        "--repo-id",
        default="yang335/search_agent_hl_to_i18n_transport",
    )
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--no-delete", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    if root != Path("/scr/yuheng/DevOpsBench/multiscale_diffusion"):
        raise RuntimeError(f"refusing unexpected repository root: {root}")

    upload_files, checkpoints, cleanup_files = collect_files(root)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest_dir = root / "eval_results/upload_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"rdd_fm_unified_l_archive_{timestamp}.json"
    verification_path = manifest_dir / f"rdd_fm_unified_l_archive_{timestamp}_verified.json"

    print(
        f"Collected {len(checkpoints)} checkpoints, "
        f"{len(upload_files) - len(checkpoints)} log/metric files, "
        f"and {len(cleanup_files)} sample intermediates to delete after verification.",
        flush=True,
    )
    if args.inventory_only:
        for path in upload_files:
            print(f"UPLOAD {path.relative_to(root)} {path.stat().st_size}")
        for path in cleanup_files:
            print(f"CLEANUP {path.relative_to(root)} {path.stat().st_size}")
        return

    entries: list[dict[str, Any]] = []
    for index, path in enumerate(upload_files, start=1):
        if not is_below(path, root):
            raise RuntimeError(f"upload path escaped root: {path}")
        print(f"HASH [{index}/{len(upload_files)}] {path.relative_to(root)}", flush=True)
        entries.append(
            {
                "local_path": str(path),
                "repo_path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
                "kind": "checkpoint" if path in checkpoints else "log_or_metric",
            }
        )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_id": args.repo_id,
        "root": str(root),
        "scope": [
            "RDD-FM-Unified-L-corrected-300-300-400",
            "RDD-FM-Unified-L-corrected-400-300-300",
        ],
        "explicitly_excluded": ["DMD", "LPIPS", "reuse", "unknown experiments", "sample images"],
        "entries": entries,
        "cleanup_candidates": [str(path) for path in cleanup_files],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    api = HfApi()
    identity = api.whoami()
    username = identity.get("name") or identity.get("fullname")
    if username != "yang335":
        raise RuntimeError(f"authenticated Hugging Face user is {username!r}, expected 'yang335'")
    api.repo_info(args.repo_id, repo_type="model")
    print(f"Authenticated as {username}; uploading to {args.repo_id}", flush=True)

    manifest_entry = {
        "local_path": str(manifest_path),
        "repo_path": manifest_path.relative_to(root).as_posix(),
        "size": manifest_path.stat().st_size,
        "sha256": sha256_file(manifest_path),
        "kind": "manifest",
    }
    entries_with_manifest = entries + [manifest_entry]

    for index, entry in enumerate(entries_with_manifest, start=1):
        print(
            f"UPLOAD [{index}/{len(entries_with_manifest)}] "
            f"{entry['repo_path']} ({entry['size'] / 1024**3:.3f} GiB)",
            flush=True,
        )
        api.upload_file(
            path_or_fileobj=entry["local_path"],
            path_in_repo=entry["repo_path"],
            repo_id=args.repo_id,
            repo_type="model",
            revision="main",
            commit_message=f"Archive {entry['repo_path']}",
        )

    print("Upload calls completed; verifying every remote object.", flush=True)
    verification = verify_remote(api, args.repo_id, root, entries_with_manifest)
    report = {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "repo_id": args.repo_id,
        "all_verified": True,
        "files": verification,
    }
    verification_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(verification_path),
        path_in_repo=verification_path.relative_to(root).as_posix(),
        repo_id=args.repo_id,
        repo_type="model",
        revision="main",
        commit_message="Add verified RDD-FM Unified-L archive report",
    )
    verify_remote(
        api,
        args.repo_id,
        root,
        [
            {
                "local_path": str(verification_path),
                "repo_path": verification_path.relative_to(root).as_posix(),
                "size": verification_path.stat().st_size,
                "sha256": sha256_file(verification_path),
                "kind": "verification_report",
            }
        ],
    )
    print("REMOTE VERIFICATION PASSED FOR EVERY FILE.", flush=True)

    if args.no_delete:
        print("--no-delete supplied; local files retained.", flush=True)
        return

    checkpoint_roots = {
        (root / "checkpoints/RDD-FM-Unified-L-corrected-300-300-400/imagenet256-fm-unified/checkpoints").resolve(),
        (root / "checkpoints/RDD-FM-Unified-L-corrected-400-300-300/imagenet256-fm-unified/checkpoints").resolve(),
    }
    for checkpoint in checkpoints:
        match = CHECKPOINT_RE.fullmatch(checkpoint.name)
        if checkpoint.parent not in checkpoint_roots or not match or int(match.group(1)) % 50_000:
            raise RuntimeError(f"refusing out-of-scope checkpoint deletion: {checkpoint}")
    for path in cleanup_files:
        allowed_eval_roots = [
            (root / "eval_results/rdd_fm_unified_l_corrected_300300400_convergence").resolve(),
            (root / "eval_results/rdd_fm_unified_l_corrected_400300300_convergence").resolve(),
        ]
        if not any(is_below(path, allowed / "samples") or is_below(path, allowed / "npz_tmp") for allowed in allowed_eval_roots):
            raise RuntimeError(f"refusing out-of-scope sample deletion: {path}")

    for checkpoint in checkpoints:
        checkpoint.unlink()
        print(f"DELETED checkpoint {checkpoint}", flush=True)
    for path in cleanup_files:
        if path.exists():
            path.unlink()
    for eval_root in [
        root / "eval_results/rdd_fm_unified_l_corrected_300300400_convergence",
        root / "eval_results/rdd_fm_unified_l_corrected_400300300_convergence",
    ]:
        remove_empty_children(eval_root / "samples")
        remove_empty_children(eval_root / "npz_tmp")

    print(
        f"CLEANUP COMPLETE: deleted {len(checkpoints)} checkpoints and "
        f"{len(cleanup_files)} sample/NPZ files. Logs and metric text were retained locally.",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
