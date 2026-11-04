#!/usr/bin/env python3
"""Archive every experiment checkpoint and text result, then clean safe items.

Upload scope:
  * all checkpoint-like files below ``checkpoints`` (except downloaded
    ``pretrained`` dependencies), ``results_*``, and
    ``/home/yuheng/rdd_experiments``;
  * experiment logs/configs/metric tables, including every textual FID result;
  * no generated sample images.

Deletion scope is deliberately much narrower: only 50K-multiple checkpoints
from an explicit allowlist of the historical RDD/DiT/FM runs, plus generated
sample images whose paths clearly identify those runs. DMD, LPIPS, CSRA,
reuse, 3D, video, third-party, pretrained, and unknown artifacts are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download


PROJECT_ROOT = Path("/scr/yuheng/DevOpsBench/multiscale_diffusion")
HOME_EXPERIMENT_ROOT = Path("/home/yuheng/rdd_experiments")
WEIGHT_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors"}
TEXT_SUFFIXES = {".log", ".txt", ".tsv", ".csv", ".json", ".jsonl", ".yaml", ".yml", ".out"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
STEP_RE = re.compile(r"^(\d{7})\.pt$")

FORMAL_DELETE_GROUPS = {
    "DiT-XL-2",
    "RDD-FM-Unified-XL-v2",
    "RDD-Multistage-100-300-600",
    "RDD-Multistage-100-300-600-8x8-continued-from400k",
    "RDD-Multistage-300-300-400",
    "RDD-Unified-L",
    "RDD-Unified-L-100-300-600",
    "RDD-Unified-L-2Stage-500-500",
    "RDD-Unified-L-2Stage-8to32-500-500",
    "RDD-Unified-L-400-300-300",
    "RDD-Unified-L-600-300-100",
    "RDD-Unified-XL",
}

PROTECTED_SAMPLE_TOKENS = {
    "dmd", "lpips", "reuse", "csra", "wan", "video", "3d",
    "attention", "diagnostic", "preview", "third-party", "pretrained",
}
FORMAL_SAMPLE_TOKENS = {
    "unified", "multistage", "rdd", "rddfm", "dit", "imagenet", "ms100",
    "ms300", "ms600", "500500", "300300400", "400300300", "600300100",
    "100300600",
}
DYNAMIC_FILENAMES = {
    "archive_all_experiments_to_hf.log",
    "all_experiments_hash_cache.jsonl",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def walk_files(root: Path, pruned_names: set[str] | None = None) -> Iterator[Path]:
    if not root.exists():
        return
    pruned = pruned_names or set()
    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in pruned]
        base = Path(directory)
        for filename in filenames:
            yield base / filename


def project_repo_path(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def home_repo_path(path: Path) -> str:
    relative = path.resolve().relative_to(HOME_EXPERIMENT_ROOT).as_posix()
    return f"external_artifacts/rdd_experiments/{relative}"


def discover_entries() -> list[dict[str, Any]]:
    by_repo_path: dict[str, dict[str, Any]] = {}

    def add(path: Path, repo_path: str, kind: str) -> None:
        path = path.resolve()
        if not path.is_file() or path.name in DYNAMIC_FILENAMES:
            return
        existing = by_repo_path.get(repo_path)
        if existing and existing["local_path"] != str(path):
            raise RuntimeError(f"repo-path collision: {repo_path}")
        by_repo_path[repo_path] = {
            "local_path": str(path),
            "repo_path": repo_path,
            "kind": kind,
        }

    checkpoint_root = PROJECT_ROOT / "checkpoints"
    for path in walk_files(checkpoint_root, {"pretrained", ".cache", "__pycache__"}):
        if path.suffix.lower() in WEIGHT_SUFFIXES:
            add(path, project_repo_path(path), "checkpoint")

    for result_root in sorted(PROJECT_ROOT.glob("results_*")):
        if not result_root.is_dir():
            continue
        for path in walk_files(result_root, {".cache", "__pycache__"}):
            if path.suffix.lower() in WEIGHT_SUFFIXES:
                add(path, project_repo_path(path), "checkpoint")

    for path in walk_files(HOME_EXPERIMENT_ROOT, {".cache", "__pycache__"}):
        if path.suffix.lower() in WEIGHT_SUFFIXES:
            add(path, home_repo_path(path), "checkpoint")

    metadata_roots = [
        PROJECT_ROOT / "logs",
        PROJECT_ROOT / "eval_results",
        PROJECT_ROOT / "unfinished_upload_logs",
        PROJECT_ROOT / "checkpoints",
    ]
    metadata_roots.extend(sorted(p for p in PROJECT_ROOT.glob("results_*") if p.is_dir()))
    metadata_roots.extend(sorted(p for p in (PROJECT_ROOT / "2d").glob("eval_results*") if p.is_dir()))
    if (PROJECT_ROOT / "2d/exp").is_dir():
        metadata_roots.append(PROJECT_ROOT / "2d/exp")

    for metadata_root in metadata_roots:
        for path in walk_files(
            metadata_root,
            {"pretrained", ".cache", "__pycache__", "samples", "npz_tmp"},
        ):
            if path.suffix.lower() in TEXT_SUFFIXES:
                add(path, project_repo_path(path), "log_or_metric")

    for path in PROJECT_ROOT.iterdir():
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            add(path, project_repo_path(path), "log_or_metric")

    for path in walk_files(HOME_EXPERIMENT_ROOT, {".cache", "__pycache__", "samples"}):
        if path.suffix.lower() in TEXT_SUFFIXES:
            add(path, home_repo_path(path), "log_or_metric")

    return sorted(by_repo_path.values(), key=lambda entry: (entry["kind"] != "checkpoint", entry["repo_path"]))


def formal_checkpoint_cleanup(entries: Iterable[dict[str, Any]]) -> list[Path]:
    checkpoint_root = (PROJECT_ROOT / "checkpoints").resolve()
    selected: list[Path] = []
    for entry in entries:
        if entry["kind"] != "checkpoint":
            continue
        path = Path(entry["local_path"]).resolve()
        if not is_below(path, checkpoint_root):
            continue
        relative = path.relative_to(checkpoint_root)
        if not relative.parts or relative.parts[0] not in FORMAL_DELETE_GROUPS:
            continue
        match = STEP_RE.fullmatch(path.name)
        if match and int(match.group(1)) % 50_000 == 0:
            selected.append(path)
    return sorted(selected)


def is_formal_sample(path: Path) -> bool:
    lower = path.as_posix().lower()
    parts = {part.lower() for part in path.parts}
    if "samples" not in parts and "npz_tmp" not in parts:
        return False
    if any(token in lower for token in PROTECTED_SAMPLE_TOKENS):
        return False
    return any(token in lower for token in FORMAL_SAMPLE_TOKENS)


def iter_formal_sample_cleanup() -> Iterator[Path]:
    for scan_root in (PROJECT_ROOT / "eval_results", PROJECT_ROOT / "2d"):
        if not scan_root.exists():
            continue
        for path in walk_files(scan_root, {".cache", "__pycache__"}):
            suffix = path.suffix.lower()
            if suffix not in IMAGE_SUFFIXES | {".npz"}:
                continue
            if is_formal_sample(path):
                yield path.resolve()


def load_hash_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
                cache[item["local_path"]] = item
            except (json.JSONDecodeError, KeyError):
                continue
    return cache


def hash_entries(entries: list[dict[str, Any]], cache_path: Path) -> None:
    cache = load_hash_cache(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as cache_handle:
        for index, entry in enumerate(entries, start=1):
            path = Path(entry["local_path"])
            stat = path.stat()
            cached = cache.get(str(path))
            if (
                cached
                and cached.get("size") == stat.st_size
                and cached.get("mtime_ns") == stat.st_mtime_ns
                and cached.get("sha256")
            ):
                digest = cached["sha256"]
            else:
                print(f"HASH [{index}/{len(entries)}] {entry['repo_path']}", flush=True)
                digest = sha256_file(path)
                cached = {
                    "local_path": str(path),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sha256": digest,
                }
                cache_handle.write(json.dumps(cached) + "\n")
                cache_handle.flush()
            entry.update(
                size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                sha256=digest,
            )


def remote_key(info: Any) -> str:
    return str(getattr(info, "path", None) or getattr(info, "rfilename", ""))


def remote_lfs_sha(info: Any) -> str | None:
    lfs = getattr(info, "lfs", None)
    if lfs is None:
        return None
    if isinstance(lfs, dict):
        return lfs.get("sha256") or lfs.get("oid")
    return getattr(lfs, "sha256", None) or getattr(lfs, "oid", None)


def retry_delay_seconds(exc: Exception, attempt: int) -> int | None:
    """Return a conservative wait for transient Hub failures."""
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    message = str(exc)
    if status == 429 or "429 Too Many Requests" in message:
        delays = [300]
        headers = getattr(response, "headers", {}) or {}
        try:
            delays.append(int(headers.get("Retry-After", 0)))
        except (TypeError, ValueError):
            pass
        match = re.search(r"Retry after (\d+) seconds", message, re.IGNORECASE)
        if match:
            delays.append(int(match.group(1)))
        # The per-repository commit quota is hourly even when the generic API
        # Retry-After header is shorter.
        if "repository commits" in message or re.search(r"about\s+1\s+hour", message, re.IGNORECASE):
            delays.append(3_660)
        return max(delays) + 15
    if status is not None and int(status) >= 500:
        return min(60 * (2**attempt), 900)
    return None


def hub_call_with_retry(label: str, operation: Any) -> Any:
    for attempt in range(20):
        try:
            return operation()
        except Exception as exc:
            delay = retry_delay_seconds(exc, attempt)
            if delay is None or attempt == 19:
                raise
            print(
                f"HUB RETRY {label}: {type(exc).__name__}; waiting {delay}s "
                f"(attempt {attempt + 1}/20)",
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def fetch_remote(api: HfApi, repo_id: str, paths: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for start in range(0, len(paths), 50):
        infos = api.get_paths_info(
            repo_id,
            paths=paths[start : start + 50],
            repo_type="model",
            revision="main",
        )
        result.update({remote_key(info): info for info in infos})
    return result


def checkpoint_matches(entry: dict[str, Any], info: Any | None) -> bool:
    if info is None or int(getattr(info, "size", -1)) != entry["size"]:
        return False
    sha = remote_lfs_sha(info)
    return bool(sha and sha.lower() == entry["sha256"].lower())


def upload_entries(api: HfApi, repo_id: str, entries: list[dict[str, Any]]) -> None:
    remote = fetch_remote(api, repo_id, [entry["repo_path"] for entry in entries])
    pending: list[dict[str, Any]] = []
    skipped = 0
    for entry in entries:
        info = remote.get(entry["repo_path"])
        if entry["kind"] == "checkpoint" and checkpoint_matches(entry, info):
            skipped += 1
        elif entry["kind"] != "checkpoint" and info is not None and remote_lfs_sha(info) == entry["sha256"]:
            skipped += 1
        else:
            pending.append(entry)
    print(f"REMOTE PRECHECK: {skipped} exact objects already present; {len(pending)} uploads pending.", flush=True)

    large = [entry for entry in pending if entry["size"] >= 512 * 1024 * 1024]
    small = [entry for entry in pending if entry["size"] < 512 * 1024 * 1024]

    for index, entry in enumerate(large, start=1):
        print(
            f"UPLOAD LARGE [{index}/{len(large)}] {entry['repo_path']} "
            f"({entry['size'] / 1024**3:.2f} GiB)",
            flush=True,
        )
        hub_call_with_retry(
            f"upload {entry['repo_path']}",
            lambda entry=entry: api.upload_file(
                path_or_fileobj=entry["local_path"],
                path_in_repo=entry["repo_path"],
                repo_id=repo_id,
                repo_type="model",
                revision="main",
                commit_message=f"Archive checkpoint {entry['repo_path']}",
            ),
        )

    batch: list[dict[str, Any]] = []
    batch_bytes = 0
    batch_index = 0

    def flush_batch() -> None:
        nonlocal batch, batch_bytes, batch_index
        if not batch:
            return
        batch_index += 1
        print(
            f"UPLOAD BATCH [{batch_index}] files={len(batch)} size={batch_bytes / 1024**3:.2f} GiB",
            flush=True,
        )
        operations = [
            CommitOperationAdd(
                path_in_repo=entry["repo_path"],
                path_or_fileobj=entry["local_path"],
            )
            for entry in batch
        ]
        hub_call_with_retry(
            f"commit batch {batch_index}",
            lambda: api.create_commit(
                repo_id=repo_id,
                repo_type="model",
                revision="main",
                operations=operations,
                commit_message=f"Archive experiment artifacts batch {batch_index}",
            ),
        )
        batch = []
        batch_bytes = 0

    for entry in small:
        if batch and (len(batch) >= 500 or batch_bytes + entry["size"] > 2 * 1024**3):
            flush_batch()
        batch.append(entry)
        batch_bytes += entry["size"]
    flush_batch()


def verify_entries(api: HfApi, repo_id: str, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    remote = fetch_remote(api, repo_id, [entry["repo_path"] for entry in entries])
    verified: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="all_experiments_hf_verify_") as temp_dir:
        for index, entry in enumerate(entries, start=1):
            info = remote.get(entry["repo_path"])
            if info is None:
                raise RuntimeError(f"missing remote object: {entry['repo_path']}")
            size = int(getattr(info, "size", -1))
            if size != entry["size"]:
                raise RuntimeError(f"size mismatch for {entry['repo_path']}: {size} != {entry['size']}")
            sha = remote_lfs_sha(info)
            if sha:
                if sha.lower() != entry["sha256"].lower():
                    raise RuntimeError(f"LFS SHA-256 mismatch for {entry['repo_path']}")
                method = "lfs_sha256_and_size"
            elif entry["kind"] == "checkpoint":
                downloaded = Path(
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=entry["repo_path"],
                        repo_type="model",
                        local_dir=temp_dir,
                        force_download=True,
                    )
                )
                if sha256_file(downloaded) != entry["sha256"]:
                    raise RuntimeError(f"downloaded SHA-256 mismatch for {entry['repo_path']}")
                method = "redownload_sha256_and_size"
            else:
                blob_id = getattr(info, "blob_id", None)
                if not blob_id:
                    raise RuntimeError(f"no remote blob metadata for {entry['repo_path']}")
                method = "git_blob_and_size"
            verified.append(
                {
                    "repo_path": entry["repo_path"],
                    "size": entry["size"],
                    "sha256": entry["sha256"],
                    "kind": entry["kind"],
                    "method": method,
                    "verified": True,
                }
            )
            if index % 100 == 0 or index == len(entries):
                print(f"VERIFY [{index}/{len(entries)}]", flush=True)
    return verified


def upload_and_verify_report(api: HfApi, repo_id: str, root: Path, report_path: Path) -> None:
    repo_path = report_path.relative_to(root).as_posix()
    digest = sha256_file(report_path)
    hub_call_with_retry(
        "upload verification report",
        lambda: api.upload_file(
            path_or_fileobj=str(report_path),
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="model",
            revision="main",
            commit_message="Add complete experiment archive verification report",
        ),
    )
    with tempfile.TemporaryDirectory(prefix="archive_report_verify_") as temp_dir:
        downloaded = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=repo_path,
                repo_type="model",
                local_dir=temp_dir,
                force_download=True,
            )
        )
        if sha256_file(downloaded) != digest:
            raise RuntimeError("verification report failed download SHA-256 check")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="yang335/search_agent_hl_to_i18n_transport")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--no-delete", action="store_true")
    args = parser.parse_args()

    if PROJECT_ROOT.resolve() != Path("/scr/yuheng/DevOpsBench/multiscale_diffusion"):
        raise RuntimeError("unexpected project root")

    entries = discover_entries()
    formal_checkpoints = formal_checkpoint_cleanup(entries)
    checkpoint_entries = [entry for entry in entries if entry["kind"] == "checkpoint"]
    metadata_entries = [entry for entry in entries if entry["kind"] != "checkpoint"]

    sample_count = 0
    sample_bytes = 0
    for path in iter_formal_sample_cleanup():
        sample_count += 1
        sample_bytes += path.stat().st_size

    print(
        f"INVENTORY checkpoints={len(checkpoint_entries)} "
        f"checkpoint_GiB={sum(Path(e['local_path']).stat().st_size for e in checkpoint_entries)/1024**3:.2f} "
        f"metadata={len(metadata_entries)} "
        f"formal_delete_checkpoints={len(formal_checkpoints)} "
        f"formal_delete_GiB={sum(p.stat().st_size for p in formal_checkpoints)/1024**3:.2f} "
        f"sample_cleanup_files={sample_count} sample_cleanup_GiB={sample_bytes/1024**3:.2f}",
        flush=True,
    )
    if args.inventory_only:
        for group in sorted(FORMAL_DELETE_GROUPS):
            count = sum(
                1 for path in formal_checkpoints
                if path.relative_to(PROJECT_ROOT / "checkpoints").parts[0] == group
            )
            print(f"DELETE_GROUP {group} {count}")
        return

    manifest_dir = PROJECT_ROOT / "eval_results/upload_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    cache_path = manifest_dir / "all_experiments_hash_cache.jsonl"
    hash_entries(entries, cache_path)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest_path = manifest_dir / f"all_experiments_archive_{timestamp}.json"
    verification_path = manifest_dir / f"all_experiments_archive_{timestamp}_verified.json"
    manifest = {
        "created_at": utc_now(),
        "repo_id": args.repo_id,
        "entries": entries,
        "prior_verified_archive": "eval_results/upload_manifests/rdd_fm_unified_l_archive_20260917T015908Z_verified.json",
        "deletion_policy": {
            "formal_checkpoint_groups": sorted(FORMAL_DELETE_GROUPS),
            "only_50k_multiples": True,
            "protected_sample_tokens": sorted(PROTECTED_SAMPLE_TOKENS),
            "formal_checkpoint_count": len(formal_checkpoints),
            "formal_sample_count": sample_count,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest_entry = {
        "local_path": str(manifest_path),
        "repo_path": manifest_path.relative_to(PROJECT_ROOT).as_posix(),
        "kind": "manifest",
        "size": manifest_path.stat().st_size,
        "mtime_ns": manifest_path.stat().st_mtime_ns,
        "sha256": sha256_file(manifest_path),
    }
    all_entries = entries + [manifest_entry]

    api = HfApi()
    identity = api.whoami()
    username = identity.get("name") or identity.get("fullname")
    if username != "yang335":
        raise RuntimeError(f"authenticated as {username!r}, expected 'yang335'")
    api.repo_info(args.repo_id, repo_type="model")

    upload_entries(api, args.repo_id, all_entries)
    verified = verify_entries(api, args.repo_id, all_entries)
    report = {
        "verified_at": utc_now(),
        "repo_id": args.repo_id,
        "all_verified": True,
        "verified_files": len(verified),
        "prior_verified_archive": manifest["prior_verified_archive"],
        "files": verified,
    }
    verification_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    upload_and_verify_report(api, args.repo_id, PROJECT_ROOT, verification_path)
    print("REMOTE VERIFICATION PASSED FOR THE COMPLETE ARCHIVE.", flush=True)

    if args.no_delete:
        print("--no-delete supplied; retaining every local file.", flush=True)
        return

    verified_paths = {item["repo_path"] for item in verified}
    for path in formal_checkpoints:
        repo_path = project_repo_path(path)
        if repo_path not in verified_paths:
            raise RuntimeError(f"refusing deletion without verified remote object: {repo_path}")
        relative = path.relative_to((PROJECT_ROOT / "checkpoints").resolve())
        if relative.parts[0] not in FORMAL_DELETE_GROUPS:
            raise RuntimeError(f"refusing out-of-scope checkpoint deletion: {path}")
        match = STEP_RE.fullmatch(path.name)
        if not match or int(match.group(1)) % 50_000:
            raise RuntimeError(f"refusing non-50K checkpoint deletion: {path}")

    for path in formal_checkpoints:
        path.unlink()
        print(f"DELETED CHECKPOINT {path}", flush=True)

    deleted_samples = 0
    deleted_sample_bytes = 0
    for path in iter_formal_sample_cleanup():
        if not path.exists():
            continue
        size = path.stat().st_size
        path.unlink()
        deleted_samples += 1
        deleted_sample_bytes += size
        if deleted_samples % 50_000 == 0:
            print(f"DELETED SAMPLES {deleted_samples}", flush=True)

    print(
        f"CLEANUP COMPLETE checkpoints={len(formal_checkpoints)} "
        f"sample_files={deleted_samples} sample_GiB={deleted_sample_bytes/1024**3:.2f}. "
        "DMD/LPIPS/CSRA/reuse/3D/video/unknown artifacts retained.",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
