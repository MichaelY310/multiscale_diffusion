#!/usr/bin/env python3
"""Read-only audit of the completed comprehensive Hugging Face archive."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path("/scr/yuheng/DevOpsBench/multiscale_diffusion")
sys.path.insert(0, str(ROOT / "scripts"))
import archive_all_experiments_to_hf as archive  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def contains_fid_result(path: Path) -> bool:
    if "fid" in path.as_posix().lower():
        return True
    if path.suffix.lower() not in archive.TEXT_SUFFIXES:
        return False
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line_number, line in enumerate(handle):
                if re.search(r"\bFID\b|Fr[eé]chet Inception", line, re.IGNORECASE):
                    return True
                if line_number >= 200_000:
                    break
    except OSError:
        return False
    return False


def main() -> None:
    verified_path = Path(
        max(
            glob.glob(str(ROOT / "eval_results/upload_manifests/all_experiments_archive_*_verified.json")),
            key=os.path.getmtime,
        )
    )
    report = json.loads(verified_path.read_text(encoding="utf-8"))
    verified = {item["repo_path"]: item for item in report["files"]}
    current = archive.discover_entries()
    current_by_repo = {entry["repo_path"]: entry for entry in current}
    report_repo_path = verified_path.relative_to(ROOT).as_posix()

    unexplained = sorted(set(current_by_repo) - set(verified) - {report_repo_path})
    missing_current_checkpoints = sorted(
        entry["repo_path"]
        for entry in current
        if entry["kind"] == "checkpoint" and entry["repo_path"] not in verified
    )
    size_mismatch = sorted(
        repo_path
        for repo_path, entry in current_by_repo.items()
        if repo_path in verified
        and Path(entry["local_path"]).stat().st_size != verified[repo_path]["size"]
    )

    fid_entries = [
        entry for entry in current
        if entry["kind"] == "log_or_metric" and contains_fid_result(Path(entry["local_path"]))
    ]
    fid_unverified = sorted(entry["repo_path"] for entry in fid_entries if entry["repo_path"] not in verified)
    fid_changed = []
    for entry in fid_entries:
        item = verified.get(entry["repo_path"])
        if item and sha256_file(Path(entry["local_path"])) != item["sha256"]:
            fid_changed.append(entry["repo_path"])

    api = HfApi()
    infos = api.get_paths_info(
        report["repo_id"],
        paths=[report_repo_path],
        repo_type="model",
        revision="main",
    )
    report_remote_present = bool(infos and int(infos[0].size) == verified_path.stat().st_size)

    result = {
        "report": str(verified_path),
        "all_verified": report.get("all_verified") is True,
        "verified_files": report.get("verified_files"),
        "verified_bytes": sum(item["size"] for item in report["files"]),
        "verified_checkpoints": sum(item.get("kind") == "checkpoint" for item in report["files"]),
        "current_discovered_files": len(current),
        "current_checkpoints": sum(entry["kind"] == "checkpoint" for entry in current),
        "current_fid_result_files": len(fid_entries),
        "unexplained_current_files": unexplained,
        "missing_current_checkpoints_from_verified_report": missing_current_checkpoints,
        "current_size_mismatches": size_mismatch,
        "fid_files_missing_from_verified_report": fid_unverified,
        "fid_files_changed_after_archive": fid_changed,
        "verification_report_present_on_hub": report_remote_present,
        "audit_passed": bool(
            report.get("all_verified") is True
            and report_remote_present
            and not unexplained
            and not missing_current_checkpoints
            and not size_mismatch
            and not fid_unverified
            and not fid_changed
        ),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
