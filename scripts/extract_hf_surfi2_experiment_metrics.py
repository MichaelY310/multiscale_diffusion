#!/usr/bin/env python3
"""Inventory every HF/surfi2 log or text file and extract FID metric blocks.

The parser is deliberately evidence-first: it never guesses an experiment
configuration from a number.  Each row keeps the source path, normalized line
number, nearby context, and all metrics found in the same evaluation block.
Large tqdm-heavy files are handled one at a time and carriage-return progress
updates are discarded before context is retained.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from huggingface_hub import HfApi, hf_hub_download


ROOT = Path("/scr/yuheng/DevOpsBench/multiscale_diffusion")
HOME_EXPERIMENTS = Path("/home/yuheng/rdd_experiments")
REPO_ID = "yang335/search_agent_hl_to_i18n_transport"
TEXT_SUFFIXES = {".log", ".txt"}
SKIP_DIRS = {
    ".git", ".cache", "__pycache__", "node_modules", "imagenet", "data",
    "samples", "npz_tmp", "pretrained", ".venv-rdd-eval", ".venv-rdd-train",
}
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
PATTERNS = {
    "fid": re.compile(rf"(?i)^\s*FID\s*[:=]\s*({NUMBER})(?:\s|$)"),
    "sfid": re.compile(rf"(?i)^\s*sFID\s*[:=]\s*({NUMBER})(?:\s|$)"),
    "inception_score": re.compile(rf"(?i)^\s*(?:Inception Score|IS)\s*[:=]\s*({NUMBER})(?:\s|$)"),
    "precision": re.compile(rf"(?i)^\s*Precision\s*[:=]\s*({NUMBER})(?:\s|$)"),
    "recall": re.compile(rf"(?i)^\s*Recall\s*[:=]\s*({NUMBER})(?:\s|$)"),
}
PROGRESS_MARKERS = ("it/s]", "%|", "Processing Files (", "New Data Upload", "Fetching ")


@dataclass
class Source:
    repo_path: str
    local_path: str
    origin: str
    size: int
    hf_present: bool
    surfi2_present: bool


def remote_text_paths(api: HfApi) -> list[str]:
    return sorted(
        path for path in api.list_repo_files(REPO_ID, repo_type="model", revision="main")
        if PurePosixPath(path).suffix.lower() in TEXT_SUFFIXES
    )


def local_repo_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        relative = resolved.relative_to(HOME_EXPERIMENTS.resolve()).as_posix()
        return f"external_artifacts/rdd_experiments/{relative}"


def path_for_repo_path(repo_path: str) -> Path:
    prefix = "external_artifacts/rdd_experiments/"
    if repo_path.startswith(prefix):
        return HOME_EXPERIMENTS / repo_path[len(prefix) :]
    return ROOT / repo_path


def walk_local_text(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
        base = Path(directory)
        for filename in filenames:
            path = base / filename
            if path.suffix.lower() in TEXT_SUFFIXES:
                yield path


def build_sources(api: HfApi, cache_dir: Path) -> list[Source]:
    remote = set(remote_text_paths(api))
    local: dict[str, Path] = {}
    for root in (ROOT, HOME_EXPERIMENTS):
        for path in walk_local_text(root):
            local[local_repo_path(path)] = path.resolve()

    sources: list[Source] = []
    for repo_path in sorted(remote | set(local)):
        candidate = local.get(repo_path) or path_for_repo_path(repo_path)
        hf_present = repo_path in remote
        surfi2_present = candidate.exists()
        if not surfi2_present and hf_present:
            candidate = Path(
                hf_hub_download(
                    repo_id=REPO_ID,
                    filename=repo_path,
                    repo_type="model",
                    revision="main",
                    local_dir=str(cache_dir),
                )
            )
            surfi2_present = candidate.exists()
        if not surfi2_present:
            continue
        origin = "both" if hf_present and repo_path in local else ("huggingface" if hf_present else "surfi2")
        sources.append(
            Source(
                repo_path=repo_path,
                local_path=str(candidate),
                origin=origin,
                size=candidate.stat().st_size,
                hf_present=hf_present,
                surfi2_present=repo_path in local,
            )
        )
    return sources


def clean_lines(path: Path) -> list[tuple[int, str]]:
    data = path.read_bytes().decode("utf-8", errors="replace")
    data = ANSI_RE.sub("", data)
    result: list[tuple[int, str]] = []
    for line_number, raw in enumerate(data.splitlines(), start=1):
        line = raw.strip()
        if not line or len(line) > 8_000:
            continue
        if any(marker in line for marker in PROGRESS_MARKERS):
            continue
        result.append((line_number, line))
    return result


def metric_on_line(line: str, key: str) -> float | None:
    match = PATTERNS[key].search(line)
    return float(match.group(1)) if match else None


def closest_metric(lines: list[tuple[int, str]], anchor: int, key: str) -> float | None:
    candidates: list[tuple[int, float]] = []
    for index in range(max(0, anchor - 12), min(len(lines), anchor + 16)):
        if key == "fid" and index != anchor:
            continue
        value = metric_on_line(lines[index][1], key)
        if value is not None:
            candidates.append((abs(index - anchor), value))
    return min(candidates, default=(0, None), key=lambda item: item[0])[1]


def context_for(lines: list[tuple[int, str]], anchor: int) -> str:
    chosen: list[str] = []
    for _, line in lines[max(0, anchor - 35) : min(len(lines), anchor + 10)]:
        if any(pattern.search(line) for pattern in PATTERNS.values()):
            continue
        if len(line) > 700:
            line = line[:700] + "…"
        chosen.append(line)
    # The last lines before evaluation are normally the most useful command,
    # checkpoint, output directory, and configuration evidence.
    return " || ".join(chosen[-18:])


def extract_fid_blocks(source: Source) -> list[dict[str, Any]]:
    path = Path(source.local_path)
    lines = clean_lines(path)
    records: list[dict[str, Any]] = []
    for index, (line_number, line) in enumerate(lines):
        fid = metric_on_line(line, "fid")
        if fid is None:
            continue
        record: dict[str, Any] = {
            "source_repo_path": source.repo_path,
            "source_local_path": source.local_path,
            "source_origin": source.origin,
            "source_size": source.size,
            "normalized_line": line_number,
            "fid": fid,
            "sfid": closest_metric(lines, index, "sfid"),
            "inception_score": closest_metric(lines, index, "inception_score"),
            "precision": closest_metric(lines, index, "precision"),
            "recall": closest_metric(lines, index, "recall"),
            "context": context_for(lines, index),
        }
        records.append(record)
    return records


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "eval_results/experiment_ledger_20260918"),
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "hf_text_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    identity = api.whoami()
    if identity.get("name") != "yang335":
        raise RuntimeError(f"unexpected Hugging Face identity: {identity.get('name')!r}")

    sources = build_sources(api, cache_dir)
    raw: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, source in enumerate(sources, start=1):
        try:
            blocks = extract_fid_blocks(source)
            raw.extend(blocks)
            item = asdict(source)
            item["fid_blocks"] = len(blocks)
            inventory.append(item)
        except Exception as exc:  # keep a complete audit trail
            failures.append({"repo_path": source.repo_path, "error": f"{type(exc).__name__}: {exc}"})
        if index % 100 == 0 or index == len(sources):
            print(f"PARSE [{index}/{len(sources)}] fid_blocks={len(raw)} failures={len(failures)}", flush=True)

    write_csv(
        output_dir / "source_inventory.csv",
        inventory,
        ["repo_path", "local_path", "origin", "size", "hf_present", "surfi2_present", "fid_blocks"],
    )
    fields = [
        "source_repo_path", "source_local_path", "source_origin", "source_size",
        "normalized_line", "fid", "sfid", "inception_score", "precision", "recall", "context",
    ]
    write_csv(output_dir / "raw_fid_blocks.csv", raw, fields)
    (output_dir / "raw_fid_blocks.json").write_text(
        json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "parse_failures.json").write_text(
        json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = {
        "repo_id": REPO_ID,
        "sources": len(sources),
        "hf_sources": sum(source.hf_present for source in sources),
        "surfi2_sources": sum(source.surfi2_present for source in sources),
        "sources_in_both": sum(source.origin == "both" for source in sources),
        "total_bytes": sum(source.size for source in sources),
        "raw_fid_blocks": len(raw),
        "parse_failures": failures,
    }
    (output_dir / "extraction_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
