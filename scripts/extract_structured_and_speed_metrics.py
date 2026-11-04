#!/usr/bin/env python3
"""Audit non-log/txt text files for FID and acceleration metrics.

This complements extract_hf_surfi2_experiment_metrics.py.  It scans structured
and documentation-like text formats on Hugging Face and surfi2, while keeping
every extracted value tied to a source path and JSON pointer / row / line.
"""

from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from huggingface_hub import HfApi, hf_hub_download


ROOT = Path("/scr/yuheng/DevOpsBench/multiscale_diffusion")
HOME_EXPERIMENTS = Path("/home/yuheng/rdd_experiments")
REPO_ID = "yang335/search_agent_hl_to_i18n_transport"
STRUCTURED_SUFFIXES = {".json", ".jsonl", ".csv", ".tsv"}
DOCUMENT_SUFFIXES = {".md", ".yaml", ".yml", ".toml", ".sh", ".py", ".ps1", ".tex", ".rst", ".html", ".svg"}
ALL_SUFFIXES = STRUCTURED_SUFFIXES | DOCUMENT_SUFFIXES
SKIP_DIRS = {
    ".git", ".cache", "__pycache__", "node_modules", "imagenet", "data", "samples",
    "npz_tmp", "pretrained", ".venv-rdd-eval", ".venv-rdd-train", "hf_all_text_cache",
}
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
FID_LINE = re.compile(rf"(?i)(?<![A-Za-z])FID(?:\s*[↓])?\s*[:=]\s*({NUMBER})")
METRIC_ALIASES = {
    "fid": {"fid", "frechet_inception_distance"},
    "sfid": {"sfid", "spatial_fid"},
    "inception_score": {"inception_score", "is"},
    "precision": {"precision"},
    "recall": {"recall"},
}
SPEED_KEY_RE = re.compile(
    r"(?i)(?:^|_)(?:latency|wall_?clock|elapsed|duration|time_?ms|ms_?per_?(?:image|sample|step|batch)|"
    r"seconds?_?per_?(?:image|sample|step|batch)|throughput|images?_?per_?s|samples?_?per_?s|steps?_?per_?s|"
    r"tokens?_?per_?s|fps|flops|tflops|gflops|peak_?(?:memory|gb|vram)|memory_?(?:gb|mb)|vram_?(?:gb|mb)|"
    r"nfe|model_?evaluations?)(?:$|_)"
)
SPEED_LINE_PATTERNS = [
    ("ms_per_image", re.compile(rf"(?i)({NUMBER})\s*(?:ms/image|ms per image)"), "ms/image"),
    ("images_per_second", re.compile(rf"(?i)({NUMBER})\s*(?:images?/s|img/s|images? per second)"), "images/s"),
    ("samples_per_second", re.compile(rf"(?i)({NUMBER})\s*(?:samples?/s|samples? per second)"), "samples/s"),
    ("latency_ms", re.compile(rf"(?i)latency[^\d\n]{{0,30}}({NUMBER})\s*ms"), "ms"),
    ("peak_memory_gb", re.compile(rf"(?i)(?:peak|memory|vram)[^\d\n]{{0,40}}({NUMBER})\s*(?:GB|GiB)"), "GB"),
    ("flops", re.compile(rf"(?i)({NUMBER})\s*(T?FLOPs?)"), "reported"),
]


@dataclass
class Source:
    repo_path: str
    local_path: str
    origin: str
    size: int
    suffix: str
    hf_present: bool
    surfi2_present: bool


def normalize_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")


def numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and re.fullmatch(NUMBER, value.strip()):
        return float(value)
    return None


def metric_from_mapping(mapping: dict[str, Any], canonical: str) -> float | None:
    aliases = METRIC_ALIASES[canonical]
    for key, value in mapping.items():
        if normalize_key(key) in aliases:
            result = numeric(value)
            if result is not None:
                return result
    return None


def config_excerpt(mapping: dict[str, Any]) -> str:
    keys = {
        "model", "model_name", "checkpoint", "ckpt", "step", "train_step", "sampler", "solver",
        "cfg", "cfg_scale", "nfe", "num_steps", "sampling_steps", "stage_lengths", "train_allocation",
        "dataset", "data_source", "num_samples", "batch_size", "world_size", "gpus",
    }
    chosen = {}
    for key, value in mapping.items():
        if normalize_key(key) in keys and isinstance(value, (str, int, float, bool, list, type(None))):
            text = json.dumps(value, ensure_ascii=False)
            chosen[str(key)] = text[:500]
    return json.dumps(chosen, ensure_ascii=False)


def remote_paths(api: HfApi) -> set[str]:
    return {
        path for path in api.list_repo_files(REPO_ID, repo_type="model", revision="main")
        if PurePosixPath(path).suffix.lower() in ALL_SUFFIXES
    }


def local_repo_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return "external_artifacts/rdd_experiments/" + resolved.relative_to(HOME_EXPERIMENTS.resolve()).as_posix()


def walk_local(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
        base = Path(directory)
        for filename in filenames:
            path = base / filename
            if path.suffix.lower() in ALL_SUFFIXES:
                yield path


def local_target(repo_path: str) -> Path:
    prefix = "external_artifacts/rdd_experiments/"
    return (HOME_EXPERIMENTS / repo_path[len(prefix):]) if repo_path.startswith(prefix) else (ROOT / repo_path)


def build_sources(api: HfApi, cache_dir: Path) -> list[Source]:
    remote = remote_paths(api)
    local: dict[str, Path] = {}
    for root in (ROOT, HOME_EXPERIMENTS):
        for path in walk_local(root):
            local[local_repo_path(path)] = path.resolve()
    sources: list[Source] = []
    for repo_path in sorted(remote | set(local)):
        candidate = local.get(repo_path) or local_target(repo_path)
        hf_present = repo_path in remote
        was_local = repo_path in local
        if not candidate.exists() and hf_present:
            candidate = Path(hf_hub_download(
                repo_id=REPO_ID, filename=repo_path, repo_type="model", revision="main", local_dir=str(cache_dir)
            ))
        if not candidate.exists():
            continue
        sources.append(Source(
            repo_path=repo_path,
            local_path=str(candidate),
            origin="both" if hf_present and was_local else ("huggingface" if hf_present else "surfi2"),
            size=candidate.stat().st_size,
            suffix=candidate.suffix.lower(),
            hf_present=hf_present,
            surfi2_present=was_local,
        ))
    return sources


def append_fid(rows: list[dict[str, Any]], source: Source, location: str, mapping: dict[str, Any], context: str) -> None:
    fid = metric_from_mapping(mapping, "fid")
    if fid is None:
        return
    rows.append({
        "source_repo_path": source.repo_path,
        "source_local_path": source.local_path,
        "source_origin": source.origin,
        "source_suffix": source.suffix,
        "source_size": source.size,
        "location": location,
        "fid": fid,
        "sfid": metric_from_mapping(mapping, "sfid"),
        "inception_score": metric_from_mapping(mapping, "inception_score"),
        "precision": metric_from_mapping(mapping, "precision"),
        "recall": metric_from_mapping(mapping, "recall"),
        "config_excerpt": config_excerpt(mapping),
        "context": context[:3000],
    })


def append_speed(rows: list[dict[str, Any]], source: Source, location: str, key: str, value: Any, context: str) -> None:
    number = numeric(value)
    normalized = normalize_key(key)
    if number is None or not SPEED_KEY_RE.search(normalized):
        return
    rows.append({
        "source_repo_path": source.repo_path,
        "source_local_path": source.local_path,
        "source_origin": source.origin,
        "source_suffix": source.suffix,
        "source_size": source.size,
        "location": location,
        "metric_name": normalized,
        "value": number,
        "unit": "from_key_or_context",
        "context": context[:3000],
    })


def walk_json(value: Any, pointer: str, source: Source, fid_rows: list[dict], speed_rows: list[dict]) -> None:
    if isinstance(value, dict):
        append_fid(fid_rows, source, pointer or "/", value, json.dumps(value, ensure_ascii=False)[:3000])
        excerpt = config_excerpt(value)
        for key, item in value.items():
            append_speed(speed_rows, source, f"{pointer}/{key}", key, item, excerpt)
            walk_json(item, f"{pointer}/{key}", source, fid_rows, speed_rows)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            walk_json(item, f"{pointer}/{index}", source, fid_rows, speed_rows)


def parse_json(source: Source, fid_rows: list[dict], speed_rows: list[dict]) -> None:
    path = Path(source.local_path)
    if source.suffix == ".jsonl":
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                walk_json(obj, f"line:{line_number}", source, fid_rows, speed_rows)
    else:
        try:
            obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError:
            return
        walk_json(obj, "", source, fid_rows, speed_rows)


def parse_table(source: Source, fid_rows: list[dict], speed_rows: list[dict]) -> None:
    delimiter = "\t" if source.suffix == ".tsv" else ","
    with Path(source.local_path).open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames:
            return
        for row_number, row in enumerate(reader, start=2):
            append_fid(fid_rows, source, f"row:{row_number}", row, json.dumps(row, ensure_ascii=False))
            excerpt = config_excerpt(row)
            for key, value in row.items():
                append_speed(speed_rows, source, f"row:{row_number}/{key}", key, value, excerpt)


def parse_document(source: Source, fid_rows: list[dict], speed_rows: list[dict]) -> None:
    # Documentation is secondary evidence.  Only inline labeled metrics are
    # extracted; Markdown table columns are left for provenance review to avoid
    # guessing column alignment.
    with Path(source.local_path).open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            match = FID_LINE.search(line)
            if match:
                fid_rows.append({
                    "source_repo_path": source.repo_path,
                    "source_local_path": source.local_path,
                    "source_origin": source.origin,
                    "source_suffix": source.suffix,
                    "source_size": source.size,
                    "location": f"line:{line_number}",
                    "fid": float(match.group(1)),
                    "sfid": None,
                    "inception_score": None,
                    "precision": None,
                    "recall": None,
                    "config_excerpt": "",
                    "context": line.strip()[:3000],
                })
            lower_path = source.repo_path.lower()
            if any(token in lower_path for token in ("bench", "speed", "latency", "throughput", "profile", "timing")):
                for metric_name, pattern, unit in SPEED_LINE_PATTERNS:
                    for speed_match in pattern.finditer(line):
                        speed_rows.append({
                            "source_repo_path": source.repo_path,
                            "source_local_path": source.local_path,
                            "source_origin": source.origin,
                            "source_suffix": source.suffix,
                            "source_size": source.size,
                            "location": f"line:{line_number}",
                            "metric_name": metric_name,
                            "value": float(speed_match.group(1)),
                            "unit": unit,
                            "context": line.strip()[:3000],
                        })


def deduplicate(rows: list[dict], fields: list[str]) -> list[dict]:
    seen = set()
    result = []
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in fields)
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    output_dir = ROOT / "eval_results/experiment_ledger_20260918/structured_text_audit"
    cache_dir = output_dir / "hf_all_text_cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    if api.whoami().get("name") != "yang335":
        raise RuntimeError("unexpected Hugging Face identity")
    sources = build_sources(api, cache_dir)
    fid_rows: list[dict] = []
    speed_rows: list[dict] = []
    failures: list[dict] = []
    inventory: list[dict] = []
    for index, source in enumerate(sources, start=1):
        before_fid, before_speed = len(fid_rows), len(speed_rows)
        try:
            if source.suffix in {".json", ".jsonl"}:
                parse_json(source, fid_rows, speed_rows)
            elif source.suffix in {".csv", ".tsv"}:
                parse_table(source, fid_rows, speed_rows)
            else:
                parse_document(source, fid_rows, speed_rows)
        except Exception as exc:
            failures.append({"repo_path": source.repo_path, "error": f"{type(exc).__name__}: {exc}"})
        item = asdict(source)
        item["fid_records"] = len(fid_rows) - before_fid
        item["speed_records"] = len(speed_rows) - before_speed
        inventory.append(item)
        if index % 100 == 0 or index == len(sources):
            print(f"SCAN [{index}/{len(sources)}] fid={len(fid_rows)} speed={len(speed_rows)} failures={len(failures)}", flush=True)

    fid_fields = [
        "source_repo_path", "source_local_path", "source_origin", "source_suffix", "source_size", "location",
        "fid", "sfid", "inception_score", "precision", "recall", "config_excerpt", "context",
    ]
    speed_fields = [
        "source_repo_path", "source_local_path", "source_origin", "source_suffix", "source_size", "location",
        "metric_name", "value", "unit", "context",
    ]
    fid_rows = deduplicate(fid_rows, ["source_repo_path", "location", "fid", "sfid", "inception_score", "precision", "recall"])
    speed_rows = deduplicate(speed_rows, ["source_repo_path", "location", "metric_name", "value"])
    write_csv(output_dir / "structured_fid_records.csv", fid_rows, fid_fields)
    write_csv(output_dir / "structured_speed_records.csv", speed_rows, speed_fields)
    write_csv(output_dir / "structured_source_inventory.csv", inventory, list(inventory[0]) if inventory else [])
    (output_dir / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "sources": len(sources),
        "suffix_counts": Counter(source.suffix for source in sources),
        "hf_sources": sum(source.hf_present for source in sources),
        "surfi2_sources": sum(source.surfi2_present for source in sources),
        "total_bytes": sum(source.size for source in sources),
        "fid_records": len(fid_rows),
        "speed_records": len(speed_rows),
        "failures": len(failures),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
