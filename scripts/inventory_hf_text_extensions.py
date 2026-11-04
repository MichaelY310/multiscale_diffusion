#!/usr/bin/env python3
"""Summarize text-like files in the private RDD Hugging Face repository."""

from __future__ import annotations

from collections import defaultdict
from pathlib import PurePosixPath

from huggingface_hub import HfApi


REPO_ID = "yang335/search_agent_hl_to_i18n_transport"
TEXT_SUFFIXES = {
    ".log", ".txt", ".json", ".jsonl", ".csv", ".tsv", ".md", ".yaml", ".yml",
    ".toml", ".sh", ".py", ".ps1", ".tex", ".rst", ".html", ".svg",
}


def main() -> None:
    api = HfApi()
    if api.whoami().get("name") != "yang335":
        raise RuntimeError("unexpected Hugging Face identity")
    stats: dict[str, list[int]] = defaultdict(list)
    total = 0
    for entry in api.list_repo_tree(REPO_ID, repo_type="model", revision="main", recursive=True, expand=False):
        if getattr(entry, "type", "file") not in {"file", None} and entry.__class__.__name__ != "RepoFile":
            continue
        suffix = PurePosixPath(entry.path).suffix.lower()
        if suffix not in TEXT_SUFFIXES:
            continue
        size = int(getattr(entry, "size", 0) or 0)
        stats[suffix].append(size)
        total += size
    for suffix in sorted(stats):
        sizes = stats[suffix]
        print(f"{suffix or '<none>'}\t{len(sizes)}\t{sum(sizes)}\t{max(sizes)}")
    print(f"TOTAL\t{sum(len(v) for v in stats.values())}\t{total}\t")


if __name__ == "__main__":
    main()
