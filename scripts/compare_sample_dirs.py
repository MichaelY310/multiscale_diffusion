#!/usr/bin/env python3
"""Verify sample-directory byte parity for a fixed seed/config smoke run."""

import argparse
import hashlib
from pathlib import Path


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    args = parser.parse_args()

    reference = Path(args.reference)
    candidate = Path(args.candidate)
    ref_files = {path.name: path for path in reference.glob("*.png")}
    cand_files = {path.name: path for path in candidate.glob("*.png")}
    if set(ref_files) != set(cand_files):
        missing = sorted(set(ref_files) - set(cand_files))
        extra = sorted(set(cand_files) - set(ref_files))
        raise SystemExit(f"File-set mismatch: missing={missing[:10]}, extra={extra[:10]}")

    mismatches = [
        name for name in sorted(ref_files)
        if digest(ref_files[name]) != digest(cand_files[name])
    ]
    if mismatches:
        raise SystemExit(f"Byte parity failed for {len(mismatches)} files: {mismatches[:10]}")
    print(f"Byte-identical: {len(ref_files)} PNG files")


if __name__ == "__main__":
    main()
