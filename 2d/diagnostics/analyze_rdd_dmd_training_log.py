#!/usr/bin/env python3
"""Summarize the scalar trajectory emitted by train_rdd_phased_dmd3.py."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path


PATTERN = re.compile(
    r"step=(?P<step>\d+) phase=(?P<phase>\d+)\(res=(?P<res>\d+)\) "
    r"critic=(?P<critic>[-+0-9.eE]+) dmd=(?P<dmd>[-+0-9.eE]+) "
    r"preserve=(?P<preserve>[-+0-9.eE]+) student_grad=(?P<grad>[-+0-9.eE]+)"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--bin-size", type=int, default=5000)
    args = parser.parse_args()

    bins: dict[int, list[dict[str, float]]] = defaultdict(list)
    rows = []
    for line in args.log.read_text(encoding="utf-8").splitlines():
        match = PATTERN.search(line)
        if match is None:
            continue
        row = {key: float(match.group(key)) for key in ("critic", "dmd", "preserve", "grad")}
        row["step"] = int(match.group("step"))
        rows.append(row)
        bins[(row["step"] - 1) // args.bin_size].append(row)

    print("step_range\tcount\tcritic\tdmd\tpreserve\tweighted_preserve\tstudent_grad")
    for bin_index in sorted(bins):
        group = bins[bin_index]
        means = {
            key: sum(row[key] for row in group) / len(group)
            for key in ("critic", "dmd", "preserve", "grad")
        }
        start = bin_index * args.bin_size
        end = (bin_index + 1) * args.bin_size
        print(
            f"{start:06d}-{end:06d}\t{len(group)}\t{means['critic']:.7f}"
            f"\t{means['dmd']:.7f}\t{means['preserve']:.7f}"
            f"\t{0.25 * means['preserve']:.7f}\t{means['grad']:.7f}"
        )
    if rows:
        print(f"parsed_rows={len(rows)}, first_step={rows[0]['step']}, last_step={rows[-1]['step']}")


if __name__ == "__main__":
    main()
