#!/usr/bin/env python3
"""Print non-weight checkpoint metadata without materializing tensor storage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "__dict__"):
        return jsonable(vars(value))
    return f"<{type(value).__module__}.{type(value).__name__}>"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.checkpoint:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        metadata = {
            key: jsonable(value)
            for key, value in payload.items()
            if key not in {"model", "ema", "opt", "optimizer"}
        }
        print(json.dumps({"checkpoint": str(path), "keys": list(payload), "metadata": metadata}, indent=2))


if __name__ == "__main__":
    main()
