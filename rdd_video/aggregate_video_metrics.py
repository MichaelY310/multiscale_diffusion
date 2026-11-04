"""Aggregate sampler sidecars without confusing scheduled steps with NFE."""

import argparse
import json
import math
import statistics
from pathlib import Path


FIELDS = (
    "scheduled_updates",
    "actual_model_forward_calls",
    "actual_model_forwards",
    "text_encode_seconds",
    "denoise_and_boundary_seconds",
    "vae_decode_seconds",
    "end_to_end_inference_seconds",
    "peak_cuda_memory_bytes",
)


def summarize(values):
    values = sorted(float(value) for value in values)
    if not values:
        return None

    def percentile(q):
        if len(values) == 1:
            return values[0]
        pos = (len(values) - 1) * q
        low = math.floor(pos)
        high = math.ceil(pos)
        if low == high:
            return values[low]
        return values[low] * (high - pos) + values[high] * (pos - low)

    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p05": percentile(0.05),
        "p95": percentile(0.95),
        "min": values[0],
        "max": values[-1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--warmup-videos",
        type=int,
        default=1,
        help="Drop this many earliest records per shard from latency summaries.",
    )
    args = parser.parse_args()

    paths = sorted(args.metrics_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No JSON sidecars in {args.metrics_dir}")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]

    by_shard = {}
    for record in records:
        by_shard.setdefault(int(record.get("shard_rank", 0)), []).append(record)
    latency_records = []
    for shard_records in by_shard.values():
        shard_records.sort(key=lambda record: int(record.get("shard_record_index", 0)))
        latency_records.extend(shard_records[args.warmup_videos :])
    result = {
        "metrics_dir": str(args.metrics_dir.resolve()),
        "videos": len(records),
        "shards": len(by_shard),
        "warmup_records_dropped": len(records) - len(latency_records),
        "fields": {},
    }
    for field in FIELDS:
        source = latency_records if field.endswith("seconds") else records
        values = [record[field] for record in source if record.get(field) is not None]
        if values:
            summary = summarize(values)
            if field == "peak_cuda_memory_bytes":
                summary["mean_gib"] = summary["mean"] / (1024**3)
                summary["max_gib"] = summary["max"] / (1024**3)
            result["fields"][field] = summary

    # Normalize the two intentional field names into one paper-facing statistic.
    nfe = [
        record.get("actual_model_forward_calls", record.get("actual_model_forwards"))
        for record in records
    ]
    result["fields"]["actual_model_forwards_normalized"] = summarize(
        [value for value in nfe if value is not None]
    )

    output = args.output or args.metrics_dir.parent / "metrics_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
