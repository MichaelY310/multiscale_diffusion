"""Long-lived official Wan2.1 baseline sampler for a JSONL video manifest."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_DIR / "Wan2.1"))

from generate_training_data import save_video_mp4


def load_manifest(path):
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for key in ("prompt", "seed", "relative_path"):
                if key not in record:
                    raise ValueError(f"Manifest line {line_number} is missing {key!r}")
            records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", choices=["t2v-1.3B", "t2v-14B"], default="t2v-1.3B")
    parser.add_argument("--size", default="832x448")
    parser.add_argument("--frame-num", type=int, default=81)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--sample-solver", choices=["unipc", "dpm++"], default="unipc")
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-rank", type=int, default=0)
    parser.add_argument("--offload-model", action="store_true")
    parser.add_argument("--t5-cpu", action="store_true")
    args = parser.parse_args()

    if not (0 <= args.shard_rank < args.num_shards):
        parser.error("--shard-rank must be in [0, --num-shards)")
    if not torch.cuda.is_available():
        raise RuntimeError("VBench sampling requires CUDA")

    from wan.configs import WAN_CONFIGS
    from wan.text2video import WanT2V

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    all_records = load_manifest(args.manifest)
    records = all_records[args.shard_rank :: args.num_shards]
    print(
        f"Shard {args.shard_rank}/{args.num_shards}: {len(records)}/{len(all_records)} videos"
    )
    pipe = WanT2V(
        WAN_CONFIGS[args.model],
        checkpoint_dir=args.pretrained_dir,
        device_id=args.device,
        t5_cpu=args.t5_cpu,
    )
    width, height = [int(value) for value in args.size.split("x")]

    completed = 0
    for record_index, record in enumerate(records, 1):
        output_path = output_dir / record["relative_path"]
        metric_path = metrics_dir / (output_path.name + ".json")
        if output_path.is_file() and metric_path.is_file():
            completed += 1
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        video = pipe.generate(
            record["prompt"],
            size=(width, height),
            frame_num=args.frame_num,
            shift=args.shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sampling_steps,
            guide_scale=args.guidance_scale,
            seed=int(record["seed"]),
            offload_model=args.offload_model,
        )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        save_video_mp4(video, str(output_path), fps=args.fps)
        metrics = {
            "shard_rank": args.shard_rank,
            "shard_record_index": record_index,
            "pretrained_dir": str(Path(args.pretrained_dir).resolve()),
            "prompt": record["prompt"],
            "seed": int(record["seed"]),
            "sample_index": record.get("sample_index"),
            "dimensions": record.get("dimensions", []),
            "model": args.model,
            "solver": args.sample_solver,
            "scheduled_updates": args.sampling_steps,
            "conditional_model_forwards": args.sampling_steps,
            "unconditional_model_forwards": args.sampling_steps,
            "actual_model_forwards": 2 * args.sampling_steps,
            "guidance_scale": args.guidance_scale,
            "shift": args.shift,
            "size": args.size,
            "frame_num": args.frame_num,
            "end_to_end_inference_seconds": elapsed,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "output": str(output_path.resolve()),
        }
        metric_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        completed += 1
        print(f"[{record_index}/{len(records)}] {output_path.name} | total={elapsed:.2f}s")

    print(f"Shard complete: {completed}/{len(records)} videos available")


if __name__ == "__main__":
    main()
