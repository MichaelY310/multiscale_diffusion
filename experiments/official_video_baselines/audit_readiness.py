"""CPU-only, small readiness audit; never downloads or initializes CUDA."""
import datetime as dt
import importlib.metadata as md
import json
import os
from pathlib import Path
import shutil
import subprocess
from zoneinfo import ZoneInfo
from launch_smoke import inventory, occupants


def main():
    root = Path(__file__).resolve().parent
    model = root / "assets/pyramid-flow-miniflux"
    now = dt.datetime.now(dt.timezone.utc)
    cards = inventory()
    available = shutil.disk_usage(root).free
    # Native training loads FP32 weights and AdamW has two FP32 moments.
    checkpoint = 1972059200 * 4 * 3
    components = ["diffusion_transformer_384p", "causal_video_vae", "text_encoder",
                  "text_encoder_2", "tokenizer", "tokenizer_2"]
    caches = [Path.home() / ".cache/huggingface/hub", Path("/scr/yuheng/.cache/huggingface/hub")]
    result = {"utc": now.isoformat(), "la": now.astimezone(ZoneInfo("America/Los_Angeles")).isoformat(),
              "gpu_cutoff_la": "08:00", "gpu_task_max_seconds": 290, "gpu_max_cards": 2,
              "gpu_inventory": cards, "selected_occupants": occupants({cards[g]["uuid"] for g in [1, 2]}),
              "disk_free_gib": available / 2**30, "smoke_reserve_gib": 80,
              "full_native_checkpoint_min_gib": checkpoint / 2**30,
              "single_checkpoint_room_with_reserve": available >= 80*2**30 + checkpoint + 2**30,
              "two_checkpoint_room_with_reserve": available >= 80*2**30 + 2*checkpoint + 2**30,
              "components_at_staged_model": {c: (model/c).is_dir() for c in components},
              "existing_pyramid_hf_cache_dirs": [str(p) for cache in caches if cache.is_dir()
                    for p in cache.glob("models--*pyramid*") if p.is_dir()],
              "packages": {p: md.version(p) for p in ["torch", "accelerate", "transformers", "diffusers", "flash-attn"]},
              "fastvideo_pin": "37dd31d5eadfa726e102702ec1ba1cf7c977eaa1",
              "snapshot_created_from_tracked_git_archive": True,
              "quality_evaluation": "not run"}
    (root / "readiness_audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
