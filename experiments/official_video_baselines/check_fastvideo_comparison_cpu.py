"""Run the migrated controlled suite on pinned FastVideo CPU components."""
import argparse
import importlib.metadata as md
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fastvideo", type=Path, required=True)
    p.add_argument("--rdd", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--short-deadline", action="store_true")
    args = p.parse_args()
    if args.short_deadline:
        os.environ["RDD_DEADLINE_UNIX"] = str(time.time() + 120)
    from fastvideo_cpu_import import bootstrap
    compatibility = bootstrap(args.fastvideo)
    sys.path.append(str(args.rdd.resolve()))  # Pinned FastVideo stays first.
    # This existing test supplies a Conv3d CPU surrogate, never a Wan pipeline.
    # Exclude the unused inference-preset lookup which eagerly imports CUDA.
    import yaml
    configs = Path(tempfile.mkdtemp(prefix="rdd-controlled-cpu-configs-"))
    for name in ("fixed", "rdd", "pyramid_flow", "pyramidal_wan"):
        cfg = yaml.safe_load((args.rdd / f"rdd/comparison/configs/{name}.yaml").read_text())
        cfg.pop("pipeline", None)
        (configs / f"{name}.yaml").write_text(yaml.safe_dump(cfg))
    sys.argv = ["cpu_smoke", "--config-root", str(configs),
                "--output", str(args.output)]
    runpy.run_module("rdd.comparison.cpu_smoke", run_name="__main__")
    result = json.loads(args.output.read_text())
    result.update(compatibility=compatibility, temporary_configs=str(configs),
        config_change="CPU surrogate only: unused inference pipeline stanza omitted",
        native_pyramidal_wan=False,
        short_120s_deadline_check=args.short_deadline,
        packages={name: md.version(name) for name in ("torch", "accelerate", "diffusers", "transformers")})
    if result["cuda_context_initialized"]:
        raise AssertionError("Unexpected CUDA context")
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
