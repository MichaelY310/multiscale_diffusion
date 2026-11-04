"""CPU component-test bootstrap; load actual modules, skip eager inference exports.

The unmodified FastVideo root/pipelines __init__ imports unrelated CUDA-only
Triton kernels. Namespace packages avoid those eager exports. No operators,
classes, model forwards or training functions are mocked or replaced. This is
test-only and does not establish normal GPU entrypoint import compatibility.
"""
import importlib
import os
from pathlib import Path
import sys
import types


def bootstrap(snapshot):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("The test bootstrap requires CUDA_VISIBLE_DEVICES empty")
    root = Path(snapshot).resolve()
    if "fastvideo" in sys.modules:
        raise RuntimeError("Bootstrap must run before importing FastVideo")
    packages = ["fastvideo", "fastvideo.configs.pipelines", "fastvideo.pipelines",
                "fastvideo.training", "fastvideo.train", "fastvideo.train.callbacks"]
    for name in packages:
        package = types.ModuleType(name)
        package.__path__ = [str(root.joinpath(*name.split(".")))]
        package.__package__ = name
        sys.modules[name] = package
    pipeline_config = importlib.import_module("fastvideo.configs.pipelines.base")
    sys.modules["fastvideo.configs.pipelines"].PipelineConfig = pipeline_config.PipelineConfig
    info = importlib.import_module("fastvideo.pipelines.pipeline_batch_info")
    for name in ("ForwardBatch", "TrainingBatch"):
        setattr(sys.modules["fastvideo.pipelines"], name, getattr(info, name))
    return {"mode": "CPU component imports", "skipped_eager_package_initializers": packages,
            "replacement_operators": [], "snapshot": str(root)}
