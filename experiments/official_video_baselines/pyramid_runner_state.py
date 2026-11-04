"""External AR counter/RNG checkpoint state for the pinned native PF runner.

This complements, never replaces, the full Accelerator model/optimizer state.
Install the save hook before save_state; restore explicitly AFTER load_state
and before any next forward or data iteration. Data/bucket cursor, LR position,
EMA and accumulation boundaries still require their own recovery validation.
"""
import random
from pathlib import Path

import numpy as np
import torch

from pyramid_smoke import PIN


def _recipe(runner):
    return {
        "max_temporal_length": runner.max_temporal_length,
        "frame_per_unit": runner.frame_per_unit,
        "video_sync_group": runner.video_sync_group,
        "stages": list(runner.stages),
        "sample_ratios": list(runner.sample_ratios),
        "corrupt_ratio": runner.corrupt_ratio,
        "cfg_rate": runner.cfg_rate,
    }


def _path(directory, accelerator):
    return Path(directory) / f"pyramid_runner_rank_{accelerator.process_index}.pt"


def capture_runner_state(runner, accelerator):
    """Capture rank-local state using only weights_only-compatible values."""
    counter = runner.accumulate_steps
    if type(counter) is not int or counter < 0:
        raise ValueError("Invalid native AR counter")
    device = accelerator.device
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU/CUDA RNG state is supported")
    numpy_state = np.random.get_state()
    return {
        "schema_version": 1,
        "source_commit": PIN,
        "rank": accelerator.process_index,
        "world_size": accelerator.num_processes,
        "device_type": device.type,
        "recipe": _recipe(runner),
        "accumulate_steps": counter,
        "python_rng": random.getstate(),
        "numpy_rng": {
            "algorithm": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].astype(np.int64)),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu_rng": torch.get_rng_state(),
        # Access only this rank's device, never all physical GPUs.
        "torch_cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def apply_runner_state(state, runner, accelerator):
    """Reject incompatible metadata before restoring the counter and RNGs."""
    expected = {
        "schema_version": 1,
        "source_commit": PIN,
        "rank": accelerator.process_index,
        "world_size": accelerator.num_processes,
        "device_type": accelerator.device.type,
        "recipe": _recipe(runner),
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(f"Native runner checkpoint mismatch: {key}")
    counter = state["accumulate_steps"]
    if type(counter) is not int or counter < 0:
        raise ValueError("Invalid native AR counter")
    # Validate the CPU RNG payloads without changing the active streams.
    random.Random().setstate(state["python_rng"])
    numpy_rng = state["numpy_rng"]
    keys = numpy_rng["keys"]
    if keys.dtype != torch.int64 or keys.shape != (624,) or bool(((keys < 0) | (keys > 2**32 - 1)).any()):
        raise ValueError("Invalid NumPy MT19937 keys")
    numpy_state = (
        numpy_rng["algorithm"], keys.numpy().astype(np.uint32),
        numpy_rng["position"], numpy_rng["has_gauss"], numpy_rng["cached_gaussian"],
    )
    np.random.RandomState().set_state(numpy_state)
    torch.Generator(device="cpu").set_state(state["torch_cpu_rng"])
    cuda_rng = state["torch_cuda_rng"]
    if accelerator.device.type == "cuda":
        if not isinstance(cuda_rng, torch.Tensor) or cuda_rng.dtype != torch.uint8 or cuda_rng.ndim != 1:
            raise ValueError("Invalid rank-local CUDA RNG state")
        torch.cuda.set_rng_state(cuda_rng, accelerator.device)
    elif cuda_rng is not None:
        raise ValueError("Unexpected CUDA state in CPU checkpoint")
    runner.accumulate_steps = counter
    random.setstate(state["python_rng"])
    np.random.set_state(numpy_state)
    torch.set_rng_state(state["torch_cpu_rng"])


def register_runner_save_hook(accelerator, runner):
    """Save one small sidecar per rank through Accelerator's public hook API."""
    def save_hook(models, weights, output_dir):
        destination = _path(output_dir, accelerator)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(capture_runner_state(runner, accelerator), destination)

    return accelerator.register_save_state_pre_hook(save_hook)


def restore_runner_state_after_load(accelerator, runner, input_dir):
    """Call after Accelerator.load_state; missing sidecars fail explicitly."""
    state = torch.load(_path(input_dir, accelerator), map_location="cpu", weights_only=True)
    apply_runner_state(state, runner, accelerator)
    return state["accumulate_steps"]
