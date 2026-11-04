"""Opt-in state for the pinned native PF temporal data pipeline.

The official bucket selection/collation methods are inherited unchanged.
TorchData saves the reader/sampler position and prefetched worker snapshots;
the dataset adapter adds each worker's Python/NumPy/Torch CPU RNG state.
Restore the data iterator first, then the rank-local runner/main RNG sidecar.
No model, trainer, optimizer or method implementation is introduced here.
"""
import copy
import importlib.metadata as md
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from dataset.bucket_loader import TemporalLengthBucketeer
from dataset.dataloaders import identity
from pyramid_smoke import PIN


class WorkerRNGDataset(Dataset):
    """Wrap a cached map-style dataset without changing __getitem__ values."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def state_dict(self):
        state = np.random.get_state()
        return {
            "python": random.getstate(), "torch": torch.get_rng_state(),
            "numpy": (state[0], state[1].tolist(), state[2], state[3], state[4]),
        }

    def load_state_dict(self, state):
        numpy_state = state["numpy"]
        numpy_state = (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
        # Validate before changing any active RNG stream.
        random.Random().setstate(state["python"])
        np.random.RandomState().set_state(numpy_state)
        torch.Generator(device="cpu").set_state(state["torch"])
        random.setstate(state["python"])
        np.random.set_state(numpy_state)
        torch.set_rng_state(state["torch"])


class StatefulNativeTemporalBucketeer(TemporalLengthBucketeer):
    """Only state/load/count additions; __next__ delegates to official code."""
    def __init__(self, dataloader, max_frames, epoch, dataset_fingerprint):
        if not isinstance(dataloader, StatefulDataLoader):
            raise ValueError("Native recovery requires a StatefulDataLoader")
        if not isinstance(dataloader.sampler, DistributedSampler):
            raise ValueError("Only the official distributed sampler path is supported")
        if not isinstance(dataset_fingerprint, str) or not dataset_fingerprint:
            raise ValueError("An immutable dataset/cache fingerprint is required")
        if md.version("torchdata") != "0.10.0":
            raise ValueError("This recovery adapter is validated against TorchData 0.10.0")
        self.max_frames = max_frames
        self.dataset_fingerprint = dataset_fingerprint
        self.emitted_batches = 0
        super().__init__(dataloader, max_frames=max_frames, epoch=epoch)

    def __next__(self):
        result = super().__next__()
        self.emitted_batches += 1
        return result

    def _config(self):
        loader = self._dataloader
        sampler = loader.sampler
        return {
            "batch_size": loader.batch_size, "num_workers": loader.num_workers,
            "max_frames": self.max_frames, "dataset_length": len(loader.dataset),
            "dataset_fingerprint": self.dataset_fingerprint,
            "sampler_rank": sampler.rank, "sampler_world_size": sampler.num_replicas,
            "sampler_seed": sampler.seed, "sampler_shuffle": sampler.shuffle,
            "sampler_drop_last": sampler.drop_last, "loader_drop_last": loader.drop_last,
            "prefetch_factor": loader.prefetch_factor,
            "multiprocessing_context": loader.multiprocessing_context.get_start_method()
                if loader.multiprocessing_context is not None else None,
            # TorchData 0.10 has ordered delivery and no in_order argument.
            "in_order": True, "snapshot_every_n_steps": loader.snapshot_every_n_steps,
            "torchdata_version": md.version("torchdata"), "torch_version": str(torch.__version__),
        }

    def state_dict(self):
        return {
            "schema_version": 1, "source_commit": PIN, "config": self._config(),
            # The bucket epoch initially equals args.seed whereas sampler.epoch
            # initially equals 0. They must be saved separately.
            "bucket_epoch": self._epoch, "sampler_epoch": self._dataloader.sampler.epoch,
            "emitted_batches": self.emitted_batches,
            "buckets": copy.deepcopy(self.buckets),
            "reader": self._dataloader.state_dict(),
        }

    def load_state_dict(self, state):
        for key, value in {"schema_version": 1, "source_commit": PIN, "config": self._config()}.items():
            if state.get(key) != value:
                raise ValueError(f"Native data state mismatch: {key}")
        for key in ["bucket_epoch", "sampler_epoch", "emitted_batches"]:
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError(f"Invalid native data counter: {key}")
        buckets = state["buckets"]
        if set(buckets) != set(range(1, self.max_frames + 1)):
            raise ValueError("Native temporal bucket keys changed")
        for frames, entries in buckets.items():
            if not isinstance(entries, list):
                raise ValueError("Invalid bucket payload")
            for entry in entries:
                video = entry.get("video")
                if not isinstance(video, torch.Tensor) or video.device.type != "cpu" or video.ndim != 5 or video.shape[2] != frames:
                    raise ValueError("Invalid cached bucket video")
        self.close()
        self._epoch = state["bucket_epoch"]
        self._dataloader.sampler.set_epoch(state["sampler_epoch"])
        self._dataloader.load_state_dict(state["reader"])
        # Materialize now: lazy reader/dataset restoration can otherwise consume
        # or overwrite the main RNG *after* the runner's final restore.
        self.iterator = iter(self._dataloader)
        self.buckets = copy.deepcopy(buckets)
        self.emitted_batches = state["emitted_batches"]

    def close(self):
        # TorchData exposes no public close; this version-pinned shutdown acts
        # only on the workers created by this loader, with no external signals.
        iterator = getattr(self, "iterator", None)
        if hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()


def build_stateful_native_loader(dataset, *, batch_size, max_frames, num_workers,
                                 rank, world_size, epoch, dataset_fingerprint):
    """Match the official builder's sampler/collation/drop_last semantics."""
    sampler = DistributedSampler(dataset, shuffle=True, num_replicas=world_size, rank=rank, seed=epoch)
    if len(sampler)<batch_size:
        raise ValueError('Native drop_last loader requires at least one complete batch per rank')
    kwargs = {"multiprocessing_context": "spawn"} if num_workers else {}
    loader = StatefulDataLoader(
        WorkerRNGDataset(dataset), batch_size=batch_size, num_workers=num_workers,
        pin_memory=True, sampler=sampler, collate_fn=identity, drop_last=True,
        snapshot_every_n_steps=1, **kwargs,
    )
    return StatefulNativeTemporalBucketeer(loader, max_frames, epoch, dataset_fingerprint)
