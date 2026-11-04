#!/usr/bin/env python3
"""DDP-safe TRELLIS.2 entry point for matched flat and RDD fine-tuning.

The official entry point uses ``mp.spawn``.  Runtime monkey patches made by a
parent launcher do not survive that spawn, while using ``sitecustomize`` also
pollutes every DataLoader worker.  This entry point performs the small trainer
registration and DataLoader patch *inside each DDP rank* before delegating the
actual training loop to the pinned upstream implementation.
"""

from __future__ import annotations

import argparse
import functools
import glob
import importlib.util
import json
import os
import sys
from pathlib import Path


def _patch_sparse_dataloader() -> None:
    from torch.utils.data import DataLoader

    from trellis2.trainers.flow_matching.sparse_flow_matching import (
        SparseFlowMatchingTrainer,
    )
    from trellis2.utils.data_utils import BalancedResumableSampler, cycle

    # Zero is the conservative default: importing the CUDA sparse extensions in
    # fork/spawned DataLoader workers aborts inside their C++ teardown on the
    # Blackwell pilot host.  The batch is only one SLat + one PNG, so synchronous
    # loading is cheap relative to DINO and the 1.3B denoiser.  Larger runs may
    # opt in after validating their local backend.
    workers = int(os.environ.get("RDD3D_DATALOADER_WORKERS", "0"))
    if workers < 0:
        raise ValueError("RDD3D_DATALOADER_WORKERS must be non-negative")

    def prepare_dataloader(self, **kwargs):
        self.data_sampler = BalancedResumableSampler(
            self.dataset,
            shuffle=True,
            batch_size=self.batch_size_per_gpu,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=workers > 0,
            collate_fn=functools.partial(
                self.dataset.collate_fn, split_size=self.batch_split
            ),
            sampler=self.data_sampler,
        )
        self.data_iterator = cycle(self.dataloader)

    SparseFlowMatchingTrainer.prepare_dataloader = prepare_dataloader


def _register_rdd_trainer() -> None:
    from trellis2 import trainers
    from trellis2_rdd.rdd_sparse_flow_matching import (
        ImageConditionedSparseRDDDetailFlowMatchingCFGTrainer,
    )

    trainers.ImageConditionedSparseRDDDetailFlowMatchingCFGTrainer = (
        ImageConditionedSparseRDDDetailFlowMatchingCFGTrainer
    )


def _rank_main(local_rank, cfg, trainer_mode: str) -> None:
    # Imports stay inside the rank target so DataLoader spawn workers can import
    # this main module without initializing all of TRELLIS/TensorFlow again.
    root = Path(__file__).resolve().parents[1]
    trellis_root = Path(os.environ.get("TRELLIS2_ROOT", root / "third_party" / "TRELLIS.2"))
    entry = trellis_root / "train.py"
    spec = importlib.util.spec_from_file_location("_trellis2_upstream_train", entry)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load TRELLIS.2 training entry: {entry}")
    upstream_train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream_train)

    _patch_sparse_dataloader()
    if trainer_mode == "rdd":
        _register_rdd_trainer()
    try:
        upstream_train.main(local_rank, cfg)
    finally:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()


def _find_ckpt(cfg):
    cfg["load_ckpt"] = None
    if cfg.load_dir:
        if cfg.ckpt == "latest":
            files = glob.glob(os.path.join(cfg.load_dir, "ckpts", "misc_*.pt"))
            if files:
                cfg.load_ckpt = max(
                    int(os.path.basename(path).split("step")[-1].split(".")[0])
                    for path in files
                )
        elif cfg.ckpt != "none":
            cfg.load_ckpt = int(cfg.ckpt)
    return cfg


def main() -> None:
    import torch
    import torch.multiprocessing as mp
    from easydict import EasyDict as edict

    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer-mode", choices=("rdd", "flat"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--load_dir", default="")
    parser.add_argument("--ckpt", default="latest")
    parser.add_argument("--data_dir", default="./data/")
    parser.add_argument("--auto_retry", type=int, default=3)
    parser.add_argument("--tryrun", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--node_rank", type=int, default=0)
    parser.add_argument("--num_gpus", type=int, default=-1)
    parser.add_argument("--master_addr", default="localhost")
    parser.add_argument("--master_port", default="12345")
    opt = parser.parse_args()

    trainer_mode = opt.trainer_mode
    delattr(opt, "trainer_mode")
    opt.load_dir = opt.load_dir or opt.output_dir
    opt.num_gpus = torch.cuda.device_count() if opt.num_gpus == -1 else opt.num_gpus
    config = json.load(open(opt.config, "r", encoding="utf-8"))
    cfg = edict()
    cfg.update(opt.__dict__)
    cfg.update(config)
    print("\n\nConfig:")
    print("=" * 80)
    print(json.dumps(cfg.__dict__, indent=4))

    if cfg.node_rank == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
        with open(os.path.join(cfg.output_dir, "command.txt"), "w", encoding="utf-8") as file:
            print(" ".join(["python"] + sys.argv), file=file)
        with open(os.path.join(cfg.output_dir, "config.json"), "w", encoding="utf-8") as file:
            json.dump(config, file, indent=4)

    attempts = 1 if cfg.auto_retry == 0 else cfg.auto_retry
    for attempt in range(attempts):
        try:
            cfg = _find_ckpt(cfg)
            if cfg.num_gpus > 1:
                mp.spawn(
                    _rank_main,
                    args=(cfg, trainer_mode),
                    nprocs=cfg.num_gpus,
                    join=True,
                )
            else:
                _rank_main(0, cfg, trainer_mode)
            break
        except Exception:
            if cfg.auto_retry == 0 or attempt + 1 == attempts:
                raise
            print(f"Retrying ({attempt + 1}/{attempts})...")


if __name__ == "__main__":
    main()
