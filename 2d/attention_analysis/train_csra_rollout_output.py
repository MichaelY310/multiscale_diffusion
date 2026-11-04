#!/usr/bin/env python3
"""Pilot: distill final network outputs on saved real RDD teacher trajectories.

This is offline teacher-trajectory distillation, not online student rollout
training and not backpropagation through the whole diffusion trajectory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from csra_output_distillation import CSRAOutputStudent, prediction_loss
from csra_runtime import predict_adapter
from probe_rdd_vs_dit_block_reuse import manual_forward
from sample_csra_adapter_preview import load_adapter
from train_rdd_csra_adapter import (CSRAAdapterBank, atomic_save, initialize_adapter_weights,
                                   load_frozen_teacher, upsample_tokens)


class RolloutDataset(Dataset):
    def __init__(self, root, split, layers, base_checkpoint, hidden_root=""):
        manifests = sorted(Path(root).glob("shard_*/manifest.json"))
        if not manifests:
            raise ValueError("No completed dataset shard manifests")
        chunks = {k: [] for k in ("x", "t", "labels", "target", "image_index")}
        coarse = {k: [] for k in layers}
        self.seeds = set()
        offset = 0
        total_shards = None
        for expected_shard, manifest_path in enumerate(manifests):
            manifest = json.loads(manifest_path.read_text())
            if not manifest["complete"] or not manifest["parity_checked"] or manifest["parity_max_abs"] != 0:
                raise ValueError("Incomplete shard or failed collector parity")
            if manifest["shard"] != expected_shard or manifest["layers"] != layers:
                raise ValueError("Unexpected shard order or layer mask")
            if total_shards is None:
                total_shards = manifest["shards"]
            if total_shards != manifest["shards"] or len(manifests) != total_shards:
                raise ValueError("Missing/inconsistent dataset shards")
            if Path(manifest["base_checkpoint"]).resolve() != Path(base_checkpoint).resolve():
                raise ValueError("Dataset and frozen model mismatch")
            hidden_folder = Path(hidden_root) / manifest_path.parent.name if hidden_root else None
            if hidden_folder is not None:
                side = json.loads((hidden_folder / "manifest.json").read_text())
                digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                if not side.get("complete") or side.get("feature") != "output" or side.get("source_manifest_sha256") != digest:
                    raise ValueError("Hidden-state sidecar metadata mismatch")
            for item in manifest["files"]:
                if item["split"] != split:
                    continue
                name = item["name"]
                if Path(name).name != name:
                    raise ValueError("Dataset file must be inside its shard")
                data = torch.load(manifest_path.parent / name, map_location="cpu", weights_only=True)
                if data["split"] != split or data["layers"] != layers or data["seed"] in self.seeds:
                    raise ValueError("Duplicate seed or mismatched example metadata")
                self.seeds.add(data["seed"])
                count, batch = data["x"].shape[:2]
                if count != 8 or data["x"].shape[2:] != (4, 32, 32) or data["target"].shape != (count, batch, 8, 32, 32):
                    raise ValueError("Unexpected trajectory tensor shape")
                if not torch.all(data["source_timestep"] == 300):
                    raise ValueError("Unexpected source timestep")
                chunks["x"].append(data["x"].flatten(0, 1))
                chunks["target"].append(data["target"].flatten(0, 1))
                chunks["t"].append(data["t"].flatten().long())
                chunks["labels"].append(data["labels"].repeat(count).long())
                chunks["image_index"].append(torch.arange(batch).repeat(count) + offset)
                if hidden_folder is not None:
                    hidden = torch.load(hidden_folder / name, map_location="cpu", weights_only=True)
                    if (hidden["seed"] != data["seed"] or hidden["split"] != split
                        or hidden["feature"] != "output" or hidden["parity_max_abs"] != 0
                        or hidden["source_manifest_sha256"] != digest):
                        raise ValueError("Invalid hidden-state sidecar record")
                    if not torch.equal(hidden["labels"], data["labels"]):
                        raise ValueError("Hidden-state labels differ")
                    data["coarse"] = hidden["coarse"]
                for layer in layers:
                    if data["coarse"][layer].shape[0:2] != (batch, 64):
                        raise ValueError("Unexpected native middle-stage cache shape")
                    coarse[layer].append(data["coarse"][layer])
                offset += batch
        if not offset:
            raise ValueError(f"Empty {split} dataset")
        self.tensors = {k: torch.cat(v) for k, v in chunks.items()}
        self.coarse = {k: torch.cat(v) for k, v in coarse.items()}
        self.images = offset

    def __len__(self):
        return self.tensors["x"].shape[0]

    def __getitem__(self, index):
        image_index = int(self.tensors["image_index"][index])
        return {
            **{k: v[index] for k, v in self.tensors.items() if k != "image_index"},
            "coarse": {k: v[image_index] for k, v in self.coarse.items()},
        }


def to_device(batch, device):
    return {k: ({l: x.to(device, non_blocking=True) for l, x in v.items()}
                if k == "coarse" else v.to(device, non_blocking=True))
            for k, v in batch.items()}


def run_student(student, batch):
    return student(batch["x"], batch["t"], batch["labels"], batch["coarse"])


@torch.no_grad()
def validate(student, loader, device, variance_weight):
    totals = torch.zeros(3, device=device, dtype=torch.float64)
    for batch in loader:
        batch = to_device(batch, device)
        output = run_student(student, batch)
        _, metrics = prediction_loss(output, batch["target"], variance_weight)
        n = len(batch["t"])
        totals += torch.tensor([float(metrics["main_mse"]) * n,
                                float(metrics["variance_mse"]) * n, n], device=device)
    dist.all_reduce(totals)
    return {"main_mse": float(totals[0] / totals[2]),
            "variance_mse": float(totals[1] / totals[2]), "states": int(totals[2])}


def train(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", local_rank)
    if world > 4 or args.global_batch_size % world:
        raise ValueError("Require at most4 GPUs and a divisible global batch")
    output = Path(args.output_dir)
    if not args.resume and output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite an existing training output")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed + rank)
    teacher = load_frozen_teacher(Path(args.base_ckpt), device, "DiT-L/2", "rdd")
    adapter, initial = load_adapter(Path(args.resume or args.init_adapter), device)
    requested_arch = getattr(args, "adapter_architecture", "") or adapter.architecture
    if requested_arch != adapter.architecture:
        if args.resume:
            raise ValueError("Architecture changes require weights-only initialization, not optimizer resume")
        adapter = CSRAAdapterBank(adapter.layers, int(initial["hidden_size"]),
                                  int(initial["rank"]), requested_arch).to(device)
        initialize_adapter_weights(adapter, initial)
    if Path(initial["base_checkpoint"]).resolve() != Path(args.base_ckpt).resolve():
        raise ValueError("Adapter and teacher checkpoint mismatch")
    adapter.requires_grad_(True).train()
    student = CSRAOutputStudent(teacher, adapter, checkpoint_blocks=True)
    # Matched architecture arms use identical data RNG after model construction.
    if getattr(args, "adapter_architecture", ""):
        torch.manual_seed(args.seed + rank)
    trainset = RolloutDataset(args.data_root, "train", adapter.layers, args.base_ckpt)
    validation_root = getattr(args, "validation_data_root", "") or args.data_root
    valset = RolloutDataset(validation_root, "validation", adapter.layers, args.base_ckpt)
    if trainset.seeds & valset.seeds:
        raise ValueError("Training/validation seed leakage")
    batch_size = args.global_batch_size // world
    train_sampler = DistributedSampler(trainset, world, rank, shuffle=True, seed=args.seed, drop_last=True)
    val_sampler = DistributedSampler(valset, world, rank, shuffle=False, drop_last=False)
    if len(valset) % world:
        raise ValueError("Validation must divide evenly: avoid duplicated validation rows")
    loader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler,
                        num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(valset, batch_size=batch_size, sampler=val_sampler,
                            num_workers=0, pin_memory=True)
    if len(loader) == 0:
        raise ValueError("Not enough training states for one full batch")
    # Actual-checkpoint forward parity against the existing sampling adapter.
    check = to_device(next(iter(val_loader)), device)
    with torch.no_grad():
        actual = run_student(student, check)
        callbacks = {layer: (lambda hidden, condition, layer=layer: predict_adapter(
            adapter.adapters[str(layer)], upsample_tokens(check["coarse"][layer], 16),
            hidden, check["t"], "fp32", condition=condition)) for layer in adapter.layers}
        reference, _ = manual_forward(teacher, check["x"], check["t"], check["labels"],
                                      multires=True, replacements=callbacks,
                                      conditioned_replacements=True)
        parity = (actual - reference).abs().max().float()
        dist.all_reduce(parity, op=dist.ReduceOp.MAX)
        if float(parity) != 0:
            raise RuntimeError(f"Training and sampling forwards differ: {float(parity)}")
    del check, actual, reference, callbacks
    network = DDP(student, device_ids=[local_rank], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate, weight_decay=.01)
    step = 0
    if args.resume:
        if initial.get("training_objective") != "offline_rollout_final_output":
            raise ValueError("Cannot resume optimizer from another objective")
        optimizer.load_state_dict(initial["optimizer"])
        step = int(initial["step"])
    best = initial.get("best_validation_main_mse", float("inf")) if args.resume else float("inf")

    def save(path):
        atomic_save({"version": 2, "method": "RDD-CSRA offline rollout output distillation",
                     "training_objective": "offline_rollout_final_output", "step": step,
                     "base_checkpoint": str(Path(args.base_ckpt).resolve()),
                     "adapter_architecture": adapter.architecture, "layers": adapter.layers,
                     "rank": int(initial["rank"]), "hidden_size": int(initial["hidden_size"]),
                     "adapter": adapter.state_dict(), "optimizer": optimizer.state_dict(),
                     "best_validation_main_mse": best, "args": vars(args)}, path)

    baseline = validate(student, val_loader, device, args.variance_weight)
    # The unmodified initializer is a valid candidate on a new validation set.
    # Otherwise the first update could be selected even when it made things worse.
    best = min(best, baseline["main_mse"])
    if rank == 0 and not args.resume:
        save(output / "best.pt")
    if rank == 0:
        print(json.dumps({"event": "initialized", "step": step, "world": world,
                          "per_gpu_batch": batch_size, "layers": adapter.layers,
                          "adapter_architecture": adapter.architecture,
                          "trainable_parameters": sum(p.numel() for p in adapter.parameters()),
                          "train_states": len(trainset), "train_images": trainset.images,
                          "validation_data_root": str(Path(validation_root).resolve()),
                          "val_images": valset.images, "validation": baseline,
                          "forward_parity_max_abs": float(parity)}), flush=True)
    start = time.perf_counter()
    running = 0.0
    epoch = step // len(loader)
    skip = step % len(loader)
    while step < args.max_steps:
        train_sampler.set_epoch(epoch)
        for index, batch in enumerate(loader):
            if index < skip:
                continue
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = run_student(network, batch)
            loss, _ = prediction_loss(prediction, batch["target"], args.variance_weight)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite output loss")
            loss.backward()
            if step == 0:
                if any(p.grad is not None for p in teacher.parameters()):
                    raise RuntimeError("Frozen teacher unexpectedly received gradients")
                for layer in adapter.layers:
                    norm = sum(float(p.grad.abs().sum()) for p in adapter.adapters[str(layer)].parameters() if p.grad is not None)
                    if not norm > 0:
                        raise RuntimeError(f"No gradient reaches adapter layer {layer}")
                    branch = adapter.adapters[str(layer)]
                    if getattr(branch, "requires_condition", False):
                        grad = branch.condition_mlp[-1].weight.grad
                        if grad is None or not float(grad.abs().sum()) > 0:
                            raise RuntimeError(f"No gradient reaches current-condition FiLM at layer {layer}")
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            running += float(loss.detach())
            step += 1
            if step % 10 == 0:
                value = torch.tensor(running / 10, device=device)
                dist.all_reduce(value)
                if rank == 0:
                    print(json.dumps({"event": "train", "step": step, "loss": float(value/world),
                                      "elapsed_seconds": time.perf_counter()-start,
                                      "max_memory_gib": torch.cuda.max_memory_allocated()/2**30}), flush=True)
                running = 0.0
            if step % args.validate_every == 0 or step == args.max_steps:
                metrics = validate(student, val_loader, device, args.variance_weight)
                is_best = metrics["main_mse"] < best
                best = min(best, metrics["main_mse"])
                if rank == 0:
                    print(json.dumps({"event": "validation", "step": step, **metrics}), flush=True)
                    if is_best:
                        save(output / "best.pt")
            if step % args.checkpoint_every == 0 or step == args.max_steps:
                if rank == 0:
                    save(output / f"{step:07d}.pt")
                    save(output / "latest.pt")
                dist.barrier()
            if step >= args.max_steps:
                break
        skip = 0
        epoch += 1
    if rank == 0:
        (output / "complete.json").write_text(json.dumps({"complete": True, "step": step,
            "best_validation_main_mse": best, "config": vars(args)}, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ckpt", required=True)
    parser.add_argument("--init-adapter", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--validation-data-root", default="",
                        help="Optional separate corpus for a matched validation set; defaults to data-root.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--adapter-architecture", default="",
                        choices=["", "fine_context_v2", "teacher_condition_v3"],
                        help="Optional weights-only V2->zero-initialized conditioned V3 conversion.")
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--variance-weight", type=float, default=.05)
    parser.add_argument("--validate-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20262000)
    train(parser.parse_args())
