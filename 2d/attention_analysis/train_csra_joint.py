#!/usr/bin/env python3
"""Matched frozen/joint pilot with live coarse caches and coarse anchors."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from train_csra_rollout_output import RolloutDataset
from csra_output_distillation import prediction_loss
from csra_joint_distillation import JointCSRAStudent, PreservedCoarseStudent
from sample_csra_adapter_preview import load_adapter
from train_rdd_csra_adapter import atomic_save, load_frozen_teacher, CSRAAdapterBank, initialize_adapter_weights


class JointDataset(RolloutDataset):
    def __init__(self, root, source_root, split, layers, base_checkpoint):
        super().__init__(root, split, layers, base_checkpoint)
        sources = {f"source{r}_{k}": [] for r in (8, 16) for k in ("x", "t", "target")}
        for manifest_path in sorted(Path(root).glob("shard_*/manifest.json")):
            manifest = json.loads(manifest_path.read_text())
            folder = Path(source_root) / manifest_path.parent.name
            sidecar = json.loads((folder / "manifest.json").read_text())
            if not sidecar["complete"] or sidecar["source_manifest_sha256"] != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
                raise ValueError("Source replay sidecar is incomplete or from another corpus")
            for item in manifest["files"]:
                if item["split"] != split:
                    continue
                data = torch.load(folder / item["name"], map_location="cpu", weights_only=True)
                if data["seed"] != item["seed"] or data["cache_parity_max_abs"] != 0:
                    raise ValueError("Source replay parity failed")
                for r in (8, 16):
                    for k in ("x", "t", "target"):
                        sources[f"source{r}_{k}"].append(data["sources"][str(r)][k])
        self.sources = {k: torch.cat(v) for k, v in sources.items()}
        if any(len(v) != self.images for v in self.sources.values()):
            raise ValueError("Source image count mismatch")
        # Live source forward replaces the old cache bank; release RAM.
        del self.coarse

    def __getitem__(self, index):
        image = int(self.tensors["image_index"][index])
        return {**{k: v[index] for k, v in self.tensors.items() if k != "image_index"},
                **{k: v[image] for k, v in self.sources.items()}}


def device_batch(batch, device):
    return {k: ({layer: x.to(device, non_blocking=True) for layer, x in v.items()}
                if isinstance(v, dict) else v.to(device, non_blocking=True))
            for k, v in batch.items()}


def losses(outputs, batch, args):
    if args.arm.startswith("preserved_"):
        fine, fm = prediction_loss(outputs, batch["target"], args.variance_weight)
        zero = fine.detach().new_zeros(())
        # Placeholder reduction slots only. validate() labels anchors unmeasured.
        return fine, torch.stack([fm["main_mse"], fm["variance_mse"], zero, zero, fine.detach()])
    fine, fm = prediction_loss(outputs[0], batch["target"], args.variance_weight)
    middle, mm = prediction_loss(outputs[1], batch["source16_target"], args.variance_weight)
    coarse, cm = prediction_loss(outputs[2], batch["source8_target"], args.variance_weight)
    total = fine + args.anchor_weight * (middle + coarse) / 2
    return total, torch.stack([fm["main_mse"], fm["variance_mse"], mm["main_mse"], cm["main_mse"], total.detach()])


@torch.no_grad()
def validate(net, loader, device, args):
    stats = torch.zeros(6, device=device, dtype=torch.float64)
    for batch in loader:
        batch = device_batch(batch, device)
        _, row = losses(net(batch), batch, args)
        n = len(batch["t"])
        stats[:5] += row.double() * n
        stats[5] += n
    dist.all_reduce(stats)
    keys = ["fine_main_mse", "fine_variance_mse", "middle_anchor_mse", "coarse_anchor_mse", "total_objective"]
    result = {**dict(zip(keys, (stats[:5]/stats[5]).tolist())), "states": int(stats[5])}
    if args.arm.startswith("preserved_"):
        result.update(middle_anchor_mse=None, coarse_anchor_mse=None,
                      anchor_note="Not evaluated; inference keeps the original 8/16 model unchanged")
    return result


def main(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    if world > 4 or args.global_batch % world:
        raise ValueError("At most four GPUs; global batch must divide world")
    out = Path(args.output_dir)
    if not args.resume and out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    backbone = load_frozen_teacher(Path(args.base_ckpt), device, "DiT-L/2", "rdd")
    adapter, initial = load_adapter(Path(args.init_adapter), device)
    requested_arch = getattr(args, "adapter_architecture", "") or adapter.architecture
    if requested_arch != adapter.architecture:
        if requested_arch in ("coarse_memory_v5", "hidden_memory_v6"):
            # Fresh memory weights must match between aligned/zero controls.
            # Do not change historical initialization for older architectures.
            with torch.random.fork_rng(devices=[local_rank]):
                torch.manual_seed(args.seed)
                adapter = CSRAAdapterBank(adapter.layers, initial["hidden_size"], initial["rank"], requested_arch).to(device)
        else:
            adapter = CSRAAdapterBank(adapter.layers, initial["hidden_size"], initial["rank"], requested_arch).to(device)
        initialize_adapter_weights(adapter, initial)
    if Path(initial["base_checkpoint"]).resolve() != Path(args.base_ckpt).resolve():
        raise ValueError("Initial adapter/base mismatch")
    train_backbone = args.arm in ("joint", "preserved_joint")
    if requested_arch == "hidden_memory_v6":
        if not args.arm.startswith("preserved_") or not args.hidden_root or not args.init_student:
            raise ValueError("V6 requires original-coarse hidden data and a matched zero-source student initializer")
    elif args.hidden_root or args.init_student:
        raise ValueError("Hidden data/student initialization currently supported only for V6")
    if args.coarse_source != "aligned" and not args.arm.startswith("preserved_"):
        raise ValueError("Source controls require unchanged original coarse inference")
    if args.arm.startswith("preserved_"):
        student = PreservedCoarseStudent(backbone, adapter, train_backbone=train_backbone,
                                         coarse_source=args.coarse_source)
        trainset = RolloutDataset(args.data_root, "train", adapter.layers, args.base_ckpt, args.hidden_root)
        valset = RolloutDataset(args.data_root, "validation", adapter.layers, args.base_ckpt, args.hidden_root)
    else:
        student = JointCSRAStudent(backbone, adapter, train_backbone=train_backbone)
        trainset = JointDataset(args.data_root, args.source_root, "train", adapter.layers, args.base_ckpt)
        valset = JointDataset(args.data_root, args.source_root, "validation", adapter.layers, args.base_ckpt)
    if args.init_student:
        ck = torch.load(args.init_student, map_location="cpu", mmap=True, weights_only=False)
        meta = ck["adapter_metadata"]
        if (ck["format"] != "csra_preserved_source_control_v1" or ck["args"]["coarse_source"] != "zero"
            or meta["adapter_architecture"] != "fine_context_v2"
            or Path(ck["base_checkpoint"]).resolve() != Path(args.base_ckpt).resolve()):
            raise ValueError("V6 initialization must be the matched V2 trained zero-source model")
        weights = {k.removeprefix("adapter."):v for k,v in ck["student"].items() if k.startswith("adapter.")}
        initialize_adapter_weights(adapter, {**meta, "adapter":weights})
        backbone.load_state_dict({k.removeprefix("backbone."):v for k,v in ck["student"].items() if k.startswith("backbone.")})
        del weights, ck
    if trainset.seeds & valset.seeds or len(valset) % world:
        raise ValueError("Split leakage or duplicated validation rows")
    per_gpu = args.global_batch // world
    sampler = DistributedSampler(trainset, world, rank, seed=args.seed, drop_last=True)
    val_sampler = DistributedSampler(valset, world, rank, shuffle=False, drop_last=False)
    loader = DataLoader(trainset, batch_size=per_gpu, sampler=sampler, pin_memory=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(valset, batch_size=per_gpu, sampler=val_sampler, pin_memory=True, num_workers=0)
    groups = [{"params": list(adapter.parameters()), "lr": args.adapter_lr}]
    if train_backbone:
        groups.append({"params": [p for p in backbone.parameters() if p.requires_grad], "lr": args.backbone_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=.01)
    step, best = 0, float("inf")
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        for key in ("arm", "seed", "data_root", "source_root", "base_ckpt", "init_adapter", "global_batch", "anchor_weight", "variance_weight", "adapter_lr", "backbone_lr"):
            if payload["args"][key] != vars(args)[key]:
                raise ValueError(f"Resume config changed: {key}")
        if payload["args"].get("adapter_architecture", "") != getattr(args, "adapter_architecture", ""):
            raise ValueError("Resume adapter architecture changed")
        if payload["args"].get("coarse_source", "aligned") != args.coarse_source:
            raise ValueError("Resume coarse source changed")
        for key in ("hidden_root", "init_student"):
            if payload["args"].get(key, "") != getattr(args, key):
                raise ValueError(f"Resume changed: {key}")
        student.load_state_dict(payload["student"])
        optimizer.load_state_dict(payload["optimizer"])
        step, best = payload["step"], payload["best_validation_objective"]
    torch.manual_seed(args.seed + rank)
    network = DDP(student, device_ids=[local_rank], broadcast_buffers=False)
    params = [p for p in student.parameters() if p.requires_grad]

    def save(name):
        checkpoint_format = "csra_preserved_coarse_v1" if args.arm.startswith("preserved_") else "csra_joint_v1"
        if args.coarse_source != "aligned":
            # Old inference loaders must reject this rather than silently feed
            # a coarse cache to a model explicitly trained without one.
            checkpoint_format = "csra_preserved_source_control_v1"
        if adapter.architecture == "hidden_memory_v6":
            checkpoint_format = "csra_preserved_hidden_memory_v1"
        atomic_save({"format": checkpoint_format, "step": step, "arm": args.arm,
                     "base_checkpoint": str(Path(args.base_ckpt).resolve()),
                     "student": student.state_dict(), "optimizer": optimizer.state_dict(),
                     "adapter_metadata": {**{k: initial[k] for k in ("layers", "rank", "hidden_size")},
                                          "adapter_architecture": adapter.architecture},
                     "best_validation_objective": best, "args": vars(args),
                     "trajectory": "offline_teacher_inputs_fixed_original_coarse_cache" if args.arm.startswith("preserved_") else "offline_teacher_inputs_live_student_coarse_cache"}, out / name)

    baseline = validate(student, val_loader, device, args)
    best = min(best, baseline["total_objective"])
    if rank == 0:
        print(json.dumps({"event": "initialized", "arm": args.arm, "step": step,
                          "per_gpu_batch": per_gpu, "accumulation": 1,
                          "trainable_backbone": sum(p.numel() for p in backbone.parameters() if p.requires_grad),
                          "trainable_adapter": sum(p.numel() for p in adapter.parameters()),
                          "validation": baseline, "config": vars(args)}), flush=True)
        if not args.resume and not args.smoke:
            save("best.pt")
    dist.barrier()
    start = time.perf_counter()
    epoch, skip = divmod(step, len(loader))
    while step < args.max_steps:
        sampler.set_epoch(epoch)
        for i, batch in enumerate(loader):
            if i < skip:
                continue
            batch = device_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, row = losses(network(batch), batch, args)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            if step == 0:
                missing = [n for n, p in student.named_parameters() if p.requires_grad and p.grad is None]
                if missing:
                    raise RuntimeError(f"Trainable parameters missing gradients: {missing}")
            optimizer.step()
            step += 1
            if rank == 0 and (step <= 3 or step % 10 == 0):
                print(json.dumps({"event": "train", "step": step, "loss": float(loss),
                                  "fine_mse": float(row[0]), "grad_norm": float(grad_norm),
                                  "seconds": time.perf_counter()-start,
                                  "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
                                  "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30}), flush=True)
            if step % args.validate_every == 0 or step == args.max_steps:
                metrics = validate(student, val_loader, device, args)
                improved = metrics["total_objective"] < best
                if improved:
                    best = metrics["total_objective"]
                if rank == 0:
                    print(json.dumps({"event": "validation", "step": step, **metrics, "new_best": improved}), flush=True)
                    if not args.smoke:
                        if improved:
                            save("best.pt")
                        save("latest.pt")
                dist.barrier()
            if step >= args.max_steps:
                break
        epoch, skip = epoch+1, 0
    if rank == 0:
        (out / "complete.json").write_text(json.dumps({"complete": True, "step": step,
            "smoke": args.smoke, "best_objective": best, "config": vars(args)}, indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["joint", "frozen", "preserved_joint", "preserved_frozen"], required=True)
    for key in ("base-ckpt", "init-adapter", "data-root", "source-root", "output-dir"):
        p.add_argument("--"+key, required=True)
    p.add_argument("--resume", default="")
    p.add_argument("--global-batch", type=int, default=256)
    p.add_argument("--adapter-lr", type=float, default=1e-4)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--anchor-weight", type=float, default=1.)
    p.add_argument("--variance-weight", type=float, default=.05)
    p.add_argument("--max-steps", type=int, default=800)
    p.add_argument("--validate-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=20279000)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--adapter-architecture", choices=["", "fine_context_v2", "full_channel_v4", "coarse_memory_v5", "hidden_memory_v6"], default="")
    p.add_argument("--coarse-source", choices=["aligned", "zero"], default="aligned")
    p.add_argument("--hidden-root", default="")
    p.add_argument("--init-student", default="")
    main(p.parse_args())
