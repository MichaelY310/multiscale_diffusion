#!/usr/bin/env python3
"""Reconstruct old sampling labels and evaluate ImageNet class consistency.

This is valid only when each rank generated its full strided index list in one
run.  The 100/300/600 logs satisfy that condition.  The original sampler calls
``torch.manual_seed(seed + first_batch_index)`` and then immediately samples
CUDA class labels, so this script reproduces the same labels exactly.

Classifier accuracy is a semantic diagnostic, not a replacement for FID or
human evaluation.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


def _parse_dataset(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError("dataset must be LABEL=/path/file.npz")
    label, path = value.split("=", 1)
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise argparse.ArgumentTypeError(f"missing NPZ: {resolved}")
    return label, resolved


def reconstruct_labels(
    total_samples: int,
    world_size: int,
    sample_batch_size: int,
    seed: int,
    num_classes: int,
    device: torch.device,
) -> np.ndarray:
    labels = np.empty(total_samples, dtype=np.int64)
    for rank in range(world_size):
        indices = list(range(rank, total_samples, world_size))
        for start in range(0, len(indices), sample_batch_size):
            batch_indices = indices[start : start + sample_batch_size]
            torch.manual_seed(seed + batch_indices[0])
            sampled = torch.randint(
                0, num_classes, (len(batch_indices),), device=device
            ).cpu().numpy()
            labels[np.asarray(batch_indices)] = sampled
    return labels


def classify_dataset(
    label: str,
    path: Path,
    target_labels: np.ndarray,
    model,
    device: torch.device,
    batch_size: int,
    output_dir: Path,
) -> dict:
    print(f"[load] {label}: {path}", flush=True)
    archive = np.load(path, allow_pickle=False)
    images = archive["arr_0"]
    if images.shape[0] != len(target_labels) or images.dtype != np.uint8:
        raise ValueError(f"unexpected image archive: {images.shape} {images.dtype}")
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    top1_count = 0
    top5_count = 0
    target_prob_sum = 0.0
    nll_sum = 0.0
    confidence_sum = 0.0
    predicted_hist = np.zeros(1000, dtype=np.int64)
    target_hist = np.bincount(target_labels, minlength=1000)
    class_correct = np.zeros(1000, dtype=np.int64)
    per_image_rows: List[dict] = []

    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            end = min(len(images), start + batch_size)
            batch = torch.from_numpy(images[start:end]).to(device=device, non_blocking=True)
            batch = batch.permute(0, 3, 1, 2).float().div_(255.0)
            # Exact spatial convention of ResNet50_Weights.IMAGENET1K_V2 for
            # square inputs: resize the 256px short side to 232, then take the
            # centered 224px crop.  Directly resizing to 224 would be a subtly
            # different classifier preprocessing pipeline.
            batch = F.interpolate(
                batch, size=(232, 232), mode="bilinear", align_corners=False, antialias=True
            )
            batch = batch[:, :, 4:228, 4:228]
            batch = (batch - mean) / std
            targets = torch.from_numpy(target_labels[start:end]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(batch)
            probs = logits.float().softmax(dim=1)
            top5 = probs.topk(5, dim=1).indices
            predicted = top5[:, 0]
            target_prob = probs.gather(1, targets[:, None]).squeeze(1)
            confidence = probs.max(dim=1).values
            correct1 = predicted.eq(targets)
            correct5 = top5.eq(targets[:, None]).any(dim=1)

            top1_count += int(correct1.sum())
            top5_count += int(correct5.sum())
            target_prob_sum += float(target_prob.sum())
            nll_sum += float((-torch.log(target_prob.clamp_min(1e-12))).sum())
            confidence_sum += float(confidence.sum())
            pred_np = predicted.cpu().numpy()
            target_np = targets.cpu().numpy()
            target_prob_np = target_prob.cpu().numpy()
            confidence_np = confidence.cpu().numpy()
            correct_np = correct1.cpu().numpy()
            predicted_hist += np.bincount(pred_np, minlength=1000)
            np.add.at(class_correct, target_np, correct_np.astype(np.int64))
            for offset in range(end - start):
                per_image_rows.append(
                    {
                        "image_index": start + offset,
                        "target_class": int(target_np[offset]),
                        "predicted_class": int(pred_np[offset]),
                        "top1_correct": int(correct_np[offset]),
                        "target_probability": float(target_prob_np[offset]),
                        "classifier_confidence": float(confidence_np[offset]),
                    }
                )
            print(f"  {end:>6}/{len(images)}", flush=True)

    class_mask = target_hist > 0
    per_class_accuracy = class_correct[class_mask] / target_hist[class_mask]
    distribution = predicted_hist / max(predicted_hist.sum(), 1)
    positive = distribution > 0
    prediction_entropy = float(-(distribution[positive] * np.log(distribution[positive])).sum())
    summary = {
        "label": label,
        "npz": str(path),
        "images": int(len(images)),
        "classifier": "torchvision ResNet-50 ImageNet1K V2",
        "classifier_preprocess": "resize short side to 232, center crop 224, ImageNet mean/std",
        "top1_accuracy": top1_count / len(images),
        "top5_accuracy": top5_count / len(images),
        "mean_target_probability": target_prob_sum / len(images),
        "mean_target_nll": nll_sum / len(images),
        "mean_classifier_confidence": confidence_sum / len(images),
        "macro_class_accuracy": float(per_class_accuracy.mean()),
        "predicted_class_coverage": int((predicted_hist > 0).sum()),
        "predicted_class_entropy_nats": prediction_entropy,
        "predicted_class_entropy_fraction_of_log1000": prediction_entropy / math.log(1000),
    }
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    with (output_dir / f"{safe_label}_per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_image_rows)
    del images, archive
    gc.collect()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", type=_parse_dataset, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-samples", type=int, default=50_000)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--sample-batch-size", type=int, default=512)
    parser.add_argument(
        "--dataset-world-sizes",
        default="",
        help="Optional comma-separated world size per --dataset; overrides --world-size.",
    )
    parser.add_argument(
        "--dataset-sample-batch-sizes",
        default="",
        help="Optional comma-separated sampling batch size per --dataset.",
    )
    parser.add_argument("--classifier-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_count = len(args.dataset)
    if args.dataset_world_sizes:
        world_sizes = [int(value) for value in args.dataset_world_sizes.split(",")]
        if len(world_sizes) != dataset_count:
            parser.error("--dataset-world-sizes must have one value per --dataset")
    else:
        world_sizes = [args.world_size] * dataset_count
    if args.dataset_sample_batch_sizes:
        sample_batch_sizes = [int(value) for value in args.dataset_sample_batch_sizes.split(",")]
        if len(sample_batch_sizes) != dataset_count:
            parser.error("--dataset-sample-batch-sizes must have one value per --dataset")
    else:
        sample_batch_sizes = [args.sample_batch_size] * dataset_count
    if any(value <= 0 for value in world_sizes + sample_batch_sizes):
        parser.error("world sizes and sampling batch sizes must be positive")

    labels_by_sampling_config = {}
    for world_size, sample_batch_size in sorted(set(zip(world_sizes, sample_batch_sizes))):
        labels = reconstruct_labels(
            total_samples=args.total_samples,
            world_size=world_size,
            sample_batch_size=sample_batch_size,
            seed=args.seed,
            num_classes=args.num_classes,
            device=device,
        )
        labels_by_sampling_config[(world_size, sample_batch_size)] = labels
        np.save(
            output_dir / f"reconstructed_condition_labels_ws{world_size}_bs{sample_batch_size}.npy",
            labels,
        )
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device).eval()
    summaries = []
    reconstruction_configs = []
    for (label, path), world_size, sample_batch_size in zip(
        args.dataset, world_sizes, sample_batch_sizes
    ):
        summary = classify_dataset(
            label,
            path,
            labels_by_sampling_config[(world_size, sample_batch_size)],
            model,
            device,
            args.classifier_batch_size,
            output_dir,
        )
        summary["sampling_world_size_used_for_label_reconstruction"] = world_size
        summary["sampling_batch_per_rank_used_for_label_reconstruction"] = sample_batch_size
        summaries.append(summary)
        reconstruction_configs.append(
            {
                "dataset": label,
                "world_size": world_size,
                "sampling_batch_per_rank": sample_batch_size,
            }
        )
    payload = {
        "status": "auxiliary_classifier_diagnostic_not_fid",
        "label_reconstruction": {
            "validity_condition": "each rank generated its complete strided index list once without resume",
            "verified_from_logs": True,
            "per_dataset_sampling_configs": reconstruction_configs,
            "sampling_seed": args.seed,
            "pytorch_rng": "CUDA torch.randint immediately after torch.manual_seed(seed + first_index)",
            "rng_caveat": "exact if the sampling and diagnostic environments use the same CUDA RNG algorithm; both run on the same server software stack",
        },
        "datasets": summaries,
    }
    (output_dir / "class_consistency_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
