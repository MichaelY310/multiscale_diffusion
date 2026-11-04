"""
Train a stage-specific noisy classifier for RDD / diffusion-with-drift.

This script trains one classifier per RDD stage:
    stage=64  -> coarse stage, no drift, t in [600, 1000)
    stage=128 -> middle stage, drift on, t in [300, 600)
    stage=256 -> fine stage, drift on, t in [0, 300)

The classifier input distribution matches the stage diffusion process exactly:
    x_t = q_stage(x_t | x_0_stage)

For drift stages, this means the classifier sees the RDD noised input rather
than the standard DDPM noised input.
"""

import argparse
import time
from pathlib import Path

import torch as th
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_defaults,
    create_classifier,
    create_gaussian_diffusion_with_drift,
)

T_0, T_1, T_2, T_3 = 0, 300, 600, 1000

STAGE_CONFIGS = {
    "64": dict(image_size=64, c=None, T_c=T_2, T_c_plus_1=T_3),
    "128": dict(image_size=128, c=1, T_c=T_1, T_c_plus_1=T_2),
    "256": dict(image_size=256, c=0, T_c=T_0, T_c_plus_1=T_1),
}


class ImageNetHFDataset(Dataset):
    """Streams ImageNet-1k 256x256 from HuggingFace and normalizes to [-1, 1]."""

    def __init__(self, image_size, hf_repo, split, random_flip=True):
        from datasets import load_dataset

        self.dataset = load_dataset(hf_repo, split=split)
        tfms = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
        if random_flip:
            tfms.append(transforms.RandomHorizontalFlip())
        tfms.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.transform = transforms.Compose(tfms)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.transform(image), int(item["label"])


def downsample_image(x, target_size):
    """Average-pool [B, C, H, W] to [B, C, target_size, target_size]."""
    _, _, h, w = x.shape
    assert h == w and h % target_size == 0
    factor = h // target_size
    x = x.view(x.shape[0], x.shape[1], target_size, factor, target_size, factor)
    return x.mean(dim=(3, 5))


def split_microbatches(microbatch, *args):
    batch_size = len(args[0])
    if microbatch == -1 or microbatch >= batch_size:
        yield tuple(args)
        return
    for i in range(0, batch_size, microbatch):
        yield tuple(x[i : i + microbatch] if x is not None else None for x in args)


def create_output_dir(args):
    stage_dir = Path(args.out_dir) / f"classifier_stage_{args.stage}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    return stage_dir


def save_checkpoint(stage_dir, model, opt, step, args):
    th.save(model.state_dict(), stage_dir / f"model{step:07d}.pt")
    th.save(
        {
            "optimizer": opt.state_dict(),
            "step": step,
            "args": vars(args),
        },
        stage_dir / f"opt{step:07d}.pt",
    )


def maybe_resume(model, opt, args, device):
    if not args.resume_checkpoint:
        return 0
    ckpt_path = Path(args.resume_checkpoint)
    step_str = ckpt_path.stem.replace("model", "")
    step = int(step_str) if step_str.isdigit() else 0
    model.load_state_dict(th.load(ckpt_path, map_location=device, weights_only=False))
    opt_path = ckpt_path.with_name(f"opt{step:07d}.pt")
    if opt_path.exists():
        opt_state = th.load(opt_path, map_location=device, weights_only=False)
        opt.load_state_dict(opt_state["optimizer"])
        step = int(opt_state.get("step", step))
    return step


def topk_acc(logits, labels, k):
    _, topk = th.topk(logits, k, dim=-1)
    return (topk == labels[:, None]).any(dim=-1).float().mean().item()


def main():
    args = create_argparser().parse_args()
    stage_cfg = STAGE_CONFIGS[args.stage]
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    th.backends.cuda.matmul.allow_tf32 = True
    th.backends.cudnn.allow_tf32 = True

    print(f"Device: {device}")
    print(f"RDD classifier stage: {args.stage}")
    print(f"Output dir: {args.out_dir}")

    classifier_arg_keys = set(classifier_defaults().keys()) - {"image_size"}
    classifier = create_classifier(
        image_size=stage_cfg["image_size"],
        **args_to_dict(args, classifier_arg_keys),
    ).to(device)
    diffusion = create_gaussian_diffusion_with_drift(
        steps=T_3,
        noise_schedule=args.noise_schedule,
        learn_sigma=False,
        timestep_respacing="",
        c=stage_cfg["c"],
        T_c=stage_cfg["T_c"],
        T_c_plus_1=stage_cfg["T_c_plus_1"],
        sample_scale=args.sample_scale,
    )

    dataset = ImageNetHFDataset(
        image_size=args.base_image_size,
        hf_repo=args.hf_repo,
        split=args.hf_split,
        random_flip=not args.disable_random_flip,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    loader_iter = iter(loader)

    opt = AdamW(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    out_dir = create_output_dir(args)
    start_step = maybe_resume(classifier, opt, args, device)

    scaler = th.cuda.amp.GradScaler(enabled=args.use_fp16 and device.type == "cuda")
    stage_size = stage_cfg["image_size"]
    running_loss = 0.0
    running_acc1 = 0.0
    running_acc5 = 0.0
    window_steps = 0
    start_time = time.time()
    global_map = th.tensor(diffusion.timestep_map, device=device, dtype=th.long)

    classifier.train()
    for step in range(start_step, args.iterations):
        try:
            batch, labels = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch, labels = next(loader_iter)

        batch = batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if stage_size != args.base_image_size:
            batch = downsample_image(batch, stage_size)

        t = th.randint(0, diffusion.num_timesteps, (batch.shape[0],), device=device)
        noise = th.randn_like(batch)
        x_t = diffusion.q_sample(batch, t, noise=noise)
        global_t = global_map[t]

        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        logits_for_logging = []
        labels_for_logging = []

        for sub_batch, sub_labels, sub_t in split_microbatches(
            args.microbatch, x_t, labels, global_t
        ):
            with th.cuda.amp.autocast(enabled=args.use_fp16 and device.type == "cuda"):
                logits = classifier(sub_batch, sub_t)
                loss = F.cross_entropy(logits, sub_labels)
            scaler.scale(loss * (len(sub_batch) / len(batch))).backward()
            total_loss += loss.detach().item() * len(sub_batch)
            logits_for_logging.append(logits.detach())
            labels_for_logging.append(sub_labels.detach())

        scaler.step(opt)
        scaler.update()

        logits_cat = th.cat(logits_for_logging, dim=0)
        labels_cat = th.cat(labels_for_logging, dim=0)
        running_loss += total_loss / len(batch)
        running_acc1 += topk_acc(logits_cat, labels_cat, k=1)
        running_acc5 += topk_acc(logits_cat, labels_cat, k=5)
        window_steps += 1

        if (step + 1) % args.log_interval == 0:
            elapsed = time.time() - start_time
            print(
                f"[step {step + 1:07d}] "
                f"loss={running_loss / window_steps:.4f} "
                f"acc1={running_acc1 / window_steps:.4f} "
                f"acc5={running_acc5 / window_steps:.4f} "
                f"steps_per_sec={window_steps / max(elapsed, 1e-6):.2f}"
            )
            running_loss = 0.0
            running_acc1 = 0.0
            running_acc5 = 0.0
            window_steps = 0
            start_time = time.time()

        if (step + 1) % args.save_interval == 0:
            save_checkpoint(out_dir, classifier, opt, step + 1, args)

    save_checkpoint(out_dir, classifier, opt, args.iterations, args)
    print("Training complete.")


def create_argparser():
    defaults = dict(
        stage="64",
        out_dir="guided_diffusion/results_rdd_classifier",
        resume_checkpoint="",
        iterations=300000,
        lr=3e-4,
        weight_decay=0.05,
        batch_size=8,
        microbatch=-1,
        log_interval=50,
        save_interval=10000,
        num_workers=4,
        use_fp16=False,
        noise_schedule="linear",
        sample_scale=2,
        base_image_size=256,
        hf_repo="benjamin-paine/imagenet-1k-256x256",
        hf_split="train",
        disable_random_flip=False,
    )
    defaults.update(classifier_defaults())
    defaults["image_size"] = 256
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
