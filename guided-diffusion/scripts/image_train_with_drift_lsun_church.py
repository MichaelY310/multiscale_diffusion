"""
Multi-scale UNet training with drift on local LSUN church-outdoor data.

This variant is unconditional:
  - the dataset may contain a `label` column, but it is ignored,
  - all UNets are created with `class_cond=False`,
  - training_losses() receives empty model_kwargs except for `rdd_stage`
    in the shared single-model adapter mode.

Expected dataset layout:
  /path/to/lsun_church_train
    README.md
    data/train-*.parquet
    data/test-*.parquet
"""

import copy
import gc
import os
import re
import time
from glob import glob

import torch as th
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    create_model_for_drift,
    create_gaussian_diffusion_with_drift,
)

th.backends.cuda.matmul.allow_tf32 = True
th.backends.cudnn.allow_tf32 = True

T_0, T_1, T_2, T_3 = 0, 300, 600, 1000


class LocalLSUNChurchDataset(Dataset):
    def __init__(self, dataset_path: str, split: str = "train", image_size: int = 256):
        from datasets import load_dataset

        self.dataset = load_dataset(dataset_path, split=split)
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.transform(image)


def downsample_image(x, target_size):
    b, c, h, w = x.shape
    assert h == w and h % target_size == 0
    factor = h // target_size
    x = x.view(b, c, target_size, factor, target_size, factor)
    return x.mean(dim=(3, 5))


@th.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = dict(ema_model.named_parameters())
    src_params = dict(model.named_parameters())
    for name, param in src_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def _checkpoint_path(ckpt_dir, step):
    return os.path.join(ckpt_dir, f"{step:07d}.pt")


def _available_checkpoint_steps(ckpt_dir):
    if not ckpt_dir or not os.path.isdir(ckpt_dir):
        return set()
    steps = set()
    for path in glob(os.path.join(ckpt_dir, "*.pt")):
        match = re.search(r"(\d+)\.pt$", os.path.basename(path))
        if match:
            steps.add(int(match.group(1)))
    return steps


def _resolve_resume_step(args):
    if not args.resume_dir:
        return None

    if args.resume_step is not None:
        return args.resume_step

    if args.use_single_model:
        shared_steps = _available_checkpoint_steps(
            os.path.join(args.resume_dir, "checkpoints_shared")
        )
        if not shared_steps:
            raise FileNotFoundError(
                f"No checkpoints found under {args.resume_dir}/checkpoints_shared"
            )
        return max(shared_steps)

    common_steps = (
        _available_checkpoint_steps(os.path.join(args.resume_dir, "checkpoints_64"))
        & _available_checkpoint_steps(os.path.join(args.resume_dir, "checkpoints_128"))
        & _available_checkpoint_steps(os.path.join(args.resume_dir, "checkpoints_256"))
    )
    if not common_steps:
        raise FileNotFoundError(
            "Could not find a common checkpoint step across checkpoints_64, "
            "checkpoints_128, and checkpoints_256 under "
            f"{args.resume_dir}"
        )
    return max(common_steps)


def _load_checkpoint_state(path):
    ckpt = th.load(path, map_location="cpu", weights_only=False)
    return {
        "model": {k.replace("module.", ""): v for k, v in ckpt["model"].items()},
        "ema": {k.replace("module.", ""): v for k, v in ckpt["ema"].items()},
        "opt": ckpt["opt"],
    }


def _move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, th.Tensor):
                state[key] = value.to(device, non_blocking=True)


def main(args):
    dist_util.setup_dist()
    logger.configure()

    rank = dist.get_rank()
    world = dist.get_world_size()
    device = dist_util.dev()
    use_ddp = world > 1
    resume_step = _resolve_resume_step(args)

    logger.log("Creating models ...")
    image_size_128 = args.image_size // 2
    image_size_64 = args.image_size // 4

    if args.use_single_model:
        shared_model = create_model_for_drift(
            image_size=args.image_size,
            num_channels=args.num_channels_256,
            num_res_blocks=args.num_res_blocks,
            learn_sigma=args.learn_sigma,
            class_cond=False,
            use_checkpoint=args.use_checkpoint,
            attention_resolutions=args.attention_resolutions_256,
            num_heads=args.num_heads,
            use_scale_shift_norm=args.use_scale_shift_norm,
            dropout=args.dropout,
            use_fp16=args.use_fp16,
            use_single_model=True,
            sample_scale=args.sample_scale,
        ).to(device)
        ema_shared = copy.deepcopy(shared_model)
        requires_grad(ema_shared, False)
        ema_shared.eval()
        if use_ddp:
            ddp_shared = DDP(
                shared_model,
                device_ids=[device],
                output_device=device,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        else:
            ddp_shared = shared_model
        logger.log(f"Shared RDD UNet params : {sum(p.numel() for p in shared_model.parameters()):,}")
        opt_shared = th.optim.AdamW(shared_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        update_ema(ema_shared, shared_model, decay=0)
        if resume_step is not None:
            ckpt_path = _checkpoint_path(
                os.path.join(args.resume_dir, "checkpoints_shared"), resume_step
            )
            state = _load_checkpoint_state(ckpt_path)
            shared_model.load_state_dict(state["model"])
            ema_shared.load_state_dict(state["ema"])
            opt_shared.load_state_dict(state["opt"])
            _move_optimizer_state_to_device(opt_shared, device)
            del state
            gc.collect()
            if device.type == "cuda":
                th.cuda.empty_cache()
            logger.log(f"Resumed shared model from {ckpt_path}")
        shared_model.train()
    else:
        model_256 = create_model_for_drift(
            image_size=args.image_size,
            num_channels=args.num_channels_256,
            num_res_blocks=args.num_res_blocks,
            learn_sigma=args.learn_sigma,
            class_cond=False,
            use_checkpoint=args.use_checkpoint,
            attention_resolutions=args.attention_resolutions_256,
            num_heads=args.num_heads,
            use_scale_shift_norm=args.use_scale_shift_norm,
            dropout=args.dropout,
            use_fp16=args.use_fp16,
        ).to(device)
        ema_256 = copy.deepcopy(model_256)
        requires_grad(ema_256, False)
        ema_256.eval()
        if use_ddp:
            ddp_256 = DDP(
                model_256,
                device_ids=[device],
                output_device=device,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        else:
            ddp_256 = model_256

        model_128 = create_model_for_drift(
            image_size=image_size_128,
            num_channels=args.num_channels_128,
            num_res_blocks=args.num_res_blocks,
            learn_sigma=args.learn_sigma,
            class_cond=False,
            use_checkpoint=args.use_checkpoint,
            attention_resolutions=args.attention_resolutions_128,
            num_heads=args.num_heads,
            use_scale_shift_norm=args.use_scale_shift_norm,
            dropout=args.dropout,
            use_fp16=args.use_fp16,
        ).to(device)
        ema_128 = copy.deepcopy(model_128)
        requires_grad(ema_128, False)
        ema_128.eval()
        if use_ddp:
            ddp_128 = DDP(
                model_128,
                device_ids=[device],
                output_device=device,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        else:
            ddp_128 = model_128

        model_64 = create_model_for_drift(
            image_size=image_size_64,
            num_channels=args.num_channels_64,
            num_res_blocks=args.num_res_blocks,
            learn_sigma=args.learn_sigma,
            class_cond=False,
            use_checkpoint=args.use_checkpoint,
            attention_resolutions=args.attention_resolutions_64,
            num_heads=args.num_heads,
            use_scale_shift_norm=args.use_scale_shift_norm,
            dropout=args.dropout,
            use_fp16=args.use_fp16,
        ).to(device)
        ema_64 = copy.deepcopy(model_64)
        requires_grad(ema_64, False)
        ema_64.eval()
        if use_ddp:
            ddp_64 = DDP(
                model_64,
                device_ids=[device],
                output_device=device,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        else:
            ddp_64 = model_64

        logger.log(f"UNet 256x256 params : {sum(p.numel() for p in model_256.parameters()):,}")
        logger.log(f"UNet 128x128 params : {sum(p.numel() for p in model_128.parameters()):,}")
        logger.log(f"UNet  64x64 params : {sum(p.numel() for p in model_64.parameters()):,}")

        opt_256 = th.optim.AdamW(model_256.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        opt_128 = th.optim.AdamW(model_128.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        opt_64 = th.optim.AdamW(model_64.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        update_ema(ema_256, model_256, decay=0)
        update_ema(ema_128, model_128, decay=0)
        update_ema(ema_64, model_64, decay=0)

        if resume_step is not None:
            state_256 = _load_checkpoint_state(
                _checkpoint_path(os.path.join(args.resume_dir, "checkpoints_256"), resume_step)
            )
            state_128 = _load_checkpoint_state(
                _checkpoint_path(os.path.join(args.resume_dir, "checkpoints_128"), resume_step)
            )
            state_64 = _load_checkpoint_state(
                _checkpoint_path(os.path.join(args.resume_dir, "checkpoints_64"), resume_step)
            )
            model_256.load_state_dict(state_256["model"])
            ema_256.load_state_dict(state_256["ema"])
            opt_256.load_state_dict(state_256["opt"])
            _move_optimizer_state_to_device(opt_256, device)
            model_128.load_state_dict(state_128["model"])
            ema_128.load_state_dict(state_128["ema"])
            opt_128.load_state_dict(state_128["opt"])
            _move_optimizer_state_to_device(opt_128, device)
            model_64.load_state_dict(state_64["model"])
            ema_64.load_state_dict(state_64["ema"])
            opt_64.load_state_dict(state_64["opt"])
            _move_optimizer_state_to_device(opt_64, device)
            del state_256, state_128, state_64
            gc.collect()
            if device.type == "cuda":
                th.cuda.empty_cache()
            logger.log(f"Resumed multi-scale checkpoints from {args.resume_dir} at step {resume_step:07d}")

        model_256.train()
        model_128.train()
        model_64.train()

    diffusion_256 = create_gaussian_diffusion_with_drift(
        steps=T_3,
        learn_sigma=args.learn_sigma,
        noise_schedule=args.noise_schedule,
        c=0,
        T_c=T_0,
        T_c_plus_1=T_1,
        sample_scale=args.sample_scale,
    )
    diffusion_128 = create_gaussian_diffusion_with_drift(
        steps=T_3,
        learn_sigma=args.learn_sigma,
        noise_schedule=args.noise_schedule,
        c=1,
        T_c=T_1,
        T_c_plus_1=T_2,
        sample_scale=args.sample_scale,
    )
    diffusion_64 = create_gaussian_diffusion_with_drift(
        steps=T_3,
        learn_sigma=args.learn_sigma,
        noise_schedule=args.noise_schedule,
        c=None,
        T_c=T_2,
        T_c_plus_1=T_3,
        sample_scale=args.sample_scale,
    )

    train_steps = resume_step or 0
    if rank == 0:
        if args.resume_dir:
            exp_dir = args.resume_dir
        else:
            exp_index = 0
            pattern = os.path.join(args.results_dir, "*")
            for path in glob(pattern):
                match = re.search(r"(\d+)$", os.path.basename(path))
                if match:
                    exp_index = max(exp_index, int(match.group(1)) + 1)
            exp_dir = os.path.join(args.results_dir, f"{exp_index:03d}")
        if args.use_single_model:
            ckpt_dir_shared = os.path.join(exp_dir, "checkpoints_shared")
            os.makedirs(ckpt_dir_shared, exist_ok=True)
        else:
            ckpt_dir_256 = os.path.join(exp_dir, "checkpoints_256")
            ckpt_dir_128 = os.path.join(exp_dir, "checkpoints_128")
            ckpt_dir_64 = os.path.join(exp_dir, "checkpoints_64")
            os.makedirs(ckpt_dir_256, exist_ok=True)
            os.makedirs(ckpt_dir_128, exist_ok=True)
            os.makedirs(ckpt_dir_64, exist_ok=True)
        logger.log(f"Experiment directory: {exp_dir}")
        if resume_step is not None:
            logger.log(f"Resuming training from step {resume_step:07d}")
    else:
        ckpt_dir_shared = ckpt_dir_256 = ckpt_dir_128 = ckpt_dir_64 = None

    logger.log("Creating data loader ...")
    dataset = LocalLSUNChurchDataset(
        dataset_path=args.dataset_path,
        split=args.dataset_split,
        image_size=args.image_size,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world,
        rank=rank,
        shuffle=True,
        seed=args.global_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    loader_iter = iter(loader)

    logger.log(f"Training for {args.total_steps} steps ...")
    running_loss_256 = running_loss_128 = running_loss_64 = 0.0
    log_steps = 0
    start_time = time.time()
    epoch = 0

    while train_steps < args.total_steps:
        try:
            x = next(loader_iter)
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            loader_iter = iter(loader)
            x = next(loader_iter)

        x = x.to(device)
        t_256 = th.randint(0, T_1 - T_0, (x.shape[0],), device=device)
        x_128 = downsample_image(x.detach(), image_size_128)
        t_128 = th.randint(0, T_2 - T_1, (x_128.shape[0],), device=device)
        x_64 = downsample_image(x.detach(), image_size_64)
        t_64 = th.randint(0, T_3 - T_2, (x_64.shape[0],), device=device)

        if args.use_single_model:
            loss_256 = diffusion_256.training_losses(
                ddp_shared, x, t_256, {"rdd_stage": "fine"}
            )["loss"].mean()
            loss_128 = diffusion_128.training_losses(
                ddp_shared, x_128, t_128, {"rdd_stage": "mid"}
            )["loss"].mean()
            loss_64 = diffusion_64.training_losses(
                ddp_shared, x_64, t_64, {"rdd_stage": "coarse"}
            )["loss"].mean()
            total_loss = loss_256 + loss_128 + loss_64
            opt_shared.zero_grad()
            total_loss.backward()
            opt_shared.step()
            update_ema(ema_shared, shared_model)
        else:
            model_kwargs = {}

            loss_256 = diffusion_256.training_losses(ddp_256, x, t_256, model_kwargs)["loss"].mean()
            opt_256.zero_grad()
            loss_256.backward()
            opt_256.step()
            update_ema(ema_256, model_256)

            loss_128 = diffusion_128.training_losses(ddp_128, x_128, t_128, model_kwargs)["loss"].mean()
            opt_128.zero_grad()
            loss_128.backward()
            opt_128.step()
            update_ema(ema_128, model_128)

            loss_64 = diffusion_64.training_losses(ddp_64, x_64, t_64, model_kwargs)["loss"].mean()
            opt_64.zero_grad()
            loss_64.backward()
            opt_64.step()
            update_ema(ema_64, model_64)

        running_loss_256 += loss_256.item()
        running_loss_128 += loss_128.item()
        running_loss_64 += loss_64.item()
        log_steps += 1
        train_steps += 1

        if train_steps % args.log_every == 0:
            if device.type == "cuda":
                th.cuda.synchronize()
            end_time = time.time()
            steps_per_sec = log_steps / (end_time - start_time)

            avg_256 = th.tensor(running_loss_256 / log_steps, device=device)
            avg_128 = th.tensor(running_loss_128 / log_steps, device=device)
            avg_64 = th.tensor(running_loss_64 / log_steps, device=device)
            if use_ddp:
                dist.all_reduce(avg_256, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_128, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_64, op=dist.ReduceOp.SUM)
            avg_256 = avg_256.item() / world
            avg_128 = avg_128.item() / world
            avg_64 = avg_64.item() / world

            logger.log(
                f"(step={train_steps:07d}) "
                f"Loss[256x256]={avg_256:.4f}  "
                f"Loss[128x128]={avg_128:.4f}  "
                f"Loss[64x64]={avg_64:.4f}  "
                f"Steps/s={steps_per_sec:.2f}"
            )

            running_loss_256 = running_loss_128 = running_loss_64 = 0.0
            log_steps = 0
            start_time = time.time()

        if train_steps % args.ckpt_every == 0 and train_steps > 0:
            if rank == 0:
                if args.use_single_model:
                    th.save(
                        {
                            "model": shared_model.state_dict(),
                            "ema": ema_shared.state_dict(),
                            "opt": opt_shared.state_dict(),
                            "args": args,
                            "resolution": "shared",
                            "mode": "single_model_adapter",
                            "T": [T_0, T_1, T_2, T_3],
                        },
                        f"{ckpt_dir_shared}/{train_steps:07d}.pt",
                    )
                else:
                    th.save(
                        {
                            "model": model_256.state_dict(),
                            "ema": ema_256.state_dict(),
                            "opt": opt_256.state_dict(),
                            "args": args,
                            "resolution": "256x256",
                            "c": 0,
                            "T_c": T_0,
                            "T_c_plus_1": T_1,
                        },
                        f"{ckpt_dir_256}/{train_steps:07d}.pt",
                    )
                    th.save(
                        {
                            "model": model_128.state_dict(),
                            "ema": ema_128.state_dict(),
                            "opt": opt_128.state_dict(),
                            "args": args,
                            "resolution": "128x128",
                            "c": 1,
                            "T_c": T_1,
                            "T_c_plus_1": T_2,
                        },
                        f"{ckpt_dir_128}/{train_steps:07d}.pt",
                    )
                    th.save(
                        {
                            "model": model_64.state_dict(),
                            "ema": ema_64.state_dict(),
                            "opt": opt_64.state_dict(),
                            "args": args,
                            "resolution": "64x64",
                            "c": None,
                            "T_c": T_2,
                            "T_c_plus_1": T_3,
                        },
                        f"{ckpt_dir_64}/{train_steps:07d}.pt",
                    )
                logger.log(f"Saved checkpoints at step {train_steps:07d}")
            if use_ddp:
                dist.barrier()

    logger.log("Done!")
    if dist.is_initialized():
        dist.destroy_process_group()


def create_argparser():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Multi-scale UNet training on local LSUN church-outdoor with drift.\n"
            "  default: three separate unconditional UNets for 256/128/64\n"
            "  optional: one shared 256x256 UNet + adapters for 128/64"
        )
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/tiger/multiscale_diffusion/lsun/lsun_church_train",
        help="Local HuggingFace dataset snapshot directory.",
    )
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--global_seed", type=int, default=0)

    parser.add_argument(
        "--results_dir",
        type=str,
        default="guided_diffusion/results_lsun_church",
        help="Root directory for experiment folders and checkpoints.",
    )
    parser.add_argument(
        "--resume_dir",
        type=str,
        default="",
        help="Existing experiment directory to resume into, e.g. .../results_lsun_church_gpu0/004",
    )
    parser.add_argument(
        "--resume_step",
        type=int,
        default=None,
        help="Specific step to resume from. Defaults to the latest common checkpoint under --resume_dir.",
    )

    parser.add_argument("--total_steps", type=int, default=1_000_000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--ema_rate", type=float, default=0.9999)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=5000)

    parser.add_argument("--noise_schedule", type=str, default="linear")
    parser.add_argument("--learn_sigma", action="store_true", default=False)
    parser.add_argument("--sample_scale", type=int, default=2)

    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_scale_shift_norm", action="store_true", default=True)
    parser.add_argument("--use_checkpoint", action="store_true", default=True)
    parser.add_argument("--use_fp16", action="store_true", default=False)

    parser.add_argument("--num_channels_64", type=int, default=64)
    parser.add_argument("--num_channels_128", type=int, default=128)
    parser.add_argument("--num_channels_256", type=int, default=192)
    parser.add_argument("--attention_resolutions_64", type=str, default="16,8")
    parser.add_argument("--attention_resolutions_128", type=str, default="32,16,8")
    parser.add_argument("--attention_resolutions_256", type=str, default="32,16,8")

    parser.add_argument("--use_single_model", action="store_true", default=False)

    return parser


if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)
