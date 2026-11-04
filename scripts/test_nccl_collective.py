"""Minimal two-rank NCCL diagnostic for multi-GPU training hosts."""

import argparse
import os

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--numel", type=int, default=1)
    args = parser.parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    value = torch.full(
        (args.numel,), rank + 1.0, dtype=torch.float32, device=f"cuda:{local_rank}"
    )
    dist.all_reduce(value)
    torch.cuda.synchronize()
    print(
        f"rank={rank} first={value[0].item()} numel={value.numel()} "
        f"device={torch.cuda.get_device_name(local_rank)}"
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
