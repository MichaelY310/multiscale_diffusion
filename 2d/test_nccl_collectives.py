"""Small multi-GPU collective smoke test used by the RDD DMD launcher.

Run with ``torchrun --standalone --nproc_per_node=N``.  It deliberately avoids
loading any model, so a failure here isolates the CUDA/NCCL transport from the
distillation implementation.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    value = torch.tensor([float(rank + 1)], device=device)
    dist.all_reduce(value)
    torch.cuda.synchronize(device)
    expected = dist.get_world_size() * (dist.get_world_size() + 1) / 2
    assert value.item() == expected, (value.item(), expected)

    layer = DDP(torch.nn.Linear(128, 128, bias=False).to(device), device_ids=[local_rank])
    loss = layer(torch.randn(4, 128, device=device)).square().mean()
    loss.backward()
    torch.cuda.synchronize(device)
    dist.barrier()
    if rank == 0:
        print("NCCL all_reduce + DDP backward: PASS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
