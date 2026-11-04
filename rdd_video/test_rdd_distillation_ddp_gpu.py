"""Fast two-GPU test of RDD trajectory/DMD gradients without loading Wan."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from flow_matching_video import RDDVideoSchedule
from rdd_distillation import (
    RDDDistillationSchedule,
    normalized_dmd_surrogate,
    rdd_renoise,
    trajectory_distillation_loss,
)


class TinyVelocity(nn.Module):
    def __init__(self, channels: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels, 16, 1),
            nn.SiLU(),
            nn.Conv3d(16, channels, 1),
        )
        self.time_scale = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        time = t.view(-1, 1, 1, 1, 1)
        return self.net(x) + time * self.time_scale.view(1, -1, 1, 1, 1)


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(1234 + rank)

    schedule = RDDVideoSchedule(
        [
            {"spatial_factor": 4, "temporal_factor": 4, "num_steps": 2},
            {"spatial_factor": 2, "temporal_factor": 2, "num_steps": 2},
            {"spatial_factor": 1, "temporal_factor": 1, "num_steps": 1},
        ],
        [60, 350, 590],
        drift_scale=1.0,
    )
    distilled = RDDDistillationSchedule(schedule, (2, 2, 1))
    call = distilled.calls[2 + rank]
    fm = schedule.flow_matchers[call.stage_index]

    student = DDP(TinyVelocity().to(device), device_ids=[local_rank])
    teacher = TinyVelocity().to(device).eval().requires_grad_(False)
    critic = DDP(TinyVelocity().to(device), device_ids=[local_rank])
    student_opt = torch.optim.AdamW(student.parameters(), lr=1e-3)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=1e-3)

    clean = torch.randn(2, 4, 5, 8, 8, device=device)
    start = rdd_renoise(fm, clean, call.t_start)
    trajectory, _, _ = trajectory_distillation_loss(
        lambda x, t: student(x, t),
        lambda x, t: teacher(x, t),
        start,
        call,
        teacher_substeps=2,
        teacher_solver="heun",
    )
    student_opt.zero_grad(set_to_none=True)
    trajectory.backward()
    student_opt.step()

    score_t = torch.full((clean.shape[0],), call.t_start + 0.5 * call.dt, device=device)
    generated = clean + 0.05 * student(clean, score_t)
    fake = generated.detach() + 0.05 * critic(generated.detach(), score_t)
    critic_loss = (fake - clean).square().mean()
    critic_opt.zero_grad(set_to_none=True)
    critic_loss.backward()
    critic_opt.step()

    dmd_loss, gradient = normalized_dmd_surrogate(
        generated,
        teacher_x1=clean.detach(),
        fake_x1=fake.detach(),
        fm=fm,
        score_t=score_t,
        path_correction="exact",
    )
    student_opt.zero_grad(set_to_none=True)
    dmd_loss.backward()
    student_opt.step()

    metrics = torch.tensor(
        [trajectory.item(), critic_loss.item(), dmd_loss.item(), gradient.norm().item()],
        device=device,
    )
    if not torch.isfinite(metrics).all():
        raise RuntimeError(f"non-finite GPU smoke metrics on rank {rank}: {metrics}")
    dist.all_reduce(metrics)
    metrics /= dist.get_world_size()
    if rank == 0:
        print(
            "RDD_DDP_GPU_SMOKE_OK "
            f"trajectory={metrics[0]:.6f} critic={metrics[1]:.6f} "
            f"dmd={metrics[2]:.6f} grad_norm={metrics[3]:.6f}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
