"""FastVideo DCP manager with native rank-local PF RNG save/load hooks."""
from typing import Any
from pathlib import Path

import torch

from fastvideo.train.utils.checkpoint import _resolve_resume_checkpoint
from exact_checkpoint import ExactCheckpointManager
from pyramid_runner_state import capture_runner_state, apply_runner_state


class NativePyramidCheckpointManager(ExactCheckpointManager):

    def save(self: Any, step: Any) -> Any:
        if self._last_saved_step == step:
            return
        # The parent's collective existence check runs before any rank mkdir;
        # an independent rank-local check can race another rank's commit.
        if self.config.keep_last:
            raise ValueError("Use keep_last=0; this workflow does not delete old checkpoint files")
        super().save(step)

    def _save_rng_snapshot(self: Any, checkpoint_dir: Any) -> Any:
        student = self.method.student
        state = capture_runner_state(student.runner, student.accelerator)
        torch.save(state, Path(checkpoint_dir) / f"native_rng_rank{student.accelerator.process_index}.pt")

    def load_rng_snapshot(self: Any, checkpoint_path: Any) -> Any:
        resolved = _resolve_resume_checkpoint(checkpoint_path, output_dir=self.output_dir)
        if resolved is None:
            raise FileNotFoundError("Explicit native resume request has no complete checkpoint")
        student = self.method.student
        state = torch.load(resolved / f"native_rng_rank{student.accelerator.process_index}.pt",
                           map_location="cpu",
                           weights_only=True)
        apply_runner_state(state, student.runner, student.accelerator)

    def maybe_resume(self: Any, *, resume_from_checkpoint: Any) -> Any:
        if not resume_from_checkpoint:
            return None
        resolved = _resolve_resume_checkpoint(resume_from_checkpoint, output_dir=self.output_dir)
        if resolved is None:
            raise FileNotFoundError("Explicit native resume request has no complete checkpoint")
        step = super().maybe_resume(resume_from_checkpoint=str(resolved))
        if step != self.method.global_step:
            raise ValueError("Native global step differs from checkpoint directory step")
        return step
