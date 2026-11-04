# Debug Session: gpu-cuda-mismatch [OPEN]

## Symptom
- Training starts but `torch` reports CUDA initialization failure and falls back to CPU.
- `nvidia-smi` shows `Driver Version: 535.161.08`, `CUDA Version: 12.2`, GPU idle.

## Goal
- Make the training launched from `.venv` use GPU successfully on this machine.

## Hypotheses
1. The current `.venv` has a CUDA-enabled `torch` wheel built for a newer CUDA runtime than the installed NVIDIA driver supports.
2. The training shell script activates `.venv`, but `python` may still resolve to an unexpected interpreter or package set.
3. CUDA is unavailable inside the active Python process even though the host driver is present, so the script falls back to CPU by design.
4. A secondary package mismatch (`torchvision`/`torchaudio`) may block a working CUDA stack even after `torch` is corrected.

## Evidence Plan
- Capture the active interpreter path and the installed `torch` metadata from `.venv`.
- Verify runtime CUDA availability inside the exact interpreter used by the launch script.
- Replace only the PyTorch stack if the runtime evidence confirms a wheel/driver mismatch.
- Re-run a minimal CUDA probe and then a short training smoke test.

## Status Log
- Session opened.
- Confirmed the launch script uses `/home/tiger/multiscale_diffusion/guided-diffusion/.venv/bin/python`.
- Confirmed `.venv` originally contained `torch 2.11.0+cu130`, which is newer than the host's CUDA 12.2 driver capability.
- Installed a driver-compatible PyTorch stack: `torch 2.5.1+cu121`, `torchvision 0.20.1+cu121`, `torchaudio 2.5.1+cu121`.
- Hardened the launch script to pin `PYTHONPATH`, default `DIST_BACKEND=gloo`, and fail fast if CUDA is unavailable instead of silently training on CPU.
- Confirmed sampling reaches GPU execution, but `posterior_upsample_between_scales()` OOMs at `64→128` when `num_samples=4`; the exact posterior routine materializes large covariance/inverse matrices whose memory scales with batch size.
