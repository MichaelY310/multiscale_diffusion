param(
    [int]$MaxAttempts = 120,
    [int]$RetrySeconds = 60
)

$ErrorActionPreference = 'Continue'
$Root = 'C:\Users\aa310\Desktop\multiscale_diffusion'
$Log = Join-Path $Root 'logs\surfi2_rdd3d_gate_retry.log'
New-Item -ItemType Directory -Force -Path (Split-Path $Log) | Out-Null

$RemoteCommand = @'
cd /scr/yuheng/DevOpsBench/multiscale_diffusion || exit 2
if tmux has-session -t rdd3d_ddp_smoke_then_gate 2>/dev/null; then
    echo ALREADY_RUNNING
    exit 0
fi
bash -n scripts/rdd3d/run_dynamic_sparse_pilot.sh scripts/rdd3d/run_dynamic_sparse_pilot_tmux.sh scripts/rdd3d/run_dynamic_sparse_gate_2gpu.sh || exit 2
.venv_rdd3d/bin/python -m py_compile RDD_DiT_3D/train_dynamic_sparse_rdd.py || exit 2
busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 2 | awk '$1 > 1000 {count++} END {print count+0}')
if [ "$busy" -ne 0 ]; then
    echo GPUS_0_1_BUSY
    exit 3
fi
tmux new-session -d -s rdd3d_ddp_smoke_then_gate -e 'CUDA_VISIBLE_DEVICES=0,1' "bash -lc '
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion

RDD3D_DYNAMIC_RUN_NAME=dynamic_sparse_ddp_smoke_w384d12 \
RDD3D_DYNAMIC_OUTPUT=/scr/yuheng/DevOpsBench/multiscale_diffusion/results_rdd3d/dynamic_sparse_ddp_smoke_w384d12 \
RDD3D_DYNAMIC_MAX_ASSETS=4 \
RDD3D_DYNAMIC_TRAIN_STEPS=20 \
RDD3D_DYNAMIC_WIDTH=384 \
RDD3D_DYNAMIC_DEPTH=12 \
RDD3D_DYNAMIC_WEIGHTS=model \
RDD3D_DYNAMIC_NUM_GPUS=2 \
RDD3D_DYNAMIC_SAVE_EVERY=20 \
bash scripts/rdd3d/run_dynamic_sparse_pilot.sh

echo DDP_SMOKE_PASSED_STARTING_FORMAL_GATE

RDD3D_DYNAMIC_RUN_NAME=dynamic_sparse_32asset_gate_w384d12_2gpu \
RDD3D_DYNAMIC_OUTPUT=/scr/yuheng/DevOpsBench/multiscale_diffusion/results_rdd3d/dynamic_sparse_32asset_gate_w384d12_2gpu \
RDD3D_DYNAMIC_MAX_ASSETS=32 \
RDD3D_DYNAMIC_TRAIN_STEPS=10000 \
RDD3D_DYNAMIC_WIDTH=384 \
RDD3D_DYNAMIC_DEPTH=12 \
RDD3D_DYNAMIC_WEIGHTS=model \
RDD3D_DYNAMIC_NUM_GPUS=2 \
RDD3D_DYNAMIC_SAVE_EVERY=1000 \
bash scripts/rdd3d/run_dynamic_sparse_pilot.sh
' 2>&1 | tee logs/rdd3d_ddp_smoke_then_gate.log"
echo STARTED_RDD3D_DDP_PIPELINE
'@

for ($Attempt = 1; $Attempt -le $MaxAttempts; $Attempt++) {
    $Timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Add-Content -LiteralPath $Log -Value "[$Timestamp] attempt $Attempt/$MaxAttempts"
    $Output = & ssh.exe -o ConnectTimeout=20 -o ConnectionAttempts=1 surfi2 $RemoteCommand 2>&1
    $Code = $LASTEXITCODE
    $Output | ForEach-Object { Add-Content -LiteralPath $Log -Value $_ }
    if ($Code -eq 0 -and ($Output -match 'STARTED_RDD3D_DDP_PIPELINE|ALREADY_RUNNING')) {
        Add-Content -LiteralPath $Log -Value "[$Timestamp] launcher complete"
        exit 0
    }
    if ($Attempt -lt $MaxAttempts) {
        Start-Sleep -Seconds $RetrySeconds
    }
}

Add-Content -LiteralPath $Log -Value "Retry budget exhausted without starting the pipeline."
exit 1
