$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$sshConfig = 'C:\Users\aa310\.ssh\config'
$remoteRoot = '/scr/yuheng/DevOpsBench/multiscale_diffusion'

function Send-File([string]$LocalRelative, [string]$RemoteRelative) {
    $local = Join-Path $root $LocalRelative
    & scp -F $sshConfig $local "surfi2:${remoteRoot}/${RemoteRelative}"
    if ($LASTEXITCODE -ne 0) { throw "scp failed: $LocalRelative" }
}

Send-File 'rdd_video/attention_analysis/fastwan_native_attention.py' 'rdd_video/attention_analysis/fastwan_native_attention.py'
Send-File 'rdd_video/benchmark_fastwan_paired_suite.py' 'rdd_video/benchmark_fastwan_paired_suite.py'
Send-File 'rdd_video/experiments/fastwan_cross_stage_branch_reuse_screen.json' 'rdd_video/experiments/fastwan_cross_stage_branch_reuse_screen.json'
Send-File 'rdd_video/launch_fastwan_xsr_screen_gpu2.sh' 'rdd_video/launch_fastwan_xsr_screen_gpu2.sh'
Send-File 'rdd_video/experiments/fastwan_csbr_aggressive.json' 'rdd_video/experiments/fastwan_csbr_aggressive.json'
Send-File 'rdd_video/launch_fastwan_csbr_verify.sh' 'rdd_video/launch_fastwan_csbr_verify.sh'
