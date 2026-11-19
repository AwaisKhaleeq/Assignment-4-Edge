# ---------------------------------------------------------------------------
# End-to-end pipeline for Windows PowerShell (same steps as scripts/run_all.sh).
# Run from the repository root:   powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
# ---------------------------------------------------------------------------
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force results\q1\logs, results\q2\logs | Out-Null

function Invoke-Step($cmd) {
    Write-Host "`n>>> python $cmd" -ForegroundColor Cyan
    & python @($cmd -split ' ')
    if ($LASTEXITCODE -ne 0) { throw "step failed: $cmd" }
}

# Q1 - knowledge distillation (CIFAR-100)
Invoke-Step "-m q1_kd.train_teacher --epochs 2 --res 64"
Invoke-Step "-m q1_kd.train_baseline --epochs 5"
Invoke-Step "-m q1_kd.train_kd --epochs 5 --T 4 --alpha 0.9 --beta 1.0 --kd-mode both"

# Q2 - hardware-aware NAS (CIFAR-10)
Invoke-Step "-m q2_nas.search --population 8 --sample 3 --budget 20 --epochs 3 --train-subset 10000"
Invoke-Step "-m q2_nas.train_best --epochs 15"

# Deployment
Invoke-Step "-m deploy.export_onnx --models all"
Invoke-Step "-m deploy.onnx_infer --models all --samples 3"
Invoke-Step "-m deploy.benchmark --models all --runs 100"
Invoke-Step "-m q2_nas.plots"
Invoke-Step "report/build_report.py"
