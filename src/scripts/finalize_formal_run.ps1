param(
    [Parameter(Mandatory = $true)][int]$TrainingPid,
    [string]$Config = "configs/zenodo_heft_safe_fast_2026.yaml",
    [string]$OutputDir = "outputs/zenodo_heft_safe_fast_2026"
)

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))
$log = Join-Path $OutputDir "finalize.log"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

try {
    Wait-Process -Id $TrainingPid -ErrorAction SilentlyContinue
    $checkpoint = Join-Path $OutputDir "best.pt"
    if (-not (Test-Path $checkpoint)) {
        throw "Training ended without best.pt; held-out evaluation was not run."
    }
    $env:PYTHONPATH = (Join-Path (Get-Location) "src")
    $env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
    & python scripts/evaluate.py --config $Config --policy hrl --checkpoint $checkpoint --split test *>&1 | Tee-Object -FilePath $log -Append
    & python scripts/evaluate.py --config $Config --policy hrl_safe --checkpoint $checkpoint --split test *>&1 | Tee-Object -FilePath $log -Append
    & python scripts/evaluate.py --config $Config --policy portfolio --split test *>&1 | Tee-Object -FilePath $log -Append
    & python scripts/evaluate_baselines.py --config $Config --split test *>&1 | Tee-Object -FilePath $log -Append
    & python scripts/plot_formal_results.py --evaluation-dir (Join-Path $OutputDir "eval_hrl_test") *>&1 | Tee-Object -FilePath $log -Append
    & python scripts/plot_formal_results.py --evaluation-dir (Join-Path $OutputDir "eval_hrl_safe_test") *>&1 | Tee-Object -FilePath $log -Append
    & python scripts/plot_formal_results.py --evaluation-dir (Join-Path $OutputDir "eval_portfolio_test") *>&1 | Tee-Object -FilePath $log -Append
    & python scripts/plot_policy_comparison.py --output-dir $OutputDir --split test *>&1 | Tee-Object -FilePath $log -Append
    & python -m pytest -q *>&1 | Tee-Object -FilePath $log -Append
} catch {
    $_ | Out-String | Tee-Object -FilePath $log -Append
    exit 1
}
