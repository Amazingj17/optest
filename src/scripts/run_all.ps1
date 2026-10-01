param(
    [string]$Config = "configs/cpn_hrl_dag.yaml"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot
$env:PYTHONPATH = "$(Get-Location)\src"
python scripts/audit_datasets.py --grapheonrl-root data/raw/zenodo-18927122-derived --grapheonrl-system-configs data/raw/zenodo-18927122-derived --limit-per-dataset 3
python -m pytest -q
python scripts/train.py --config $Config
$OutputDir = python -c "import yaml; print(yaml.safe_load(open(r'$Config', encoding='utf-8'))['output_dir'])"
python scripts/evaluate.py --config $Config --policy hrl --checkpoint "$OutputDir/best.pt" --split validation
python scripts/evaluate.py --config $Config --policy hrl_safe --checkpoint "$OutputDir/best.pt" --split validation
python scripts/evaluate.py --config $Config --policy portfolio --split validation
python scripts/evaluate_baselines.py --config $Config --split validation
python scripts/plot_formal_results.py --evaluation-dir "$OutputDir/eval_hrl_validation"
python scripts/plot_formal_results.py --evaluation-dir "$OutputDir/eval_hrl_safe_validation"
python scripts/plot_formal_results.py --evaluation-dir "$OutputDir/eval_portfolio_validation"
python scripts/plot_policy_comparison.py --output-dir $OutputDir --split validation
