#!/usr/bin/env bash
set -euo pipefail
CONFIG="configs/cpn_hrl_dag.yaml"
if [[ "${1:-}" == "--config" ]]; then
  if [[ -z "${2:-}" ]]; then
    echo "--config requires a YAML path" >&2
    exit 2
  fi
  CONFIG="$2"
fi
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
python scripts/audit_datasets.py --grapheonrl-root data/raw/zenodo-18927122-derived --grapheonrl-system-configs data/raw/zenodo-18927122-derived --limit-per-dataset 3
python -m pytest -q
python scripts/train.py --config "$CONFIG"
OUTPUT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG', encoding='utf-8'))['output_dir'])")
python scripts/evaluate.py --config "$CONFIG" --policy hrl --checkpoint "$OUTPUT_DIR/best.pt" --split validation
python scripts/evaluate.py --config "$CONFIG" --policy hrl_safe --checkpoint "$OUTPUT_DIR/best.pt" --split validation
python scripts/evaluate.py --config "$CONFIG" --policy portfolio --split validation
python scripts/evaluate_baselines.py --config "$CONFIG" --split validation
python scripts/plot_formal_results.py --evaluation-dir "$OUTPUT_DIR/eval_hrl_validation"
python scripts/plot_formal_results.py --evaluation-dir "$OUTPUT_DIR/eval_hrl_safe_validation"
python scripts/plot_formal_results.py --evaluation-dir "$OUTPUT_DIR/eval_portfolio_validation"
python scripts/plot_policy_comparison.py --output-dir "$OUTPUT_DIR" --split validation
