# Reproduction

Raw data remains under `data/raw` and is never modified. A fixed
`split_manifest.json` contains source-qualified base DAG IDs, not resource
realizations. Generated resources are only materialized after splitting, and
validation/test seed lists are fixed. Generated GPU-capable tiers automatically
advertise a `gpu` feature when their configured GPU capacity is positive.

```powershell
$env:PYTHONPATH = "$PWD/src"
python -m pytest -q
powershell -ExecutionPolicy Bypass -File scripts/run_all.ps1 -Config configs/zenodo_heft_safe_fast_2026.yaml
python scripts/train.py --config configs/generated_smoke.yaml
python scripts/train.py --config configs/generated_smoke.yaml --resume outputs/generated_smoke/latest.pt
```

Linux/openEuler uses `bash scripts/run_all.sh --config configs/cpn_hrl_dag.yaml`.
Each output directory preserves config, config hash, runtime metadata, split
manifest, source audit, cached HEFT references, checkpoints, logs, paired
per-scenario results and aggregate reports.

The formal fast configuration is `configs/zenodo_heft_safe_fast_2026.yaml`.
It uses HEFT-equivalent residual initialization, normalized observations,
75 high-only plus 75 joint episodes, a fixed 18-scenario validation proxy and
full validation only after proxy improvement. `run_all` evaluates pure `hrl`,
`hrl_safe`, `portfolio`, and `heft` into separate directories and renders each
non-baseline dashboard. This separation is required for honest attribution.

To generate all held-out baselines with one dataset load and then render the
combined quality/latency plot:

```powershell
python scripts/evaluate_baselines.py --config configs/zenodo_heft_safe_fast_2026.yaml --split test
python scripts/plot_policy_comparison.py --output-dir outputs/zenodo_heft_safe_fast_2026 --split test
```

Checkpoint and report hashes use the same canonical YAML mapping and exclude
the separately recorded `runtime` block. Existing checkpoints can be normalized
without changing model or optimizer state using
`scripts/normalize_checkpoint_metadata.py`.
