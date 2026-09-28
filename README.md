# CPN-HRL-DAG

Reproducible cloud-edge-end heterogeneous DAG scheduling.  The primary metric
is the validation mean of `RL_makespan / HEFT_makespan`, where all policies use
the same Scenario and insertion-based simulator.

## Repository contents and latest results

This repository includes source, tests, configurations, the fixed split manifest,
and documentation. Raw datasets, trained checkpoints, full experiment outputs,
local IDE settings and agent workspaces are excluded by `.gitignore` and remain
local. Paths under `outputs/` in historical guides require separately prepared
artifacts; cloning this repository does not download pretrained models.

The latest completed batch is `multiseed_20260925_122243` (seeds 2026/2027/2028).
Its [verified report](docs/results/multiseed_20260925_122243/三种子训练验收与结果报告.md)
and [summary CSV](docs/results/multiseed_20260925_122243/mean_std.csv) are included.
The independent hybrid reached mean ratio **0.918223** (sample standard deviation
across seeds **0.000498**) on 108 shared validation scenarios / 54 base DAGs.
Most improvement came from search; pure HRL reached **0.999058**. These are
validation-selected results, not independent test results. The report below's
older seed2026 runs are retained as historical documentation.

Before training, obtain the [Zenodo 18927122 dataset](https://zenodo.org/records/18927122)
and follow the [local dataset layout](data/README.md).
`python scripts/prepare_data.py` checks the local release; it does not download it.
After installing dependencies, use the training commands below or the
[three-seed run guide](doc/服务器三种子训练与组合评估操作手册.md).

## Recommended competition path

### Current independent hybrid (seed2026, full validation)

`IndependentHybridPolicy` independently completes HEFT, beam+blocks search,
and the frozen HRL checkpoint, validates all three schedules, and replays the
smallest makespan through `SchedulerPolicy`. The learned candidate **does not**
enter the search frontier. This differs from the older `hrl_search_safe` mode.

```bash
python scripts/evaluate_independent_hybrid.py --config configs/hrl_independent_hybrid_seed2026.yaml
```

Use `--output <new-directory>` for another run; nonempty outputs are refused.
This entry only evaluates the fixed 108 validation scenes (54 base DAGs), never
trains or evaluates test. Per-candidate schedules, timing and attribution are saved.
The completed run reached `mean_ratio=0.918677321`, 100% legal schedules,
and 9.6993 seconds/scene on this Windows CPU run. Search alone reached
`0.918681240`: HRL strictly improved 2 scenes, with mean ratio reduction
`0.000003920`. Most of the improvement over HEFT therefore comes from search.
Do not present the full 8.13% reduction as an HRL learning gain.
See [the usage and defence guide](doc/组合策略使用与答辩说明.md) and
[the measured report](outputs/hrl_independent_hybrid_seed2026/RESULTS.md).

### Earlier portfolio/search variants and results

Canonical HEFT is always candidate zero in every safe-search mode. Candidate
schedules are evaluated by the same simulator, and the smallest *actual*
makespan is replayed through the common policy interface. Consequently the
returned ratio is at most `1.0` for every valid scenario. Two deployment points
are available: the legacy 16-candidate `portfolio` is the latency-oriented
fallback, while `search` adds PEFT/OCT, scaled-communication HEFT ranks,
limited discrepancies, and configurable realized-critical-chain LNS.

Keep the following result labels separate in reports:

- `hrl`: pure LSTM/GAT residual PPO checkpoint; this is the learning result.
- `portfolio`: HEFT plus deterministic schedule search; this needs no model.
- `search`: enhanced HEFT-safe search, optionally with critical-chain LNS.
- `hrl_safe`: portfolio plus the learned schedule; this is the deployment mode.
- `hrl_search_safe`: enhanced search plus the learned residual HRL candidate.
- `heft`: the project-owned reference baseline.

On the fixed validation split in
`data/manifests/grapheonrl_mixed_iid_seed7.json`, the already materialized
16-candidate portfolio evaluated 108 scenarios from 54 unseen base DAGs at
`mean_ratio=0.957168`, `max_ratio=1.0`, and 100% valid schedules. This is an
actual portfolio result, not a claimed pure-RL improvement. See
[the optimization strategy](docs/optimization_strategy.md) for the decision
rationale and reporting rules. The frozen held-out test results are summarized
in [the formal results report](docs/formal_results.md): pure residual HRL reached
`0.974760`, portfolio `0.944391`, and HRL-safe `0.942438`, all with 100% valid
schedules. Only the two safe policies guarantee a per-scenario ratio at most
one.

Post-report development used only the fixed validation split. The two-round
LNS configuration reached `mean_ratio=0.927551`, 95% bootstrap CI
`[0.900587, 0.951886]`, `max_ratio=1.0`, 93.52% better-than-HEFT rate, and 100%
valid schedules over the same 108 validation scenarios. Its measured
end-to-end inference was 1.342 s/scenario; the legacy portfolio rerun with the
same optimized simulator took 0.652 s/scenario. These are validation results,
not new test claims. See [the neural-guided search report](docs/neural_guided_search.md).

The subsequent cached beam plus critical-block refinement reached a validation
mean ratio of 0.918681. A full 108-scenario feasibility audit places an admissible
combined lower bound at 0.894645 under the unchanged scheduling model and HEFT
denominators, excluding an overall target of 0.8. See the
[Chinese feasibility assessment and optimization roadmap](docs/validation_target_080_plan_zh.md)
for the derivation, realistic milestones, and proposed experiments. The bound
is not a claim that 0.894645 is achievable, and these remain validation-only results.

## Graph PPO and coordinated MAPPO comparisons

Two pure-learning baselines now share the same graph encoders, environment,
training coverage and complete 108-scenario validation protocol: `graph_ppo`
uses one joint task/resource actor, while `tier_mappo` uses end/edge/cloud
proposal actors, a learned coordinator and a centralized team critic. The
MAPPO variant is a project adaptation, not an exact reproduction of a paper.
Neither baseline includes HEFT fallback or search. Training and comparison
commands, probability definitions and reporting limits are documented in
[the Chinese comparison guide](doc/云边端多种策略对比.md).

## Design

`DatasetAdapter -> UnifiedScenario -> ScheduleSimulator -> Environment -> Policy -> Evaluator` is the only scheduling path. GrapheonRL format details never reach policies. The simulator is independent of Torch and supports non-preemptive insertion into the first feasible node idle gap. For task `i` and node `j`, it uses an explicit execution model, dependency-ready time, pairwise communication, EST, and EFT. Same-node communication is zero.

The high level is masked LSTM-PPO and chooses exactly one ready task. The low
level is a pure-PyTorch dense GAT-PPO and chooses a permanently feasible
resource node conditioned on task execution/communication/DRT/EST/EFT
features. Busy nodes are never masked. Both actors can be initialized as
bounded residuals around deterministic HEFT: high-level logits start from
normalized upward rank and low-level logits from normalized negative EFT. The
zero-initialized residual heads therefore reproduce HEFT before PPO updates.

The enhanced search additionally implements PEFT's Optimistic Cost Table,
CPOP, configurable HEFT communication-rank scaling, systematic task/node
discrepancies, and node moves on the realized schedule critical chain. The
list-schedule ready set and permanent task/node feasibility are cached
incrementally in the simulator; search branches can clone dynamic simulator
state without recomputing static execution matrices.

An experimental joint graph policy encodes the task DAG and resource network
once per Scenario, then jointly scores every legal `(ready task, node)` pair
from cached static embeddings and dynamic EST/EFT features. Its search-teacher
cache and distillation path are functional, but the current small-sample smoke
run selected the protected HEFT-equivalent step-zero checkpoint. It is not a
formal performance result; DAgger/regret supervision remains necessary before
full training.

Observations normalize workload, speed, graph statistics and all time-like
candidate features without using final RL outcomes. Time features use the
scenario's HEFT makespan as a fixed scale. Rollouts persist masks with old
log-probabilities so PPO ratios are evaluated against the original legal
action set. Batched same-shape PPO updates include GAE, clipped policy/value
losses, entropy, gradient clipping, LR decay and an optional target-KL stop.
With the default undiscounted makespan-delta objective, both agents use
`gamma=1.0`, so the episode reward telescopes to `-RL_makespan/HEFT_makespan`.

Training supports three explicit phases: low-level pretraining with HEFT-rank
task choice, high-level training with a frozen deterministic min-EFT placer,
and joint PPO. Checkpoint selection uses a fixed size-balanced validation proxy;
the full validation split is evaluated only when that proxy improves. Step zero
is evaluated and saved, preventing a degraded PPO update from replacing the
HEFT-equivalent initialization.

## Data and provenance

The evidence-based schema and licence audit is in
[docs/dataset_audit.md](docs/dataset_audit.md).

- GrapheonRL STG JSON, repository commit `22134e74028b0164032fbcc4c806bfa7c385e551`, Zenodo 18927122, CC-BY-4.0.

Raw data is read-only under `data/raw`. Scenario provenance includes source,
original path, original graph identity, adapter version, original metadata,
and resource seed/configuration. Split manifests use
`dataset_source:original_graph_id`, preventing a topology from crossing train,
validation, or test even when it has multiple resource realizations.

## Install and run

For server training of all three neural models, live logs, and the six-policy
full-validation comparison, use `scripts/run_server_comparison.py`.
Start with `--output outputs/server_run --gpus 1 --dry-run`; remove `--dry-run`
only after checking GPU allocation. It defaults to three complete coverage
epochs, never evaluates test, and supports `--reuse-main /path/to/best.pt`.
See [the server comparison guide](doc/服务器全模型训练与性能对比操作指南.md)
for multi-GPU execution, progress monitoring, artifacts, and failure recovery.

```bash
pip install -e '.[train,analysis,dev]'
export PYTHONPATH="$PWD/src"
python -m pytest -q
bash scripts/run_all.sh --config configs/zenodo_heft_safe_fast_2026.yaml
# Windows PowerShell equivalent:
powershell -ExecutionPolicy Bypass -File scripts/run_all.ps1 -Config configs/zenodo_heft_safe_fast_2026.yaml
```

Training is offline after data preparation. If network access is unavailable,
place the Zenodo 18927122 release at
`data/raw/zenodo-18927122-derived`. The adapter reads its workflow and system
configuration `.tar.xz` members directly and never mutates raw data. Default
experiments select the `rnc50`, `rnc100`, and `rnc300` families; adjust
`dataset.archive_task_counts` to opt into larger graph families.
`scripts/prepare_data.py` verifies the manually prepared local release. CPU execution is
supported; on openEuler, create a Python 3.10+ virtual environment and install
the same dependencies. CUDA deterministic behaviour still depends on the
installed PyTorch/CUDA operator set.

`configs/generated_smoke.yaml` additionally verifies that generated
cloud-edge-end resources, task CPU/GPU/memory affinity, HEFT cache, checkpoint
loading, and evaluation materialize the exact same fixed validation realization.

## Protocols and outputs

`scripts/create_protocol.py` creates fixed manifests for Mixed IID and scale
generalization protocols. The
resource generator supports `dataset`, `generated`, and `hybrid` modes. It
materializes resources only after base-DAG splitting, uses fixed validation
seeds, and supports distinct train/evaluation profiles for resource OOD.
`scripts/train.py --resume outputs/<experiment>/latest.pt` restores model,
optimizer and scheduler state. Training writes `best.pt`, `latest.pt`, HEFT cache,
config, split manifest, logs, per-scenario CSV, summary JSON/CSV, dataset/size/domain reports, training curve and ratio histogram.

For separate, auditable evaluation after training:

```bash
python scripts/evaluate.py --config configs/zenodo_heft_safe_fast_2026.yaml --policy hrl --checkpoint outputs/zenodo_heft_safe_fast_2026/best.pt --split test
python scripts/evaluate.py --config configs/zenodo_heft_safe_fast_2026.yaml --policy hrl_safe --checkpoint outputs/zenodo_heft_safe_fast_2026/best.pt --split test
python scripts/evaluate.py --config configs/zenodo_heft_safe_fast_2026.yaml --policy portfolio --split test
python scripts/evaluate.py --config configs/zenodo_heft_safe_fast_2026.yaml --policy heft --split test
python scripts/plot_formal_results.py --evaluation-dir outputs/zenodo_heft_safe_fast_2026/eval_hrl_safe_test
python scripts/plot_policy_comparison.py --output-dir outputs/zenodo_heft_safe_fast_2026 --split test
```

For the enhanced validation-only search and graph-distillation smoke path:

```bash
python scripts/evaluate.py --config configs/zenodo_heft_safe_search_lns2_2026.yaml --policy search --split validation
python scripts/analyze_search_candidates.py --config configs/zenodo_heft_safe_search_lns2_2026.yaml --split validation --limit 0
python scripts/plot_search_improvement.py
python scripts/train_dag_pair.py --config configs/zenodo_dag_pair_distill_smoke_2026.yaml
```

The supplied config is deliberately a tiny smoke run, not a competition
claim.  Formal results require a fixed public manifest, many DAGs and resource
realizations, plus reported mixed/cross/scale/resource/domain and ablation
experiments (hierarchy, LSTM, GAT, staged training, HEFT features, BC).

For reproducible architecture ablations, run:

```bash
python scripts/run_ablations.py --base-config configs/generated_smoke.yaml \
  --output-dir outputs/ablations_smoke --variants full no_lstm no_gat no_staged no_heft_features
```

The runner changes the actual model/runtime controls: `no_lstm` replaces the
high sequence encoder with per-task MLP embeddings; `no_gat` replaces resource
message passing with a per-node MLP; `no_staged` removes the low-level
pretraining phase; and `no_heft_features` zeros every rank-derived task input.

The separate Flat PPO baseline uses one masked categorical distribution over
all legal `(ready task, feasible node)` pairs and is intended for small graphs:

```bash
python scripts/train_flat.py --config configs/generated_smoke.yaml --episodes 2
```

## State, actions, masks and reward

The stable-topological task tensor contains workload, degree, readiness,
scheduled-predecessor progress, graph level, data volumes, execution summaries,
optional upward-rank features and DAG progress. `task_mask` marks padding while
`ready_mask` marks legal high-level actions. The resource graph contains tier,
speed/capacity, availability, utilization, assigned work, bandwidth and latency.
For the selected task, every node receives execution, communication, DRT, EST
and EFT features. `node_mask` excludes only permanent incompatibilities such as
missing GPU or insufficient memory; a busy node remains legal.

The default reward is
`-(partial_makespan_t-partial_makespan_(t-1))/(HEFT_makespan+eps)`.
Communication penalties and potential shaping are disabled unless explicitly
configured. Evaluation is deterministic and rejects illegal actions instead of
learning legality through penalties.

## Evaluation and reproducibility

All policies consume the identical immutable Scenario, execution model,
pairwise bandwidth matrix, latency matrix and insertion simulator. Reports
include mean/std/median/p90 ratio, sample count, paired makespans, per-size and
per-domain tables, validity, inference time and optional bootstrap intervals.
The manifest first splits source-qualified base DAG IDs and only then handles
resource realizations, so topology leakage is rejected by construction.

Every formal output preserves the resolved YAML, source audit, split manifest,
runtime versions, seed, HEFT cache, logs, checkpoints, CSV/JSON metrics and PNG
plots. Python, NumPy, Torch CPU and CUDA seeds are set. Exact CUDA bitwise
reproducibility still depends on the installed driver/operator set; the supplied
RTX 3050 Ti configuration disables cuDNN for the recurrent path after a
platform-specific long-run LSTM fault was observed.

## Known limitations

The Zenodo release is currently the only supported data source. Its endpoint
transfer rates must be converted to a pairwise matrix using the documented
minimum-endpoint compatibility rule. The portfolio improves makespan by
multi-candidate simulation and therefore has higher inference latency than
single-pass HEFT or pure HRL. A portfolio result must never be presented as a
pure-RL result. No optimal-solver lower bound is included, so a ratio below one
means improvement over this implementation of HEFT, not proof of global
optimality. The enhanced search was selected using validation and has not been
rerun on the frozen test split; use a new untouched final holdout or the
official hidden test for an unbiased post-optimization claim. The joint graph
model currently has only smoke-scale distillation evidence and must not be
reported as outperforming HEFT.
