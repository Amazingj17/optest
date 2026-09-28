# Critical-block joint migration: fixed validation

## Scope

The September 14, 2026 experiment uses the unchanged
`data/manifests/grapheonrl_mixed_iid_seed7.json` split, seed 2026,
dataset resource realizations, all 108 validation scenarios and 54 base DAGs.
No sampled validation, training, test evaluation or manifest changes were used.
The reference is the archived prefix-cached beam-width-3, three-round search.

## Implementation

The entire existing beam search runs first. Optional post-search refinement
then expands the current best complete schedule for up to two rounds.
It retains all original candidates, including the original beam winner and
HEFT; no new result can replace the incumbent with a worse makespan.
The audit confirms that the best non-block candidate exactly matches the
archived baseline makespan on every one of the 108 scenarios.

Each refinement round enumerates contiguous windows of two or three tasks
along one realized critical chain, including dependency and resource blockers.
It intersects their permanent feasible-node sets and excludes unchanged
placements. A heuristic ranks blocks and target nodes using execution times
and incoming communication estimates from the incumbent schedule. This score
is neither a proven lower bound nor an accurate prediction of final makespan;
the common simulator determines the actual winner.

The four highest-scoring blocks each try up to two feasible co-location
nodes. Both preserve and greedy downstream repair are evaluated. Every block
member is forced to the selected node, including during greedy repair;
decision order stays unchanged. A co-location candidate may move only one
member if others already occupy the chosen node. The search does not yet
include arbitrary cross-node swaps or multi-node block assignments.

Requested prefixes share the existing prefix cache. Duplicate generated
block decision sequences are discarded after replay. Exact-result caching
is supported but remains disabled in this experiment. At most 16 candidates
are generated per round and 32 per scenario, before duplicate suppression.
Refinement stops when a round has no improvement greater than 1e-12.
The feature is disabled by default through `local_search_block_rounds: 0`.

## Full-validation results

| Metric | Cached beam baseline | With block refinement |
| --- | ---: | ---: |
| Mean ratio | 0.919514912 | 0.918681240 |
| Mean seconds/scenario | 5.803532 | 5.896137 |
| P95 seconds/scenario | 17.384276 | 17.929929 |
| Valid schedule rate | 100% | 100% |
| Maximum ratio | 1.0 | 1.0 |
| Better-than-HEFT rate | 98.1481% | 98.1481% |

Using a 1e-9 tolerance on ratio differences, 13 scenarios improve, none
regress and 95 tie. There are 17 selected block candidates; four of these
change the ratio by less than the reporting tolerance. The instrumented run
recorded 1,799 retained block candidates across all scenarios.

The mean ratio difference is -0.000833671. A paired bootstrap using 10,000
resamples of the 54 base-DAG groups with seed 2026 yields a 95% difference
interval of [-0.002222739, -0.000086807]. Grouping keeps each DAG's resource
realizations together. This exploratory interval is not adjusted for repeated
validation selection and must not be interpreted as untouched-test evidence.
The report's scenario-bootstrap mean-ratio interval is [0.889774, 0.945408].

| Tasks | Mean ratio difference vs baseline |
| --- | ---: |
| 50 | -0.000003219 |
| 100 | -0.000000000072 |
| 300 | -0.002141887 |

Gains concentrate in 300-task scenarios. Measured mean inference time rose
by approximately 1.6%, while P95 rose by approximately 3.1%. These compare
one new run with an archived baseline, not repeated interleaved benchmarks;
small timing differences may include machine-load effects. Added candidate
evaluation work is real even if the observed timing difference is modest.

## Reproduction and artifacts

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m pytest -q
python scripts/evaluate.py --config configs/zenodo_heft_safe_search_beam3_r3_blocks_2026.yaml --policy search --split validation
```

The configuration sets two block rounds, four blocks, maximum block size
three, and two node alternatives. Other search options match the cached
beam baseline, and existing configurations are unchanged.

Artifacts are in
`outputs/zenodo_heft_safe_search_beam3_r3_blocks_2026/eval_search_validation`:

- `summary.json` and `per_scene.csv`: normal evaluation results.
- `block_audit.json`: per-scene original and selected makespans, retained
  block candidate counts and selected candidate names, captured by a reset
  wrapper during the full run.
- `comparison_vs_cached.json`: paired differences and grouped-bootstrap
  interval, computed from the two full per-scene CSV files.

The standard evaluation command regenerates normal reports; the audit and
paired comparison were generated separately in this run.
All 80 tests pass, including a two-task communication barrier where either
single preserve-mode migration worsens the schedule but joint migration
reduces makespan from 6 to 3. Tests also cover disjoint feasible-node sets,
three-task forced placements, both repair modes, determinism, prefix-cache
equivalence, result-cache compatibility and retention of the baseline candidates.

## Decision

Keep this as an explicit quality-oriented configuration, with the existing
cached beam configuration still available. It improves the measured validation
mean without any per-scenario regression against that baseline, but the gain
is small and concentrated in large DAGs. The next meaningful neighborhood
extension would explore coordinated assignments to different nodes rather
than simply increasing the number of co-location windows.
