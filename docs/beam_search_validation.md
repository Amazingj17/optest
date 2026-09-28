# Multi-start beam search: fixed validation only

## Scope and implementation

Evaluated on September 14, 2026 using the unchanged
`data/manifests/grapheonrl_mixed_iid_seed7.json` manifest, experiment seed
2026, dataset resource realizations, 108 scenarios and 54 base DAGs.
Scenario IDs and HEFT makespans match the existing wide-LNS report exactly.
No test evaluation, subset selection, model training, or split changes were performed.

The new `search.local_search_beam_width` option defaults to 1, preserving
the original single-incumbent search and early stopping. A width above 1
selects up to that many best unexpanded schedules per round from an archive.
Schedule identity includes both decision order and node assignments.
Duplicate schedules do not enter the archive twice or get expanded twice.
Duplicate moves are currently detected after simulation, not before it.
Stable ordering preserves deterministic ties and the initial HEFT candidate.
Unlike the legacy loop, a non-improving round does not terminate search.
The historical best candidate remains eligible for final selection.

This is an archive-based multi-start beam variant, not a strict beam that
permanently discards all candidates outside its width. It does not guarantee
diversity beyond distinct decision sequences and assignments. It retains
HEFT safety, but does not preserve the entire legacy wide-LNS trajectory.
Therefore it need not dominate wide-LNS on every scenario.

## Results

| Configuration | Mean ratio | Mean seconds/scene | P95 seconds/scene | Improved / worse / tied vs wide-LNS |
| --- | ---: | ---: | ---: | --- |
| Existing wide-LNS | 0.922860364 | 2.627430 | 8.336448 | baseline |
| Beam width 3, 1 round | 0.926173022 | 3.048142 | 9.535147 | 21 / 39 / 48 |
| Beam width 3, 3 rounds | 0.919514912 | 8.253199 | 25.129540 | 42 / 3 / 63 |

All configurations have 108 scenarios, 54 base DAGs, 100% valid schedules,
98.1481% better-than-HEFT rate, and maximum ratio 1.0.
Pairwise win/loss classification uses a ratio tolerance of 1e-9.

The three-round beam improves the mean ratio by 0.003345453 versus wide-LNS
(0.334545 percentage points of the HEFT-relative ratio), but takes about
3.14 times the measured inference time. Its scenario-bootstrap mean-ratio
95% interval is [0.891064, 0.946158]. The paired mean difference, using 10,000
bootstrap samples of the 54 base-DAG groups and seed 2026, has interval
[-0.005766097, -0.001416934]. Grouping keeps the resource realizations of
each base DAG together. This exploratory interval is not corrected for
repeated validation-set selection and is not evidence from an untouched test set.

Mean ratio differences by task count for the three-round beam are:

| Tasks | Beam minus wide-LNS |
| --- | ---: |
| 50 | -0.005220066 |
| 100 | -0.004272287 |
| 300 | -0.001347411 |

The one-round beam and legacy wide-LNS each permit at most three parent
expansions, but this is not an exactly matched candidate or runtime budget:
legacy early stopping, legal neighborhood sizes, and duplicates differ.
The three-round beam allows up to nine parent expansions. Its improvement
therefore cannot be attributed to beam strategy alone independently of extra compute.
Wide-LNS timings are from the existing run; there was no fresh timing rerun.

## Reproduction

From the repository root in PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m pytest -q
python scripts/evaluate.py --config configs/zenodo_heft_safe_search_beam3_r1_2026.yaml --policy search --split validation
python scripts/evaluate.py --config configs/zenodo_heft_safe_search_beam3_r3_2026.yaml --policy search --split validation
```

Reports are in each corresponding
`outputs/zenodo_heft_safe_search_beam3_r{1,3}_2026/eval_search_validation`
directory, including `summary.json` and `per_scene.csv`.
All 58 tests pass, including deterministic replay, default-width compatibility,
invalid-width rejection, duplicate suppression and plateau continuation.

## Decision

Keep wide-LNS as the latency-oriented option. The three-round beam is an
optional quality-oriented configuration, not an unconditional replacement.
Do not promote the one-round variant: its mean result is worse and it is slower.
Prefix replay is now optimized as documented below. Consider pre-simulation
duplicate filtering before increasing width or rounds further. Any subsequent comparison must continue
to use the same full 108-scenario validation split and leave test untouched.

## Prefix-cache optimization

The September 14, 2026 follow-up adds
`search.local_search_prefix_cache` (default `true`). Set it to `false`
to use the original full replay for controlled comparisons.
For each parent schedule, node and task-order moves share requested simulator
prefix snapshots. New snapshots extend the nearest cached earlier prefix;
each candidate clones its snapshot and replays only the suffix.
Only one parent's snapshots are retained at a time. The cache is cleared at
reset and after selecting the winning schedule, and never shared across scenarios.
Search width, rounds, ordering, tie-breaking and candidate selection are unchanged.

The cached configuration was evaluated on all 108 fixed validation scenarios
from 54 base DAGs, with no test run and no subset evaluation.
Every field in `per_scene.csv` except `inference_time_ms` is exactly equal
to the archived uncached three-round beam result. This verifies identical
reported per-scenario outcomes, not complete real-data schedule-trace identity.
Unit tests additionally compare every candidate's complete SimulationResult
and the winning decision sequence with caching enabled and disabled for both
single-incumbent and beam search, and confirm fewer simulator schedule calls.

| Metric | Uncached beam 3 x 3 | Cached beam 3 x 3 |
| --- | ---: | ---: |
| Mean ratio | 0.919514912 | 0.919514912 |
| Mean seconds/scenario | 8.253199 | 5.803532 |
| P95 seconds/scenario | 25.129540 | 17.384276 |
| Valid schedule rate | 100% | 100% |
| Maximum ratio | 1.0 | 1.0 |

Mean inference time decreased by approximately 29.7% (1.42x speedup), while
P95 decreased by approximately 30.8%. Mean seconds/scenario by task count:

| Tasks | Uncached | Cached |
| --- | ---: | ---: |
| 50 | 2.459435 | 0.984713 |
| 100 | 3.209545 | 2.717928 |
| 300 | 16.607576 | 11.642747 |

Timing comparisons use the archived uncached run and one subsequent cached
run, not repeated interleaved benchmarks; machine-load effects may contribute.
These figures measure evaluator inference, not total process wall-clock time.
The original reports remain untouched. Cached reports are under
`outputs/zenodo_heft_safe_search_beam3_r3_cached_2026/eval_search_validation`.

Reproduce with the same PYTHONPATH setup used above:

```powershell
python -m pytest -q
python scripts/evaluate.py --config configs/zenodo_heft_safe_search_beam3_r3_cached_2026.yaml --policy search --split validation
```

All 61 tests pass, including out-of-order snapshot lookup, branch isolation,
cache cleanup, cross-scenario reset, exact candidate equivalence and replay reduction.
The prefix-cache optimization alone still filters duplicate candidates only
after replay. The following separate experiment adds exact-result reuse.

## Exact-result cache experiment

The next September 14, 2026 experiment adds opt-in
`search.local_search_result_cache`, default `false`.
After resolving the proposed node or ready-task alternative, a preserve-mode
move has a fully specified decision order and node assignment. The cache
looks up that exact sequence before simulating the modified suffix. A hit
reuses an immutable complete SimulationResult, while preserving candidate
names, ordering, comparisons and existing beam deduplication behavior.

Initial portfolio schedules and completed local-search schedules seed the
cache. Greedy-repair moves still simulate normally because their downstream
assignments cannot be known in advance; their completed results are available
to subsequent preserve-mode lookups. This is not deduplication of all possible
candidates before any work: prefix lookup and alternative ranking still occur.
The cache is scoped to one scenario and cleared at reset and after selection.
No search neighborhoods, budgets, seeds or scoring rules change.

All 108 scenarios from the fixed validation manifest were evaluated again.
Every per-scene report field except inference time matches the preceding
prefix-cached run exactly. No test or sampled validation run was performed.

| Metric | Prefix cache only | Prefix + exact-result cache |
| --- | ---: | ---: |
| Mean ratio | 0.919514912 | 0.919514912 |
| Mean seconds/scenario | 5.803532 | 5.748661 |
| P95 seconds/scenario | 17.384276 | 17.354291 |
| Valid schedule rate | 100% | 100% |
| Maximum ratio | 1.0 | 1.0 |

The instrumented run recorded 341 result-cache hits across 76 scenarios,
skipping 43,628 task schedule calls in duplicate suffixes. These counters
do not count avoided EFT calculations, whole scenarios, or unique schedules.
The remaining 32 scenarios had no hits. Counters and per-scene audit data are
saved in `result_cache_audit.json` alongside the report.

Mean inference time decreased only 0.9455% relative to the archived
prefix-cached run. This is a single sequential comparison, not repeated
interleaved benchmarking, and is insufficient to establish a reliable speedup.
Hashing full decision sequences and retaining results have overhead; the
observed skipped work does not imply a proportional time reduction.
Keep the option disabled by default rather than replacing the existing
recommended configuration on the basis of this small timing difference.

All 68 tests pass. New tests cover exact candidate equivalence with beam
widths 1 and 3, both prefix-cache settings, scenario cleanup, exact order and
assignment matching, and repeated node/task preserve moves making no suffix
schedule calls. In the synthetic beam cases, the counted reduction in schedule
calls exactly matches the reported skipped-task counter.

Reproduction, using the PYTHONPATH setup above:

```powershell
python -m pytest -q
python scripts/evaluate.py --config configs/zenodo_heft_safe_search_beam3_r3_dedup_2026.yaml --policy search --split validation
```

Reports are under
`outputs/zenodo_heft_safe_search_beam3_r3_dedup_2026/eval_search_validation`.
The standard evaluation command reproduces scheduling and timing reports;
the stored cache audit was captured separately through a reset wrapper reading
`replay_cache_hits` and `replay_cache_skipped_tasks` after each scenario.
It requires the same complete 108-scenario evaluation, not a smaller sample.
The subsequent joint critical-block experiment is documented in
`docs/critical_block_validation.md`. It targets schedule quality rather than
assuming further memoization will materially improve mean ratio.
