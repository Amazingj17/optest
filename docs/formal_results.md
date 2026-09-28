# Formal optimization results

## Experimental identity

- Data: the local Zenodo 18927122-derived GrapheonRL release only.
- Families: rnc50, rnc100 and rnc300.
- Manifest: `data/manifests/grapheonrl_mixed_iid_seed7.json`.
- Validation: 108 scenarios / 54 unseen base DAGs.
- Test: 108 scenarios / 54 unseen base DAGs.
- Training: 75 high-only plus 75 joint residual PPO episodes.
- Best checkpoint: episode 100.
- Canonical config hash:
  `f82038b34cecde0dd7c61fb48151565967025498ffe7c18242ef46ebddacca15`.

The test split was not used by PPO, proxy validation, early stopping or
checkpoint selection. The configuration was frozen before the first test run.

## Validation

| Policy | Mean ratio | Standard deviation | P90 | Max | Valid |
| --- | ---: | ---: | ---: | ---: | ---: |
| Residual HRL | 0.995168 | 0.139152 | 1.053827 | 2.011220 | 100% |
| HEFT-safe portfolio | 0.957168 | 0.095805 | 1.000000 | 1.000000 | 100% |
| HRL-safe | 0.953510 | 0.100704 | 1.000000 | 1.000000 | 100% |

The residual model improved the portfolio on 14/108 validation scenarios. The
pure learned mean is below one, but its large maximum and confidence interval
crossing one make it unsuitable as an unguarded deployment policy.

## Held-out test

| Policy | Mean ratio | Standard deviation | Median | P90 | Max | Valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HEFT | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 | 100% |
| Greedy EFT | 1.074101 | 0.136323 | 1.080206 | 1.209520 | 1.779179 | 100% |
| Random | 2.011975 | 1.267972 | 1.505400 | 4.194320 | 6.449233 | 100% |
| Residual HRL | 0.974760 | 0.102152 | 1.000000 | 1.017563 | 1.131767 | 100% |
| HEFT-safe portfolio | 0.944391 | 0.127369 | 0.999068 | 1.000000 | 1.000000 | 100% |
| HRL-safe | 0.942438 | 0.127550 | 0.998638 | 1.000000 | 1.000000 | 100% |

Pure HRL's 95% bootstrap interval was `[0.955090, 0.992745]`; the learned
improvement therefore transferred to the frozen test split at this sample size.
HRL-safe improved the portfolio on 16/108 test scenarios and remained no worse
than canonical HEFT on every scenario.

## Scale breakdown for HRL-safe test

| Tasks | Scenarios | Mean ratio | Standard deviation | Median |
| ---: | ---: | ---: | ---: | ---: |
| 50 | 38 | 0.949357 | 0.119181 | 0.998860 |
| 100 | 36 | 0.917478 | 0.156810 | 0.998558 |
| 300 | 34 | 0.961132 | 0.093298 | 0.998442 |

## Interpretation

The best immediate result is the no-training portfolio. Formal residual PPO
adds a smaller but independently generalizing improvement. Combining them gives
the lowest mean ratio and the required per-scenario HEFT guard, at the cost of
6.55 seconds mean policy inference on the measured Windows/RTX 3050 Ti setup.
The portfolio gain is schedule search, not an RL result; pure HRL and HRL-safe
must remain separate rows in any competition report.

## Post-report optimization (validation only)

After the frozen test report above, architecture development continued on the
fixed validation split only. The test split was not rerun. These numbers are
therefore model-selection evidence, not replacement held-out test claims.

| Validation policy | Mean ratio | 95% bootstrap CI | Max | Better than HEFT | Valid | Mean inference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy HEFT-safe portfolio | 0.957168 | [0.939189, 0.973270] | 1.000000 | 72.22% | 100% | 0.652 s |
| Residual HRL-safe | 0.953510 | [0.934913, 0.970468] | 1.000000 | 76.85% | 100% | 7.916 s* |
| PEFT/rank/LDS safe search | 0.935056 | [0.910280, 0.957438] | 1.000000 | 92.59% | 100% | 1.144 s |
| Safe search + 1-round LNS | 0.930873 | not formally exported | 1.000000 | — | 100% | search audit only |
| Safe search + 2-round LNS | **0.927551** | **[0.900587, 0.951886]** | **1.000000** | **93.52%** | **100%** | **1.342 s** |

`*` The HRL-safe timing predates the incremental simulator-ready cache and is
not a fair speed comparison. The legacy portfolio and both new formal search
rows were measured through the current evaluator. The optimized simulator
preserved every audited makespan while reducing the 18-scenario broad-search
core time from 5.876 s to 1.501 s per scenario.

The selected two-round LNS scale means were 0.926234 (50 tasks), 0.901196
(100 tasks), and 0.954660 (300 tasks). A graph-policy distillation smoke run
successfully generated/cached teachers, trained, loaded its best checkpoint,
and maintained 100% legal actions. Its protected best checkpoint remained the
HEFT-equivalent step-zero model at ratio 1.0; later epochs degraded the six
scenario smoke validation subset, so no learned improvement is claimed.

The paired mean-ratio gain over the legacy portfolio was 0.029616 with a 95%
bootstrap interval of [0.016348, 0.046277]. Against the previous HRL-safe
validation result, the paired gain was 0.025959 with interval
[0.013010, 0.042307]: 66 scenarios improved, 38 tied within `1e-9`, and 4 were
worse. The HEFT incumbent still bounded all of those four ratios by one.
