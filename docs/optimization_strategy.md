# Fast optimization strategy

## Goal

Minimize the paired mean of `policy_makespan / HEFT_makespan` while preserving
legal schedules and a reproducible learned-policy path. The design optimizes
the measured contest objective directly; training reward is not used for model
selection.

## Two-track solution

The immediate-result track is a deterministic HEFT-safe portfolio. For every
scenario it evaluates canonical HEFT and 16 seeded perturbations that explore
only the top two ready tasks and top two EFT nodes. Every complete candidate is
run by `ScheduleSimulator`; the minimum measured makespan wins. Canonical HEFT
is always present, so `max_ratio <= 1` is an implementation invariant. This
track needs no GPU training and is the recommended submission fallback.

The learning track is residual hierarchical PPO. The LSTM high agent chooses a
ready task and starts from HEFT upward-rank logits. The GAT low agent chooses a
permanently feasible node and starts from negative-EFT logits. Bounded,
zero-initialized residual heads allow PPO to learn deviations without requiring
the first checkpoint to rediscover HEFT. High-only training freezes placement
to minimum EFT before joint updates. A learned schedule joins the safe
portfolio only in `hrl_safe` mode and cannot degrade its output.

## Stability and speed controls

- HEFT-normalized time observations reduce dataset-scale drift.
- `gamma_high=gamma_low=1` matches the telescoping makespan-delta reward.
- Same-shape rollout items are batched in PPO instead of forwarded one by one.
- Low learning rates, clipped values/gradients and target-KL stopping constrain
  destructive policy movement.
- A fixed, task-size-balanced 18-scenario proxy is checked every 25 episodes.
- Full validation runs only on proxy improvement; best selection still uses
  full validation mean ratio.
- Step-zero validation is checkpointed, so training cannot erase the
  HEFT-equivalent starting policy.

## Actual fixed-validation evidence

The 16-perturbation portfolio was evaluated on 108 scenarios representing 54
held-out base DAGs from rnc50, rnc100 and rnc300. It produced:

| Metric | Value |
| --- | ---: |
| Mean ratio | 0.957168 |
| Standard deviation | 0.095805 |
| Median ratio | 0.998493 |
| P90 ratio | 1.000000 |
| Maximum ratio | 1.000000 |
| Strictly better than HEFT | 72.22% |
| Equal to HEFT | 27.78% |
| Valid schedule rate | 100% |
| Mean inference time | 3185.50 ms/scenario |

By task count, mean ratios were 0.978916 (50), 0.929444 (100), and 0.972464
(300). These measurements are stored under
`outputs/zenodo_heft_safe_fast_2026/eval_portfolio_validation`. They quantify
deterministic schedule search, not pure RL.

The inference figure above includes candidate construction and full simulation.
An earlier internal report started its timer after `policy.reset()` and was
discarded because it omitted portfolio planning.

## Completed residual training and held-out test

The formal run completed all 75 high-only and 75 joint episodes. Checkpoint
selection chose episode 100 at validation mean ratio `0.995168`; optimization
and periodic selection took 3035.93 seconds, and the end-to-end process including
the final full report took 3625.7 seconds. The normalized checkpoint/config hash
is `f82038b34cecde0dd7c61fb48151565967025498ffe7c18242ef46ebddacca15`.

The configuration was frozen before its first test evaluation. On 108 test
scenarios representing 54 unseen base DAGs:

| Policy | Mean ratio | 95% CI | Max ratio | Better than HEFT | Mean inference |
| --- | ---: | ---: | ---: | ---: | ---: |
| HEFT | 1.000000 | [1.000000, 1.000000] | 1.000000 | 0.00% | 474.24 ms |
| Greedy EFT | 1.074101 | [1.047692, 1.102230] | 1.779179 | 10.19% | 432.87 ms |
| Random | 2.011975 | [1.770870, 2.252625] | 6.449233 | 0.93% | 446.77 ms |
| Residual HRL | 0.974760 | [0.955090, 0.992745] | 1.131767 | 37.04% | 3738.72 ms |
| HEFT-safe portfolio | 0.944391 | [0.919354, 0.967207] | 1.000000 | 65.74% | 2689.71 ms |
| HRL-safe | 0.942438 | [0.917141, 0.965873] | 1.000000 | 73.15% | 6551.43 ms |

All six policies produced 108/108 valid schedules. The learned candidate beat
the portfolio candidate on 16 test scenarios; the measured HRL-safe makespan
matched `min(residual HRL, portfolio)` exactly for every scenario. Canonical
HEFT self-ratio was exactly one for every row after preserving float64 EFT
tie-breaking in the common policy interface.

## Decision rule

Use `portfolio` when the deadline is immediate or a checkpoint is unavailable.
Use `hrl_safe` for deployment after training. Report `hrl` separately as the
scientific learning ablation. Reject any configuration with validity below
100%, HEFT self-ratio different from one, or a held-out result produced after
tuning on that held-out split.
