# Neural-guided safe-search optimization

## Implemented outcome

The first optimization stage replaces a single stochastic HEFT perturbation
family with an auditable collection of canonical HEFT, PEFT/OCT look-ahead,
scaled-communication upward ranks, deterministic limited discrepancies, and
realized-critical-chain node relocation. Canonical HEFT remains candidate zero
and all complete candidates are measured by `ScheduleSimulator`, so the
selected result cannot be worse than HEFT.

On the fixed 108-scenario / 54-base-DAG validation split, the selected
two-round configuration produced:

- mean ratio: 0.927551
- 95% bootstrap CI: [0.900587, 0.951886]
- median ratio: 0.994583
- p90 ratio: 1.000000
- maximum ratio: 1.000000
- better than HEFT: 93.52%
- valid schedules: 100%
- evaluator inference: 1.342 s/scenario

The result is stored under
`outputs/zenodo_heft_safe_search_lns2_2026/eval_search_validation`. It is a
validation result. The previously frozen test set has not been used to select
or assess this configuration.

## Search construction

PEFT computes an Optimistic Cost Table for every task/resource pair and ranks
ready tasks by average OCT. Its resource decision minimizes
`EFT(i,j) + weight * OCT(i,j)`. The selected teacher portfolio retains weights
0 and 0.25. Additional HEFT candidates recompute upward ranks with average
communication multipliers 0, 0.25, 2, and 4. One deterministic node
limited-discrepancy candidate explores a low-regret alternative on the
canonical HEFT trace, while 16 seeded perturbations preserve the previous
comparison baseline.

After selecting the best initial candidate, LNS traces an actual critical
chain through both predecessor arrival constraints and same-node blockers.
Four evenly spaced movable critical-chain tasks are tried on their closest
alternative node. Two repair modes either retain the incumbent's downstream
placements or greedily recompute downstream EFT placement. A second round is
run only when the previous round strictly improves the incumbent.

Incremental list-ready and permanent-feasibility caches preserve simulator
semantics but eliminate repeated whole-DAG scans inside every candidate EFT.
The simulator also supports independent dynamic-state cloning for future beam
search.

## Joint graph-policy stage

The experimental `DAGPairGraphActorCritic` replaces the LSTM/GAT hierarchy
with a single legal-pair distribution. Edge-biased graph attention separately
encodes the task DAG and resource network. Static embeddings are computed once
per Scenario; current readiness, resource load, and pair-specific EST/EFT data
are fused by a lightweight actor. The action mask is the exact Cartesian set
of ready tasks and permanently feasible nodes.

The actor is a bounded residual around an exact lexicographic HEFT prior, so a
zero-initialized model reproduces HEFT. `SearchTeacherCache` hashes all
scheduling-relevant scenario content plus the search configuration and stores
the winning complete trajectory. The distiller samples states across each
trajectory and trains joint action classification plus a normalized-makespan
value target.

The initial 12-train / 6-validation smoke run exposed distribution shift:
training loss decreased, but later pure-policy validation degraded, and the
HEFT-equivalent epoch-zero checkpoint remained best. Accordingly, the next
stage must use a larger topology-diverse teacher set and DAgger or per-action
completion regret. Increasing epochs on the same winner trajectories is not an
accepted optimization strategy.

## Next controlled stage

1. Generate teacher schedules only for train base DAGs and keep validation
   fixed.
2. At model-visited partial schedules, evaluate top legal actions with a
   heuristic completion to obtain normalized regret rather than a single hard
   label.
3. Add those states through DAgger and train listwise/ranking targets.
4. Use the model to rank beam/LNS expansions while always retaining the HEFT
   incumbent.
5. Select checkpoints solely by validation mean ratio and evaluate once on a
   new untouched final holdout or official hidden test.

## References

1. Topcuoglu, H., Hariri, S., and Wu, M.-Y. “Performance-Effective and
   Low-Complexity Task Scheduling for Heterogeneous Computing.” IEEE TPDS,
   2002. https://doi.org/10.1109/71.993206
2. Arabnejad, H. and Barbosa, J. G. “List Scheduling Algorithm for
   Heterogeneous Systems by an Optimistic Cost Table.” IEEE TPDS, 2014.
   https://doi.org/10.1109/TPDS.2013.57
3. Mao, H. et al. “Learning Scheduling Algorithms for Data Processing
   Clusters.” ACM SIGCOMM, 2019. https://doi.org/10.1145/3341302.3342080
4. Hu, Z. et al. “Heterogeneous Graph Transformer.” WWW, 2020.
   https://doi.org/10.1145/3366423.3380027
5. Ying, C. et al. “Do Transformers Really Perform Bad for Graph
   Representation?” NeurIPS, 2021. https://arxiv.org/abs/2106.05234
6. Zhang, C. et al. “Learning to Dispatch for Job Shop Scheduling via Deep
   Reinforcement Learning.” NeurIPS, 2020. https://arxiv.org/abs/2010.12367
7. Ross, S., Gordon, G., and Bagnell, D. “A Reduction of Imitation Learning and
   Structured Prediction to No-Regret Online Learning.” AISTATS, 2011.
   https://arxiv.org/abs/1011.0686
8. Kwon, Y.-D. et al. “POMO: Policy Optimization with Multiple Optima for
   Reinforcement Learning.” NeurIPS, 2020. https://arxiv.org/abs/2010.16011
9. Hottung, A., Kwon, Y.-D., and Tierney, K. “Efficient Active Search for
   Combinatorial Optimization Problems.” ICLR, 2022.
   https://arxiv.org/abs/2106.05126
