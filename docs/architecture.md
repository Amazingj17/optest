# Architecture

All inputs follow one path: `DatasetAdapter -> Scenario -> ScheduleSimulator ->
Environment -> SchedulerPolicy -> Evaluator`. Immutable scenario preprocessing
builds the topological order, predecessor/successor arrays, edge matrix and
execution-time matrix. The simulator has no Torch or policy dependency.

The high policy is a masked LSTM actor-critic over stable-topological task
features, and chooses one ready task. The low policy is a pure-PyTorch dense
GAT actor-critic. It receives task-conditioned execution/communication/DRT/
EST/EFT node features and normalized bandwidth/latency edge features, then
chooses a permanently feasible node. Busy nodes are deliberately not masked.

For the fast competition configuration, each actor is a bounded neural
residual around a HEFT score. The high prior is normalized upward rank and the
low prior is normalized negative EFT. Zero initialization of only the residual
output heads makes the initial deterministic policy exactly HEFT-equivalent
while retaining trainable LSTM/GAT representations. Time-like observations are
scaled by a precomputed HEFT makespan; no final learned-policy result enters an
observation.

PPO stores each sampled categorical mask alongside the old log probability,
then uses that same mask during updates. Same-shape transitions are batched for
LSTM/GAT evaluation. Training supports HEFT-rank low-level pretraining, a
high-only phase with the low-level placement frozen to deterministic minimum
EFT, joint PPO, optional HEFT behaviour cloning, curriculum, clipped values,
target-KL stopping, learning-rate decay, proxy/full validation-ratio checkpoint
selection and checkpoint resume.

`HEFTSafePortfolioPolicy` is the deployment guard. It simulates canonical HEFT,
an optional learned schedule and seeded top-k task/node HEFT perturbations with
the shared simulator, then replays the candidate with minimum measured
makespan. Since canonical HEFT is always candidate zero and ties prefer it, the
portfolio cannot be worse than HEFT under the same scenario semantics.

For small-scale ablation, Flat PPO scores the complete task-node Cartesian
product with one categorical policy. Its joint legality mask combines the ready
task mask with each task's permanent node mask; it is not factorized into a
high/low policy.
