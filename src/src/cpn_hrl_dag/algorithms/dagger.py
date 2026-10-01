"""DAgger-style completion-regret training for the joint graph scheduler.

The original trajectory distiller only labels states visited by a fixed search
winner.  This module instead rolls out the current model, queries a deterministic
completion oracle at model-visited states, and learns a listwise distribution
over several legal ``(task, node)`` alternatives.  Oracle costs are used only
during training; deployment observations remain free of future information.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch.nn import functional as F

from cpn_hrl_dag.algorithms.ppo import masked_distribution
from cpn_hrl_dag.env.dag_env import CloudEdgeEndDAGEnv
from cpn_hrl_dag.models.dag_pair import DAGPairGraphActorCritic
from cpn_hrl_dag.policies.dag_pair import graph_dynamic_tensors, graph_static_tensors
from cpn_hrl_dag.scenario.types import Scenario
from cpn_hrl_dag.scheduling.simulator import ScheduleSimulator, SimulationResult


@dataclass(frozen=True, slots=True)
class CompletionRegretTargets:
    """Completion costs and a temperature-scaled target over legal actions."""

    actions: tuple[int, ...]
    makespans: tuple[float, ...]
    normalized_regrets: tuple[float, ...]
    probabilities: tuple[float, ...]
    best_action: int


@dataclass(frozen=True, slots=True)
class RegretDAggerResult:
    """Aggregate diagnostics for one or more online DAgger passes."""

    actor_loss: float
    value_loss: float
    expected_regret_loss: float
    residual_loss: float
    action_accuracy: float
    mean_student_regret: float
    oracle_improvement_rate: float
    target_entropy: float
    rollout_teacher_rate: float
    decisions: int
    completion_evaluations: int
    scenarios: int


def rank_eft_completion(
    simulator: ScheduleSimulator,
    upward_rank: np.ndarray,
) -> SimulationResult:
    """Complete a partial schedule with deterministic HEFT-rank/EFT choices."""

    if upward_rank.shape != (simulator.scenario.num_tasks,):
        raise ValueError("upward-rank shape does not match the scenario")
    branch = simulator.clone()
    return _rank_eft_complete_in_place(branch, upward_rank)


def _rank_eft_complete_in_place(
    branch: ScheduleSimulator,
    upward_rank: np.ndarray,
) -> SimulationResult:
    """Internal completion helper for an already isolated simulator branch."""

    while not branch.done:
        task = min(
            branch.ready_task_positions(),
            key=lambda position: (
                -float(upward_rank[position]),
                str(branch.scenario.tasks[position].id),
            ),
        )
        feasible = branch.feasible_node_positions(task)
        if not feasible:
            raise ValueError(f"task position {task} has no feasible completion node")
        node = min(
            feasible,
            key=lambda position: (
                branch.candidate_timing(task, position).finish,
                int(position),
            ),
        )
        branch.schedule(task, node)
    return branch.result()


def completion_regret_targets(
    simulator: ScheduleSimulator,
    upward_rank: np.ndarray,
    actions: Iterable[int],
    *,
    heft_makespan: float,
    temperature: float,
) -> CompletionRegretTargets:
    """Evaluate legal first actions by deterministic full-schedule completion.

    ``actions`` are flattened as ``task_position * num_nodes + node_position``.
    The normalized regret is measured against the best evaluated completion and
    divided by the scenario's canonical HEFT makespan, matching the competition
    metric's scale.
    """

    if not np.isfinite(heft_makespan) or heft_makespan <= 0.0:
        raise ValueError("heft_makespan must be finite and positive")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    unique_actions = tuple(dict.fromkeys(int(action) for action in actions))
    if not unique_actions:
        raise ValueError("completion-regret supervision requires at least one action")

    nodes = simulator.scenario.num_nodes
    ready = set(simulator.ready_task_positions())
    makespans: list[float] = []
    for action in unique_actions:
        task, node = divmod(action, nodes)
        if task not in ready or node not in simulator.feasible_node_positions(task):
            raise ValueError(f"completion oracle received illegal action {(task, node)}")
        branch = simulator.clone()
        branch.schedule(task, node)
        result = _rank_eft_complete_in_place(branch, upward_rank)
        makespans.append(float(result.makespan))

    costs = np.asarray(makespans, dtype=np.float64)
    if not np.isfinite(costs).all() or np.any(costs <= 0.0):
        raise ValueError("completion oracle produced invalid makespans")
    best_index = min(range(len(unique_actions)), key=lambda index: (costs[index], unique_actions[index]))
    regrets = np.maximum(costs - costs[best_index], 0.0) / float(heft_makespan)
    scaled = -regrets / float(temperature)
    scaled -= float(np.max(scaled))
    probabilities = np.exp(scaled)
    probabilities /= float(probabilities.sum())
    return CompletionRegretTargets(
        actions=unique_actions,
        makespans=tuple(float(value) for value in costs),
        normalized_regrets=tuple(float(value) for value in regrets),
        probabilities=tuple(float(value) for value in probabilities),
        best_action=unique_actions[best_index],
    )


class CompletionRegretDAgger:
    """Train on model-visited states using listwise completion-regret targets."""

    def __init__(
        self,
        model: DAGPairGraphActorCritic,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        *,
        normalize_observations: bool = True,
        states_per_scenario: int = 8,
        max_candidate_actions: int = 6,
        heuristic_task_candidates: int = 2,
        heuristic_node_candidates: int = 2,
        student_candidates: int = 2,
        temperature: float = 0.025,
        value_weight: float = 0.1,
        expected_regret_weight: float = 1.0,
        residual_weight: float = 0.01,
        max_grad_norm: float = 1.0,
    ) -> None:
        integer_values = (
            states_per_scenario,
            max_candidate_actions,
            heuristic_task_candidates,
            heuristic_node_candidates,
            student_candidates,
        )
        if any(int(value) <= 0 for value in integer_values):
            raise ValueError("DAgger sample and candidate counts must be positive")
        if max_candidate_actions < 2:
            raise ValueError("max_candidate_actions must be at least two")
        numeric_values = (
            temperature,
            value_weight,
            expected_regret_weight,
            residual_weight,
            max_grad_norm,
        )
        if any(not np.isfinite(value) for value in numeric_values):
            raise ValueError("DAgger weights must be finite")
        if temperature <= 0.0 or max_grad_norm <= 0.0:
            raise ValueError("temperature and max_grad_norm must be positive")
        if min(value_weight, expected_regret_weight, residual_weight) < 0.0:
            raise ValueError("DAgger loss weights must be non-negative")
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.normalize_observations = bool(normalize_observations)
        self.states_per_scenario = int(states_per_scenario)
        self.max_candidate_actions = int(max_candidate_actions)
        self.heuristic_task_candidates = int(heuristic_task_candidates)
        self.heuristic_node_candidates = int(heuristic_node_candidates)
        self.student_candidates = int(student_candidates)
        self.temperature = float(temperature)
        self.value_weight = float(value_weight)
        self.expected_regret_weight = float(expected_regret_weight)
        self.residual_weight = float(residual_weight)
        self.max_grad_norm = float(max_grad_norm)

    @staticmethod
    def _ranked_actions(actions: np.ndarray, scores: np.ndarray) -> list[int]:
        return sorted((int(action) for action in actions), key=lambda action: (-float(scores[action]), action))

    def _candidate_actions(
        self,
        observation: dict[str, object],
        student_logits: torch.Tensor,
        rng: np.random.Generator,
    ) -> tuple[int, ...]:
        pair_mask = np.asarray(observation["pair_mask"], dtype=bool)
        heuristic = np.asarray(observation["heuristic_pair_logits"], dtype=np.float64)
        nodes = pair_mask.shape[1]
        legal = np.flatnonzero(pair_mask.reshape(-1))
        if legal.size == 0:
            raise RuntimeError("DAgger encountered a state without a legal joint action")
        student = student_logits.detach().reshape(-1).cpu().numpy()
        chosen: list[int] = []

        ready_tasks = np.flatnonzero(pair_mask.any(axis=1))
        ranked_tasks = sorted(
            (int(task) for task in ready_tasks),
            key=lambda task: (-float(np.max(heuristic[task, pair_mask[task]])), task),
        )
        for task in ranked_tasks[: self.heuristic_task_candidates]:
            feasible = np.flatnonzero(pair_mask[task])
            node = min(
                (int(value) for value in feasible),
                key=lambda value: (-float(heuristic[task, value]), value),
            )
            chosen.append(task * nodes + node)

        if ranked_tasks:
            task = ranked_tasks[0]
            feasible = np.flatnonzero(pair_mask[task])
            node_actions = np.asarray([task * nodes + int(node) for node in feasible], dtype=np.int64)
            chosen.extend(
                self._ranked_actions(node_actions, heuristic.reshape(-1))[
                    : self.heuristic_node_candidates
                ]
            )
        chosen.extend(self._ranked_actions(legal, student)[: self.student_candidates])
        chosen = list(dict.fromkeys(chosen))

        remaining = np.asarray([int(action) for action in legal if int(action) not in set(chosen)])
        if len(chosen) < self.max_candidate_actions and remaining.size:
            take = min(self.max_candidate_actions - len(chosen), int(remaining.size))
            sampled = rng.choice(remaining, size=take, replace=False)
            chosen.extend(int(action) for action in np.atleast_1d(sampled))
        return tuple(chosen[: self.max_candidate_actions])

    def fit(
        self,
        scenarios: Iterable[Scenario],
        *,
        epochs: int = 1,
        seed: int = 0,
        teacher_beta: float = 0.5,
    ) -> RegretDAggerResult:
        """Run online DAgger passes and update once per scenario trajectory."""

        values = tuple(scenarios)
        if not values or epochs <= 0:
            raise ValueError("DAgger requires scenarios and a positive epoch count")
        if not 0.0 <= teacher_beta <= 1.0:
            raise ValueError("teacher_beta must lie in [0, 1]")

        actor_history: list[float] = []
        value_history: list[float] = []
        regret_loss_history: list[float] = []
        residual_history: list[float] = []
        student_regrets: list[float] = []
        target_entropies: list[float] = []
        correct = 0
        oracle_improvements = 0
        rollout_teacher_actions = 0
        rollout_actions = 0
        decisions = 0
        completion_evaluations = 0
        rng = np.random.default_rng(seed)
        self.model.train()

        for _ in range(epochs):
            for scenario_index in rng.permutation(len(values)):
                scenario = values[int(scenario_index)]
                env = CloudEdgeEndDAGEnv(normalize_observations=self.normalize_observations)
                env.reset(scenario)
                initial = env.get_flat_observation()
                static = self.model.encode_static(*graph_static_tensors(initial, self.device))
                selected_depths = set(
                    int(index)
                    for index in np.linspace(
                        0,
                        scenario.num_tasks - 1,
                        min(self.states_per_scenario, scenario.num_tasks),
                        dtype=int,
                    )
                )
                actor_losses: list[torch.Tensor] = []
                value_losses: list[torch.Tensor] = []
                expected_regret_losses: list[torch.Tensor] = []
                residual_losses: list[torch.Tensor] = []

                for depth in range(scenario.num_tasks):
                    observation = env.get_flat_observation()
                    pair_mask = np.asarray(observation["pair_mask"], dtype=bool)
                    flat_mask = torch.as_tensor(
                        pair_mask.reshape(1, -1),
                        device=self.device,
                        dtype=torch.bool,
                    )
                    if depth in selected_depths:
                        logits, prediction = self.model.score_dynamic(
                            static,
                            *graph_dynamic_tensors(observation, self.device),
                        )
                    else:
                        with torch.no_grad():
                            logits, prediction = self.model.score_dynamic(
                                static,
                                *graph_dynamic_tensors(observation, self.device),
                            )
                    distribution = masked_distribution(logits, flat_mask)
                    student_action = int(torch.argmax(distribution.logits, dim=1).item())
                    heuristic_values = np.where(
                        pair_mask.reshape(-1),
                        np.asarray(observation["heuristic_pair_logits"]).reshape(-1),
                        -np.inf,
                    )
                    heuristic_action = int(np.argmax(heuristic_values))
                    rollout_action = student_action

                    if depth in selected_depths:
                        if env.simulator is None or env.upward_rank is None:
                            raise RuntimeError("DAgger environment is missing simulator analysis")
                        candidates = self._candidate_actions(observation, logits, rng)
                        targets = completion_regret_targets(
                            env.simulator,
                            env.upward_rank,
                            candidates,
                            heft_makespan=env.heft_makespan,
                            temperature=self.temperature,
                        )
                        candidate_tensor = torch.as_tensor(
                            targets.actions,
                            device=self.device,
                            dtype=torch.long,
                        )
                        candidate_logits = logits[0].index_select(0, candidate_tensor)
                        probabilities = torch.as_tensor(
                            targets.probabilities,
                            device=self.device,
                            dtype=candidate_logits.dtype,
                        )
                        regrets = torch.as_tensor(
                            targets.normalized_regrets,
                            device=self.device,
                            dtype=candidate_logits.dtype,
                        )
                        log_probabilities = F.log_softmax(candidate_logits, dim=0)
                        policy_probabilities = torch.softmax(candidate_logits, dim=0)
                        actor_losses.append(-(probabilities * log_probabilities).sum())
                        expected_regret_losses.append((policy_probabilities * regrets).sum())
                        base_logits = self.model.heuristic_scale * torch.as_tensor(
                            np.asarray(observation["heuristic_pair_logits"]).reshape(-1),
                            device=self.device,
                            dtype=candidate_logits.dtype,
                        ).index_select(0, candidate_tensor)
                        residual_losses.append(torch.mean((candidate_logits - base_logits) ** 2))
                        best_completion = min(targets.makespans) / env.heft_makespan
                        target_value = torch.tensor(
                            [best_completion],
                            device=self.device,
                            dtype=prediction.dtype,
                        )
                        value_losses.append(F.smooth_l1_loss(prediction, target_value))

                        predicted_candidate = targets.actions[int(torch.argmax(candidate_logits).item())]
                        correct += int(predicted_candidate == targets.best_action)
                        oracle_improvements += int(targets.best_action != heuristic_action)
                        action_to_regret = dict(zip(targets.actions, targets.normalized_regrets))
                        student_regrets.append(float(action_to_regret[student_action]))
                        probability_array = np.asarray(targets.probabilities, dtype=np.float64)
                        target_entropies.append(
                            float(-np.sum(probability_array * np.log(np.maximum(probability_array, 1e-12))))
                        )
                        decisions += 1
                        completion_evaluations += len(targets.actions)
                        if rng.random() < teacher_beta:
                            rollout_action = targets.best_action
                            rollout_teacher_actions += 1
                    elif rng.random() < teacher_beta:
                        rollout_action = heuristic_action
                        rollout_teacher_actions += 1

                    task, node = divmod(rollout_action, scenario.num_nodes)
                    if not bool(pair_mask[task, node]):
                        raise RuntimeError("DAgger selected an illegal rollout action")
                    env.step(task, node)
                    rollout_actions += 1

                if not actor_losses:
                    raise RuntimeError("DAgger trajectory did not collect any supervised state")
                actor_loss = torch.stack(actor_losses).mean()
                value_loss = torch.stack(value_losses).mean()
                expected_regret_loss = torch.stack(expected_regret_losses).mean()
                residual_loss = torch.stack(residual_losses).mean()
                total_loss = (
                    actor_loss
                    + self.value_weight * value_loss
                    + self.expected_regret_weight * expected_regret_loss
                    + self.residual_weight * residual_loss
                )
                self.optimizer.zero_grad()
                total_loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                if not torch.isfinite(torch.as_tensor(gradient_norm)):
                    raise ValueError("DAgger produced non-finite gradients")
                self.optimizer.step()
                actor_history.append(float(actor_loss.detach().cpu()))
                value_history.append(float(value_loss.detach().cpu()))
                regret_loss_history.append(float(expected_regret_loss.detach().cpu()))
                residual_history.append(float(residual_loss.detach().cpu()))

        self.model.eval()
        return RegretDAggerResult(
            actor_loss=float(np.mean(actor_history)),
            value_loss=float(np.mean(value_history)),
            expected_regret_loss=float(np.mean(regret_loss_history)),
            residual_loss=float(np.mean(residual_history)),
            action_accuracy=correct / decisions,
            mean_student_regret=float(np.mean(student_regrets)),
            oracle_improvement_rate=oracle_improvements / decisions,
            target_entropy=float(np.mean(target_entropies)),
            rollout_teacher_rate=rollout_teacher_actions / rollout_actions,
            decisions=decisions,
            completion_evaluations=completion_evaluations,
            scenarios=len(values) * epochs,
        )
