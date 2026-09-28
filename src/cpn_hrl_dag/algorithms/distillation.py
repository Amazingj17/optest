"""Safe-search teacher caching and joint graph-policy distillation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from cpn_hrl_dag.algorithms.ppo import masked_distribution
from cpn_hrl_dag.env.dag_env import CloudEdgeEndDAGEnv
from cpn_hrl_dag.models.dag_pair import DAGPairGraphActorCritic
from cpn_hrl_dag.policies.dag_pair import graph_dynamic_tensors, graph_static_tensors
from cpn_hrl_dag.policies.portfolio import HEFTSafePortfolioPolicy
from cpn_hrl_dag.scenario.types import Scenario


@dataclass(frozen=True, slots=True)
class TeacherSchedule:
    """One exact, simulator-validated teacher trajectory."""

    scenario_id: str
    scenario_hash: str
    decisions: tuple[tuple[int, int], ...]
    makespan: float
    heft_makespan: float
    selected_candidate: str

    @property
    def ratio(self) -> float:
        return self.makespan / self.heft_makespan


class SearchTeacherCache:
    """Content-addressed cache for expensive safe-search teacher schedules."""

    cache_version = "search-teacher-v1-insertion-list-schedule"

    def __init__(self, root: str | Path, search_signature: str) -> None:
        if not search_signature:
            raise ValueError("search_signature cannot be empty")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.search_signature = str(search_signature)

    def scenario_hash(self, scenario: Scenario) -> str:
        """Hash all scheduling-relevant static scenario content."""

        payload: dict[str, Any] = {
            "cache_version": self.cache_version,
            "search_signature": self.search_signature,
            "scenario_id": scenario.scenario_id,
            "tasks": [
                (
                    str(task.id),
                    task.workload,
                    task.cpu_requirement,
                    task.gpu_requirement,
                    task.memory_requirement,
                    task.device_requirement,
                )
                for task in scenario.tasks
            ],
            "dependencies": [
                (str(edge.src), str(edge.dst), edge.data_size)
                for edge in scenario.dependencies
            ],
            "nodes": [
                (
                    str(node.id),
                    node.node_type,
                    node.compute_speed,
                    node.cpu_capacity,
                    node.gpu_capacity,
                    node.memory_capacity,
                    dict(node.metadata),
                )
                for node in scenario.compute_nodes
            ],
            "bandwidth": scenario.bandwidth_matrix.tolist(),
            "latency": None if scenario.latency_matrix is None else scenario.latency_matrix.tolist(),
        }
        serialized = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def get_or_compute(
        self,
        scenario: Scenario,
        teacher_policy: HEFTSafePortfolioPolicy,
    ) -> TeacherSchedule:
        """Load a matching trajectory or execute safe search exactly once."""

        scenario_hash = self.scenario_hash(scenario)
        target = self.root / f"{scenario_hash}.json"
        if target.is_file():
            raw = json.loads(target.read_text(encoding="utf-8"))
            schedule = self._decode(raw)
            self._validate(schedule, scenario, scenario_hash)
            return schedule

        teacher_policy.reset(scenario)
        heft_candidates = [candidate for candidate in teacher_policy.candidates if candidate.name == "heft"]
        if len(heft_candidates) != 1:
            raise RuntimeError("teacher portfolio must contain exactly one canonical HEFT candidate")
        schedule = TeacherSchedule(
            scenario_id=scenario.scenario_id,
            scenario_hash=scenario_hash,
            decisions=teacher_policy.planned_decisions,
            makespan=float(teacher_policy.selected_makespan),
            heft_makespan=float(heft_candidates[0].result.makespan),
            selected_candidate=teacher_policy.selected_candidate,
        )
        self._validate(schedule, scenario, scenario_hash)
        payload = {
            "cache_version": self.cache_version,
            "search_signature": self.search_signature,
            "scenario_id": schedule.scenario_id,
            "scenario_hash": schedule.scenario_hash,
            "decisions": [list(pair) for pair in schedule.decisions],
            "makespan": schedule.makespan,
            "heft_makespan": schedule.heft_makespan,
            "ratio": schedule.ratio,
            "selected_candidate": schedule.selected_candidate,
        }
        target.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        return schedule

    @staticmethod
    def _decode(raw: dict[str, Any]) -> TeacherSchedule:
        return TeacherSchedule(
            scenario_id=str(raw["scenario_id"]),
            scenario_hash=str(raw["scenario_hash"]),
            decisions=tuple((int(task), int(node)) for task, node in raw["decisions"]),
            makespan=float(raw["makespan"]),
            heft_makespan=float(raw["heft_makespan"]),
            selected_candidate=str(raw["selected_candidate"]),
        )

    @staticmethod
    def _validate(schedule: TeacherSchedule, scenario: Scenario, scenario_hash: str) -> None:
        if schedule.scenario_id != scenario.scenario_id or schedule.scenario_hash != scenario_hash:
            raise ValueError("teacher cache identity does not match the requested scenario")
        if len(schedule.decisions) != scenario.num_tasks:
            raise ValueError("teacher schedule does not contain exactly one decision per task")
        if len({task for task, _ in schedule.decisions}) != scenario.num_tasks:
            raise ValueError("teacher schedule contains duplicate or missing task decisions")
        if not np.isfinite(schedule.makespan) or not np.isfinite(schedule.heft_makespan):
            raise ValueError("teacher schedule contains a non-finite makespan")
        if schedule.makespan <= 0.0 or schedule.heft_makespan <= 0.0:
            raise ValueError("teacher makespans must be positive")
        if schedule.makespan > schedule.heft_makespan + 1e-9:
            raise ValueError("safe-search teacher is worse than canonical HEFT")


@dataclass(frozen=True, slots=True)
class DistillationResult:
    """Aggregate supervised-learning diagnostics."""

    actor_loss: float
    value_loss: float
    action_accuracy: float
    decisions: int
    scenarios: int


class DAGPairDistiller:
    """Distil complete safe-search trajectories into a joint graph policy."""

    def __init__(
        self,
        model: DAGPairGraphActorCritic,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        *,
        normalize_observations: bool = True,
        states_per_scenario: int = 16,
        value_weight: float = 0.2,
        max_grad_norm: float = 1.0,
        disagreement_weight: float = 3.0,
    ) -> None:
        if states_per_scenario <= 0:
            raise ValueError("states_per_scenario must be positive")
        if value_weight < 0.0 or max_grad_norm <= 0.0 or disagreement_weight < 1.0:
            raise ValueError("distillation weights/gradient norm are invalid")
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.normalize_observations = bool(normalize_observations)
        self.states_per_scenario = int(states_per_scenario)
        self.value_weight = float(value_weight)
        self.max_grad_norm = float(max_grad_norm)
        self.disagreement_weight = float(disagreement_weight)

    def fit(
        self,
        scenarios: Iterable[Scenario],
        teachers: dict[str, TeacherSchedule],
        *,
        epochs: int = 1,
        seed: int = 0,
    ) -> DistillationResult:
        """Train on deterministic state samples spread across each trajectory."""

        values = tuple(scenarios)
        if not values or epochs <= 0:
            raise ValueError("distillation requires scenarios and a positive epoch count")
        missing = [scenario.scenario_id for scenario in values if scenario.scenario_id not in teachers]
        if missing:
            raise KeyError(f"missing teacher schedules for {len(missing)} scenarios")

        actor_history: list[float] = []
        value_history: list[float] = []
        correct = 0
        decisions = 0
        rng = np.random.default_rng(seed)
        self.model.train()
        for _ in range(epochs):
            order = rng.permutation(len(values))
            for scenario_index in order:
                scenario = values[int(scenario_index)]
                teacher = teachers[scenario.scenario_id]
                env = CloudEdgeEndDAGEnv(normalize_observations=self.normalize_observations)
                env.reset(scenario)
                initial = env.get_flat_observation()
                static = self.model.encode_static(*graph_static_tensors(initial, self.device))
                count = min(self.states_per_scenario, scenario.num_tasks)
                selected_depths = set(
                    int(index)
                    for index in np.linspace(0, scenario.num_tasks - 1, count, dtype=int)
                )
                actor_losses: list[torch.Tensor] = []
                value_losses: list[torch.Tensor] = []
                scenario_correct = 0
                for depth, (task, node) in enumerate(teacher.decisions):
                    observation = env.get_flat_observation()
                    pair_mask = observation["pair_mask"]
                    if not bool(pair_mask[task, node]):
                        raise ValueError(
                            f"cached teacher action {(task, node)} is illegal at depth {depth}"
                        )
                    if depth in selected_depths:
                        logits, prediction = self.model.score_dynamic(
                            static,
                            *graph_dynamic_tensors(observation, self.device),
                        )
                        flat_mask = torch.as_tensor(
                            pair_mask.reshape(1, -1),
                            device=self.device,
                            dtype=torch.bool,
                        )
                        distribution = masked_distribution(logits, flat_mask)
                        action = torch.tensor(
                            [task * scenario.num_nodes + node],
                            device=self.device,
                            dtype=torch.long,
                        )
                        heuristic_action = int(
                            np.argmax(
                                np.where(
                                    pair_mask.reshape(-1),
                                    observation["heuristic_pair_logits"].reshape(-1),
                                    -np.inf,
                                )
                            )
                        )
                        weight = self.disagreement_weight if heuristic_action != action.item() else 1.0
                        actor_losses.append(-weight * distribution.log_prob(action).mean())
                        target = torch.tensor([teacher.ratio], device=self.device, dtype=prediction.dtype)
                        value_losses.append(torch.nn.functional.smooth_l1_loss(prediction, target))
                        scenario_correct += int(torch.argmax(distribution.logits, dim=1).item() == action.item())
                    env.step(task, node)

                actor_loss = torch.stack(actor_losses).mean()
                value_loss = torch.stack(value_losses).mean()
                total_loss = actor_loss + self.value_weight * value_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                actor_history.append(float(actor_loss.detach().cpu()))
                value_history.append(float(value_loss.detach().cpu()))
                correct += scenario_correct
                decisions += len(selected_depths)

        self.model.eval()
        return DistillationResult(
            actor_loss=float(np.mean(actor_history)),
            value_loss=float(np.mean(value_history)),
            action_accuracy=correct / decisions,
            decisions=decisions,
            scenarios=len(values) * epochs,
        )
