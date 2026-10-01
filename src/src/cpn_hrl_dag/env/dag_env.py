"""Gym-like environment for the hierarchical `(ready task, compute node)` action."""

from __future__ import annotations

from typing import Any

import numpy as np

from cpn_hrl_dag.baselines.heft import HEFTScheduler
from cpn_hrl_dag.env.masks import node_mask, ready_mask
from cpn_hrl_dag.scenario.types import Scenario
from cpn_hrl_dag.scheduling.communication_model import CommunicationModel, MatrixCommunicationModel
from cpn_hrl_dag.scheduling.execution_model import ExecutionTimeModel, execution_model_for_scenario
from cpn_hrl_dag.scheduling.simulator import InvalidSchedulingAction, ScheduleSimulator


class CloudEdgeEndDAGEnv:
    """Offline DAG list-scheduling environment with hierarchical observations.

    High-level selection chooses one ready task. Low-level selection assigns a
    permanently feasible node. The same masks protect both PPO sampling and
    deterministic policies; node busyness changes EST/EFT features, never mask
    membership.
    """

    def __init__(
        self,
        execution_model: ExecutionTimeModel | None = None,
        communication_model: CommunicationModel | None = None,
        *,
        heft_makespan: float | None = None,
        include_heft_features: bool = True,
        normalize_observations: bool = False,
        reward_config: dict[str, float | bool] | None = None,
    ) -> None:
        self.execution_model = execution_model
        self.communication_model = communication_model or MatrixCommunicationModel()
        self._fixed_heft_makespan = heft_makespan
        self.include_heft_features = include_heft_features
        self.normalize_observations = bool(normalize_observations)
        self.reward_config = dict(reward_config or {})
        self.scenario: Scenario | None = None
        self.simulator: ScheduleSimulator | None = None
        self.selected_task_position: int | None = None
        self.heft_makespan = 0.0
        self.upward_rank: np.ndarray | None = None
        self._previous_partial_makespan = 0.0

    def _potential(self) -> float:
        """Admissible-style, state-only remaining-work potential for optional PBRS."""
        simulator = self._simulator()
        if simulator.done:
            return 0.0
        unscheduled = [i for i in range(simulator.scenario.num_tasks) if i not in simulator.entries]
        remaining = 0.0 if not unscheduled or self.upward_rank is None else float(np.max(self.upward_rank[unscheduled]))
        return -(simulator.partial_makespan + remaining) / max(self.heft_makespan, 1e-9)

    def reset(self, scenario: Scenario, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """Start one episode and return high-level observation plus provenance info."""

        del seed  # stochasticity belongs to policies/resource realization, not deterministic simulation
        self.scenario = scenario
        execution_model = self.execution_model or execution_model_for_scenario(scenario.dataset_source)
        self.simulator = ScheduleSimulator(scenario, execution_model, self.communication_model)
        self.selected_task_position = None
        self._previous_partial_makespan = 0.0
        heft_scheduler = HEFTScheduler(
            execution_model, self.communication_model
        )
        analysis = heft_scheduler.analyze(scenario)
        self.upward_rank = analysis.upward_rank
        self.heft_makespan = self._fixed_heft_makespan or heft_scheduler.schedule(scenario)[0].makespan
        if not np.isfinite(self.heft_makespan) or self.heft_makespan <= 0.0:
            raise ValueError("HEFT reference makespan must be finite and positive")
        return self.get_high_observation(), self._info(final=False)

    def get_ready_mask(self) -> np.ndarray:
        return ready_mask(self._simulator())

    def get_node_mask(self, task_position: int) -> np.ndarray:
        self._assert_ready_position(task_position)
        return node_mask(self._simulator(), task_position)

    def select_task(self, task_position: int) -> None:
        """Validate and remember a high-level ready-task action."""

        self._assert_ready_position(task_position)
        self.selected_task_position = task_position

    def get_high_observation(self) -> dict[str, Any]:
        simulator = self._simulator()
        scenario = simulator.scenario
        workload = np.asarray([task.workload for task in scenario.tasks], dtype=np.float64)
        workload_scale = max(float(workload.mean()), 1e-9)
        scheduled = np.asarray([index in simulator.entries for index in range(scenario.num_tasks)], dtype=np.float64)
        ready = self.get_ready_mask().astype(np.float64)
        scheduled_predecessors = np.asarray(
            [sum(parent in simulator.entries for parent in predecessors) for predecessors in scenario.cache.predecessors],
            dtype=np.float64,
        )
        execution = simulator.execution_times
        incoming_data = scenario.cache.edge_data_matrix.sum(axis=0)
        outgoing_data = scenario.cache.edge_data_matrix.sum(axis=1)
        rank = self.upward_rank if self.upward_rank is not None else np.zeros(scenario.num_tasks)
        rank_scale = max(float(rank.max()), 1e-9)
        rank_feature = rank / rank_scale if self.include_heft_features else np.zeros(scenario.num_tasks)
        time_scale = max(float(self.heft_makespan), 1e-9)
        if self.normalize_observations:
            degree_scale = max(float(scenario.num_tasks - 1), 1.0)
            level_scale = max(float(np.max(scenario.cache.topological_level)), 1.0)
            predecessor_progress = scheduled_predecessors / np.maximum(scenario.cache.in_degree, 1.0)
            in_degree = scenario.cache.in_degree / degree_scale
            out_degree = scenario.cache.out_degree / degree_scale
            topological_level = scenario.cache.topological_level / level_scale
            execution_features = execution / time_scale
            rank_scale_feature = rank_scale / time_scale if self.include_heft_features else 0.0
        else:
            predecessor_progress = scheduled_predecessors
            in_degree = scenario.cache.in_degree
            out_degree = scenario.cache.out_degree
            topological_level = scenario.cache.topological_level
            execution_features = execution
            rank_scale_feature = rank_scale if self.include_heft_features else 0.0
        task_features = np.column_stack(
            (
                workload / workload_scale,
                in_degree,
                out_degree,
                predecessor_progress,
                ready,
                scheduled,
                (scenario.cache.in_degree == 0).astype(np.float64),
                (scenario.cache.out_degree == 0).astype(np.float64),
                topological_level,
                execution_features.mean(axis=1),
                execution_features.min(axis=1),
                execution_features.max(axis=1),
                incoming_data / max(float(incoming_data.mean()), 1e-9),
                outgoing_data / max(float(outgoing_data.mean()), 1e-9),
                rank_feature,
                np.full(scenario.num_tasks, rank_scale_feature),
                np.full(scenario.num_tasks, len(simulator.entries) / scenario.num_tasks),
            )
        ).astype(np.float32)
        resource_features = self._resource_features().astype(np.float32)
        if not np.isfinite(task_features).all() or not np.isfinite(resource_features).all():
            raise ValueError("high-level observation contains a non-finite feature")
        return {
            "task_features": task_features,
            "task_mask": np.ones(scenario.num_tasks, dtype=bool),
            "ready_mask": self.get_ready_mask(),
            # Preserve the float64 heuristic key for deterministic residual
            # inference.  The network still consumes the normalized float32
            # column above, while exact HEFT ties are resolved without losing
            # precision at deployment.
            "heuristic_priority": np.asarray(rank_feature, dtype=np.float64),
            "resource_features": resource_features,
            "progress": np.float32(len(simulator.entries) / scenario.num_tasks),
        }

    def get_low_observation(self, task_position: int) -> dict[str, Any]:
        """Return task-conditioned EST/EFT features for all candidate nodes."""

        self._assert_ready_position(task_position)
        simulator = self._simulator()
        mask = self.get_node_mask(task_position)
        features, heuristic_eft = self._task_node_features(task_position, mask)
        return {
            "task_position": task_position,
            "task_features": self.get_high_observation()["task_features"][task_position],
            "node_features": features,
            "heuristic_eft": heuristic_eft,
            "node_feature_names": ("speed", "execution_time", "communication_time", "dependency_ready_time", "est", "eft", "available_at", "assigned_task_count"),
            "node_mask": mask,
            "adjacency": np.isfinite(simulator.scenario.bandwidth_matrix)
            & (simulator.scenario.bandwidth_matrix > 0.0),
            "edge_features": self._edge_features(),
        }

    def _task_node_features(
        self,
        task_position: int,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build pair timing features without rebuilding the DAG observation."""

        simulator = self._simulator()
        if mask.shape != (simulator.scenario.num_nodes,):
            raise ValueError("node mask shape does not match the scenario")
        features = np.zeros((simulator.scenario.num_nodes, 8), dtype=np.float32)
        heuristic_eft = np.zeros(simulator.scenario.num_nodes, dtype=np.float64)
        for node_position in np.flatnonzero(mask):
            timing = simulator.candidate_timing(task_position, int(node_position))
            node = simulator.scenario.compute_nodes[int(node_position)]
            availability = max((entry.finish for entry in simulator.node_timelines[int(node_position)]), default=0.0)
            features[int(node_position)] = (
                node.compute_speed,
                timing.execution_time,
                timing.communication_time,
                timing.dependency_ready_time,
                timing.start,
                timing.finish,
                availability,
                len(simulator.node_timelines[int(node_position)]),
            )
            heuristic_eft[int(node_position)] = timing.finish
        if self.normalize_observations:
            speed_scale = max(float(np.mean([node.compute_speed for node in simulator.scenario.compute_nodes])), 1e-9)
            features[:, 0] /= speed_scale
            features[:, 1:7] /= max(float(self.heft_makespan), 1e-9)
            features[:, 7] /= max(float(simulator.scenario.num_tasks), 1.0)
            heuristic_eft /= max(float(self.heft_makespan), 1e-9)
        if not np.isfinite(features).all():
            raise ValueError("low-level observation contains a non-finite feature")
        return features, heuristic_eft

    def get_flat_observation(self) -> dict[str, Any]:
        """Build a graph-aware joint-action observation without future leakage."""
        high = self.get_high_observation()
        scenario = self._simulator().scenario
        node_features = np.zeros((scenario.num_tasks, scenario.num_nodes, 8), dtype=np.float32)
        pair_mask = np.zeros((scenario.num_tasks, scenario.num_nodes), dtype=bool)
        pair_heuristic_eft = np.zeros((scenario.num_tasks, scenario.num_nodes), dtype=np.float64)
        resource_adjacency = np.isfinite(scenario.bandwidth_matrix) & (scenario.bandwidth_matrix > 0.0)
        resource_edge_features = self._edge_features()
        current_ready = np.asarray(high["ready_mask"], dtype=bool)
        for task in np.flatnonzero(current_ready):
            task_position = int(task)
            feasible = np.zeros(scenario.num_nodes, dtype=bool)
            feasible[list(self._simulator().feasible_node_positions(task_position))] = True
            features, heuristic_eft = self._task_node_features(task_position, feasible)
            node_features[task_position] = features
            pair_mask[task_position] = feasible
            pair_heuristic_eft[task_position] = heuristic_eft

        heuristic_pair_logits = np.full(
            (scenario.num_tasks, scenario.num_nodes),
            -1e9,
            dtype=np.float64,
        )
        ready_tasks = sorted(
            (int(task) for task in np.flatnonzero(current_ready)),
            key=lambda task: (
                -float(high["heuristic_priority"][task]),
                str(scenario.tasks[task].id),
            ),
        )
        for task_rank, task in enumerate(ready_tasks):
            feasible_nodes = sorted(
                (int(node) for node in np.flatnonzero(pair_mask[task])),
                key=lambda node: (float(pair_heuristic_eft[task, node]), node),
            )
            for node_rank, node in enumerate(feasible_nodes):
                # Unit gaps make the prior exactly lexicographic: even the
                # worst feasible node of task-rank k beats the best node of
                # task-rank k+1. A bounded residual can still learn local
                # deviations when explicitly configured to exceed that gap.
                heuristic_pair_logits[task, node] = -float(
                    task_rank * (scenario.num_nodes + 1) + node_rank
                )

        task_adjacency = np.zeros((scenario.num_tasks, scenario.num_tasks), dtype=bool)
        for source, successors in enumerate(scenario.cache.successors):
            task_adjacency[source, list(successors)] = True
        edge_data = scenario.cache.edge_data_matrix
        positive_edge_data = edge_data[task_adjacency & (edge_data > 0.0)]
        edge_scale = max(float(np.mean(positive_edge_data)), 1e-9) if positive_edge_data.size else 1.0
        task_edge_features = (edge_data / edge_scale).astype(np.float32)[..., None]
        if not np.isfinite(task_edge_features).all() or not np.isfinite(resource_edge_features).all():
            raise ValueError("flat graph observation contains a non-finite edge feature")
        return {
            **high,
            "pair_node_features": node_features,
            "pair_heuristic_eft": pair_heuristic_eft,
            "heuristic_pair_logits": heuristic_pair_logits,
            "pair_mask": pair_mask,
            "task_adjacency": task_adjacency,
            "task_edge_features": task_edge_features,
            "resource_adjacency": resource_adjacency,
            "resource_edge_features": resource_edge_features,
        }

    def _edge_features(self) -> np.ndarray:
        scenario = self._simulator().scenario
        bandwidth = scenario.bandwidth_matrix
        finite_bandwidth = bandwidth[np.isfinite(bandwidth) & (bandwidth > 0.0)]
        scale = max(float(np.mean(finite_bandwidth)), 1e-9) if finite_bandwidth.size else 1.0
        normalized_bandwidth = np.where(np.isfinite(bandwidth), bandwidth / scale, 0.0)
        latency = np.zeros_like(bandwidth) if scenario.latency_matrix is None else scenario.latency_matrix
        latency_scale = max(float(self.heft_makespan), 1e-9) if self.normalize_observations else max(float(np.mean(latency)), 1e-9)
        features = np.stack((normalized_bandwidth, latency / latency_scale), axis=-1).astype(np.float32)
        if not np.isfinite(features).all():
            raise ValueError("resource edge observation contains a non-finite feature")
        return features

    def step(self, task_position: int, node_position: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Commit one complete hierarchical action and return a Gymnasium-like tuple."""

        if self.selected_task_position is not None and task_position != self.selected_task_position:
            raise InvalidSchedulingAction("low-level task differs from the selected high-level task")
        self._assert_ready_position(task_position)
        mask = self.get_node_mask(task_position)
        if not 0 <= node_position < len(mask) or not mask[node_position]:
            raise InvalidSchedulingAction(f"node action {node_position!r} is not legal for task position {task_position}")
        simulator = self._simulator()
        potential_before = self._potential()
        communication_before = simulator.total_communication_time
        simulator.schedule(task_position, node_position)
        current = simulator.partial_makespan
        reward = -(current - self._previous_partial_makespan) / max(self.heft_makespan, 1e-9)
        communication_delta = simulator.total_communication_time - communication_before
        reward -= float(self.reward_config.get("communication_weight", 0.0)) * communication_delta / max(self.heft_makespan, 1e-9)
        if bool(self.reward_config.get("potential_based", False)):
            reward += float(self.reward_config.get("potential_gamma", 0.99)) * self._potential() - potential_before
        self._previous_partial_makespan = current
        self.selected_task_position = None
        terminated = simulator.done
        return self.get_high_observation(), float(reward), terminated, False, self._info(final=terminated)

    def _resource_features(self) -> np.ndarray:
        simulator = self._simulator()
        values = np.zeros((simulator.scenario.num_nodes, 5), dtype=np.float64)
        for position, node in enumerate(simulator.scenario.compute_nodes):
            availability = max((entry.finish for entry in simulator.node_timelines[position]), default=0.0)
            values[position] = (node.compute_speed, availability, len(simulator.node_timelines[position]), node.cpu_capacity or 0.0, node.memory_capacity or 0.0)
        if self.normalize_observations:
            values[:, 0] /= max(float(np.mean(values[:, 0])), 1e-9)
            values[:, 1] /= max(float(self.heft_makespan), 1e-9)
            values[:, 2] /= max(float(simulator.scenario.num_tasks), 1.0)
            for column in (3, 4):
                positive = values[:, column][values[:, column] > 0.0]
                values[:, column] /= max(float(np.mean(positive)), 1e-9) if positive.size else 1.0
        return values

    def _assert_ready_position(self, task_position: int) -> None:
        simulator = self._simulator()
        if not isinstance(task_position, int) or not 0 <= task_position < simulator.scenario.num_tasks:
            raise InvalidSchedulingAction(f"unknown task position: {task_position!r}")
        if task_position not in simulator.ready_task_positions():
            raise InvalidSchedulingAction(f"task position {task_position} is not ready")

    def _simulator(self) -> ScheduleSimulator:
        if self.simulator is None:
            raise RuntimeError("environment must be reset with a Scenario before use")
        return self.simulator

    def _info(self, final: bool) -> dict[str, Any]:
        simulator = self._simulator()
        info: dict[str, Any] = {
            "scenario_id": simulator.scenario.scenario_id,
            "dataset_source": simulator.scenario.dataset_source,
            "num_tasks": simulator.scenario.num_tasks,
            "num_nodes": simulator.scenario.num_nodes,
            "partial_makespan": simulator.partial_makespan,
            "communication_cost": simulator.total_communication_time,
            "computation_cost": simulator.total_computation_time,
            "heft_makespan": self.heft_makespan,
            "valid_schedule": False,
        }
        if final:
            result = simulator.result()
            info.update({"final_makespan": result.makespan, "valid_schedule": True})
        return info
