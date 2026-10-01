"""RL-free, insertion-based simulator shared by all scheduling policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

import numpy as np

from cpn_hrl_dag.scenario.types import Scenario

from .communication_model import CommunicationModel, MatrixCommunicationModel
from .execution_model import ExecutionTimeModel, SimpleSpeedExecutionModel
from .ready_semantics import ListScheduleReadySemantics, ReadySemantics
from .timeline import EPSILON, TimelineEntry, first_feasible_slot


class InvalidSchedulingAction(ValueError):
    """An action violates task readiness, node existence, or task feasibility."""


@dataclass(frozen=True, slots=True)
class CandidateTiming:
    """Non-mutating placement calculation for a legal task/node pair."""

    task_position: int
    node_position: int
    execution_time: float
    dependency_ready_time: float
    start: float
    finish: float
    communication_time: float


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Final schedule facts, independently sufficient for evaluator validation."""

    entries: tuple[TimelineEntry, ...]
    makespan: float
    total_communication_time: float
    total_computation_time: float
    decision_order: tuple[int, ...]


class ScheduleSimulator:
    """Deterministic non-preemptive DAG list scheduler with timeline insertion.

    The simulator deliberately has no model, policy, Torch, or environment
    dependency. A task may be selected as soon as every predecessor is placed;
    its actual start time remains constrained by predecessor completion and the
    earliest idle gap on its selected node.
    """

    def __init__(
        self,
        scenario: Scenario,
        execution_model: ExecutionTimeModel | None = None,
        communication_model: CommunicationModel | None = None,
        ready_semantics: ReadySemantics | None = None,
    ) -> None:
        self.scenario = scenario
        self.execution_model = execution_model or SimpleSpeedExecutionModel()
        self.communication_model = communication_model or MatrixCommunicationModel()
        self.ready_semantics = ready_semantics or ListScheduleReadySemantics()
        self.execution_times = self.execution_model.matrix(scenario.tasks, scenario.compute_nodes)
        self._uses_incremental_ready = isinstance(self.ready_semantics, ListScheduleReadySemantics)
        self._feasible_nodes_by_task = tuple(
            tuple(node for node in range(scenario.num_nodes) if self.is_node_feasible(task, node))
            for task in range(scenario.num_tasks)
        )
        self.reset()

    def reset(self) -> None:
        """Reset dynamic schedule state without touching immutable scenario caches."""

        self.entries: dict[int, TimelineEntry] = {}
        self.node_timelines: list[list[TimelineEntry]] = [[] for _ in range(self.scenario.num_nodes)]
        self.decision_order: list[int] = []
        self.total_communication_time = 0.0
        self.total_computation_time = 0.0
        self.decision_time = 0.0
        self._partial_makespan = 0.0
        self._remaining_predecessors = np.asarray(self.scenario.cache.in_degree, dtype=np.int64).copy()
        self._incremental_ready = {
            task for task, count in enumerate(self._remaining_predecessors) if count == 0
        }

    @property
    def done(self) -> bool:
        return len(self.entries) == self.scenario.num_tasks

    @property
    def partial_makespan(self) -> float:
        return self._partial_makespan

    def ready_task_positions(self) -> tuple[int, ...]:
        positions = (
            tuple(sorted(self._incremental_ready))
            if self._uses_incremental_ready
            else self.ready_semantics.ready_positions(self)
        )
        if not self.done and not positions:
            raise RuntimeError(f"no ready task under {self.ready_semantics.name} before schedule completion")
        return positions

    def clone(self) -> "ScheduleSimulator":
        """Copy dynamic state while sharing immutable scenario/model caches.

        Search policies use this operation to branch partial schedules without
        recomputing execution matrices or mutating a sibling branch.
        """

        duplicate = object.__new__(type(self))
        duplicate.scenario = self.scenario
        duplicate.execution_model = self.execution_model
        duplicate.communication_model = self.communication_model
        duplicate.ready_semantics = self.ready_semantics
        duplicate.execution_times = self.execution_times
        duplicate._uses_incremental_ready = self._uses_incremental_ready
        duplicate._feasible_nodes_by_task = self._feasible_nodes_by_task
        duplicate.entries = dict(self.entries)
        duplicate.node_timelines = [list(timeline) for timeline in self.node_timelines]
        duplicate.decision_order = list(self.decision_order)
        duplicate.total_communication_time = self.total_communication_time
        duplicate.total_computation_time = self.total_computation_time
        duplicate.decision_time = self.decision_time
        duplicate._partial_makespan = self._partial_makespan
        duplicate._remaining_predecessors = self._remaining_predecessors.copy()
        duplicate._incremental_ready = set(self._incremental_ready)
        return duplicate

    def advance_to_next_completion(self) -> float:
        """Advance online decision time; useful only with runtime-completion semantics."""

        future = [entry.finish for entry in self.entries.values() if entry.finish > self.decision_time + EPSILON]
        if not future:
            raise RuntimeError("no future task completion is available")
        self.decision_time = min(future)
        return self.decision_time

    def task_position(self, task_id_or_position: Hashable | int) -> int:
        if task_id_or_position in self.scenario.cache.task_index:
            return self.scenario.task_position(task_id_or_position)
        if isinstance(task_id_or_position, int) and 0 <= task_id_or_position < self.scenario.num_tasks:
            return task_id_or_position
        raise InvalidSchedulingAction(f"unknown task action: {task_id_or_position!r}")

    def node_position(self, node_id_or_position: Hashable | int) -> int:
        if node_id_or_position in self.scenario.cache.node_index:
            return self.scenario.node_position(node_id_or_position)
        if isinstance(node_id_or_position, int) and 0 <= node_id_or_position < self.scenario.num_nodes:
            return node_id_or_position
        raise InvalidSchedulingAction(f"unknown compute-node action: {node_id_or_position!r}")

    def is_node_feasible(self, task_position: int, node_position: int) -> bool:
        """Check permanent capacity/device feasibility; busy state is intentionally ignored."""

        task, node = self.scenario.tasks[task_position], self.scenario.compute_nodes[node_position]
        checks = (
            (task.cpu_requirement, node.cpu_capacity),
            (task.gpu_requirement, node.gpu_capacity),
            (task.memory_requirement, node.memory_capacity),
        )
        for requirement, capacity in checks:
            if requirement is not None and capacity is not None and requirement > capacity + EPSILON:
                return False
        if task.device_requirement is not None:
            features = {str(value).lower() for value in node.metadata.get("features", [])}
            device = task.device_requirement.lower()
            if device == "gpu" and (node.gpu_capacity is None or node.gpu_capacity <= 0.0) and "gpu" not in features:
                return False
            if device not in {"cpu", "gpu"} and device not in features:
                return False
        return True

    def feasible_node_positions(self, task_position: int) -> tuple[int, ...]:
        return self._feasible_nodes_by_task[task_position]

    def dependency_ready_time(self, task_position: int, node_position: int) -> tuple[float, float]:
        """Return max predecessor arrival time and sum of communication-time terms."""

        ready_time = 0.0
        communication_sum = 0.0
        for parent in self.scenario.cache.predecessors[task_position]:
            parent_entry = self.entries.get(parent)
            if parent_entry is None:
                raise InvalidSchedulingAction(f"task position {task_position} is not ready")
            transfer = self.communication_model.duration(
                self.scenario, parent, task_position, parent_entry.node_position, node_position
            )
            ready_time = max(ready_time, parent_entry.finish + transfer)
            communication_sum += transfer
        return ready_time, communication_sum

    def candidate_timing(self, task_id_or_position: Hashable | int, node_id_or_position: Hashable | int) -> CandidateTiming:
        task_position = self.task_position(task_id_or_position)
        node_position = self.node_position(node_id_or_position)
        if task_position in self.entries:
            raise InvalidSchedulingAction(f"task {self.scenario.tasks[task_position].id!r} is already scheduled")
        task_is_ready = (
            task_position in self._incremental_ready
            if self._uses_incremental_ready
            else task_position in self.ready_task_positions()
        )
        if not task_is_ready:
            raise InvalidSchedulingAction(f"task {self.scenario.tasks[task_position].id!r} is not ready")
        if not self.is_node_feasible(task_position, node_position):
            raise InvalidSchedulingAction(
                f"node {self.scenario.compute_nodes[node_position].id!r} is permanently infeasible for task {self.scenario.tasks[task_position].id!r}"
            )
        execution_time = float(self.execution_times[task_position, node_position])
        dependency_ready_time, communication_time = self.dependency_ready_time(task_position, node_position)
        start, finish = first_feasible_slot(self.node_timelines[node_position], dependency_ready_time, execution_time)
        return CandidateTiming(task_position, node_position, execution_time, dependency_ready_time, start, finish, communication_time)

    def schedule(self, task_id_or_position: Hashable | int, node_id_or_position: Hashable | int) -> TimelineEntry:
        """Commit exactly one legal placement using its earliest feasible insertion slot."""

        timing = self.candidate_timing(task_id_or_position, node_id_or_position)
        entry = TimelineEntry(timing.task_position, timing.node_position, timing.start, timing.finish)
        self.entries[timing.task_position] = entry
        timeline = self.node_timelines[timing.node_position]
        timeline.append(entry)
        timeline.sort(key=lambda item: (item.start, item.finish, item.task_position))
        self.decision_order.append(timing.task_position)
        self.total_communication_time += timing.communication_time
        self.total_computation_time += timing.execution_time
        self._partial_makespan = max(self._partial_makespan, timing.finish)
        if self._uses_incremental_ready:
            self._incremental_ready.remove(timing.task_position)
            for child in self.scenario.cache.successors[timing.task_position]:
                self._remaining_predecessors[child] -= 1
                if self._remaining_predecessors[child] < 0:
                    raise RuntimeError("incremental predecessor count became negative")
                if self._remaining_predecessors[child] == 0:
                    self._incremental_ready.add(child)
        return entry

    def result(self) -> SimulationResult:
        if not self.done:
            raise RuntimeError("cannot create a final result before every task is scheduled")
        entries = tuple(self.entries[position] for position in self.decision_order)
        result = SimulationResult(
            entries=entries,
            makespan=self.partial_makespan,
            total_communication_time=self.total_communication_time,
            total_computation_time=self.total_computation_time,
            decision_order=tuple(self.decision_order),
        )
        self.validate_result(result)
        return result

    def validate_result(self, result: SimulationResult) -> None:
        """Validate no overlap, exact execution duration, and all DAG precedence edges."""

        if len(result.entries) != self.scenario.num_tasks:
            raise ValueError("schedule result does not contain every task")
        by_task = {entry.task_position: entry for entry in result.entries}
        if len(by_task) != self.scenario.num_tasks:
            raise ValueError("schedule result contains duplicate tasks")
        for entry in result.entries:
            expected = float(self.execution_times[entry.task_position, entry.node_position])
            if abs((entry.finish - entry.start) - expected) > EPSILON:
                raise ValueError("schedule result has an incorrect execution duration")
        for dependency in self.scenario.dependencies:
            source = self.scenario.task_position(dependency.src)
            target = self.scenario.task_position(dependency.dst)
            parent, child = by_task[source], by_task[target]
            transfer = self.communication_model.duration(self.scenario, source, target, parent.node_position, child.node_position)
            if child.start + EPSILON < parent.finish + transfer:
                raise ValueError(f"dependency {source}->{target} is violated")
        for timeline in self.node_timelines:
            for previous, current in zip(timeline, timeline[1:]):
                if current.start + EPSILON < previous.finish:
                    raise ValueError("node timeline contains overlapping intervals")
        if abs(result.makespan - max(entry.finish for entry in result.entries)) > EPSILON:
            raise ValueError("schedule result makespan is inconsistent")
