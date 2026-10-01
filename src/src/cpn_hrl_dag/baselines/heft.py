"""Project-owned HEFT implementation using the shared insertion simulator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cpn_hrl_dag.scenario.types import Scenario
from cpn_hrl_dag.scheduling.communication_model import CommunicationModel, MatrixCommunicationModel
from cpn_hrl_dag.scheduling.execution_model import ExecutionTimeModel, SimpleSpeedExecutionModel
from cpn_hrl_dag.scheduling.simulator import ScheduleSimulator, SimulationResult


@dataclass(frozen=True, slots=True)
class HEFTAnalysis:
    """Static HEFT quantities indexed by the deterministic Scenario task order."""

    average_execution_cost: np.ndarray
    upward_rank: np.ndarray
    average_communication_cost: dict[tuple[int, int], float]
    task_order: tuple[int, ...]


class HEFTScheduler:
    """Canonical upward-rank / earliest-finish-time DAG list scheduling.

    Task priority is descending upward rank with a stable task-ID tie-break.
    Processor selection delegates *every* candidate EST/EFT calculation to
    :class:`ScheduleSimulator`, so HEFT and learned policies share exactly the
    same insertion, dependency, execution, and communication semantics.
    """

    name = "heft"

    def __init__(
        self,
        execution_model: ExecutionTimeModel | None = None,
        communication_model: CommunicationModel | None = None,
        *,
        include_same_node_in_average_communication: bool = False,
    ) -> None:
        self.execution_model = execution_model or SimpleSpeedExecutionModel()
        self.communication_model = communication_model or MatrixCommunicationModel()
        self.include_same_node_in_average_communication = include_same_node_in_average_communication

    def analyze(self, scenario: Scenario) -> HEFTAnalysis:
        """Compute HEFT average costs and upward ranks without scheduling state."""

        simulator = ScheduleSimulator(scenario, self.execution_model, self.communication_model)
        average_execution = np.empty(scenario.num_tasks, dtype=np.float64)
        feasible_nodes: dict[int, tuple[int, ...]] = {}
        for task_position in range(scenario.num_tasks):
            candidates = simulator.feasible_node_positions(task_position)
            if not candidates:
                task_id = scenario.tasks[task_position].id
                raise ValueError(f"HEFT cannot schedule task {task_id!r}: no permanently feasible node")
            feasible_nodes[task_position] = candidates
            average_execution[task_position] = float(np.mean(simulator.execution_times[task_position, list(candidates)]))

        average_communication: dict[tuple[int, int], float] = {}
        for dependency in scenario.dependencies:
            source, target = scenario.task_position(dependency.src), scenario.task_position(dependency.dst)
            costs = [
                self.communication_model.duration(scenario, source, target, source_node, target_node)
                for source_node in feasible_nodes[source]
                for target_node in feasible_nodes[target]
                if self.include_same_node_in_average_communication or source_node != target_node
            ]
            average_communication[source, target] = float(np.mean(costs)) if costs else 0.0

        upward = np.zeros(scenario.num_tasks, dtype=np.float64)
        for task_position in reversed(scenario.cache.topological_order):
            successors = scenario.cache.successors[task_position]
            downstream = max(
                average_communication[task_position, child] + upward[child] for child in successors
            ) if successors else 0.0
            upward[task_position] = average_execution[task_position] + downstream
        order = tuple(
            sorted(
                range(scenario.num_tasks),
                key=lambda position: (-float(upward[position]), str(scenario.tasks[position].id)),
            )
        )
        for values in (average_execution, upward):
            values.setflags(write=False)
        return HEFTAnalysis(average_execution, upward, average_communication, order)

    def schedule(self, scenario: Scenario) -> tuple[SimulationResult, HEFTAnalysis]:
        """Schedule a Scenario and return both the schedule result and HEFT analysis."""

        analysis = self.analyze(scenario)
        simulator = ScheduleSimulator(scenario, self.execution_model, self.communication_model)
        for task_position in analysis.task_order:
            if task_position not in simulator.ready_task_positions():
                task_id = scenario.tasks[task_position].id
                raise RuntimeError(f"HEFT rank order is not precedence feasible at task {task_id!r}")
            node_position = min(
                simulator.feasible_node_positions(task_position),
                key=lambda node: (simulator.candidate_timing(task_position, node).finish, node),
            )
            simulator.schedule(task_position, node_position)
        return simulator.result(), analysis
