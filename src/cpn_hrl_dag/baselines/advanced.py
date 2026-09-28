"""Deterministic look-ahead list schedulers for heterogeneous DAGs.

The implementations in this module deliberately delegate all EST/EFT and
timeline insertion decisions to :class:`ScheduleSimulator`.  They are useful
both as standalone baselines and as diverse, auditable teachers for learned
schedule search.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cpn_hrl_dag.scenario.types import Scenario
from cpn_hrl_dag.scheduling.communication_model import CommunicationModel, MatrixCommunicationModel
from cpn_hrl_dag.scheduling.execution_model import ExecutionTimeModel, SimpleSpeedExecutionModel
from cpn_hrl_dag.scheduling.simulator import ScheduleSimulator, SimulationResult

from .heft import HEFTAnalysis, HEFTScheduler


@dataclass(frozen=True, slots=True)
class PEFTAnalysis:
    """Optimistic Cost Table (OCT) quantities used by PEFT."""

    optimistic_cost_table: np.ndarray
    priority: np.ndarray
    task_order: tuple[int, ...]


class PEFTScheduler:
    """Predict Earliest Finish Time scheduling using an OCT look-ahead.

    For a task ``i`` tentatively placed on processor ``p``, ``OCT[i, p]`` is
    the optimistic cost of the longest remaining successor path.  Processor
    selection minimizes ``EFT(i, p) + lookahead_weight * OCT[i, p]``.  A
    weight of one is the PEFT rule; other non-negative values are exposed for
    deterministic teacher diversity.
    """

    name = "peft"

    def __init__(
        self,
        execution_model: ExecutionTimeModel | None = None,
        communication_model: CommunicationModel | None = None,
        *,
        lookahead_weight: float = 1.0,
    ) -> None:
        if not np.isfinite(lookahead_weight) or lookahead_weight < 0.0:
            raise ValueError("PEFT lookahead_weight must be finite and non-negative")
        self.execution_model = execution_model or SimpleSpeedExecutionModel()
        self.communication_model = communication_model or MatrixCommunicationModel()
        self.lookahead_weight = float(lookahead_weight)

    def analyze(self, scenario: Scenario) -> PEFTAnalysis:
        """Compute the OCT without consulting mutable scheduling state."""

        simulator = ScheduleSimulator(scenario, self.execution_model, self.communication_model)
        feasible = {
            task: simulator.feasible_node_positions(task)
            for task in range(scenario.num_tasks)
        }
        for task, nodes in feasible.items():
            if not nodes:
                raise ValueError(
                    f"PEFT cannot schedule task {scenario.tasks[task].id!r}: no permanently feasible node"
                )

        oct_table = np.full((scenario.num_tasks, scenario.num_nodes), np.inf, dtype=np.float64)
        for task in reversed(scenario.cache.topological_order):
            successors = scenario.cache.successors[task]
            for source_node in feasible[task]:
                if not successors:
                    oct_table[task, source_node] = 0.0
                    continue
                child_costs: list[float] = []
                for child in successors:
                    best_child = min(
                        float(oct_table[child, target_node])
                        + float(simulator.execution_times[child, target_node])
                        + self.communication_model.duration(
                            scenario,
                            task,
                            child,
                            source_node,
                            target_node,
                        )
                        for target_node in feasible[child]
                    )
                    child_costs.append(best_child)
                oct_table[task, source_node] = max(child_costs)

        priority = np.asarray(
            [float(np.mean(oct_table[task, list(feasible[task])])) for task in range(scenario.num_tasks)],
            dtype=np.float64,
        )
        task_order = tuple(
            sorted(
                range(scenario.num_tasks),
                key=lambda task: (-float(priority[task]), str(scenario.tasks[task].id)),
            )
        )
        if not np.isfinite(oct_table[np.isfinite(oct_table)]).all() or not np.isfinite(priority).all():
            raise ValueError("PEFT analysis produced a non-finite feasible cost")
        oct_table.setflags(write=False)
        priority.setflags(write=False)
        return PEFTAnalysis(oct_table, priority, task_order)

    def schedule(self, scenario: Scenario) -> tuple[SimulationResult, PEFTAnalysis]:
        """Build a complete PEFT schedule through the shared simulator."""

        analysis = self.analyze(scenario)
        simulator = ScheduleSimulator(scenario, self.execution_model, self.communication_model)
        task_ids = tuple(str(task.id) for task in scenario.tasks)
        while not simulator.done:
            task = min(
                simulator.ready_task_positions(),
                key=lambda item: (-float(analysis.priority[item]), task_ids[item]),
            )
            node = min(
                simulator.feasible_node_positions(task),
                key=lambda item: (
                    simulator.candidate_timing(task, item).finish
                    + self.lookahead_weight * float(analysis.optimistic_cost_table[task, item]),
                    simulator.candidate_timing(task, item).finish,
                    int(item),
                ),
            )
            simulator.schedule(task, node)
        return simulator.result(), analysis


@dataclass(frozen=True, slots=True)
class CPOPAnalysis:
    """Static CPOP priority and selected critical-processor information."""

    upward_rank: np.ndarray
    downward_rank: np.ndarray
    priority: np.ndarray
    critical_path: tuple[int, ...]
    critical_node: int | None


class CPOPScheduler:
    """Critical-Path-on-a-Processor list scheduling.

    A deterministic averaged-cost critical path is pinned to the feasible
    processor with minimum total path execution time.  If heterogeneous task
    constraints leave no processor feasible for the full path, those tasks
    fall back to ordinary minimum-EFT placement instead of producing an
    illegal schedule.
    """

    name = "cpop"

    def __init__(
        self,
        execution_model: ExecutionTimeModel | None = None,
        communication_model: CommunicationModel | None = None,
    ) -> None:
        self.execution_model = execution_model or SimpleSpeedExecutionModel()
        self.communication_model = communication_model or MatrixCommunicationModel()

    def analyze(self, scenario: Scenario) -> CPOPAnalysis:
        """Compute CPOP upward/downward priorities and one stable critical path."""

        heft = HEFTScheduler(self.execution_model, self.communication_model).analyze(scenario)
        downward = np.zeros(scenario.num_tasks, dtype=np.float64)
        for task in scenario.cache.topological_order:
            predecessors = scenario.cache.predecessors[task]
            if predecessors:
                downward[task] = max(
                    downward[parent]
                    + float(heft.average_execution_cost[parent])
                    + float(heft.average_communication_cost[parent, task])
                    for parent in predecessors
                )
        priority = heft.upward_rank + downward
        task_ids = tuple(str(task.id) for task in scenario.tasks)
        sources = [task for task in range(scenario.num_tasks) if not scenario.cache.predecessors[task]]
        current = min(sources, key=lambda task: (-float(heft.upward_rank[task]), task_ids[task]))
        critical_path = [current]
        while scenario.cache.successors[current]:
            current = min(
                scenario.cache.successors[current],
                key=lambda child: (
                    -(
                        float(heft.average_communication_cost[current, child])
                        + float(heft.upward_rank[child])
                    ),
                    task_ids[child],
                ),
            )
            critical_path.append(current)

        simulator = ScheduleSimulator(scenario, self.execution_model, self.communication_model)
        common_nodes = set(simulator.feasible_node_positions(critical_path[0]))
        for task in critical_path[1:]:
            common_nodes.intersection_update(simulator.feasible_node_positions(task))
        critical_node = None
        if common_nodes:
            critical_node = min(
                common_nodes,
                key=lambda node: (
                    float(np.sum(simulator.execution_times[critical_path, node])),
                    int(node),
                ),
            )
        downward.setflags(write=False)
        priority.setflags(write=False)
        return CPOPAnalysis(
            heft.upward_rank,
            downward,
            priority,
            tuple(critical_path),
            critical_node,
        )

    def schedule(self, scenario: Scenario) -> tuple[SimulationResult, CPOPAnalysis]:
        """Build a complete CPOP schedule through the shared simulator."""

        analysis = self.analyze(scenario)
        simulator = ScheduleSimulator(scenario, self.execution_model, self.communication_model)
        critical_tasks = set(analysis.critical_path)
        task_ids = tuple(str(task.id) for task in scenario.tasks)
        while not simulator.done:
            task = min(
                simulator.ready_task_positions(),
                key=lambda item: (-float(analysis.priority[item]), task_ids[item]),
            )
            if analysis.critical_node is not None and task in critical_tasks:
                node = analysis.critical_node
            else:
                node = min(
                    simulator.feasible_node_positions(task),
                    key=lambda item: (simulator.candidate_timing(task, item).finish, int(item)),
                )
            simulator.schedule(task, node)
        return simulator.result(), analysis


class ScaledHEFTScheduler:
    """HEFT variant with a configurable communication weight in task ranking."""

    name = "scaled_heft"

    def __init__(
        self,
        execution_model: ExecutionTimeModel | None = None,
        communication_model: CommunicationModel | None = None,
        *,
        communication_scale: float = 1.0,
    ) -> None:
        if not np.isfinite(communication_scale) or communication_scale < 0.0:
            raise ValueError("communication_scale must be finite and non-negative")
        self.execution_model = execution_model or SimpleSpeedExecutionModel()
        self.communication_model = communication_model or MatrixCommunicationModel()
        self.communication_scale = float(communication_scale)

    def analyze(self, scenario: Scenario) -> HEFTAnalysis:
        """Recompute upward rank after scaling average communication costs."""

        base = HEFTScheduler(self.execution_model, self.communication_model).analyze(scenario)
        communication = {
            edge: self.communication_scale * float(cost)
            for edge, cost in base.average_communication_cost.items()
        }
        upward = np.zeros(scenario.num_tasks, dtype=np.float64)
        for task in reversed(scenario.cache.topological_order):
            successors = scenario.cache.successors[task]
            downstream = max(
                communication[task, child] + upward[child]
                for child in successors
            ) if successors else 0.0
            upward[task] = float(base.average_execution_cost[task]) + downstream
        order = tuple(
            sorted(
                range(scenario.num_tasks),
                key=lambda task: (-float(upward[task]), str(scenario.tasks[task].id)),
            )
        )
        upward.setflags(write=False)
        return HEFTAnalysis(base.average_execution_cost, upward, communication, order)

    def schedule(self, scenario: Scenario) -> tuple[SimulationResult, HEFTAnalysis]:
        """Build the scaled-rank schedule using canonical minimum-EFT placement."""

        analysis = self.analyze(scenario)
        simulator = ScheduleSimulator(scenario, self.execution_model, self.communication_model)
        task_ids = tuple(str(task.id) for task in scenario.tasks)
        while not simulator.done:
            task = min(
                simulator.ready_task_positions(),
                key=lambda item: (-float(analysis.upward_rank[item]), task_ids[item]),
            )
            node = min(
                simulator.feasible_node_positions(task),
                key=lambda item: (simulator.candidate_timing(task, item).finish, int(item)),
            )
            simulator.schedule(task, node)
        return simulator.result(), analysis
