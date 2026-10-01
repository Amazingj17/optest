from __future__ import annotations

import numpy as np
import pytest

from cpn_hrl_dag.scenario.types import ComputeNode, Dependency, Scenario, Task
from cpn_hrl_dag.scheduling.simulator import InvalidSchedulingAction, ScheduleSimulator


def test_dependency_communication_and_same_node_zero() -> None:
    scenario = Scenario(
        scenario_id="communication", dataset_source="unit",
        tasks=[Task("a", 10.0), Task("b", 10.0)], dependencies=[Dependency("a", "b", 20.0)],
        compute_nodes=[ComputeNode("slow", "end", 1.0), ComputeNode("fast", "cloud", 2.0)],
        bandwidth_matrix=np.array([[1e9, 10.0], [10.0, 1e9]]), latency_matrix=np.array([[0.0, 1.0], [1.0, 0.0]]),
    )
    simulator = ScheduleSimulator(scenario)
    first = simulator.schedule("a", "slow")
    assert (first.start, first.finish) == (0.0, 10.0)
    cross = simulator.candidate_timing("b", "fast")
    assert cross.dependency_ready_time == 13.0  # 10 + 20/10 + 1 latency
    same = simulator.candidate_timing("b", "slow")
    assert same.dependency_ready_time == 10.0
    simulator.schedule("b", "fast")
    assert simulator.result().makespan == 18.0


def test_insertion_scheduling_fills_idle_gap() -> None:
    scenario = Scenario(
        scenario_id="gap", dataset_source="unit",
        tasks=[Task("parent", 10.0), Task("late", 10.0), Task("root", 2.0)],
        dependencies=[Dependency("parent", "late", 0.0)],
        compute_nodes=[ComputeNode("node-0", "edge", 1.0), ComputeNode("node-1", "cloud", 1.0)],
        bandwidth_matrix=np.full((2, 2), 1e9),
    )
    simulator = ScheduleSimulator(scenario)
    simulator.schedule("parent", "node-1")
    simulator.schedule("late", "node-0")  # [10, 20] on node-0
    inserted = simulator.schedule("root", "node-0")
    assert (inserted.start, inserted.finish) == (0.0, 2.0)
    assert [(entry.start, entry.finish) for entry in simulator.node_timelines[0]] == [(0.0, 2.0), (10.0, 20.0)]
    assert simulator.result().makespan == 20.0


def test_simulator_rejects_not_ready_and_infeasible_actions() -> None:
    scenario = Scenario(
        scenario_id="mask", dataset_source="unit",
        tasks=[Task("gpu", 1.0, device_requirement="gpu"), Task("after", 1.0)],
        dependencies=[Dependency("gpu", "after", 1.0)],
        compute_nodes=[ComputeNode("cpu", "edge", 1.0, gpu_capacity=0.0)], bandwidth_matrix=np.ones((1, 1)),
    )
    simulator = ScheduleSimulator(scenario)
    with pytest.raises(InvalidSchedulingAction, match="infeasible"):
        simulator.schedule("gpu", "cpu")
    with pytest.raises(InvalidSchedulingAction, match="not ready"):
        simulator.candidate_timing("after", "cpu")


def test_incremental_ready_cache_and_clone_branches_are_independent() -> None:
    scenario = Scenario(
        scenario_id="clone",
        dataset_source="unit",
        tasks=[Task("root", 1.0), Task("left", 2.0), Task("right", 3.0)],
        dependencies=[Dependency("root", "left", 0.0), Dependency("root", "right", 0.0)],
        compute_nodes=[ComputeNode("n0", "edge", 1.0), ComputeNode("n1", "cloud", 2.0)],
        bandwidth_matrix=np.full((2, 2), 1e9),
    )
    simulator = ScheduleSimulator(scenario)
    assert simulator.ready_task_positions() == (0,)
    simulator.schedule(0, 0)
    assert simulator.ready_task_positions() == (1, 2)

    branch = simulator.clone()
    branch.schedule(1, 1)
    assert 1 in branch.entries and 1 not in simulator.entries
    assert branch.ready_task_positions() == (2,)
    assert simulator.ready_task_positions() == (1, 2)

    simulator.schedule(2, 0)
    assert 2 in simulator.entries and 2 not in branch.entries
    assert simulator.partial_makespan != branch.partial_makespan
