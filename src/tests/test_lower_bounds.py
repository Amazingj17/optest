from __future__ import annotations

from itertools import permutations, product

import numpy as np
import pytest

from cpn_hrl_dag.evaluation.lower_bounds import fractional_load_bound, optimistic_path_bound
from cpn_hrl_dag.scenario.types import ComputeNode, Dependency, Scenario, Task
from cpn_hrl_dag.scheduling.simulator import ScheduleSimulator


def _scenario(dependencies=(), tasks=None, nodes=None) -> Scenario:
    return Scenario(
        scenario_id='bound-test', dataset_source='unit',
        tasks=tasks or [Task('a', 4.0), Task('b', 2.0), Task('c', 3.0)],
        dependencies=dependencies,
        compute_nodes=nodes or [ComputeNode('slow', 'edge', 1.0), ComputeNode('fast', 'cloud', 2.0)],
        bandwidth_matrix=np.asarray([[1e9, 1.0], [1.0, 1e9]]),
    )


@pytest.mark.parametrize('dependencies', [(), (Dependency('a', 'b', 20.0),),
                                         (Dependency('a', 'c', 5.0), Dependency('b', 'c', 7.0))])
def test_bounds_do_not_exceed_exhaustive_optimum(dependencies) -> None:
    pytest.importorskip('scipy')
    scenario = _scenario(dependencies)
    best = float('inf')
    for order in permutations(range(scenario.num_tasks)):
        positions = {task: index for index, task in enumerate(order)}
        if any(positions[parent] > positions[task] for task in order for parent in scenario.cache.predecessors[task]):
            continue
        for nodes in product(range(scenario.num_nodes), repeat=scenario.num_tasks):
            simulator = ScheduleSimulator(scenario)
            for task in order:
                simulator.schedule(task, nodes[task])
            best = min(best, simulator.result().makespan)
    template = ScheduleSimulator(scenario)
    load, weights = fractional_load_bound(template)
    assert 0 < load <= best + 1e-9
    assert 0 < optimistic_path_bound(template) <= best + 1e-9
    assert min(weights) >= 0 and sum(weights) <= 1.0
    assert not template.entries


def test_load_bound_detects_exclusive_resource_bottleneck() -> None:
    pytest.importorskip('scipy')
    scenario = _scenario(
        tasks=[Task('a', 4.0, gpu_requirement=1.0), Task('b', 6.0, gpu_requirement=1.0)],
        nodes=[ComputeNode('gpu', 'edge', 1.0, gpu_capacity=1.0), ComputeNode('cpu', 'cloud', 2.0, gpu_capacity=0.0)],
    )
    template = ScheduleSimulator(scenario)
    load, weights = fractional_load_bound(template)
    assert load == pytest.approx(10.0)
    assert optimistic_path_bound(template) == 6.0
    assert weights[0] == pytest.approx(1.0)


def test_path_bound_retains_forced_cross_node_communication() -> None:
    scenario = _scenario(
        tasks=[Task('a', 4.0, cpu_requirement=1.0), Task('b', 2.0, memory_requirement=1.0)],
        nodes=[ComputeNode('cpu', 'edge', 1.0, cpu_capacity=1.0, memory_capacity=0.0),
               ComputeNode('memory', 'cloud', 2.0, cpu_capacity=0.0, memory_capacity=1.0)],
        dependencies=[Dependency('a', 'b', 7.0)],
    )
    assert optimistic_path_bound(ScheduleSimulator(scenario)) == 12.0
