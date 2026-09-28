"""Optimistic scheduling bounds for feasibility audits, not scheduling policies."""

from __future__ import annotations

import numpy as np

from cpn_hrl_dag.scheduling.simulator import ScheduleSimulator


def optimistic_path_bound(simulator: ScheduleSimulator) -> float:
    """Relax node contention while retaining precedence and placement-dependent transfer."""

    scenario = simulator.scenario
    earliest = np.full((scenario.num_tasks, scenario.num_nodes), np.inf)
    for task in scenario.cache.topological_order:
        ready = np.zeros(scenario.num_nodes)
        for parent in scenario.cache.predecessors[task]:
            transfer = np.asarray([
                [simulator.communication_model.duration(scenario, parent, task, source, target)
                 for target in range(scenario.num_nodes)]
                for source in range(scenario.num_nodes)
            ])
            ready = np.maximum(ready, np.min(earliest[parent][:, None] + transfer, axis=0))
        feasible = list(simulator.feasible_node_positions(task))
        earliest[task, feasible] = ready[feasible] + simulator.execution_times[task, feasible]
    return float(np.min(earliest, axis=1).max())


def fractional_load_bound(simulator: ScheduleSimulator) -> tuple[float, list[float]]:
    """Derive a feasible load-dual certificate from a fractional assignment LP."""

    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    scenario = simulator.scenario
    pairs = [(task, node) for task in range(scenario.num_tasks)
             for node in simulator.feasible_node_positions(task)]
    count = len(pairs)
    scale = float(simulator.execution_times.max())
    objective = np.zeros(count + 1)
    objective[-1] = 1.0
    assignment = coo_matrix(
        (np.ones(count), ([task for task, node in pairs], list(range(count)))),
        shape=(scenario.num_tasks, count + 1),
    ).tocsr()
    loads = coo_matrix(
        ([float(simulator.execution_times[task, node]) / scale for task, node in pairs]
         + [-1.0] * scenario.num_nodes,
         ([node for task, node in pairs] + list(range(scenario.num_nodes)),
          list(range(count)) + [count] * scenario.num_nodes)),
        shape=(scenario.num_nodes, count + 1),
    ).tocsr()
    result = linprog(objective, A_ub=loads, b_ub=np.zeros(scenario.num_nodes),
                     A_eq=assignment, b_eq=np.ones(scenario.num_tasks), method="highs")
    if not result.success:
        raise RuntimeError(f"load-bound LP failed: {result.message}")
    weights = np.maximum(-np.asarray(result.ineqlin.marginals), 0.0)
    weights /= max(1.0, float(weights.sum())) * (1.0 + 1e-12)
    bound = sum(
        min(float(weights[node] * simulator.execution_times[task, node])
            for node in simulator.feasible_node_positions(task))
        for task in range(scenario.num_tasks)
    )
    return float(bound), weights.tolist()
