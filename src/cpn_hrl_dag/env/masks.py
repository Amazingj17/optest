"""Single legality implementation shared by training and deterministic inference."""

from __future__ import annotations

import numpy as np

from cpn_hrl_dag.scheduling.simulator import ScheduleSimulator


def ready_mask(simulator: ScheduleSimulator) -> np.ndarray:
    """True only for unscheduled tasks whose required predecessors are placed."""

    mask = np.zeros(simulator.scenario.num_tasks, dtype=bool)
    mask[list(simulator.ready_task_positions())] = True
    return mask


def node_mask(simulator: ScheduleSimulator, task_position: int) -> np.ndarray:
    """True only for permanently executable nodes; busy nodes remain valid."""

    mask = np.zeros(simulator.scenario.num_nodes, dtype=bool)
    mask[list(simulator.feasible_node_positions(task_position))] = True
    return mask
