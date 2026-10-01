"""One-time validation and array preprocessing for :class:`Scenario`."""

from __future__ import annotations

from collections import deque

import numpy as np

from .types import Scenario, ScenarioCache, ScenarioValidationError


def _require_finite(value: float, label: str, *, positive: bool = False) -> None:
    if not np.isfinite(value) or (positive and value <= 0.0):
        qualifier = "finite and > 0" if positive else "finite"
        raise ScenarioValidationError(f"{label} must be {qualifier}, got {value!r}")


def preprocess_scenario(scenario: Scenario) -> ScenarioCache:
    """Validate a scenario and derive all static graph data without NetworkX.

    Task order is source order; adapters must make that order deterministic. The
    topological order uses this positional order as its deterministic tie-break.
    """

    if not scenario.scenario_id:
        raise ScenarioValidationError("scenario_id must be non-empty")
    if not scenario.dataset_source:
        raise ScenarioValidationError("dataset_source must be non-empty")
    if not scenario.tasks:
        raise ScenarioValidationError("a scenario needs at least one task")
    if not scenario.compute_nodes:
        raise ScenarioValidationError("a scenario needs at least one compute node")

    task_index: dict[object, int] = {}
    for index, task in enumerate(scenario.tasks):
        if task.id in task_index:
            raise ScenarioValidationError(f"duplicate task id: {task.id!r}")
        _require_finite(float(task.workload), f"task {task.id!r} workload", positive=True)
        for label, requirement in (
            ("cpu_requirement", task.cpu_requirement),
            ("gpu_requirement", task.gpu_requirement),
            ("memory_requirement", task.memory_requirement),
        ):
            if requirement is not None:
                _require_finite(float(requirement), f"task {task.id!r} {label}")
                if requirement < 0:
                    raise ScenarioValidationError(f"task {task.id!r} {label} cannot be negative")
        task_index[task.id] = index

    node_index: dict[object, int] = {}
    for index, node in enumerate(scenario.compute_nodes):
        if node.id in node_index:
            raise ScenarioValidationError(f"duplicate compute-node id: {node.id!r}")
        if not node.node_type:
            raise ScenarioValidationError(f"compute node {node.id!r} has an empty node_type")
        _require_finite(float(node.compute_speed), f"node {node.id!r} compute_speed", positive=True)
        for label, capacity in (
            ("cpu_capacity", node.cpu_capacity),
            ("gpu_capacity", node.gpu_capacity),
            ("memory_capacity", node.memory_capacity),
        ):
            if capacity is not None:
                _require_finite(float(capacity), f"node {node.id!r} {label}")
                if capacity < 0:
                    raise ScenarioValidationError(f"node {node.id!r} {label} cannot be negative")
        node_index[node.id] = index

    expected_shape = (len(scenario.compute_nodes), len(scenario.compute_nodes))
    if scenario.bandwidth_matrix.shape != expected_shape:
        raise ScenarioValidationError(
            f"bandwidth_matrix shape must be {expected_shape}, got {scenario.bandwidth_matrix.shape}"
        )
    if np.isnan(scenario.bandwidth_matrix).any() or (scenario.bandwidth_matrix <= 0.0).any():
        raise ScenarioValidationError("bandwidth_matrix must contain only positive non-NaN values")
    if scenario.latency_matrix is not None:
        if scenario.latency_matrix.shape != expected_shape:
            raise ScenarioValidationError(
                f"latency_matrix shape must be {expected_shape}, got {scenario.latency_matrix.shape}"
            )
        if not np.isfinite(scenario.latency_matrix).all() or (scenario.latency_matrix < 0.0).any():
            raise ScenarioValidationError("latency_matrix must be finite and non-negative")

    predecessors: list[list[int]] = [[] for _ in scenario.tasks]
    successors: list[list[int]] = [[] for _ in scenario.tasks]
    edge_data = np.zeros((len(scenario.tasks), len(scenario.tasks)), dtype=np.float64)
    edge_pairs: set[tuple[int, int]] = set()
    for dependency in scenario.dependencies:
        if dependency.src not in task_index or dependency.dst not in task_index:
            raise ScenarioValidationError(
                f"dependency {dependency.src!r}->{dependency.dst!r} references an unknown task"
            )
        source, target = task_index[dependency.src], task_index[dependency.dst]
        if source == target:
            raise ScenarioValidationError(f"self dependency at task {dependency.src!r}")
        if (source, target) in edge_pairs:
            raise ScenarioValidationError(f"duplicate dependency {dependency.src!r}->{dependency.dst!r}")
        _require_finite(float(dependency.data_size), f"dependency {dependency.src!r}->{dependency.dst!r} data_size")
        if dependency.data_size < 0:
            raise ScenarioValidationError("dependency data_size cannot be negative")
        edge_pairs.add((source, target))
        predecessors[target].append(source)
        successors[source].append(target)
        edge_data[source, target] = float(dependency.data_size)

    for items in predecessors:
        items.sort()
    for items in successors:
        items.sort()
    in_degree = np.asarray([len(items) for items in predecessors], dtype=np.int64)
    out_degree = np.asarray([len(items) for items in successors], dtype=np.int64)
    remaining = in_degree.copy()
    ready: deque[int] = deque(index for index, degree in enumerate(remaining) if degree == 0)
    topological_order: list[int] = []
    while ready:
        task_position = ready.popleft()
        topological_order.append(task_position)
        for child in successors[task_position]:
            remaining[child] -= 1
            if remaining[child] == 0:
                ready.append(child)
    if len(topological_order) != len(scenario.tasks):
        raise ScenarioValidationError("task dependencies contain a directed cycle")

    levels = np.zeros(len(scenario.tasks), dtype=np.int64)
    for task_position in topological_order:
        if predecessors[task_position]:
            levels[task_position] = max(levels[parent] + 1 for parent in predecessors[task_position])
    edge_data.setflags(write=False)
    in_degree.setflags(write=False)
    out_degree.setflags(write=False)
    levels.setflags(write=False)
    return ScenarioCache(
        task_index=task_index,
        node_index=node_index,
        predecessors=tuple(tuple(items) for items in predecessors),
        successors=tuple(tuple(items) for items in successors),
        topological_order=tuple(topological_order),
        topological_level=levels,
        in_degree=in_degree,
        out_degree=out_degree,
        edge_data_matrix=edge_data,
    )
