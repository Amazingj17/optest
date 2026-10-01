"""Canonical, dataset-independent DAG scheduling data model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, Mapping, Sequence

import numpy as np

TaskId = Hashable


class ScenarioValidationError(ValueError):
    """Raised when a scenario cannot represent a feasible DAG instance."""


@dataclass(frozen=True, slots=True)
class Task:
    """One DAG task, with source-specific details retained in ``metadata``."""

    id: TaskId
    workload: float
    cpu_requirement: float | None = None
    gpu_requirement: float | None = None
    memory_requirement: float | None = None
    device_requirement: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Dependency:
    """A directed task dependency carrying predecessor output data."""

    src: TaskId
    dst: TaskId
    data_size: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ComputeNode:
    """A compute resource in the cloud, edge, end, or configured extra tier."""

    id: TaskId
    node_type: str
    compute_speed: float
    cpu_capacity: float | None = None
    gpu_capacity: float | None = None
    memory_capacity: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ScenarioCache:
    """Immutable preprocessing products indexed by stable task/node positions."""

    task_index: Mapping[TaskId, int]
    node_index: Mapping[TaskId, int]
    predecessors: tuple[tuple[int, ...], ...]
    successors: tuple[tuple[int, ...], ...]
    topological_order: tuple[int, ...]
    topological_level: np.ndarray
    in_degree: np.ndarray
    out_degree: np.ndarray
    edge_data_matrix: np.ndarray


@dataclass(slots=True)
class Scenario:
    """A complete scheduling instance consumed by every policy and simulator.

    ``bandwidth_matrix[i, j]`` is the usable direct/pairwise bandwidth between
    compute nodes in the same order as ``compute_nodes``.  The source dataset
    never leaks beyond an adapter: all source provenance is stored in metadata.
    """

    scenario_id: str
    dataset_source: str
    tasks: Sequence[Task]
    dependencies: Sequence[Dependency]
    compute_nodes: Sequence[ComputeNode]
    bandwidth_matrix: np.ndarray
    latency_matrix: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    cache: ScenarioCache = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.tasks = tuple(self.tasks)
        self.dependencies = tuple(self.dependencies)
        self.compute_nodes = tuple(self.compute_nodes)
        self.bandwidth_matrix = np.asarray(self.bandwidth_matrix, dtype=np.float64)
        if self.latency_matrix is not None:
            self.latency_matrix = np.asarray(self.latency_matrix, dtype=np.float64)
        self.metadata = dict(self.metadata)
        from .preprocessing import preprocess_scenario

        self.cache = preprocess_scenario(self)

    @property
    def num_tasks(self) -> int:
        return len(self.tasks)

    @property
    def num_nodes(self) -> int:
        return len(self.compute_nodes)

    def task_position(self, task_id: TaskId) -> int:
        try:
            return self.cache.task_index[task_id]
        except KeyError as exc:
            raise KeyError(f"unknown task id: {task_id!r}") from exc

    def node_position(self, node_id: TaskId) -> int:
        try:
            return self.cache.node_index[node_id]
        except KeyError as exc:
            raise KeyError(f"unknown compute-node id: {node_id!r}") from exc
