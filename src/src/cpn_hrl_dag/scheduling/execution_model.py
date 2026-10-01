"""Explicit execution-time models; adapters never hard-code this policy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Final

import numpy as np

from cpn_hrl_dag.scenario.types import ComputeNode, Task


class ExecutionTimeModel(ABC):
    """Maps a task/node pair to a positive execution duration."""

    name: str

    @abstractmethod
    def duration(self, task: Task, node: ComputeNode) -> float:
        """Return the execution duration for a compatible task/node pair."""

    def matrix(self, tasks: tuple[Task, ...], nodes: tuple[ComputeNode, ...]) -> np.ndarray:
        values = np.asarray(
            [[self.duration(task, node) for node in nodes] for task in tasks], dtype=np.float64
        )
        if not np.isfinite(values).all() or (values <= 0.0).any():
            raise ValueError(f"{self.name} produced non-positive or non-finite execution times")
        values.setflags(write=False)
        return values


class SimpleSpeedExecutionModel(ExecutionTimeModel):
    """Physical/relative workload model: ``task.workload / node.compute_speed``."""

    name: Final[str] = "simple_speed"

    def duration(self, task: Task, node: ComputeNode) -> float:
        return float(task.workload) / float(node.compute_speed)


class DurationExecutionModel(ExecutionTimeModel):
    """Treat ``Task.workload`` as an already measured duration on every node."""

    name: Final[str] = "fixed_duration"

    def duration(self, task: Task, node: ComputeNode) -> float:
        del node
        return float(task.workload)


class HeterogeneousDeviceExecutionModel(ExecutionTimeModel):
    """GrapheonRL-compatible duration divided by an explicitly selected device speed."""

    name: Final[str] = "heterogeneous_device"

    def duration(self, task: Task, node: ComputeNode) -> float:
        device = (task.device_requirement or "cpu").lower()
        speeds = node.metadata.get("device_speeds", {})
        if not isinstance(speeds, dict):
            raise ValueError(f"node {node.id!r} metadata.device_speeds must be a dictionary")
        normalized = {str(key).lower(): float(value) for key, value in speeds.items()}
        speed = normalized.get(device, normalized.get("cpu", float(node.compute_speed)))
        if not np.isfinite(speed) or speed <= 0.0:
            raise ValueError(f"node {node.id!r} has invalid speed for device {device!r}")
        return float(task.workload) / speed


def execution_model_for_scenario(source: str) -> ExecutionTimeModel:
    """Return the audited default execution semantics for a dataset source.

    The dispatch happens at the scenario boundary rather than inside an
    adapter, preserving the adapter-to-model separation.  Callers may always
    pass an explicit model to override this compatibility default.
    """

    if source.lower() == "grapheonrl":
        return HeterogeneousDeviceExecutionModel()
    return SimpleSpeedExecutionModel()
