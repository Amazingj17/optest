"""Replaceable semantics for defining which unscheduled DAG tasks are ready."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .simulator import ScheduleSimulator


class ReadySemantics(ABC):
    """Defines ready tasks while leaving timing and placement to the simulator."""

    name: str

    @abstractmethod
    def ready_positions(self, simulator: "ScheduleSimulator") -> tuple[int, ...]:
        """Return deterministic task positions that can be selected now."""


class ListScheduleReadySemantics(ReadySemantics):
    """Offline list scheduling: predecessors must be placed, not wall-clock complete."""

    name = "list_schedule"

    def ready_positions(self, simulator: "ScheduleSimulator") -> tuple[int, ...]:
        return tuple(
            task_position
            for task_position in range(simulator.scenario.num_tasks)
            if task_position not in simulator.entries
            and all(parent in simulator.entries for parent in simulator.scenario.cache.predecessors[task_position])
        )


class RuntimeCompletionReadySemantics(ReadySemantics):
    """Online alternative: predecessors must also have completed by decision time."""

    name = "runtime_completion"

    def ready_positions(self, simulator: "ScheduleSimulator") -> tuple[int, ...]:
        return tuple(
            task_position
            for task_position in range(simulator.scenario.num_tasks)
            if task_position not in simulator.entries
            and all(
                parent in simulator.entries
                and simulator.entries[parent].finish <= simulator.decision_time
                for parent in simulator.scenario.cache.predecessors[task_position]
            )
        )
