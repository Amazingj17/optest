"""Deterministic insertion-slot search over a non-preemptive node timeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class TimelineEntry:
    """A committed non-preemptive interval on one compute node."""

    task_position: int
    node_position: int
    start: float
    finish: float


def first_feasible_slot(
    intervals: Sequence[TimelineEntry], dependency_ready_time: float, duration: float
) -> tuple[float, float]:
    """Find the earliest gap at or after dependency readiness that fits duration."""

    if duration <= 0.0:
        raise ValueError("scheduled duration must be positive")
    start = dependency_ready_time
    for interval in intervals:
        if start + duration <= interval.start + EPSILON:
            return start, start + duration
        if start < interval.finish - EPSILON:
            start = interval.finish
    return start, start + duration
