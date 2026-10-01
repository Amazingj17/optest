"""Evaluator-facing unified policy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from cpn_hrl_dag.scenario.types import Scenario


class SchedulerPolicy(ABC):
    """Every learned or heuristic scheduler chooses task then node."""

    name: str

    @abstractmethod
    def reset(self, scenario: Scenario) -> None: ...

    @abstractmethod
    def select_task(self, observation: dict[str, Any], ready_mask: np.ndarray, deterministic: bool = True) -> int: ...

    @abstractmethod
    def select_node(self, observation: dict[str, Any], task_id: int, node_mask: np.ndarray, deterministic: bool = True) -> int: ...
