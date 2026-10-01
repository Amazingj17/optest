"""Legal two-stage heuristic policies expressed through environment observations."""

from __future__ import annotations

from typing import Any

import numpy as np

from cpn_hrl_dag.baselines.heft import HEFTScheduler
from cpn_hrl_dag.scenario.types import Scenario
from cpn_hrl_dag.scheduling.execution_model import execution_model_for_scenario

from .base import SchedulerPolicy


def _argmin_eft(observation: dict[str, Any], node_mask: np.ndarray) -> int:
    candidates = np.flatnonzero(node_mask)
    if len(candidates) == 0:
        raise ValueError("node mask has no legal action")
    # Neural features are float32, but deterministic heuristics must preserve
    # the simulator's float64 EFT ordering so HEFT self-ratio is exactly one.
    eft = np.asarray(observation.get("heuristic_eft", observation["node_features"][:, 5]), dtype=np.float64)
    return int(min(candidates, key=lambda index: (float(eft[index]), int(index))))


class RandomPolicy(SchedulerPolicy):
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self, scenario: Scenario) -> None:
        self.rng = np.random.default_rng(self.seed + int.from_bytes(scenario.scenario_id.encode("utf-8"), "little") % (2**32))

    def select_task(self, observation: dict[str, Any], ready_mask: np.ndarray, deterministic: bool = True) -> int:
        del observation, deterministic
        return int(self.rng.choice(np.flatnonzero(ready_mask)))

    def select_node(self, observation: dict[str, Any], task_id: int, node_mask: np.ndarray, deterministic: bool = True) -> int:
        del observation, task_id, deterministic
        return int(self.rng.choice(np.flatnonzero(node_mask)))


class GreedyEFTPolicy(SchedulerPolicy):
    name = "greedy_eft"

    def reset(self, scenario: Scenario) -> None:
        self._topological_level = scenario.cache.topological_level

    def select_task(self, observation: dict[str, Any], ready_mask: np.ndarray, deterministic: bool = True) -> int:
        del observation, deterministic
        candidates = np.flatnonzero(ready_mask)
        return int(min(candidates, key=lambda index: (-int(self._topological_level[index]), int(index))))

    def select_node(self, observation: dict[str, Any], task_id: int, node_mask: np.ndarray, deterministic: bool = True) -> int:
        del task_id, deterministic
        return _argmin_eft(observation, node_mask)


class HEFTPolicy(SchedulerPolicy):
    name = "heft"

    def reset(self, scenario: Scenario) -> None:
        self._rank = HEFTScheduler(execution_model_for_scenario(scenario.dataset_source)).analyze(scenario).upward_rank
        self._task_ids = tuple(str(task.id) for task in scenario.tasks)

    def select_task(self, observation: dict[str, Any], ready_mask: np.ndarray, deterministic: bool = True) -> int:
        del observation, deterministic
        candidates = np.flatnonzero(ready_mask)
        return int(min(candidates, key=lambda index: (-float(self._rank[index]), self._task_ids[index])))

    def select_node(self, observation: dict[str, Any], task_id: int, node_mask: np.ndarray, deterministic: bool = True) -> int:
        del task_id, deterministic
        return _argmin_eft(observation, node_mask)
