"""Inference adapter for the Flat PPO `(task,node)` baseline."""
from __future__ import annotations

import numpy as np
import torch

from cpn_hrl_dag.algorithms.flat_trainer import flat_tensors
from cpn_hrl_dag.algorithms.ppo import masked_distribution
from cpn_hrl_dag.scenario.types import Scenario
from cpn_hrl_dag.env.dag_env import CloudEdgeEndDAGEnv
from .base import SchedulerPolicy


class FlatPPOPolicy(SchedulerPolicy):
    name = "flat_ppo"

    def __init__(self, model: torch.nn.Module, device: str | torch.device = "cpu") -> None:
        self.model, self.device, self._pending = model.eval(), torch.device(device), None

    def reset(self, scenario: Scenario) -> None:
        self._nodes = scenario.num_nodes
        self._pending = None
        self._env = CloudEdgeEndDAGEnv()
        self._env.reset(scenario)

    @torch.no_grad()
    def select_task(self, observation: dict, ready_mask: np.ndarray, deterministic: bool = True) -> int:
        del observation, ready_mask
        task, node = self.select_pair(self._env.get_flat_observation(), deterministic)
        self._pending = (task, node)
        return task

    def select_node(self, observation: dict, task_id: int, node_mask: np.ndarray, deterministic: bool = True) -> int:
        del observation, node_mask, deterministic
        if self._pending is None or task_id != self._pending[0]:
            raise RuntimeError("flat low action does not match the pending joint action")
        task, node = self._pending
        self._env.step(task, node)
        self._pending = None
        return node

    @torch.no_grad()
    def select_pair(self, observation: dict, deterministic: bool = True) -> tuple[int, int]:
        logits, _ = self.model(*flat_tensors(observation, self.device))
        mask = torch.as_tensor(observation["pair_mask"].reshape(-1), device=self.device).unsqueeze(0)
        dist = masked_distribution(logits, mask)
        action = torch.argmax(dist.logits, 1) if deterministic else dist.sample()
        task, node = divmod(int(action.item()), self._nodes)
        return task, node
