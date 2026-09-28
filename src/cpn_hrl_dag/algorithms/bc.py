"""Optional HEFT behaviour-cloning warm start for hierarchical actors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from cpn_hrl_dag.algorithms.hierarchical_trainer import high_tensors, low_tensors
from cpn_hrl_dag.algorithms.ppo import masked_distribution
from cpn_hrl_dag.baselines.heft import HEFTScheduler
from cpn_hrl_dag.env.dag_env import CloudEdgeEndDAGEnv
from cpn_hrl_dag.scenario.types import Scenario
from cpn_hrl_dag.scheduling.execution_model import execution_model_for_scenario


@dataclass(frozen=True, slots=True)
class BCResult:
    """Losses and number of complete HEFT decision pairs consumed."""

    high_loss: float
    low_loss: float
    decisions: int


class BehaviorCloner:
    """Supervised actor warm start; critics are intentionally untouched."""

    def __init__(self, high_model: torch.nn.Module, low_model: torch.nn.Module, high_optimizer: torch.optim.Optimizer, low_optimizer: torch.optim.Optimizer, device: torch.device, normalize_observations: bool = False) -> None:
        self.high_model, self.low_model = high_model, low_model
        self.high_optimizer, self.low_optimizer, self.device = high_optimizer, low_optimizer, device
        self.normalize_observations = bool(normalize_observations)

    def fit(self, scenarios: Iterable[Scenario], epochs: int = 1) -> BCResult:
        items = tuple(scenarios)
        if not items or epochs <= 0:
            return BCResult(0.0, 0.0, 0)
        high_losses: list[float] = []
        low_losses: list[float] = []
        decisions = 0
        for _ in range(epochs):
            for scenario in items:
                execution = execution_model_for_scenario(scenario.dataset_source)
                env = CloudEdgeEndDAGEnv(execution, normalize_observations=self.normalize_observations)
                high_observation, _ = env.reset(scenario)
                rank = HEFTScheduler(execution).analyze(scenario).upward_rank
                while not env.simulator.done:
                    ready = env.get_ready_mask()
                    task = int(min(np.flatnonzero(ready), key=lambda index: (-float(rank[index]), int(index))))
                    high_logits, _ = self.high_model(*high_tensors(high_observation, self.device))
                    high_dist = masked_distribution(high_logits, torch.as_tensor(ready, device=self.device).unsqueeze(0))
                    high_loss = -high_dist.log_prob(torch.tensor([task], device=self.device)).mean()
                    self.high_optimizer.zero_grad(); high_loss.backward(); self.high_optimizer.step()
                    env.select_task(task)
                    low_observation = env.get_low_observation(task)
                    node_mask = env.get_node_mask(task)
                    node = int(min(np.flatnonzero(node_mask), key=lambda index: (float(low_observation['node_features'][index, 5]), int(index))))
                    low_logits, _ = self.low_model(*low_tensors(low_observation, self.device))
                    low_dist = masked_distribution(low_logits, torch.as_tensor(node_mask, device=self.device).unsqueeze(0))
                    low_loss = -low_dist.log_prob(torch.tensor([node], device=self.device)).mean()
                    self.low_optimizer.zero_grad(); low_loss.backward(); self.low_optimizer.step()
                    high_losses.append(float(high_loss.item())); low_losses.append(float(low_loss.item())); decisions += 1
                    high_observation, _, _, _, _ = env.step(task, node)
        return BCResult(float(np.mean(high_losses)), float(np.mean(low_losses)), decisions)
