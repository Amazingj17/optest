"""Inference policy for the cached graph-encoded joint action model."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from cpn_hrl_dag.algorithms.ppo import masked_distribution
from cpn_hrl_dag.env.dag_env import CloudEdgeEndDAGEnv
from cpn_hrl_dag.models.dag_pair import DAGPairGraphActorCritic, StaticGraphEmbeddings
from cpn_hrl_dag.scenario.types import Scenario

from .base import SchedulerPolicy


def graph_static_tensors(
    observation: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """Convert one unpadded scenario's static graph inputs to batch tensors."""

    task_features = torch.as_tensor(observation["task_features"], device=device).unsqueeze(0)
    task_mask = torch.as_tensor(observation["task_mask"], device=device, dtype=torch.bool).unsqueeze(0)
    task_adjacency = torch.as_tensor(
        observation["task_adjacency"], device=device, dtype=torch.bool
    ).unsqueeze(0)
    task_edges = torch.as_tensor(observation["task_edge_features"], device=device).unsqueeze(0)
    resource_features = torch.as_tensor(observation["resource_features"], device=device).unsqueeze(0)
    resource_mask = torch.ones(
        (1, resource_features.shape[1]), device=device, dtype=torch.bool
    )
    resource_adjacency = torch.as_tensor(
        observation["resource_adjacency"], device=device, dtype=torch.bool
    ).unsqueeze(0)
    resource_edges = torch.as_tensor(observation["resource_edge_features"], device=device).unsqueeze(0)
    return (
        task_features,
        task_mask,
        task_adjacency,
        task_edges,
        resource_features,
        resource_mask,
        resource_adjacency,
        resource_edges,
    )


def graph_dynamic_tensors(
    observation: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """Convert one current simulator state to graph-policy batch tensors."""

    task_features = torch.as_tensor(observation["task_features"], device=device).unsqueeze(0)
    task_mask = torch.as_tensor(observation["task_mask"], device=device, dtype=torch.bool).unsqueeze(0)
    resource_features = torch.as_tensor(observation["resource_features"], device=device).unsqueeze(0)
    resource_mask = torch.ones(
        (1, resource_features.shape[1]), device=device, dtype=torch.bool
    )
    pair_features = torch.as_tensor(observation["pair_node_features"], device=device).unsqueeze(0)
    heuristic_logits = torch.as_tensor(
        observation["heuristic_pair_logits"], device=device, dtype=pair_features.dtype
    ).unsqueeze(0)
    progress = torch.as_tensor([observation["progress"]], device=device)
    return (
        task_features,
        task_mask,
        resource_features,
        resource_mask,
        pair_features,
        heuristic_logits,
        progress,
    )


class DAGPairGraphPolicy(SchedulerPolicy):
    """Choose one legal task/resource pair using cached static graph embeddings."""

    name = "dag_pair_graph"

    def __init__(
        self,
        model: DAGPairGraphActorCritic,
        device: str | torch.device = "cpu",
        *,
        normalize_observations: bool = True,
    ) -> None:
        self.model = model.eval()
        self.device = torch.device(device)
        self.normalize_observations = bool(normalize_observations)
        self._pending: tuple[int, int] | None = None
        self._static: StaticGraphEmbeddings | None = None

    @torch.no_grad()
    def reset(self, scenario: Scenario) -> None:
        self._nodes = scenario.num_nodes
        self._pending = None
        self._env = CloudEdgeEndDAGEnv(normalize_observations=self.normalize_observations)
        self._env.reset(scenario)
        initial = self._env.get_flat_observation()
        self._static = self.model.encode_static(*graph_static_tensors(initial, self.device))

    @torch.no_grad()
    def select_task(
        self,
        observation: dict[str, Any],
        ready_mask: np.ndarray,
        deterministic: bool = True,
    ) -> int:
        del observation
        joint = self._env.get_flat_observation()
        if not np.array_equal(joint["ready_mask"], ready_mask):
            raise RuntimeError("graph policy mirror environment is out of sync")
        task, node = self.select_pair(joint, deterministic)
        self._pending = (task, node)
        return task

    def select_node(
        self,
        observation: dict[str, Any],
        task_id: int,
        node_mask: np.ndarray,
        deterministic: bool = True,
    ) -> int:
        del observation, deterministic
        if self._pending is None or task_id != self._pending[0]:
            raise RuntimeError("joint graph action does not match the pending task")
        task, node = self._pending
        if node < 0 or node >= len(node_mask) or not bool(node_mask[node]):
            raise RuntimeError("joint graph policy selected an externally infeasible node")
        self._env.step(task, node)
        self._pending = None
        return node

    @torch.no_grad()
    def select_pair(
        self,
        observation: dict[str, Any],
        deterministic: bool = True,
    ) -> tuple[int, int]:
        if self._static is None:
            raise RuntimeError("graph policy must be reset before selecting an action")
        logits, _ = self.model.score_dynamic(
            self._static,
            *graph_dynamic_tensors(observation, self.device),
        )
        mask = torch.as_tensor(
            observation["pair_mask"].reshape(1, -1),
            device=self.device,
            dtype=torch.bool,
        )
        distribution = masked_distribution(logits, mask)
        action = torch.argmax(distribution.logits, dim=1) if deterministic else distribution.sample()
        task, node = divmod(int(action.item()), self._nodes)
        return task, node
