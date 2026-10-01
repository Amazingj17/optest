"""Common evaluator adapter for the two graph-based PPO comparison policies."""

from __future__ import annotations

import numpy as np
import torch

from cpn_hrl_dag.models.graph_baselines import GraphBaselineActorCritic
from cpn_hrl_dag.policies.dag_pair import DAGPairGraphPolicy, graph_dynamic_tensors


def scenario_tiers(scenario) -> np.ndarray:
    mapping = {tier: index for index, tier in enumerate(GraphBaselineActorCritic.tiers)}
    try:
        return np.asarray([mapping[node.node_type.lower()] for node in scenario.compute_nodes], dtype=np.int64)
    except KeyError as error:
        raise ValueError('comparison policies support only end, edge and cloud resource tiers') from error


def action_tensors(observation, tiers: np.ndarray, device: torch.device) -> tuple[torch.Tensor, ...]:
    return (*graph_dynamic_tensors(observation, device),
            torch.as_tensor(observation['pair_mask'], device=device, dtype=torch.bool).unsqueeze(0),
            torch.as_tensor(tiers, device=device, dtype=torch.long).unsqueeze(0))


class GraphBaselinePolicy(DAGPairGraphPolicy):
    def __init__(self, model: GraphBaselineActorCritic, device='cpu', *, normalize_observations=True) -> None:
        super().__init__(model, device, normalize_observations=normalize_observations)
        self.name = model.method
        self.placement_counts: dict[str, dict[str, int]] = {}

    @torch.no_grad()
    def reset(self, scenario) -> None:
        self.model.eval()
        self._tiers = scenario_tiers(scenario)
        self._scenario_id = scenario.scenario_id
        self.placement_counts[self._scenario_id] = dict.fromkeys(self.model.tiers, 0)
        super().reset(scenario)

    @torch.no_grad()
    def select_pair(self, observation, deterministic=True) -> tuple[int, int]:
        if self._static is None:
            raise RuntimeError('policy must be reset before selecting a pair')
        heads = self.model.action_heads(self._static, *action_tensors(observation, self._tiers, self.device))
        actions = heads.choose(deterministic)
        pair = int(self.model.executed_pair(actions).item())
        task, node = divmod(pair, self._nodes)
        self.placement_counts[self._scenario_id][self.model.tiers[int(self._tiers[node])]] += 1
        return task, node
