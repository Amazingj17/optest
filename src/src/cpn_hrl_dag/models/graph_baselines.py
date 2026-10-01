"""Pure graph PPO and coordinated three-tier MAPPO with a shared graph backbone."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from cpn_hrl_dag.algorithms.ppo import masked_distribution
from cpn_hrl_dag.models.dag_pair import EdgeBiasedGraphEncoder, StaticGraphEmbeddings


@dataclass
class ActionHeads:
    distributions: tuple[torch.distributions.Categorical, ...]
    active: torch.Tensor
    value: torch.Tensor

    def choose(self, deterministic: bool = False) -> torch.Tensor:
        return torch.stack([
            distribution.logits.argmax(-1) if deterministic else distribution.sample()
            for distribution in self.distributions
        ], dim=-1)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return torch.stack([distribution.log_prob(actions[:, index])
                            for index, distribution in enumerate(self.distributions)], dim=-1)

    def entropy(self) -> torch.Tensor:
        return torch.stack([distribution.entropy() for distribution in self.distributions], dim=-1)


class GraphBaselineActorCritic(nn.Module):
    """Shared encoders and critic; independent tier actors plus a categorical coordinator.

    MAPPO samples three proposals and a coordinator choice. PPO clips each
    active head's own likelihood ratio using the shared team advantage.
    No HEFT residual, search candidate, or learned-policy safety wrapper is used.
    """

    methods = ('graph_ppo', 'tier_mappo')
    tiers = ('end', 'edge', 'cloud')

    def __init__(self, task_dim: int, resource_dim: int, pair_dim: int,
                 task_edge_dim: int, resource_edge_dim: int, *, method: str,
                 hidden_dim: int = 32, heads: int = 4, task_layers: int = 1,
                 resource_layers: int = 1, bid_temperature: float = 1.0) -> None:
        super().__init__()
        if method not in self.methods:
            raise ValueError(f'unknown graph baseline: {method}')
        if not 0.0 < bid_temperature < float('inf'):
            raise ValueError('bid_temperature must be finite and positive')
        self.method = method
        self.bid_temperature = float(bid_temperature)
        self.task_encoder = EdgeBiasedGraphEncoder(task_dim, task_edge_dim, hidden_dim, heads, task_layers)
        self.resource_encoder = EdgeBiasedGraphEncoder(resource_dim, resource_edge_dim, hidden_dim, heads, resource_layers)
        self.dynamic_task = nn.Sequential(nn.Linear(task_dim, hidden_dim), nn.GELU())
        self.dynamic_resource = nn.Sequential(nn.Linear(resource_dim, hidden_dim), nn.GELU())
        self.pair_encoder = nn.Sequential(nn.Linear(pair_dim, hidden_dim), nn.GELU())
        self.task_fusion = nn.LayerNorm(hidden_dim)
        self.resource_fusion = nn.LayerNorm(hidden_dim)
        self.actors = nn.ModuleList([
            nn.Sequential(nn.Linear(4 * hidden_dim, 2 * hidden_dim), nn.GELU(),
                          nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
            for _ in range(1 if method == 'graph_ppo' else 3)
        ])
        self.bidders = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(2 * hidden_dim), nn.Linear(2 * hidden_dim, hidden_dim),
                          nn.GELU(), nn.Linear(hidden_dim, 1), nn.Tanh())
            for _ in range(3 if method == 'tier_mappo' else 0)
        ])
        self.critic = nn.Sequential(nn.Linear(2 * hidden_dim + 1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def encode_static(self, task_features, task_mask, task_adjacency, task_edges,
                      resource_features, resource_mask, resource_adjacency, resource_edges) -> StaticGraphEmbeddings:
        return StaticGraphEmbeddings(
            self.task_encoder(task_features, task_mask, task_adjacency, task_edges),
            self.resource_encoder(resource_features, resource_mask, resource_adjacency, resource_edges),
        )

    def action_heads(self, static: StaticGraphEmbeddings, task_features, task_mask,
                     resource_features, resource_mask, pair_features, heuristic_logits,
                     progress, pair_mask, resource_tiers) -> ActionHeads:
        del heuristic_logits
        task_values = self.task_fusion(static.tasks + self.dynamic_task(task_features))
        resource_values = self.resource_fusion(static.resources + self.dynamic_resource(resource_features))
        batch, tasks, _ = task_values.shape
        nodes = resource_values.shape[1]
        if pair_mask.shape != (batch, tasks, nodes) or resource_tiers.shape != (batch, nodes):
            raise ValueError('inconsistent graph action masks or tier labels')
        pair_mask = pair_mask & task_mask.unsqueeze(-1) & resource_mask.unsqueeze(1)
        if not bool(pair_mask.flatten(1).any(-1).all()):
            raise ValueError('graph policy needs at least one legal pair')
        if bool(((resource_tiers < 0) | (resource_tiers > 2)).any()):
            raise ValueError('resource tiers must map to end, edge or cloud')
        task_pairs = task_values.unsqueeze(2).expand(-1, -1, nodes, -1)
        node_pairs = resource_values.unsqueeze(1).expand(-1, tasks, -1, -1)
        features = torch.cat((task_pairs, node_pairs, task_pairs * node_pairs,
                              self.pair_encoder(pair_features)), dim=-1)
        pooled_tasks = (task_values * task_mask.unsqueeze(-1)).sum(1) / task_mask.sum(1, keepdim=True).clamp_min(1)
        pooled_resources = (resource_values * resource_mask.unsqueeze(-1)).sum(1) / resource_mask.sum(1, keepdim=True).clamp_min(1)
        value = self.critic(torch.cat((pooled_tasks, pooled_resources, progress.view(batch, 1)), dim=-1)).squeeze(-1)
        if self.method == 'graph_ppo':
            logits = self.actors[0](features).squeeze(-1).flatten(1)
            return ActionHeads((masked_distribution(logits, pair_mask.flatten(1)),),
                               torch.ones((batch, 1), dtype=torch.bool, device=logits.device), value)
        distributions, active_tiers, bids = [], [], []
        for tier, actor in enumerate(self.actors):
            tier_nodes = (resource_tiers == tier) & resource_mask
            legal = (pair_mask & tier_nodes.unsqueeze(1)).flatten(1)
            active = legal.any(-1)
            safe_mask = legal.clone()
            safe_mask[~active, 0] = True
            distributions.append(masked_distribution(actor(features).squeeze(-1).flatten(1), safe_mask))
            active_tiers.append(active)
            local_resources = (resource_values * tier_nodes.unsqueeze(-1)).sum(1) / tier_nodes.sum(1, keepdim=True).clamp_min(1)
            bids.append(self.bidders[tier](torch.cat((pooled_tasks, local_resources), dim=-1)).squeeze(-1))
        active = torch.stack(active_tiers, dim=-1)
        distributions.append(masked_distribution(torch.stack(bids, dim=-1) / self.bid_temperature, active))
        return ActionHeads(tuple(distributions), torch.cat((active, torch.ones_like(active[:, :1])), dim=-1), value)

    def executed_pair(self, actions: torch.Tensor) -> torch.Tensor:
        if self.method == 'graph_ppo':
            return actions[:, 0]
        return actions[:, :3].gather(1, actions[:, 3:4]).squeeze(1)
