"""Flat PPO actor-critic over the legal Cartesian task-node action space."""
from __future__ import annotations

import torch
from torch import nn


class FlatTaskNodeActorCritic(nn.Module):
    """Score every `(task,node)` pair without a hierarchical policy factorization."""

    def __init__(self, task_dim: int, resource_dim: int, node_dim: int, hidden_dim: int = 96) -> None:
        super().__init__()
        self.task_encoder = nn.Linear(task_dim, hidden_dim)
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.actor = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.critic = nn.Sequential(nn.Linear(hidden_dim + resource_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def forward(self, task_features: torch.Tensor, task_mask: torch.Tensor, resource_features: torch.Tensor, node_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if task_features.ndim != 3 or node_features.ndim != 4 or node_features.shape[:2] != task_features.shape[:2]:
            raise ValueError("flat task/node observation shapes are inconsistent")
        task = torch.tanh(self.task_encoder(task_features)).unsqueeze(2)
        node = torch.tanh(self.node_encoder(node_features))
        pairs = torch.cat((task.expand_as(node), node), dim=-1)
        logits = self.actor(pairs).squeeze(-1).flatten(1)
        denom = task_mask.sum(1, keepdim=True).clamp_min(1)
        pooled = (task_features.new_zeros(task_features.shape[0], task_features.shape[-1]))
        # Critic pools learned task embeddings rather than raw padded task rows.
        task_embedding = task.squeeze(2)
        pooled = (task_embedding * task_mask.unsqueeze(-1)).sum(1) / denom
        value = self.critic(torch.cat((pooled, resource_features.mean(1)), dim=-1)).squeeze(-1)
        return logits, value
