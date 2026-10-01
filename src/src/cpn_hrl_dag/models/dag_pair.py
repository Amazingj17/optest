"""Graph-encoded actor-critic for joint ready-task/resource decisions."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class StaticGraphEmbeddings:
    """Scenario-static task and resource embeddings reusable across decisions."""

    tasks: torch.Tensor
    resources: torch.Tensor


class EdgeBiasedGraphAttentionBlock(nn.Module):
    """Dense multi-head attention restricted to graph neighbours and self."""

    def __init__(self, hidden_dim: int, heads: int, edge_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_dim <= 0 or heads <= 0 or hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be positive and divisible by heads")
        if edge_dim <= 0:
            raise ValueError("edge_dim must be positive")
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.head_dim = hidden_dim // heads
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_bias = nn.Linear(edge_dim, heads, bias=False)
        self.relation_bias = nn.Parameter(torch.zeros(2, heads))
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.feed_forward_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        values: torch.Tensor,
        node_mask: torch.Tensor,
        adjacency: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        if values.ndim != 3 or node_mask.shape != values.shape[:2]:
            raise ValueError("graph values/mask shapes are inconsistent")
        batch, nodes, hidden = values.shape
        if hidden != self.hidden_dim or adjacency.shape != (batch, nodes, nodes):
            raise ValueError("graph adjacency shape is inconsistent")
        if edge_features.shape[:3] != (batch, nodes, nodes):
            raise ValueError("graph edge-feature shape is inconsistent")

        query = self.query(values).view(batch, nodes, self.heads, self.head_dim).transpose(1, 2)
        key = self.key(values).view(batch, nodes, self.heads, self.head_dim).transpose(1, 2)
        value = self.value(values).view(batch, nodes, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.einsum("bhid,bhjd->bhij", query, key) / sqrt(self.head_dim)

        # adjacency[src, dst]. Attention scores are indexed [query, key], so
        # incoming edges are transposed while outgoing edges retain direction.
        incoming = adjacency.transpose(-1, -2)
        outgoing = adjacency
        identity = torch.eye(nodes, dtype=torch.bool, device=values.device).unsqueeze(0)
        connected = incoming | outgoing | identity
        connected = connected & node_mask.unsqueeze(1) & node_mask.unsqueeze(2)

        undirected_edge_features = edge_features + edge_features.transpose(1, 2)
        edge_bias = self.edge_bias(undirected_edge_features).permute(0, 3, 1, 2)
        direction_bias = (
            incoming.unsqueeze(1) * self.relation_bias[0].view(1, self.heads, 1, 1)
            + outgoing.unsqueeze(1) * self.relation_bias[1].view(1, self.heads, 1, 1)
        )
        scores = scores + edge_bias + direction_bias
        scores = scores.masked_fill(~connected.unsqueeze(1), torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        context = torch.einsum("bhij,bhjd->bhid", attention, value)
        context = context.transpose(1, 2).reshape(batch, nodes, hidden)
        context = context * node_mask.unsqueeze(-1)
        values = self.attention_norm(values + self.output(context))
        values = self.feed_forward_norm(values + self.feed_forward(values))
        return values * node_mask.unsqueeze(-1)


class EdgeBiasedGraphEncoder(nn.Module):
    """Project raw node features and apply edge-aware attention blocks."""

    def __init__(
        self,
        input_dim: int,
        edge_dim: int,
        hidden_dim: int,
        heads: int,
        layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if layers <= 0:
            raise ValueError("graph encoder requires at least one layer")
        self.input = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.layers = nn.ModuleList(
            EdgeBiasedGraphAttentionBlock(hidden_dim, heads, edge_dim, dropout)
            for _ in range(layers)
        )

    def forward(
        self,
        node_features: torch.Tensor,
        node_mask: torch.Tensor,
        adjacency: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        values = self.input(node_features)
        for layer in self.layers:
            values = layer(values, node_mask, adjacency, edge_features)
        return values


class DAGPairGraphActorCritic(nn.Module):
    """Jointly score every legal ``(ready task, feasible resource)`` pair.

    Static graph encodings are separated from lightweight dynamic feature
    fusion.  Inference policies therefore run expensive graph attention once
    per scenario, not once per scheduling decision.
    """

    def __init__(
        self,
        task_dim: int,
        resource_dim: int,
        pair_dim: int,
        task_edge_dim: int,
        resource_edge_dim: int,
        *,
        hidden_dim: int = 64,
        heads: int = 4,
        task_layers: int = 2,
        resource_layers: int = 2,
        dropout: float = 0.0,
        heuristic_residual: bool = True,
        residual_limit: float = 2.0,
        heuristic_scale: float = 0.05,
    ) -> None:
        super().__init__()
        if residual_limit <= 0.0 or heuristic_scale <= 0.0:
            raise ValueError("residual_limit and heuristic_scale must be positive")
        self.task_encoder = EdgeBiasedGraphEncoder(
            task_dim, task_edge_dim, hidden_dim, heads, task_layers, dropout
        )
        self.resource_encoder = EdgeBiasedGraphEncoder(
            resource_dim, resource_edge_dim, hidden_dim, heads, resource_layers, dropout
        )
        self.dynamic_task = nn.Sequential(nn.Linear(task_dim, hidden_dim), nn.GELU())
        self.dynamic_resource = nn.Sequential(nn.Linear(resource_dim, hidden_dim), nn.GELU())
        self.pair_encoder = nn.Sequential(nn.Linear(pair_dim, hidden_dim), nn.GELU())
        self.task_fusion = nn.LayerNorm(hidden_dim)
        self.resource_fusion = nn.LayerNorm(hidden_dim)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.hidden_dim = int(hidden_dim)
        self.heuristic_residual = bool(heuristic_residual)
        self.residual_limit = float(residual_limit)
        self.heuristic_scale = float(heuristic_scale)
        if self.heuristic_residual:
            nn.init.zeros_(self.actor[-1].weight)
            nn.init.zeros_(self.actor[-1].bias)

    def encode_static(
        self,
        task_features: torch.Tensor,
        task_mask: torch.Tensor,
        task_adjacency: torch.Tensor,
        task_edge_features: torch.Tensor,
        resource_features: torch.Tensor,
        resource_mask: torch.Tensor,
        resource_adjacency: torch.Tensor,
        resource_edge_features: torch.Tensor,
    ) -> StaticGraphEmbeddings:
        """Encode immutable topology/resource structure for one or more scenarios."""

        tasks = self.task_encoder(task_features, task_mask, task_adjacency, task_edge_features)
        resources = self.resource_encoder(
            resource_features,
            resource_mask,
            resource_adjacency,
            resource_edge_features,
        )
        if not torch.isfinite(tasks).all() or not torch.isfinite(resources).all():
            raise ValueError("static graph encoder produced non-finite embeddings")
        return StaticGraphEmbeddings(tasks, resources)

    def score_dynamic(
        self,
        static: StaticGraphEmbeddings,
        task_features: torch.Tensor,
        task_mask: torch.Tensor,
        resource_features: torch.Tensor,
        resource_mask: torch.Tensor,
        pair_features: torch.Tensor,
        heuristic_logits: torch.Tensor,
        progress: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse current simulator state with cached graph embeddings."""

        if task_features.ndim != 3 or resource_features.ndim != 3 or pair_features.ndim != 4:
            raise ValueError("dynamic graph-policy tensors have invalid ranks")
        batch, tasks, _ = task_features.shape
        resources = resource_features.shape[1]
        if pair_features.shape[:3] != (batch, tasks, resources):
            raise ValueError("pair feature shape is inconsistent with task/resource tensors")
        if heuristic_logits.shape != (batch, tasks, resources):
            raise ValueError("heuristic pair-logit shape is inconsistent")
        if static.tasks.shape[:2] != (batch, tasks) or static.resources.shape[:2] != (batch, resources):
            raise ValueError("cached graph embeddings have incompatible shapes")

        task_values = self.task_fusion(static.tasks + self.dynamic_task(task_features))
        resource_values = self.resource_fusion(static.resources + self.dynamic_resource(resource_features))
        task_pairs = task_values.unsqueeze(2).expand(-1, -1, resources, -1)
        resource_pairs = resource_values.unsqueeze(1).expand(-1, tasks, -1, -1)
        pair_values = self.pair_encoder(pair_features)
        actor_input = torch.cat(
            (task_pairs, resource_pairs, task_pairs * resource_pairs, pair_values),
            dim=-1,
        )
        residual = self.actor(actor_input).squeeze(-1)
        logits = (
            self.heuristic_scale * heuristic_logits
            + self.residual_limit * torch.tanh(residual)
            if self.heuristic_residual
            else residual
        ).flatten(1)

        task_denominator = task_mask.sum(1, keepdim=True).clamp_min(1).to(task_values.dtype)
        resource_denominator = resource_mask.sum(1, keepdim=True).clamp_min(1).to(resource_values.dtype)
        pooled_tasks = (task_values * task_mask.unsqueeze(-1)).sum(1) / task_denominator
        pooled_resources = (resource_values * resource_mask.unsqueeze(-1)).sum(1) / resource_denominator
        value = self.critic(torch.cat((pooled_tasks, pooled_resources, progress.view(batch, 1)), dim=-1)).squeeze(-1)
        if not torch.isfinite(logits).all() or not torch.isfinite(value).all():
            raise ValueError("joint graph policy produced non-finite outputs")
        return logits, value
