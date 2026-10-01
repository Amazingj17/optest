"""Variable-size masked LSTM actor-critic for ready-task selection."""
from __future__ import annotations
import torch
from torch import nn

class HighLevelLSTMActorCritic(nn.Module):
    def __init__(self, task_dim: int, resource_dim: int, hidden_dim: int = 128, use_lstm: bool = True, heuristic_residual: bool = False, heuristic_weight: float = 8.0, residual_limit: float = 0.5) -> None:
        super().__init__(); self.use_lstm=bool(use_lstm); self.heuristic_residual=bool(heuristic_residual); self.heuristic_weight=float(heuristic_weight); self.residual_limit=float(residual_limit); self.task_encoder=nn.Linear(task_dim,hidden_dim); self.lstm=nn.LSTM(hidden_dim,hidden_dim,batch_first=True) if self.use_lstm else None; self.actor=nn.Linear(hidden_dim,1); self.critic=nn.Sequential(nn.Linear(hidden_dim+resource_dim,hidden_dim),nn.Tanh(),nn.Linear(hidden_dim,1))
        if self.heuristic_residual:
            if task_dim <= 14: raise ValueError("HEFT residual high actor requires normalized upward rank feature 14")
            if self.heuristic_weight <= 0.0 or self.residual_limit < 0.0: raise ValueError("invalid high heuristic residual scales")
            nn.init.zeros_(self.actor.weight); nn.init.zeros_(self.actor.bias)
    def forward(self, task_features: torch.Tensor, task_mask: torch.Tensor, resource_features: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        if task_features.ndim!=3 or task_mask.shape!=task_features.shape[:2]: raise ValueError("invalid high-level task shapes")
        encoded=torch.tanh(self.task_encoder(task_features)); sequence=encoded if self.lstm is None else self.lstm(encoded)[0]; actor_logits=self.actor(sequence).squeeze(-1); logits=actor_logits
        if self.heuristic_residual:
            logits=self.heuristic_weight*task_features[...,14]+self.residual_limit*torch.tanh(actor_logits)
        denom=task_mask.sum(1,keepdim=True).clamp_min(1); pooled=(sequence*task_mask.unsqueeze(-1)).sum(1)/denom
        resource=resource_features.mean(1); value=self.critic(torch.cat((pooled,resource),-1)).squeeze(-1); return logits,value
