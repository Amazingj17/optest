from __future__ import annotations

import numpy as np
import torch

from cpn_hrl_dag.algorithms.ppo import PPOAgent, RolloutBuffer


class TinyActorCritic(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(2, 2)
        self.value = torch.nn.Linear(2, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.layer(features), self.value(features).squeeze(-1)


def test_ppo_update_restores_training_mode_after_evaluation() -> None:
    model = TinyActorCritic()
    agent = PPOAgent(
        model,
        torch.optim.Adam(model.parameters(), lr=1e-3),
        lambda observation: (torch.as_tensor(observation["x"], dtype=torch.float32).unsqueeze(0),),
        model.forward,
        torch.device("cpu"),
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        entropy_coef=0.0,
    )
    observation = {"x": np.array([0.5, -0.5], dtype=np.float32)}
    model.eval()
    action, log_prob, value = agent.act(observation, np.array([True, True]))
    assert model.training
    buffer = RolloutBuffer()
    buffer.add(observation=observation, mask=np.array([True, True]), action=action, log_prob=log_prob, value=value, reward=1.0, done=True)
    model.eval()
    agent.update(buffer, epochs=1)
    assert model.training
