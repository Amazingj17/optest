"""Flat-PPO small-scale training path used only as an ablation baseline."""
from __future__ import annotations

import torch

from cpn_hrl_dag.algorithms.ppo import PPOAgent, RolloutBuffer
from cpn_hrl_dag.env.dag_env import CloudEdgeEndDAGEnv
from cpn_hrl_dag.scenario.types import Scenario


def flat_tensors(obs: dict, device: torch.device):
    return (
        torch.as_tensor(obs["task_features"], device=device).unsqueeze(0),
        torch.as_tensor(obs["task_mask"], device=device).unsqueeze(0),
        torch.as_tensor(obs["resource_features"], device=device).unsqueeze(0),
        torch.as_tensor(obs["pair_node_features"], device=device).unsqueeze(0),
    )


class FlatPPOTrainer:
    def __init__(self, agent: PPOAgent, device: torch.device) -> None:
        self.agent, self.device = agent, device

    def episode(self, scenario: Scenario) -> dict[str, float]:
        env = CloudEdgeEndDAGEnv()
        env.reset(scenario)
        buffer, total, done = RolloutBuffer(), 0.0, False
        while not done:
            observation = env.get_flat_observation()
            mask = observation["pair_mask"].reshape(-1)
            action, log_prob, value = self.agent.act(observation, mask)
            task, node = divmod(action, scenario.num_nodes)
            _, reward, done, truncated, _ = env.step(int(task), int(node))
            if truncated:
                raise RuntimeError("offline Flat PPO episodes cannot truncate")
            buffer.add(observation=observation, mask=mask, action=action, log_prob=log_prob, value=value, reward=reward, done=done)
            total += reward
        return {"reward": total, "loss": self.agent.update(buffer)["loss"]}
