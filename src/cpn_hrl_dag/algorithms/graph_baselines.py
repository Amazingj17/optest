"""On-policy PPO with full proposal/coordinator records and a shared team critic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from cpn_hrl_dag.env.dag_env import CloudEdgeEndDAGEnv
from cpn_hrl_dag.models.dag_pair import StaticGraphEmbeddings
from cpn_hrl_dag.policies.dag_pair import graph_static_tensors
from cpn_hrl_dag.policies.graph_baselines import action_tensors, scenario_tiers


@dataclass
class TeamTransition:
    observation: dict
    actions: np.ndarray
    log_probs: np.ndarray
    active: np.ndarray
    value: float
    reward: float
    done: bool


def team_gae(records, gamma: float, gae_lambda: float):
    advantages = np.zeros(len(records), dtype=np.float32)
    next_value, advantage = 0.0, 0.0
    for index in range(len(records) - 1, -1, -1):
        record = records[index]
        continuation = 1.0 - float(record.done)
        delta = record.reward + gamma * next_value * continuation - record.value
        advantage = delta + gamma * gae_lambda * continuation * advantage
        advantages[index] = advantage
        next_value = record.value
    returns = advantages + np.asarray([record.value for record in records], dtype=np.float32)
    return advantages, returns


def clipped_team_loss(heads, actions, old_log_probs, advantages, returns, clip_coef, value_coef, entropy_coef):
    log_ratios = heads.log_prob(actions) - old_log_probs
    ratios = log_ratios.exp()
    surrogate = torch.minimum(ratios * advantages[:, None],
                              ratios.clamp(1.0 - clip_coef, 1.0 + clip_coef) * advantages[:, None])
    active = heads.active.to(surrogate.dtype)
    denominator = active.sum(-1).clamp_min(1)
    actor_loss = -((surrogate * active).sum(-1) / denominator).mean()
    entropy = ((heads.entropy() * active).sum(-1) / denominator).mean()
    value_loss = (heads.value - returns).square().mean()
    approximate_kl = ((((ratios - 1.0) - log_ratios) * active).sum(-1) / denominator).mean()
    return actor_loss + value_coef * value_loss - entropy_coef * entropy, approximate_kl


class GraphBaselineTrainer:
    def __init__(self, model, optimizer, device='cpu', *, normalize_observations=True,
                 gamma=1.0, gae_lambda=0.95, clip_coef=0.2, entropy_coef=0.01,
                 value_coef=0.5, max_grad_norm=0.5, update_epochs=4, batch_size=8,
                 target_kl=0.03) -> None:
        if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
            raise ValueError('gamma and gae_lambda must be in [0, 1]')
        if not 0 < clip_coef < 1 or batch_size <= 0 or update_epochs <= 0 or max_grad_norm <= 0:
            raise ValueError('invalid PPO update settings')
        if entropy_coef < 0 or value_coef < 0 or (target_kl is not None and target_kl <= 0):
            raise ValueError('invalid PPO loss settings')
        self.model, self.optimizer, self.device = model, optimizer, torch.device(device)
        self.normalize_observations = normalize_observations
        self.gamma, self.gae_lambda, self.clip_coef = gamma, gae_lambda, clip_coef
        self.entropy_coef, self.value_coef, self.max_grad_norm = entropy_coef, value_coef, max_grad_norm
        self.update_epochs, self.batch_size, self.target_kl = update_epochs, batch_size, target_kl

    def collect(self, scenario):
        self.model.eval()
        env = CloudEdgeEndDAGEnv(normalize_observations=self.normalize_observations)
        env.reset(scenario)
        initial = env.get_flat_observation()
        tiers = scenario_tiers(scenario)
        records = []
        with torch.no_grad():
            static = self.model.encode_static(*graph_static_tensors(initial, self.device))
            done = False
            while not done:
                observation = env.get_flat_observation()
                heads = self.model.action_heads(static, *action_tensors(observation, tiers, self.device))
                actions = heads.choose()
                pair = int(self.model.executed_pair(actions).item())
                task, node = divmod(pair, scenario.num_nodes)
                _, reward, done, truncated, info = env.step(task, node)
                if truncated:
                    raise RuntimeError('offline comparison episodes cannot truncate')
                records.append(TeamTransition(observation, actions[0].cpu().numpy(),
                                               heads.log_prob(actions)[0].cpu().numpy(),
                                               heads.active[0].cpu().numpy(), float(heads.value.item()),
                                               float(reward), done))
        return initial, tiers, records, info

    def update(self, initial, tiers, records):
        if not records or not records[-1].done:
            raise ValueError('PPO update requires a complete nonempty episode')
        self.model.train()
        advantages, returns = team_gae(records, self.gamma, self.gae_lambda)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        losses, divergences = [], []
        for _ in range(self.update_epochs):
            order = np.random.permutation(len(records))
            for start in range(0, len(order), self.batch_size):
                indices = order[start:start + self.batch_size]
                arguments = [action_tensors(records[index].observation, tiers, self.device) for index in indices]
                batched = tuple(torch.cat([item[position] for item in arguments], dim=0) for position in range(len(arguments[0])))
                static = self.model.encode_static(*graph_static_tensors(initial, self.device))
                static = StaticGraphEmbeddings(static.tasks.expand(len(indices), -1, -1),
                                               static.resources.expand(len(indices), -1, -1))
                heads = self.model.action_heads(static, *batched)
                old_active = torch.as_tensor(np.stack([records[index].active for index in indices]), device=self.device)
                if not torch.equal(heads.active, old_active):
                    raise RuntimeError('rollout and update agent masks differ')
                actions = torch.as_tensor(np.stack([records[index].actions for index in indices]), device=self.device)
                old_logs = torch.as_tensor(np.stack([records[index].log_probs for index in indices]), device=self.device)
                loss, divergence = clipped_team_loss(
                    heads, actions, old_logs, torch.as_tensor(advantages[indices], device=self.device),
                    torch.as_tensor(returns[indices], device=self.device), self.clip_coef, self.value_coef, self.entropy_coef,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError('non-finite graph baseline PPO loss')
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm, error_if_nonfinite=True)
                self.optimizer.step()
                losses.append(float(loss.item()))
                divergences.append(float(divergence.item()))
                if self.target_kl is not None and divergences[-1] > 1.5 * self.target_kl:
                    return {'loss': float(np.mean(losses)), 'approx_kl': float(np.mean(divergences))}
        return {'loss': float(np.mean(losses)), 'approx_kl': float(np.mean(divergences))}

    def episode(self, scenario):
        initial, tiers, records, info = self.collect(scenario)
        result = self.update(initial, tiers, records)
        return dict(result, reward=sum(record.reward for record in records), transitions=len(records),
                    makespan=float(info['final_makespan']), ratio=float(info['final_makespan']) / float(info['heft_makespan']))
