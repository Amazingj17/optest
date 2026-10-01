from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from cpn_hrl_dag.algorithms.graph_baselines import GraphBaselineTrainer, TeamTransition, clipped_team_loss, team_gae
from cpn_hrl_dag.env.dag_env import CloudEdgeEndDAGEnv
from cpn_hrl_dag.evaluation import Evaluator
from cpn_hrl_dag.experiments.graph_baselines import make_model
from cpn_hrl_dag.models.graph_baselines import GraphBaselineActorCritic
from cpn_hrl_dag.policies.dag_pair import graph_static_tensors
from cpn_hrl_dag.policies.graph_baselines import GraphBaselinePolicy, action_tensors, scenario_tiers
from cpn_hrl_dag.scenario.types import ComputeNode, Dependency, Scenario, Task


def scenario(only_cloud=False):
    nodes = [ComputeNode('device', 'end', 0.8, gpu_capacity=0.0),
             ComputeNode('edge', 'edge', 1.2, gpu_capacity=0.0),
             ComputeNode('cloud', 'cloud', 2.0, gpu_capacity=1.0)]
    if only_cloud:
        nodes = nodes[-1:]
    bandwidth = np.full((len(nodes), len(nodes)), 5.0)
    np.fill_diagonal(bandwidth, 1e9)
    return Scenario('baseline-unit', 'unit', [Task('a', 3.0), Task('b', 2.0), Task('c', 4.0), Task('d', 1.0)],
                    [Dependency('a', 'c', 1.0), Dependency('b', 'c', 2.0), Dependency('c', 'd', 1.0)],
                    nodes, bandwidth, metadata={'original_graph_id': 'baseline-unit'})


def model_observation(method, value=None):
    value = value or scenario()
    env = CloudEdgeEndDAGEnv(normalize_observations=True)
    env.reset(value)
    observation = env.get_flat_observation()
    config = dict(baseline=method, model=dict(hidden_dim=8, heads=2, task_layers=1, resource_layers=1))
    return make_model(config, observation, 'cpu'), observation, env


@pytest.mark.parametrize('method', GraphBaselineActorCritic.methods)
def test_pair_sampling_masks_and_deterministic_evaluator(method):
    model, observation, env = model_observation(method)
    static = model.encode_static(*graph_static_tensors(observation, torch.device('cpu')))
    heads = model.action_heads(static, *action_tensors(observation, scenario_tiers(env.scenario), torch.device('cpu')))
    assert len(heads.distributions) == (1 if method == 'graph_ppo' else 4)
    for _ in range(20):
        actions = heads.choose()
        pair = int(model.executed_pair(actions).item())
        assert observation['pair_mask'].reshape(-1)[pair]
        assert torch.isfinite(heads.log_prob(actions)).all()
        if method == 'tier_mappo':
            for tier in range(3):
                task, node = divmod(int(actions[0, tier]), env.scenario.num_nodes)
                assert scenario_tiers(env.scenario)[node] == tier
                assert observation['pair_mask'][task, node]
    first, _ = Evaluator({'normalize_observations': True}).evaluate(GraphBaselinePolicy(model), [env.scenario])
    second, _ = Evaluator({'normalize_observations': True}).evaluate(GraphBaselinePolicy(model), [env.scenario])
    assert first[0].valid_schedule and first[0].makespan == second[0].makespan
    assert first[0].policy == method


def test_missing_tiers_and_illegal_devices_are_inactive():
    value = replace(scenario(), tasks=[Task('a', 3.0, gpu_requirement=1.0)], dependencies=[])
    model, observation, env = model_observation('tier_mappo', value)
    static = model.encode_static(*graph_static_tensors(observation, torch.device('cpu')))
    heads = model.action_heads(static, *action_tensors(observation, scenario_tiers(value), torch.device('cpu')))
    assert heads.active.tolist() == [[False, False, True, True]]
    actions = heads.choose()
    assert actions[0, 3] == 2
    assert model.executed_pair(actions).item() == 2
    model, observation, env = model_observation('tier_mappo', scenario(only_cloud=True))
    records, _ = Evaluator({'normalize_observations': True}).evaluate(GraphBaselinePolicy(model), [env.scenario])
    assert records[0].valid_schedule


@pytest.mark.parametrize('method', GraphBaselineActorCritic.methods)
def test_rollout_update_probability_identity_and_gradients(method):
    torch.manual_seed(42)
    model, _, _ = model_observation(method)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = GraphBaselineTrainer(model, optimizer, update_epochs=1, batch_size=4, target_kl=None)
    initial, tiers, records, info = trainer.collect(scenario())
    assert len(records) == scenario().num_tasks
    assert records[-1].done and info['valid_schedule']
    assert sum(record.reward for record in records) == pytest.approx(-info['final_makespan'] / info['heft_makespan'])
    for record in records:
        static = model.encode_static(*graph_static_tensors(initial, torch.device('cpu')))
        heads = model.action_heads(static, *action_tensors(record.observation, tiers, torch.device('cpu')))
        logs = heads.log_prob(torch.as_tensor(record.actions).unsqueeze(0))
        np.testing.assert_allclose(logs.detach().numpy()[0], record.log_probs, atol=1e-6)
        assert np.array_equal(heads.active.numpy()[0], record.active)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    metrics = trainer.update(initial, tiers, records)
    assert np.isfinite(metrics['loss'])
    assert any(not torch.equal(before[name], parameter) for name, parameter in model.named_parameters())
    for name in ['task_encoder', 'resource_encoder', 'actors', 'critic'] + (['bidders'] if method == 'tier_mappo' else []):
        gradients = [parameter.grad for key, parameter in model.named_parameters() if key.startswith(name)]
        assert any(gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0 for gradient in gradients)


def test_team_gae_uses_terminal_boundary():
    records = [TeamTransition({}, np.zeros(1), np.zeros(1), np.ones(1), 0.0, -1.0, False),
               TeamTransition({}, np.zeros(1), np.zeros(1), np.ones(1), 0.0, -2.0, True)]
    advantages, returns = team_gae(records, 1.0, 1.0)
    np.testing.assert_allclose(advantages, [-3.0, -2.0])
    np.testing.assert_allclose(returns, advantages)


def test_inactive_actor_and_bidder_get_no_policy_gradient():
    model, observation, env = model_observation('tier_mappo', scenario(only_cloud=True))
    static = model.encode_static(*graph_static_tensors(observation, torch.device('cpu')))
    heads = model.action_heads(static, *action_tensors(observation, scenario_tiers(env.scenario), torch.device('cpu')))
    actions = heads.choose()
    loss, _ = clipped_team_loss(heads, actions, heads.log_prob(actions).detach(), torch.ones(1),
                               heads.value.detach(), 0.2, 0.0, 0.01)
    loss.backward()
    for tier in (0, 1):
        for parameter in list(model.actors[tier].parameters()) + list(model.bidders[tier].parameters()):
            assert parameter.grad is None or parameter.grad.abs().sum() == 0


@pytest.mark.parametrize('method', GraphBaselineActorCritic.methods)
def test_static_cache_matches_recomputation(method):
    model, initial, env = model_observation(method)
    static = model.encode_static(*graph_static_tensors(initial, torch.device('cpu')))
    env.step(0, 2)
    observation = env.get_flat_observation()
    arguments = action_tensors(observation, scenario_tiers(env.scenario), torch.device('cpu'))
    cached = model.action_heads(static, *arguments)
    fresh = model.action_heads(model.encode_static(*graph_static_tensors(initial, torch.device('cpu'))), *arguments)
    for left, right in zip(cached.distributions, fresh.distributions):
        torch.testing.assert_close(left.logits, right.logits)


def test_invalid_tiers_and_empty_actions_fail():
    model, observation, env = model_observation('tier_mappo')
    static = model.encode_static(*graph_static_tensors(observation, torch.device('cpu')))
    empty = copy.deepcopy(observation)
    empty['pair_mask'][:] = False
    with pytest.raises(ValueError, match='legal pair'):
        model.action_heads(static, *action_tensors(empty, scenario_tiers(env.scenario), torch.device('cpu')))
    with pytest.raises(ValueError, match='tiers'):
        model.action_heads(static, *action_tensors(observation, np.array([0, 1, 9]), torch.device('cpu')))
