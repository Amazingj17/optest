"""Regression tests for correctness of the stage-one optimisation experiments."""
from __future__ import annotations

import copy
from dataclasses import replace
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from test_stage1_diagnostics import _Fixture, _scenario
from cpn_hrl_dag.algorithms.hierarchical_trainer import HierarchicalTrainer
from cpn_hrl_dag.algorithms.phases import decision_policy, parse_phase_schedule
from cpn_hrl_dag.baselines.heft import HEFTScheduler
from cpn_hrl_dag.datasets.split import SplitManifest
from cpn_hrl_dag.env.dag_env import CloudEdgeEndDAGEnv
from cpn_hrl_dag.evaluation.diagnostics import DecisionDiagnostics
from cpn_hrl_dag.policies.hrl import CPNHRLDAGPolicy, HRLExecutionModePolicy
from cpn_hrl_dag.utils.config import config_hash

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import train_main_comparison as runner
from train import make_agents
from prepare_stage1_forks import build_branch_config
from run_server_comparison import build_configs


def _config(tmp_path):
    config = build_configs(tmp_path, device='cpu', threads=1)['residual_hrl']
    config['model']['high']['hidden_dim'] = 8
    config['model']['low'].update(hidden_dim=8, heads=2)
    config['training'].update(high_train_episodes=2, joint_train_episodes=0, diagnostics=True)
    config['training']['ppo'].update(update_epochs=2, batch_size=2, target_kl=None)
    config['evaluation']['bootstrap_samples'] = 0
    return config


def test_legacy_low_pretrain_never_calls_the_high_network(monkeypatch):
    scenario = _scenario()
    # Multiple ready tasks make sampling the frozen network observably different.
    scenario = replace(scenario, dependencies=[])
    fixture = _Fixture(scenario, diagnostics=True)

    def forbidden(*args, **kwargs):
        raise AssertionError('legacy low_pretrain must use fixed HEFT task ranks')

    monkeypatch.setattr(fixture.high, 'act_with_stats', forbidden)
    monkeypatch.setattr(fixture.high, 'greedy_action', forbidden)
    seen = []
    original = CloudEdgeEndDAGEnv.select_task

    def capture(env, task):
        ranks = env.upward_rank
        expected = min(np.flatnonzero(env.get_ready_mask()), key=lambda x: (-ranks[x], int(x)))
        assert task == expected
        seen.append(task)
        return original(env, task)

    monkeypatch.setattr(CloudEdgeEndDAGEnv, 'select_task', capture)
    fixture.trainer._episode(scenario, 'low_pretrain')
    assert len(seen) == scenario.num_tasks
    assert fixture.high.lr_scheduler.last_epoch == 0


def test_default_frozen_high_uses_deployment_ties_without_sampling(monkeypatch):
    scenario = _scenario()
    scenario = replace(scenario, dependencies=[])
    fixture = _Fixture(scenario)
    assert decision_policy('low_only_frozen_high').frozen_high_mode == 'deterministic'
    calls = []
    original = CPNHRLDAGPolicy.select_task

    def capture(policy, observation, mask, deterministic=True):
        assert deterministic
        action = original(policy, observation, mask, deterministic)
        calls.append(action)
        return action

    monkeypatch.setattr(CPNHRLDAGPolicy, 'select_task', capture)
    fixture.trainer._episode(scenario, 'low_only_frozen_high')
    assert len(calls) == scenario.num_tasks
    assert not fixture.high.model.training
    assert all(p.grad is None for p in fixture.high.model.parameters())


def test_model_greedy_and_eft_are_separate_diagnostics():
    stats = {'argmax_action': 1, 'masked_entropy': .6, 'chosen_probability': .7}
    diagnostics = DecisionDiagnostics()
    diagnostics.record_low(np.array([True, True]), np.array([1., 2.]), 1, stats)
    result = diagnostics.aggregate()
    assert result['low_deviation_rate_from_greedy'] == 0
    assert result['low_deviation_rate_from_min_eft'] == 1
    assert result['low_suboptimal_rate_multi'] == 1
    diagnostics.record_low(np.array([True, False]), np.array([1., 2.]), 0, dict(stats, argmax_action=0))
    result = diagnostics.aggregate()
    assert result['low_single_legal_action_decisions'] == 1
    assert result['low_suboptimal_rate_active'] == .5
    assert result['low_suboptimal_rate_multi'] == 1


def test_diagnostics_off_does_not_compute_action_statistics(monkeypatch):
    fixture = _Fixture(_scenario())

    def forbidden(*args, **kwargs):
        raise AssertionError('diagnostics are disabled')

    monkeypatch.setattr(fixture.high, '_action_stats', forbidden)
    monkeypatch.setattr(fixture.low, '_action_stats', forbidden)
    fixture.trainer._episode(fixture.scenario, 'joint')


def test_diagnostic_toggle_preserves_actions_and_updated_parameters(monkeypatch):
    scenario = replace(_scenario(), dependencies=[])
    original = CloudEdgeEndDAGEnv.step
    traces, snapshots, tails = [], [], []
    for enabled in (False, True):
        trace = []
        def capture(env, task, node):
            trace.append((task, node))
            return original(env, task, node)
        monkeypatch.setattr(CloudEdgeEndDAGEnv, 'step', capture)
        fixture = _Fixture(scenario, diagnostics=enabled)
        fixture.trainer._episode(scenario, 'high_only_eft')
        fixture.trainer._episode(scenario, 'joint')
        traces.append(trace)
        snapshots.append(fixture.snapshot())
        tails.append(torch.rand(4))
    assert traces[0] == traces[1]
    assert torch.equal(tails[0], tails[1])
    for level in ('high', 'low'):
        assert all(torch.equal(a, b) for a, b in zip(snapshots[0][level], snapshots[1][level]))


def test_freezing_clears_stale_gradients_and_preserves_populated_optimizer():
    fixture = _Fixture(_scenario())
    fixture.trainer._episode(fixture.scenario, 'joint')
    old = copy.deepcopy(fixture.high.optimizer.state_dict())
    scheduler = copy.deepcopy(fixture.high.lr_scheduler.state_dict())
    fixture.trainer._episode(fixture.scenario, 'low_only_frozen_high')
    new = fixture.high.optimizer.state_dict()
    for key, state in old['state'].items():
        for name, value in state.items():
            assert torch.equal(value, new['state'][key][name])
    assert scheduler == fixture.high.lr_scheduler.state_dict()
    assert all(p.grad is None for p in fixture.high.model.parameters())


def test_fresh_fork_scheduler_starts_at_declared_learning_rate(tmp_path):
    config = _config(tmp_path)
    high, low = make_agents(_scenario(), config, torch.device('cpu'))
    trainer = HierarchicalTrainer(high, low, torch.device('cpu'), normalize_observations=True)
    trainer._episode(_scenario(), 'joint')
    checkpoint = tmp_path / 'source.pt'
    trainer.save(checkpoint, high, low, config, 864, .1)
    assert high.learning_rate < config['model']['high']['learning_rate']
    fork = build_branch_config(config, 'low_only_frozen_high', tmp_path / 'branch', 2, 1, True)
    new_high, new_low = make_agents(_scenario(), fork, torch.device('cpu'))
    runner._restore_for_training(fork, new_high, new_low, checkpoint, 'cpu')
    assert new_high.learning_rate == config['model']['high']['learning_rate']
    assert new_low.learning_rate == config['model']['low']['learning_rate']
    assert new_low.lr_scheduler.last_epoch == 0
    assert new_low.lr_scheduler.total_iters == 2
    assert new_low.optimizer.state  # moments restored, while scheduler reset
    payload = torch.load(checkpoint, weights_only=False)
    payload['global_step'] = 2592
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match='step 864'):
        runner._restore_for_training(fork, new_high, new_low, checkpoint, 'cpu')


def test_fork_revalidates_and_retains_starting_best_and_counts_steps(tmp_path, monkeypatch):
    config = _config(tmp_path)
    samples = [_scenario('train-a'), _scenario('train-b')]
    manifest = SplitManifest(('unit:a', 'unit:b'), ('unit:v',), (), 7)
    monkeypatch.setattr(runner, 'load_fixed_splits', lambda _: ({'train': samples, 'validation': [_scenario('val')]}, manifest))
    monkeypatch.setattr(runner, 'write_training_curve', lambda *args: None)
    monkeypatch.setattr(runner, 'write_report', lambda *args: None)
    high, low = make_agents(samples[0], config, torch.device('cpu'))
    source = tmp_path / 'source.pt'
    # Historical best intentionally differs from these weights' actual ratio.
    HierarchicalTrainer.save(source, high, low, config, 864, .01)
    original_evaluate = runner.Evaluator.evaluate
    orders = []
    for name in ('high_only_eft', 'low_only_frozen_high'):
        calls = []

        def evaluation(evaluator, policy, scenarios):
            records, summary = original_evaluate(evaluator, policy, scenarios)
            calls.append(summary['mean_ratio'])
            if len(calls) == 2:
                summary['mean_ratio'] += 10  # forced regression at epoch end
            return records, summary

        monkeypatch.setattr(runner.Evaluator, 'evaluate', evaluation)
        fork = build_branch_config(config, name, tmp_path / name, 2, 1, True)
        path = runner.train_main(fork, source)
        selected = torch.load(path, weights_only=False)
        latest = torch.load(path.parent / 'latest.pt', weights_only=False)
        assert selected['global_step'] == 864
        assert selected['best_validation_ratio'] == pytest.approx(calls[0])
        for key, value in high.model.state_dict().items():
            assert torch.equal(selected['high_model'][key], value)
        for key, value in low.model.state_dict().items():
            assert torch.equal(selected['low_model'][key], value)
        budget = json.loads((path.parent / 'budget.json').read_text())['executed']
        active = 'high' if name == 'high_only_eft' else 'low'
        frozen = 'low' if active == 'high' else 'high'
        actual_steps = max(int(state['step']) for state in latest[f'{active}_optimizer']['state'].values())
        assert budget['optimizer_updates'][active] == actual_steps > 2
        assert budget['ppo_update_calls'][active] == 2
        assert budget['optimizer_updates'][frozen] == 0
        with (path.parent / 'train_log.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        orders.append([row['scenario_id'] for row in rows])
        assert (path.parent / 'diagnostics_by_phase.json').is_file()
        with pytest.raises(FileExistsError):
            runner.train_main(fork, source)
    assert orders[0] == orders[1]


def test_explicit_schedule_rejects_irrelevant_frozen_mode():
    with pytest.raises(ValueError, match='only valid'):
        parse_phase_schedule([{'name': 'joint', 'episodes': 1, 'frozen_high_mode': 'deterministic'}])


def test_fork_requires_source_and_preflight_never_deletes(tmp_path):
    from preflight_stage1_run import _check_output_dir
    target = tmp_path / 'existing'
    target.mkdir()
    (target / 'keep').write_text('important')
    with pytest.raises(SystemExit):
        _check_output_dir(target)
    with pytest.raises(ValueError):
        _check_output_dir(target, True)
    assert (target / 'keep').read_text() == 'important'
    config = _config(tmp_path)
    config['fork'] = {'require_checkpoint': True}
    with pytest.raises(ValueError, match='requires --fork-from'):
        runner.train_main(config)
