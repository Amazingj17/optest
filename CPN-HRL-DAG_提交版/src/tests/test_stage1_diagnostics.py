"""Stage-one behaviour tests for training diagnostics and explicit phase semantics.

The tests cover the eight behaviours the stage-one specification requires:

1. identical sampled trajectories with diagnostics off and on;
2. frozen modules keep their parameters while active modules update;
3. frozen modules never advance their optimizer/scheduler;
4. sampled action, stored log-probability and mask describe one distribution;
5. correct handling of a single legal action and of exact EFT ties;
6. default legacy configuration and legacy checkpoints still load;
7. a new experiment can never overwrite an existing result directory;
8. a small smoke run trains, saves, loads and evaluates.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from cpn_hrl_dag.algorithms.hierarchical_trainer import HierarchicalTrainer, high_tensors, low_tensors
from cpn_hrl_dag.algorithms.phases import (
    DecisionPolicy,
    decision_policy,
    parse_phase_schedule,
    resolve_phase_schedule,
    summarize_phase_budget,
)
from cpn_hrl_dag.algorithms.ppo import PPOAgent, RolloutBuffer, explained_variance, masked_distribution
from cpn_hrl_dag.env.dag_env import CloudEdgeEndDAGEnv
from cpn_hrl_dag.evaluation import Evaluator
from cpn_hrl_dag.evaluation.diagnostics import DecisionDiagnostics, min_eft_choice
from cpn_hrl_dag.experiments.graph_baselines import prepare_output
from cpn_hrl_dag.models.high_lstm import HighLevelLSTMActorCritic
from cpn_hrl_dag.models.low_gat import LowLevelGATActorCritic
from cpn_hrl_dag.policies.hrl import CPNHRLDAGPolicy
from cpn_hrl_dag.scenario.types import ComputeNode, Dependency, Scenario, Task
from cpn_hrl_dag.scheduling.communication_model import MatrixCommunicationModel
from cpn_hrl_dag.scheduling.execution_model import SimpleSpeedExecutionModel
from cpn_hrl_dag.utils.config import config_hash
from cpn_hrl_dag.utils.seed import seed_everything

TASK_IDS = ('t0', 't1', 't2', 't3', 't4', 't5')


def _scenario(identifier: str = 'stage1', tasks: int = 6, nodes: int = 3) -> Scenario:
    """Small deterministic DAG with real speeds and a real bandwidth matrix."""
    return Scenario(
        scenario_id=identifier,
        dataset_source='stage1_unit',
        tasks=[Task(TASK_IDS[index], 1.0 + index) for index in range(tasks)],
        dependencies=[Dependency(TASK_IDS[index], TASK_IDS[index + 1], 1.0) for index in range(tasks - 1)],
        compute_nodes=[ComputeNode(f'n{index}', 'cloud', 1.0 + index) for index in range(nodes)],
        bandwidth_matrix=np.full((nodes, nodes), 5.0),
        metadata={'original_graph_id': identifier},
    )


class _Fixture:
    """Tiny but real two-level agents wired exactly like `scripts/train.make_agents`."""

    def __init__(self, scenario: Scenario, *, diagnostics: bool = False, entropy_coef: float = 0.05):
        seed_everything(7, disable_cudnn=True)
        self.scenario = scenario
        self.device = torch.device('cpu')
        env = CloudEdgeEndDAGEnv(normalize_observations=True)
        high_observation, _ = env.reset(scenario)
        self.task_dim = high_observation['task_features'].shape[1]
        ready = int(env.get_ready_mask().nonzero()[0][0])
        env.select_task(ready)
        low_observation = env.get_low_observation(ready)
        self.node_dim = low_observation['node_features'].shape[1]

        def build():
            high_model = HighLevelLSTMActorCritic(self.task_dim, high_observation['resource_features'].shape[1], 16,
                                                 True, True, 8.0, 0.5)
            low_model = LowLevelGATActorCritic(self.node_dim, self.task_dim, 16, 2, True, True, 8.0, 0.5)
            high_optimizer = torch.optim.Adam(high_model.parameters(), lr=1e-3)
            low_optimizer = torch.optim.Adam(low_model.parameters(), lr=1e-3)
            high_scheduler = torch.optim.lr_scheduler.LinearLR(high_optimizer, start_factor=1.0, end_factor=0.5, total_iters=8)
            low_scheduler = torch.optim.lr_scheduler.LinearLR(low_optimizer, start_factor=1.0, end_factor=0.5, total_iters=8)
            high = PPOAgent(high_model, high_optimizer, lambda obs: high_tensors(obs, self.device), high_model.forward,
                            self.device, 1.0, 0.95, 0.2, entropy_coef, update_epochs=1, batch_size=16,
                            lr_scheduler=high_scheduler,
                            lr_scheduler_factory=lambda: torch.optim.lr_scheduler.LinearLR(high_optimizer, start_factor=1.0, end_factor=0.5, total_iters=8))
            low = PPOAgent(low_model, low_optimizer, lambda obs: low_tensors(obs, self.device), low_model.forward,
                           self.device, 1.0, 0.95, 0.2, entropy_coef, update_epochs=1, batch_size=16,
                           lr_scheduler=low_scheduler,
                           lr_scheduler_factory=lambda: torch.optim.lr_scheduler.LinearLR(low_optimizer, start_factor=1.0, end_factor=0.5, total_iters=8))
            return high, low

        self.high, self.low = build()
        self.rows: list[dict] = []
        self.trainer = HierarchicalTrainer(self.high, self.low, self.device, None, {}, True, True,
                                           diagnostics=self.rows.append if diagnostics else None)

    def snapshot(self) -> dict[str, list[torch.Tensor]]:
        return {name: [parameter.detach().clone() for parameter in agent.model.parameters()]
                for name, agent in (('high', self.high), ('low', self.low))}


def _parameters_equal(before, after) -> bool:
    return all(torch.equal(left, right) for left, right in zip(before, after))


# --------------------------------------------------------------------------- 1
def _run_fixed_seed(scenario: Scenario, phase: str, diagnostics: bool, seed: int = 11) -> tuple[dict, list[float], list[float]]:
    """Run one episode from a fixed seed and return its row plus the RNG tail."""
    fixture = _Fixture(scenario, diagnostics=diagnostics)
    seed_everything(seed, disable_cudnn=True)
    row = fixture.trainer._episode(scenario, phase)
    return row, np.random.random(8).tolist(), torch.rand(8).tolist()


def test_diagnostics_do_not_change_sampled_actions_or_the_rng_stream():
    scenario = _scenario()
    plain_row, plain_numpy_tail, plain_torch_tail = _run_fixed_seed(scenario, 'joint', diagnostics=False)
    instrumented_row, instrumented_numpy_tail, instrumented_torch_tail = _run_fixed_seed(scenario, 'joint', diagnostics=True)
    for key in ('reward', 'final_makespan', 'makespan_ratio', 'decisions', 'high_loss', 'low_loss'):
        assert plain_row[key] == instrumented_row[key], key
    assert plain_numpy_tail == instrumented_numpy_tail
    assert plain_torch_tail == instrumented_torch_tail
    assert not any(key.startswith('diag_') for key in plain_row)
    assert instrumented_row['diag_decisions'] == instrumented_row['decisions'] > 0
    assert instrumented_row['diag_makespan_over_heft'] == pytest.approx(instrumented_row['makespan_ratio'])
    assert instrumented_row['diag_reward_equals_negative_makespan_ratio'] is True


def test_diagnostics_disable_themselves_without_extra_random_draws():
    scenario = _scenario()
    first, first_numpy, _ = _run_fixed_seed(scenario, 'high_only_eft', diagnostics=True, seed=21)
    second, second_numpy, _ = _run_fixed_seed(scenario, 'high_only_eft', diagnostics=True, seed=21)
    assert first['final_makespan'] == second['final_makespan']
    assert first_numpy == second_numpy
    assert first['diag_high_module_frozen'] is False
    assert first['diag_low_module_frozen'] is True
    assert first['diag_low_loss'] is None
    assert first['diag_low_entropy_active'] is None
    assert first['diag_low_deviation_rate_from_min_eft'] == 0.0
    assert first['diag_low_min_eft_selected_decisions'] == first['decisions']


# --------------------------------------------------------------------------- 2
@pytest.mark.parametrize('phase,updating,frozen', [('high_only_eft', 'high', 'low'),
                                                  ('low_only_frozen_high', 'low', 'high')])
def test_frozen_module_parameters_stay_identical_while_active_module_updates(phase, updating, frozen):
    fixture = _Fixture(_scenario(), diagnostics=True, entropy_coef=0.05)
    before = fixture.snapshot()
    for _ in range(2):
        fixture.trainer._episode(fixture.scenario, phase)
    after = fixture.snapshot()
    assert _parameters_equal(before[frozen], after[frozen])
    assert not _parameters_equal(before[updating], after[updating])
    assert fixture.trainer.high.frozen is (frozen == 'high')
    assert fixture.trainer.low.frozen is (frozen == 'low')
    # A frozen level stays in evaluation mode for the whole episode; the active
    # level is left in training mode by its own PPO update.
    assert getattr(fixture, frozen).model.training is False
    assert getattr(fixture, updating).model.training is True
    row = fixture.rows[-1]
    assert row[f'diag_{frozen}_module_frozen'] is True
    assert row[f'diag_{frozen}_loss'] is None and row[f'diag_{frozen}_approx_kl'] is None
    assert row[f'diag_{frozen}_explained_variance'] is None
    assert row[f'diag_{updating}_loss'] is not None


# --------------------------------------------------------------------------- 3
def test_frozen_module_optimizer_and_scheduler_never_advance():
    fixture = _Fixture(_scenario(), diagnostics=True)
    fixture.high.lr_scheduler.last_epoch = 5
    frozen_scheduler_epoch = fixture.high.lr_scheduler.last_epoch
    frozen_lr = fixture.high.optimizer.param_groups[0]['lr']
    before = fixture.snapshot()
    for _ in range(2):
        fixture.trainer._episode(fixture.scenario, 'low_only_frozen_high')
    assert fixture.high.lr_scheduler.last_epoch == frozen_scheduler_epoch
    assert fixture.high.optimizer.param_groups[0]['lr'] == frozen_lr
    assert fixture.low.lr_scheduler.last_epoch == 2
    assert _parameters_equal(before['high'], fixture.snapshot()['high'])
    assert not any(parameter.requires_grad for parameter in fixture.high.model.parameters())
    with pytest.raises(RuntimeError):
        fixture.high.update(RolloutBuffer())


def test_high_only_eft_never_touches_the_low_optimizer():
    fixture = _Fixture(_scenario(), diagnostics=True)
    low_optimizer_state = json.dumps({key: str(value) for key, value in fixture.low.optimizer.state_dict()['state'].items()})
    before = fixture.snapshot()
    fixture.trainer._episode(fixture.scenario, 'high_only_eft')
    assert fixture.low.lr_scheduler.last_epoch == 0
    assert _parameters_equal(before['low'], fixture.snapshot()['low'])
    assert json.dumps({key: str(value) for key, value in fixture.low.optimizer.state_dict()['state'].items()}) == low_optimizer_state


# --------------------------------------------------------------------------- 4
def test_sampled_action_log_probability_and_mask_describe_one_distribution():
    seed_everything(5, disable_cudnn=True)
    model = HighLevelLSTMActorCritic(17, 5, 16)
    agent = PPOAgent(model, torch.optim.Adam(model.parameters(), lr=1e-3),
                     lambda observation: (torch.as_tensor(observation['task_features']).unsqueeze(0),
                                          torch.as_tensor(observation['task_mask']).unsqueeze(0),
                                          torch.as_tensor(observation['resource_features']).unsqueeze(0)),
                     model.forward, torch.device('cpu'), 1.0, 0.95, 0.2, 0.0)
    task_features = np.random.default_rng(0).normal(size=(4, 17)).astype(np.float32)
    mask = np.array([True, False, True, True])
    observation = {'task_features': task_features, 'task_mask': np.ones(4, dtype=bool),
                   'resource_features': np.ones((2, 5), dtype=np.float32)}
    captured: list[dict[str, float]] = []
    action, log_prob, _ = agent.act_with_stats(observation, mask, hook=captured.append)
    assert mask[action]
    logits, _ = model(torch.as_tensor(task_features).unsqueeze(0), torch.ones(1, 4, dtype=torch.bool), torch.ones(1, 2, 5))
    distribution = masked_distribution(logits, torch.as_tensor(mask).unsqueeze(0))
    assert float(distribution.log_prob(torch.tensor([action])).item()) == pytest.approx(log_prob)
    assert captured[0]['chosen_log_prob'] == pytest.approx(log_prob)
    assert captured[0]['argmax_action'] == int(torch.argmax(distribution.logits, 1).item())
    assert mask[int(captured[0]['argmax_action'])]
    assert np.isfinite(captured[0]['masked_entropy']) and captured[0]['masked_entropy'] >= 0.0
    buffer = RolloutBuffer()
    buffer.add(observation=observation, mask=mask, action=action, log_prob=log_prob, value=0.0, reward=-1.0, done=True)
    masks = torch.as_tensor(np.stack([item.mask for item in buffer.items]), dtype=torch.bool)
    replay = masked_distribution(logits, masks)
    assert float(replay.log_prob(torch.as_tensor([item.action for item in buffer.items])).item()) == pytest.approx(buffer.items[0].log_prob)


def test_explained_variance_handles_zero_variance_targets_explicitly():
    assert explained_variance(np.array([1.0, 2.0]), np.array([5.0, 5.0])) == 0.0
    assert explained_variance(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == pytest.approx(1.0)
    assert explained_variance(np.array([2.0, 2.0]), np.array([1.0, 3.0])) == pytest.approx(0.0)


# --------------------------------------------------------------------------- 5
def test_single_legal_action_and_eft_tie_reporting():
    diagnostics = DecisionDiagnostics()
    ready = np.array([True, False, False])
    priority = np.array([0.5, 0.9, 0.1])
    diagnostics.record_high(ready, greedy_action=0, chosen_action=0, stats={'masked_entropy': 0.0, 'expected_log_prob': 0.0, 'chosen_probability': 1.0}, rank_priority=priority)
    assert diagnostics.single_legal_action_decisions == 1
    assert diagnostics.high_deviation_from_greedy == 0
    aggregate = diagnostics.aggregate()
    assert aggregate['high_deviation_rate_from_greedy'] == 0.0
    assert aggregate['high_deviation_rate_from_heft_rank'] == 0.0

    tied = DecisionDiagnostics()
    node_mask = np.array([True, True, True])
    heuristic_eft = np.array([3.0, 1.0, 1.0])
    tied.record_low(node_mask, heuristic_eft, chosen_action=2, stats=None)
    assert tied.low_min_eft_tie == 1
    assert tied.low_min_eft_selected == 1
    assert tied.low_suboptimal == 0
    assert tied.low_eft_gaps == [0.0]
    assert tied.low_deviation_flags == [1]  # tied optimum, but not the lowest-index tie-break
    tied_summary = tied.aggregate()
    assert tied_summary['low_suboptimal_decisions'] == 0
    assert tied_summary['low_suboptimal_rate_active'] == 0.0
    assert tied_summary['low_min_eft_tie_decisions'] == 1
    assert tied_summary['low_min_eft_selected_decisions'] == 1
    assert tied_summary['low_deviation_rate_from_greedy'] is None
    assert tied_summary['low_deviation_rate_from_min_eft'] == 1.0

    exact_greedy = DecisionDiagnostics()
    exact_greedy.record_low(node_mask, heuristic_eft, chosen_action=1, stats=None)
    assert exact_greedy.low_deviation_flags == [0]
    assert exact_greedy.aggregate()['low_deviation_rate_from_greedy'] is None
    assert exact_greedy.aggregate()['low_deviation_rate_from_min_eft'] == 0.0

    suboptimal = DecisionDiagnostics()
    suboptimal.record_low(node_mask, heuristic_eft, chosen_action=0, stats=None)
    assert suboptimal.low_min_eft_tie == 1
    assert suboptimal.low_min_eft_selected == 0
    assert suboptimal.low_suboptimal == 1
    assert suboptimal.low_eft_gaps == [2.0]
    assert suboptimal.low_deviation_flags == [1]
    aggregate = suboptimal.aggregate()
    assert aggregate['low_suboptimal_decisions'] == 1
    assert aggregate['low_suboptimal_rate_active'] == 1.0
    assert aggregate['low_selected_eft_gap_mean_active'] == pytest.approx(2.0)

    minimum, optimal_count, greedy_action, is_optimal, deviation = min_eft_choice(heuristic_eft, node_mask, 1)
    assert (minimum, optimal_count, greedy_action, is_optimal, deviation) == (1.0, 2, 1, 1, 0.0)
    with pytest.raises(ValueError):
        min_eft_choice(heuristic_eft, np.array([True, False, False]), 2)


def test_high_rank_tie_is_counted_separately():
    diagnostics = DecisionDiagnostics()
    ready = np.array([True, True, False])
    priority = np.array([0.7, 0.7, 0.0])
    diagnostics.record_high(ready, greedy_action=0, chosen_action=1, stats={'masked_entropy': 1.0, 'expected_log_prob': -0.7, 'chosen_probability': 0.5}, rank_priority=priority)
    aggregate = diagnostics.aggregate()
    assert aggregate['high_rank_tie_decisions'] == 1
    assert aggregate['high_deviation_rate_from_heft_rank'] == 0.0  # a tied rank is not a deviation


def test_missing_exact_eft_is_reported_instead_of_guessed():
    diagnostics = DecisionDiagnostics()
    diagnostics.record_low(np.array([True, True]), None, chosen_action=1, stats=None)
    aggregate = diagnostics.aggregate()
    assert aggregate['low_eft_field_missing_decisions'] == 1
    assert aggregate['low_suboptimal_rate_active'] is None
    assert aggregate['low_selected_eft_gap_mean_active'] is None
    assert aggregate['diagnostic_warning_count'] == 1


# --------------------------------------------------------------------------- 6
def test_legacy_phase_labels_keep_their_original_meaning():
    assert decision_policy('high_train') is decision_policy('high_only_eft')
    assert decision_policy(False) is decision_policy('joint')
    assert decision_policy(True) is decision_policy('low_pretrain')
    legacy = {'training': {'low_pretrain_episodes': 0, 'high_train_episodes': 4, 'joint_train_episodes': 6}}
    schedule = resolve_phase_schedule(legacy)
    assert [item['name'] for item in schedule] == ['high_only_eft', 'joint']
    assert [item['episodes'] for item in schedule] == [4, 6]
    assert schedule[0]['policy'].low == 'fixed_min_eft'
    assert schedule[1]['policy'].update_high and schedule[1]['policy'].update_low
    budget = summarize_phase_budget(schedule)
    assert budget['episodes'] == 10
    assert budget['optimizer_updates'] == {'high': 10, 'low': 6}
    assert budget['fixed_low_rule_episodes'] == 4
    assert budget['frozen_high_policy_activation_episodes'] == 0
    assert budget['high_ppo_sampling_episodes'] == 10
    assert budget['low_ppo_sampling_episodes'] == 6


def test_explicit_phase_schedule_validates_names_and_frozen_modes():
    schedule = parse_phase_schedule([{'name': 'low_only_frozen_high', 'episodes': 3, 'frozen_high_mode': 'deterministic'}])
    assert schedule[0]['policy'].frozen_high_mode == 'deterministic'
    assert schedule[0]['policy'].high_is_frozen
    with pytest.raises(ValueError):
        parse_phase_schedule([{'name': 'not_a_phase', 'episodes': 1}])
    with pytest.raises(ValueError):
        parse_phase_schedule([{'name': 'joint', 'episodes': 1, 'unknown_key': 1}])
    with pytest.raises(ValueError):
        DecisionPolicy(name='broken', high='ppo_sample', low='ppo_sample', update_high=True, update_low=False)


def test_legacy_checkpoint_without_training_state_still_loads(scratch):
    fixture = _Fixture(_scenario())
    config = {'experiment': {'seed': 7}, 'training': {'high_train_episodes': 1, 'joint_train_episodes': 0}}
    path = scratch / 'legacy.pt'
    torch.save({'high_model': fixture.high.model.state_dict(), 'low_model': fixture.low.model.state_dict(),
                'high_optimizer': fixture.high.optimizer.state_dict(), 'low_optimizer': fixture.low.optimizer.state_dict(),
                'high_lr_scheduler': fixture.high.lr_scheduler.state_dict(), 'low_lr_scheduler': fixture.low.lr_scheduler.state_dict(),
                'global_step': 864, 'config': config, 'config_hash': config_hash(config),
                'best_validation_ratio': 0.5}, path)
    restored = HierarchicalTrainer.load(path, fixture.high, fixture.low, 'cpu')
    assert restored['global_step'] == 864
    assert 'training_state' not in restored
    assert restored['config_hash'] == config_hash(config)


def test_legacy_and_local_checkpoint_formats_are_readable():
    from cpn_hrl_dag.algorithms.hierarchical_trainer import HierarchicalTrainer as Trainer
    assert Trainer.config_hash({'runtime': {'a': 1}, 'x': 1}) == config_hash({'x': 1})


# --------------------------------------------------------------------------- 7
def test_new_experiment_refuses_to_overwrite_existing_results(scratch):
    directory = scratch / 'run'
    (directory / 'nested').mkdir(parents=True, exist_ok=True)
    (directory / 'nested' / 'summary.json').write_text('{}', encoding='utf-8')
    with pytest.raises(FileExistsError):
        prepare_output(directory)
    fresh = prepare_output(scratch / 'brand_new')
    assert fresh.is_dir() and not any(fresh.iterdir())


def test_diagnostics_recorder_writes_incremental_files(scratch):
    from cpn_hrl_dag.evaluation import DiagnosticsRecorder
    recorder = DiagnosticsRecorder(scratch)
    recorder.record({'step': 1, 'phase': 'high_only_eft', 'diag_decisions': 3, 'diag_high_module_frozen': False})
    recorder.record({'step': 2, 'phase': 'high_only_eft', 'diag_decisions': 4, 'diag_high_module_frozen': False})
    assert recorder.jsonl_path.read_text(encoding='utf-8').count('\n') == 2
    header = recorder.csv_path.read_text(encoding='utf-8').splitlines()
    assert header[0].startswith('step,phase,diag_decisions')
    assert len(header) == 3
    recorder.close()
    duplicate = DiagnosticsRecorder(scratch)
    with pytest.raises(FileExistsError):
        duplicate.record({'step': 3})


# --------------------------------------------------------------------------- 8
def test_smoke_training_save_load_and_evaluation(scratch):
    scenario = _scenario('smoke', tasks=5, nodes=2)
    fixture = _Fixture(scenario, diagnostics=True)
    schedule = parse_phase_schedule([{'name': 'high_only_eft', 'episodes': 2}, {'name': 'joint', 'episodes': 2}])
    for item in schedule:
        fixture.trainer._episode(scenario, item['policy'])
    config = {'experiment': {'seed': 7}, 'environment': {'normalize_observations': True},
              'training': {'phases': [{'name': 'high_only_eft', 'episodes': 2}]}}
    checkpoint = scratch / 'latest.pt'
    HierarchicalTrainer.save(checkpoint, fixture.high, fixture.low, config, 4, 0.9,
                             training_state={'episodes_completed': 4, 'exact_resume_supported': False})
    restored = HierarchicalTrainer.load(checkpoint, fixture.high, fixture.low, 'cpu')
    assert restored['global_step'] == 4
    assert restored['training_state']['episodes_completed'] == 4
    assert restored['training_state']['exact_resume_supported'] is False
    policy = CPNHRLDAGPolicy(fixture.high.model, fixture.low.model, 'cpu')
    records, summary = Evaluator({'normalize_observations': True}).evaluate(policy, [scenario])
    assert len(records) == 1
    assert records[0].valid_schedule
    assert summary['mean_ratio'] > 0.0
    assert summary['valid_schedule_rate'] == 1.0


def test_fixed_min_eft_rule_matches_heft_processor_selection():
    scenario = _scenario()
    env = CloudEdgeEndDAGEnv(normalize_observations=True)
    env.reset(scenario)
    ready = int(env.get_ready_mask().nonzero()[0][0])
    env.select_task(ready)
    observation = env.get_low_observation(ready)
    mask = env.get_node_mask(ready)
    from cpn_hrl_dag.policies.hrl import min_eft_node_action
    chosen = min_eft_node_action(observation, mask)
    exact = observation['heuristic_eft']
    assert chosen == int(min(np.flatnonzero(mask), key=lambda index: (float(exact[index]), index)))
    assert float(exact[chosen]) == pytest.approx(min(float(exact[index]) for index in np.flatnonzero(mask)))


def test_fixed_min_eft_falls_back_to_the_normalized_feature_when_exact_eft_is_absent():
    from cpn_hrl_dag.policies.hrl import min_eft_node_action
    observation = {'node_features': np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.7], [0.0, 0.0, 0.0, 0.0, 0.0, 0.2]], dtype=np.float32)}
    assert min_eft_node_action(observation, np.array([True, True])) == 1
    with pytest.raises(ValueError):
        min_eft_node_action(observation, np.array([False, False]))


def test_generated_fork_config_is_runnable_and_does_not_inherit_legacy_fields(scratch):
    """A generated fork config must satisfy the full-validation gate and schema.

    Two real bugs are pinned here: the base template's legacy
    ``evaluation.proxy_scenarios`` used to leak into fork configs (which
    `train_main` rejects outright), and the base template's extra
    ``environment`` keys used to make `load_fixed_splits` refuse the config.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    import prepare_stage1_forks

    base = {'experiment': {'name': 'base', 'seed': 2026},
            'model': {'high': {'include_heft_features': True}, 'low': {}},
            'training': {'low_pretrain_episodes': 0, 'high_train_episodes': 864, 'joint_train_episodes': 1728, 'ppo': {}},
            'environment': {'ready_semantics': 'list_schedule', 'insertion_scheduling': True, 'normalize_observations': True},
            'evaluation': {'deterministic': True, 'evaluate_initial': True, 'proxy_scenarios': 18},
            'reward': {'type': 'normalized_makespan_delta'}}
    for name in ('high_only_eft', 'low_only_frozen_high'):
        config = prepare_stage1_forks.build_branch_config(base, name, scratch / name, 1728, 2, True)
        assert config['evaluation']['proxy_scenarios'] == 0
        assert config['evaluation']['deterministic'] is True
        assert config['environment'] == {'normalize_observations': True}
        assert config['training']['phases'][0]['name'] == name
        assert config['training']['phases'][0]['episodes'] == 1728
        if name == 'low_only_frozen_high':
            assert config['training']['phases'][0]['frozen_high_mode'] == 'deterministic'
        assert config['training']['coverage_epochs'] == 2
        assert config['training']['diagnostics'] is True
        assert config['fork'] == {'scheduler_restore': 'fresh', 'optimizer_restore': 'restore', 'require_checkpoint': True, 'expected_source_step': 864}
        plan = prepare_stage1_forks.branch_plan(config)
        expected = {'high': 1728, 'low': 0} if name == 'high_only_eft' else {'high': 0, 'low': 1728}
        assert plan['budget']['optimizer_updates'] == expected
        assert plan['phases'][0]['frozen'] == (['low'] if name == 'high_only_eft' else ['high'])
    assert base['evaluation']['proxy_scenarios'] == 18  # the template is never mutated


def test_environment_comparison_ignores_default_only_keys():
    """A fork may load a checkpoint whose saved config dropped default keys."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from train_main_comparison import behavioural_environment

    template = {'environment': {'ready_semantics': 'list_schedule', 'insertion_scheduling': True, 'normalize_observations': True}}
    saved = {'environment': {'normalize_observations': True}}
    assert behavioural_environment(template) == behavioural_environment(saved)
    assert behavioural_environment({'environment': {'normalize_observations': None}})['ready_semantics'] == 'list_schedule'
    changed = {'environment': {'normalize_observations': True, 'insertion_scheduling': False}}
    assert behavioural_environment(changed) != behavioural_environment(saved)
    no_normalization = {'environment': {'normalize_observations': False}}
    assert behavioural_environment(no_normalization) != behavioural_environment(saved)


def test_ablations_with_misleading_names_are_refused(scratch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    import run_ablations

    base = {'experiment': {'name': 'base'}, 'model': {'high': {'use_lstm': True}, 'low': {'use_gat': True}},
            'training': {'low_pretrain_episodes': 0, 'high_train_episodes': 864, 'joint_train_episodes': 1728}}
    for name in ('no_staged', 'no_heft_features'):
        with pytest.raises(ValueError, match='semantics'):
            run_ablations.variant_config(base, name, scratch)
    config = run_ablations.variant_config(base, 'no_lstm', scratch)
    assert config['model']['high']['use_lstm'] is False and config['training'] == base['training']
    assert base['model']['high']['use_lstm'] is True  # the base config is never mutated
