"""Reevaluate main models and graph baselines on the fixed full validation split."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cpn_hrl_dag.evaluation import Evaluator, summarize, write_report
from cpn_hrl_dag.experiments.graph_baselines import load_fixed_splits, prepare_output, protocol_signature, read_checkpoint, restore_model
from cpn_hrl_dag.models.high_lstm import HighLevelLSTMActorCritic
from cpn_hrl_dag.models.low_gat import LowLevelGATActorCritic
from cpn_hrl_dag.policies.base import SchedulerPolicy
from cpn_hrl_dag.policies.graph_baselines import GraphBaselinePolicy
from cpn_hrl_dag.policies.heuristics import HEFTPolicy
from cpn_hrl_dag.policies.hrl import CPNHRLDAGPolicy
from cpn_hrl_dag.policies.portfolio import HEFTSafePortfolioPolicy
from cpn_hrl_dag.utils.config import config_hash, load_config
from cpn_hrl_dag.utils.seed import seed_everything
from cpn_hrl_dag.utils.progress import atomic_json, console_evaluation_progress


LABELS = {'heft': 'HEFT', 'graph_ppo': 'Graph PPO', 'tier_mappo': 'Tier MAPPO',
          'residual_hrl': 'Main: Residual HRL', 'hrl_safe': 'Main: HRL-safe',
          'search_blocks': 'Search: beam + blocks'}


def configured_phase_budget(config):
    """Expected per-phase episode budget from an explicit or legacy training config.

    The legacy fields keep their original names so every pre-existing
    checkpoint and log stays valid; an explicit ``training.phases`` schedule
    takes precedence when present.
    """
    training = config['training']
    entries = training.get('phases')
    if entries:
        budget = {}
        for item in entries:
            name = str(item['name'])
            budget[name] = budget.get(name, 0) + int(item['episodes'])
        return budget
    return {phase: int(training.get(field, 0)) for phase, field in
            [('low_pretrain', 'low_pretrain_episodes'), ('high_only_eft', 'high_train_episodes'), ('joint', 'joint_train_episodes')]}


def inspect_main_training(path):
    path = Path(path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    if checkpoint.get('config_hash') != config_hash(config):
        raise ValueError('main checkpoint config hash mismatch')
    log = pd.read_csv(path.parent / 'train_log.csv')
    phase_column = 'phase'
    completed = log.loc[log[phase_column].isin(['low_pretrain', 'high_train', 'high_only_eft', 'joint'])].copy()
    budgets = configured_phase_budget(config)
    total = sum(budgets.values())
    # New logs expose both `global_step` and `episode`; historical logs only
    # expose `step`.  Either one must enumerate the configured budget exactly,
    # and for a forked run the episode index starts at 1 while the global step
    # carries the fork offset.
    episode_column = 'episode' if 'episode' in completed.columns else 'step'
    if total < 1 or len(completed) != total or not completed[episode_column].is_unique or set(completed[episode_column]) != set(range(1, total + 1)):
        raise ValueError('main training log does not show completed configured budget')
    observed = completed[phase_column].value_counts().to_dict()
    expected = {name: count for name, count in budgets.items() if count}
    if expected.get('high_only_eft') and observed.get('high_train'):
        # A legacy log labels the high-level-only phase "high_train"; the two
        # names denote exactly the same semantics.
        moved = observed.pop('high_train')
        observed['high_only_eft'] = observed.get('high_only_eft', 0) + moved
    if {name: int(count) for name, count in observed.items()} != expected:
        raise ValueError('main training phases do not match configured budget')
    global_step_column = 'global_step' if 'global_step' in completed.columns else 'step'
    offset = int(completed[global_step_column].min()) - 1
    if offset < 0 or int(completed[global_step_column].max()) != offset + total:
        raise ValueError('main checkpoint step outside training budget')
    if not 0 <= int(checkpoint['global_step']) <= offset + total:
        raise ValueError('main checkpoint step outside training budget')
    history_path = path.parent / 'validation_history.json'
    if history_path.is_file():
        history = pd.DataFrame(json.loads(history_path.read_text(encoding='utf-8')))
        history = history.rename(columns={'mean_ratio': 'validation_mean_ratio'})
    else:
        history = log
    points = history[['step', 'validation_mean_ratio']].dropna().sort_values('step')
    if points.empty or not np.isfinite(points.validation_mean_ratio).all():
        raise ValueError('main model needs finite full-validation selection history')
    selected_step = int(points.loc[points.validation_mean_ratio.idxmin(), 'step'])
    if selected_step != int(checkpoint['global_step']):
        raise ValueError('requires the selected best main checkpoint, not latest or an intermediate model')
    summary = json.loads((path.parent / 'summary.json').read_text(encoding='utf-8'))
    if summary.get('split') != 'validation' or summary.get('num_scenarios') != 108 or summary.get('num_base_dags') != 54 or summary.get('valid_schedule_rate') != 1.0:
        raise ValueError('main model requires complete valid 108-scenario validation')
    if summary.get('config_hash') != config_hash(config) or not np.isclose(summary['mean_ratio'], checkpoint['best_validation_ratio']):
        raise ValueError('main checkpoint and validation report disagree')
    transitions = None
    if 'num_tasks' in completed:
        transitions = int(completed.num_tasks.sum())
    elif (path.parent / 'dataset_audit.json').is_file():
        audit = json.loads((path.parent / 'dataset_audit.json').read_text(encoding='utf-8'))
        sizes = {row['scenario_id']: row['num_tasks'] for row in audit['scenarios']}
        transitions = sum(int(sizes[identity]) for identity in completed.scenario_id)
    facts = dict(training_episodes=total, training_transitions=transitions,
                 training_unique_scenarios=int(completed.scenario_id.nunique()),
                 training_time_seconds=summary.get('training_time_seconds'),
                 training_time_scope=summary.get('training_time_scope', 'legacy loop including intermediate validation'),
                 sampling='full_coverage' if config['training'].get('coverage_epochs') or config['training'].get('phases') else 'legacy_sampling',
                 phase_budget=budgets)
    return checkpoint, facts


def validate_protocol(candidate, reference):
    normalized = copy.deepcopy(candidate)
    sampling = normalized['dataset'].pop('sampling', None)
    if sampling is not None and sampling != {'grapheonrl': 1.0}:
        raise ValueError('comparison forbids altered dataset sampling')
    environment = normalized['environment']
    if environment.get('ready_semantics', 'list_schedule') != 'list_schedule' or environment.get('insertion_scheduling', True) is not True:
        raise ValueError('comparison requires unchanged scheduling semantics')
    normalized['environment'] = {'normalize_observations': environment.get('normalize_observations', False)}
    if set(environment) - {'normalize_observations', 'ready_semantics', 'insertion_scheduling'}:
        raise ValueError('unsupported environment changes')
    if protocol_signature(normalized) != protocol_signature(reference):
        raise ValueError('comparison dataset/resource/normalization protocol mismatch')


def restore_hrl(checkpoint):
    config = checkpoint['config']
    if checkpoint.get('config_hash') != config_hash(config):
        raise ValueError('main checkpoint config hash mismatch')
    if config['model']['high'].get('include_heft_features', True) is not True:
        raise ValueError('main checkpoint requires a different observation schema')
    high_state, low_state = checkpoint['high_model'], checkpoint['low_model']
    task_dim = high_state['task_encoder.weight'].shape[1]
    high_hidden = config['model']['high']['hidden_dim']
    resource_dim = high_state['critic.0.weight'].shape[1] - high_hidden
    high_options, low_options = config['model']['high'], config['model']['low']
    high = HighLevelLSTMActorCritic(task_dim, resource_dim, high_hidden,
        high_options.get('use_lstm', True), high_options.get('heuristic_residual', False),
        high_options.get('heuristic_weight', 8.0), high_options.get('residual_limit', .5))
    low = LowLevelGATActorCritic(low_state['node_encoder.weight'].shape[1], task_dim,
        low_options['hidden_dim'], low_options['heads'], low_options.get('use_gat', True),
        low_options.get('heuristic_residual', False), low_options.get('heuristic_weight', 8.0),
        low_options.get('residual_limit', .5))
    high.load_state_dict(high_state)
    low.load_state_dict(low_state)
    return CPNHRLDAGPolicy(high, low, 'cpu')


class CountingPolicy(SchedulerPolicy):
    def __init__(self, policy, name):
        self.policy, self.name = policy, name
        self.placements = {}

    def reset(self, scenario):
        self.scenario = scenario
        self.placements[scenario.scenario_id] = {'end': 0, 'edge': 0, 'cloud': 0}
        self.policy.reset(scenario)

    def select_task(self, observation, ready_mask, deterministic=True):
        return self.policy.select_task(observation, ready_mask, deterministic)

    def select_node(self, observation, task_id, node_mask, deterministic=True):
        node = self.policy.select_node(observation, task_id, node_mask, deterministic)
        tier = self.scenario.compute_nodes[node].node_type.lower()
        self.placements[self.scenario.scenario_id][tier] += 1
        return node


def paired_statistics(frame, reference, candidate, seed=2026):
    selected = frame.loc[frame.method.isin([reference, candidate])]
    if selected.duplicated(['method', 'scenario_id']).any():
        raise ValueError('duplicate paired scenario IDs')
    pivot = selected.pivot(index='scenario_id', columns='method', values='ratio')
    if pivot[[reference, candidate]].isna().any().any():
        raise ValueError('missing paired scenarios')
    left = selected.loc[selected.method == reference].set_index('scenario_id').loc[pivot.index]
    right = selected.loc[selected.method == candidate].set_index('scenario_id').loc[pivot.index]
    if not left.base_dag_id.equals(right.base_dag_id):
        raise ValueError('paired base DAG identities differ')
    if not np.allclose(left.heft_makespan, right.heft_makespan, rtol=0, atol=1e-9):
        raise ValueError('paired HEFT denominators differ')
    delta = pivot[candidate] - pivot[reference]
    grouped = pd.DataFrame({'delta': delta, 'dag': left.base_dag_id}).groupby('dag').delta.agg(['sum', 'count'])
    indices = np.random.default_rng(seed).integers(0, len(grouped), size=(1000, len(grouped)))
    means = grouped['sum'].to_numpy()[indices].sum(1) / grouped['count'].to_numpy()[indices].sum(1)
    return dict(reference=reference, candidate=candidate, mean_delta=float(delta.mean()),
                ci95=np.quantile(means, [.025, .975]).tolist(), candidate_wins=int((delta < -1e-9).sum()),
                candidate_losses=int((delta > 1e-9).sum()), ties=int((delta.abs() <= 1e-9).sum()),
                base_dags=len(grouped), bootstrap_unit='base_dag', bootstrap_samples=1000)


def compare(main_path, graph_paths, search_path, output):
    torch.set_num_threads(1)
    main, main_facts = inspect_main_training(main_path)
    seed = int(main['config']['experiment']['seed'])
    seed_everything(seed, disable_cudnn=True)
    graphs = [read_checkpoint(path) for path in graph_paths]
    if {checkpoint['method'] for checkpoint in graphs} != {'graph_ppo', 'tier_mappo'} or len(graphs) != 2:
        raise ValueError('requires one completed checkpoint per graph baseline')
    reference = graphs[0]['config']
    if any(not checkpoint['training_complete'] or checkpoint['config']['training']['seed'] != seed for checkpoint in graphs):
        raise ValueError('requires completed graph training with the same training seed as the main model')
    if graphs[0]['config']['training'] != graphs[1]['config']['training'] or graphs[0]['transitions'] != graphs[1]['transitions']:
        raise ValueError('graph baseline training budgets differ')
    for checkpoint in [main, *graphs]:
        validate_protocol(checkpoint['config'], reference)
    search_config = load_config(search_path)
    validate_protocol(search_config, reference)
    main_policy = restore_hrl(main)
    main_episodes = main_facts['training_episodes']
    main_count = sum(parameter.numel() for model in [main_policy.high, main_policy.low] for parameter in model.parameters())
    policies = [('heft', HEFTPolicy(), 'reference', 0, 0, None, '')]
    for path, checkpoint in zip(graph_paths, graphs):
        model = restore_model(checkpoint, 'cpu')
        policies.append((model.method, GraphBaselinePolicy(model, 'cpu'), 'pure_learning',
                         sum(parameter.numel() for parameter in model.parameters()), checkpoint['episodes'],
                         checkpoint['selected_epoch'], str(path)))
    policies.append(('residual_hrl', main_policy, 'pure_learning', main_count, main_episodes, main['global_step'], str(main_path)))
    policies.append(('hrl_safe', HEFTSafePortfolioPolicy(seed=seed, learned_policy=main_policy,
                    normalize_observations=True, **main['config'].get('portfolio', {})),
                    'learning_plus_search', main_count, main_episodes, main['global_step'], str(main_path)))
    policies.append(('search_blocks', HEFTSafePortfolioPolicy(seed=seed, normalize_observations=True,
                    **search_config['search']), 'search_only', 0, 0, None, ''))
    output = prepare_output(output)
    status = dict(status='running', split='validation', test_evaluated=False, retrained=False,
                  evaluation_device='cpu', torch_num_threads=1, torch_version=str(torch.__version__),
                  seed=seed, matched_training_budget=False, completed_methods=[],
                  main_training=main_facts, main_proxy_scenarios=main['config'].get('evaluation', {}).get('proxy_scenarios', 0),
                  portfolio=main['config'].get('portfolio', {}),
                  matched_scene_coverage=bool(main_facts['sampling'] == 'full_coverage' and
                      all(checkpoint['episodes'] == main_episodes and checkpoint['transitions'] == main_facts['training_transitions'] for checkpoint in graphs)),
                  checkpoint_sha256={str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in [main_path, *graph_paths]},
                  protocol_signature=protocol_signature(reference), started_at=datetime.now(timezone.utc).isoformat())

    def save_status(stage):
        status.update(stage=stage, updated_at=datetime.now(timezone.utc).isoformat())
        atomic_json(output / 'status.json', status)
        print(status['updated_at'], stage, flush=True)

    save_status('loading_fixed_splits')
    try:
        splits, manifest = load_fixed_splits(reference)
        manifest.write(output / 'split_manifest.json')
        summaries, frames = [], []
        for method, raw_policy, category, parameters, episodes, selected, checkpoint_path in policies:
            save_status('evaluating_' + method)
            policy = CountingPolicy(raw_policy, method)
            seed_everything(seed, disable_cudnn=True)
            records, _ = Evaluator({'normalize_observations': True}, console_evaluation_progress(method)).evaluate(policy, splits['validation'])
            if len(records) != 108 or not all(record.valid_schedule and np.isfinite(record.ratio) for record in records):
                raise RuntimeError(f'{method}: incomplete or invalid full validation')
            report = summarize(records, model=method, split='validation', seed=seed, config_hash=config_hash(reference), bootstrap_samples=1000)
            folder = output / method
            report.update(category=category, parameter_count=parameters, training_episodes=episodes,
                          selected_checkpoint=selected, checkpoint_unit='episode' if method in ['residual_hrl', 'hrl_safe'] else 'epoch',
                          checkpoint_path=checkpoint_path, p95_inference_time_ms=float(np.quantile([record.inference_time_ms for record in records], .95)),
                          evaluation_device='cpu', training_time_seconds=None, timing_scope='policy reset plus full schedule replay')
            if method in ['residual_hrl', 'hrl_safe']:
                report.update(main_facts)
            elif method in ['graph_ppo', 'tier_mappo']:
                checkpoint = next(item for item in graphs if item['method'] == method)
                report.update(training_transitions=checkpoint['transitions'], training_time_seconds=checkpoint.get('training_time_seconds'),
                              training_time_scope='episode collection and PPO updates', sampling='full_coverage')
            write_report(folder, records, report)
            pd.DataFrame([dict(scenario_id=record.scenario_id, **policy.placements[record.scenario_id]) for record in records]).to_csv(folder / 'tier_placements.csv', index=False)
            frame = pd.DataFrame([asdict(record) for record in records])
            frame['method'] = method
            frames.append(frame)
            summaries.append(report)
            status['completed_methods'].append(method)
            print(f'{method}: ratio={report["mean_ratio"]:.9f}, scenes={len(records)}', flush=True)
            save_status('finished_' + method)
        frame = pd.concat(frames, ignore_index=True)
        paired = [paired_statistics(frame, baseline, candidate, seed=seed) for baseline in ['graph_ppo', 'tier_mappo']
                  for candidate in ['residual_hrl', 'hrl_safe', 'search_blocks']]
        pd.DataFrame(summaries).to_csv(output / 'comparison.csv', index=False)
        frame.to_csv(output / 'per_scene_all.csv', index=False)
        (output / 'paired_statistics.json').write_text(json.dumps(paired, indent=2) + '\n', encoding='utf-8')
        status.update(status='evaluated', num_scenarios=len(splits['validation']), base_dags=frame.base_dag_id.nunique())
        save_status('evaluation_complete')
        return output
    except Exception as error:
        status.update(status='failed', error=str(error))
        save_status(status['stage'])
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-checkpoint', required=True)
    parser.add_argument('--graph-checkpoints', nargs=2, required=True)
    parser.add_argument('--search-config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(compare(args.main_checkpoint, args.graph_checkpoints, args.search_config, args.output))


if __name__ == '__main__':
    main()
