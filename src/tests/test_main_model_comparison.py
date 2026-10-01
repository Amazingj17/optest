from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cpn_hrl_dag.env.dag_env import CloudEdgeEndDAGEnv
from cpn_hrl_dag.models.high_lstm import HighLevelLSTMActorCritic
from cpn_hrl_dag.models.low_gat import LowLevelGATActorCritic
from cpn_hrl_dag.scenario.types import ComputeNode, Dependency, Scenario, Task
from cpn_hrl_dag.utils.config import config_hash, load_config


SPEC = importlib.util.spec_from_file_location('main_comparison', Path(__file__).resolve().parents[1] / 'scripts/compare_main_models.py')
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def test_main_protocol_accepts_equivalent_defaults():
    main = load_config('configs/zenodo_heft_safe_fast_2026.yaml')
    reference = load_config('configs/zenodo_graph_ppo_comparison.yaml')
    comparison.validate_protocol(main, reference)


@pytest.mark.parametrize('mutation', ['sampling', 'normalization', 'semantics', 'seed'])
def test_main_protocol_rejects_changes(mutation):
    main = load_config('configs/zenodo_heft_safe_fast_2026.yaml')
    reference = load_config('configs/zenodo_graph_ppo_comparison.yaml')
    if mutation == 'sampling':
        main['dataset']['sampling'] = {'grapheonrl': .5}
    elif mutation == 'normalization':
        main['environment']['normalize_observations'] = False
    elif mutation == 'semantics':
        main['environment']['ready_semantics'] = 'runtime_completion'
    else:
        main['experiment']['seed'] += 1
    with pytest.raises(ValueError):
        comparison.validate_protocol(main, reference)


def test_paired_statistics_sign_grouping_and_identity():
    rows = [dict(method=method, scenario_id=f'{dag}:{resource}', base_dag_id=str(dag), ratio=ratio, heft_makespan=10.)
            for method, ratio in [('baseline', 1.1), ('main', .9)] for dag in range(3) for resource in range(2)]
    frame = pd.DataFrame(rows)
    result = comparison.paired_statistics(frame, 'baseline', 'main')
    assert result['base_dags'] == 3
    assert result['candidate_wins'] == 6
    assert result['mean_delta'] == pytest.approx(-.2)
    assert result['ci95'] == pytest.approx([-.2, -.2])
    changed = frame.copy()
    changed.loc[changed.method == 'main', 'heft_makespan'] = 20.
    with pytest.raises(ValueError, match='denominators'):
        comparison.paired_statistics(changed, 'baseline', 'main')
    with pytest.raises(ValueError, match='missing'):
        comparison.paired_statistics(frame.iloc[:-1], 'baseline', 'main')


def test_restore_main_checkpoint_and_count_executed_placements():
    config = load_config('configs/zenodo_heft_safe_fast_2026.yaml')
    config['model']['high']['hidden_dim'] = 8
    config['model']['low']['hidden_dim'] = 8
    config['model']['low']['heads'] = 2
    scenario = Scenario('fixture', 'unit', [Task('a', 2.), Task('b', 3.)], [Dependency('a', 'b', 1.)],
                        [ComputeNode('edge', 'edge', 1.), ComputeNode('cloud', 'cloud', 2.)],
                        np.asarray([[1e9, 10.], [10., 1e9]]), metadata={'original_graph_id': 'fixture'})
    env = CloudEdgeEndDAGEnv(normalize_observations=True)
    observation, _ = env.reset(scenario)
    task_dim = observation['task_features'].shape[-1]
    high = HighLevelLSTMActorCritic(task_dim, 5, 8, heuristic_residual=True)
    low = LowLevelGATActorCritic(8, task_dim, 8, 2, heuristic_residual=True)
    payload = dict(config=config, config_hash=config_hash(config), high_model=high.state_dict(), low_model=low.state_dict())
    policy = comparison.CountingPolicy(comparison.restore_hrl(payload), 'residual_hrl')
    records, _ = comparison.Evaluator({'normalize_observations': True}).evaluate(policy, [scenario])
    assert records[0].valid_schedule
    assert sum(policy.placements['fixture'].values()) == 2
    assert policy.placements['fixture']['end'] == 0
    altered = copy.deepcopy(payload)
    altered['config_hash'] = 'invalid'
    with pytest.raises(ValueError, match='hash'):
        comparison.restore_hrl(altered)


def test_six_method_report_exports_and_rejects_missing_scenes(tmp_path):
    plot_spec = importlib.util.spec_from_file_location('plot_main', Path(__file__).resolve().parents[1] / 'scripts/plot_main_model_comparison.py')
    plotter = importlib.util.module_from_spec(plot_spec)
    plot_spec.loader.exec_module(plotter)
    status = dict(status='evaluated', split='validation', test_evaluated=False, updated_at='2026-09-18T01:52:53+00:00')
    (tmp_path / 'status.json').write_text(json.dumps(status))
    frames, summaries = [], []
    for index, method in enumerate(plotter.METHODS):
        folder = tmp_path / method
        folder.mkdir()
        ratio = 1.0 if method == 'heft' else 1.15 - .04 * index
        rows = [dict(method=method, scenario_id=f'fixture:{dag}:{resource}', base_dag_id=f'fixture:{dag}',
                     ratio=ratio, heft_makespan=10., num_tasks=50, resource_type=resource)
                for dag in range(54) for resource in ['homogeneous', 'heterogeneous']]
        frame = pd.DataFrame(rows)
        frames.append(frame)
        summaries.append(dict(model=method, category='fixture', mean_ratio=ratio, max_ratio=ratio,
                              mean_inference_time_ms=1000., p95_inference_time_ms=2000.,
                              valid_schedule_rate=1., parameter_count=10, training_episodes=150))
        pd.DataFrame([dict(scenario_id=row['scenario_id'], end=0, edge=0, cloud=50) for row in rows]).to_csv(folder / 'tier_placements.csv', index=False)
    all_scenes = pd.concat(frames, ignore_index=True)
    all_scenes.to_csv(tmp_path / 'per_scene_all.csv', index=False)
    pd.DataFrame(summaries).to_csv(tmp_path / 'comparison.csv', index=False)
    pairs = [comparison.paired_statistics(all_scenes, reference, candidate)
             for reference in ['graph_ppo', 'tier_mappo'] for candidate in ['residual_hrl', 'hrl_safe', 'search_blocks']]
    (tmp_path / 'paired_statistics.json').write_text(json.dumps(pairs))
    output = plotter.plot_main_comparison(tmp_path)
    for name in ['main_comparison_dashboard', 'main_paired_differences', 'main_tier_placements']:
        for suffix in ['png', 'svg']:
            assert (output / f'{name}.{suffix}').stat().st_size > 1000
    completed = json.loads((tmp_path / 'status.json').read_text())
    assert completed['status'] == 'complete'
    assert completed['evaluation_completed_at'] == status['updated_at']
    assert completed['updated_at'] == completed['plots_generated_at']
    assert '150' in (output / 'RESULTS.md').read_text(encoding='utf-8')
    all_scenes.iloc[:-1].to_csv(tmp_path / 'per_scene_all.csv', index=False)
    with pytest.raises(ValueError, match='identity'):
        plotter.plot_main_comparison(tmp_path)
