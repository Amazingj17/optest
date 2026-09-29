from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _load_script(name):
    path = Path(__file__).resolve().parents[1] / 'scripts' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plotter = _load_script('plot_graph_baseline_comparison')


@pytest.fixture
def comparison_fixture(tmp_path):
    source = tmp_path / 'synthetic_comparison'
    source.mkdir()
    paired, reports = [], []
    scene_ids = [f'fixture:{index}:{kind}' for index in range(54) for kind in ('homogeneous', 'heterogeneous')]
    graph_ratios = np.linspace(.95, 1.4, 108)
    multi_ratios = graph_ratios + np.linspace(-.1, .1, 108)
    for index, scene_id in enumerate(scene_ids):
        paired.append(dict(seed=2026, scenario_id=scene_id, base_dag_id=f'fixture:{index // 2}',
                           graph_ppo_ratio=graph_ratios[index], tier_mappo_ratio=multi_ratios[index],
                           delta_mappo_minus_ppo=multi_ratios[index] - graph_ratios[index]))
    for method, ratios in [('graph_ppo', graph_ratios), ('tier_mappo', multi_ratios)]:
        folder = source / f'{method}_seed_2026'
        folder.mkdir()
        (folder / 'summary.json').write_text(json.dumps({'training_time_seconds': 100}))
        run = tmp_path / method
        run.mkdir()
        (run / 'validation_history.json').write_text(json.dumps([
            dict(epoch=epoch, mean_ratio=1.4 - epoch * .05, best_mean_ratio=1.4 - epoch * .05)
            for epoch in range(1, 4)]))
        pd.DataFrame([dict(epoch=1, episode=episode, ratio=1.2, transitions=50) for episode in range(1, 6)]).to_csv(run / 'train_log.csv', index=False)
        scenes, placements = [], []
        for index, scene_id in enumerate(scene_ids):
            size = [50, 100, 300][index % 3]
            scenes.append(dict(scenario_id=scene_id, base_dag_id=f'fixture:{index // 2}', num_tasks=size, ratio=ratios[index]))
            placements.append(dict(scenario_id=scene_id, end=0, edge=0 if index % 2 == 0 else 10,
                                   cloud=size if index % 2 == 0 else size - 10))
        pd.DataFrame(scenes).to_csv(folder / 'per_scene.csv', index=False)
        pd.DataFrame(placements).to_csv(folder / 'tier_placements.csv', index=False)
        reports.append(dict(method=method, seed=2026, checkpoint=str(run / 'best.pt'), mean_ratio=float(ratios.mean()),
                            max_ratio=float(ratios.max()), valid_schedule_rate=1.0, mean_inference_time_ms=2000,
                            p95_inference_time_ms=3000, parameter_count=100, selected_epoch=3))
    pd.DataFrame(reports).to_csv(source / 'comparison.csv', index=False)
    pd.DataFrame(paired).to_csv(source / 'paired_per_scene.csv', index=False)
    metadata = dict(split='validation', test_evaluated=False, num_scenarios=108, evaluation_device='cpu',
                    mappo_better=54, mappo_worse=54, tied=0,
                    paired_by_seed={'2026': dict(mean_delta=0, ci95=[-.02, .02])})
    (source / 'comparison.json').write_text(json.dumps(metadata))
    return source


def test_plot_complete_comparison_and_export_artifacts(comparison_fixture):
    output = plotter.plot_comparison(comparison_fixture)
    for name in ('comparison_dashboard', 'learning_curves', 'tier_placements'):
        assert (output / f'{name}.png').stat().st_size > 1000
        assert (output / f'{name}.svg').stat().st_size > 1000
    shares = pd.read_csv(output / 'placement_shares.csv')
    assert np.allclose(shares[['end_percent', 'edge_percent', 'cloud_percent']].sum(axis=1), 100)
    assert (shares.loc[shares.resource_type == 'homogeneous', 'cloud_percent'] == 100).all()
    assert 'not independent test evidence' in (output / 'RESULTS.md').read_text(encoding='utf-8')


@pytest.mark.parametrize('mutation', ['test', 'incomplete', 'duplicate', 'ratio', 'placement', 'delta'])
def test_plot_rejects_invalid_comparisons(comparison_fixture, mutation):
    source = comparison_fixture
    if mutation == 'test':
        path = source / 'comparison.json'
        metadata = json.loads(path.read_text())
        metadata['test_evaluated'] = True
        path.write_text(json.dumps(metadata))
    elif mutation in ('incomplete', 'duplicate', 'delta'):
        path = source / 'paired_per_scene.csv'
        frame = pd.read_csv(path)
        if mutation == 'incomplete':
            frame = frame.iloc[:-1]
        elif mutation == 'duplicate':
            frame.loc[1] = frame.loc[0]
        else:
            frame.loc[0, 'delta_mappo_minus_ppo'] *= -1
        frame.to_csv(path, index=False)
    elif mutation == 'ratio':
        path = source / 'graph_ppo_seed_2026' / 'per_scene.csv'
        frame = pd.read_csv(path)
        frame.loc[0, 'ratio'] += 1
        frame.to_csv(path, index=False)
    else:
        path = source / 'graph_ppo_seed_2026' / 'tier_placements.csv'
        frame = pd.read_csv(path)
        frame.loc[0, 'cloud'] += 1
        frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        plotter.plot_comparison(source)
    assert not (source / 'figures').exists()
