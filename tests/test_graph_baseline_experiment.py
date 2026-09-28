from __future__ import annotations

import copy
import csv
import json

import numpy as np
import pytest
import torch

import cpn_hrl_dag.experiments.graph_baselines as experiment
from cpn_hrl_dag.datasets.split import SplitManifest
from cpn_hrl_dag.env.dag_env import CloudEdgeEndDAGEnv
from cpn_hrl_dag.evaluation import reporting
from cpn_hrl_dag.scenario.types import ComputeNode, Dependency, Scenario, Task
from cpn_hrl_dag.utils.config import load_config


def _scenario(name):
    return Scenario(name, 'unit', [Task('a', 2.0), Task('b', 3.0), Task('c', 1.0)],
                    [Dependency('a', 'c', 2.0), Dependency('b', 'c', 1.0)],
                    [ComputeNode('edge', 'edge', 1.0), ComputeNode('cloud', 'cloud', 2.0)],
                    np.asarray([[1e9, 10.0], [10.0, 1e9]]), metadata={'original_graph_id': name})


@pytest.fixture
def tiny_protocol(tmp_path, monkeypatch):
    manifest = SplitManifest(('unit:train-a', 'unit:train-b'), ('unit:val-a', 'unit:val-b'), ('unit:test',), seed=7)
    manifest_path = tmp_path / 'grapheonrl_mixed_iid_seed7.json'
    manifest.write(manifest_path)
    config = load_config('configs/zenodo_graph_ppo_comparison.yaml')
    config['dataset']['split_manifest'] = str(manifest_path)
    config['model'].update(hidden_dim=8, heads=2)
    config['training']['epochs'] = 2
    config['training']['ppo'].update(update_epochs=1, batch_size=3)
    config['evaluation']['bootstrap_samples'] = 0
    config['device'] = 'cpu'
    splits = {'train': [_scenario('train-a'), _scenario('train-b')],
              'validation': [_scenario('val-a'), _scenario('val-b')], 'test': [_scenario('test')]}
    monkeypatch.setattr(experiment, 'load_fixed_splits', lambda unused: (splits, manifest))
    monkeypatch.setattr(reporting, '_write_plots', lambda *args: None)
    return config, splits, manifest_path


def test_train_reload_and_paired_comparison_end_to_end(tiny_protocol, tmp_path):
    config, splits, _ = tiny_protocol
    checkpoints = []
    for method in ('graph_ppo', 'tier_mappo'):
        selected = copy.deepcopy(config)
        selected['baseline'] = method
        path = experiment.train_baseline(selected, tmp_path / method)
        checkpoints.append(path)
        checkpoint = experiment.read_checkpoint(path)
        assert checkpoint['training_complete']
        assert checkpoint['episodes'] == 4
        assert checkpoint['transitions'] == 12
        assert checkpoint['completed_epochs'] == 2
        with (path.parent / 'train_log.csv').open() as handle:
            records = list(csv.DictReader(handle))
        for epoch in ('1', '2'):
            assert {record['scenario_id'] for record in records if record['epoch'] == epoch} == {'train-a', 'train-b'}
        with (path.parent / 'validation_best/per_scene.csv').open() as handle:
            records = list(csv.DictReader(handle))
        assert {record['scenario_id'] for record in records} == {'val-a', 'val-b'}
        with (path.parent / 'validation_best/tier_placements.csv').open() as handle:
            placements = list(csv.DictReader(handle))
        assert len(placements) == 2
        assert all(sum(int(row[tier]) for tier in ('end', 'edge', 'cloud')) == 3 for row in placements)
        model = experiment.restore_model(checkpoint)
        probe = CloudEdgeEndDAGEnv(normalize_observations=True)
        probe.reset(splits['validation'][0])
        assert experiment.model_dimensions(probe.get_flat_observation()) == checkpoint['model_dimensions']
        assert all(torch.equal(parameter, checkpoint['model_state'][name]) for name, parameter in model.state_dict().items())
    result = experiment.compare_checkpoints(checkpoints, tmp_path / 'comparison', 'cpu')
    assert result['num_scenarios'] == 2
    assert result['pair_count'] == 2
    assert result['mappo_better'] + result['mappo_worse'] + result['tied'] == 2
    assert not result['test_evaluated']
    assert set(result['aggregate']) == {'graph_ppo', 'tier_mappo'}
    assert result['paired_by_seed']['2026']['base_dags'] == 2
    assert result['paired_by_seed']['2026']['bootstrap_unit'] == 'base_dag'
    assert result['evaluation_device'] == 'cpu'
    assert (tmp_path / 'comparison/paired_per_scene.csv').is_file()
    assert json.loads((tmp_path / 'comparison/graph_ppo_seed_2026/summary.json').read_text())['heft_safety_fallback'] is False
    with pytest.raises(FileExistsError):
        experiment.train_baseline(config, tmp_path / 'graph_ppo')
    with pytest.raises(ValueError, match='duplicate'):
        experiment.compare_checkpoints([checkpoints[0], checkpoints[0]], tmp_path / 'duplicate', 'cpu')


def test_checkpoint_protocol_tampering_is_rejected(tiny_protocol, tmp_path):
    config, _, manifest_path = tiny_protocol
    config['training']['epochs'] = 1
    path = experiment.train_baseline(config, tmp_path / 'train')
    original = manifest_path.read_bytes()
    manifest_path.write_bytes(original.replace(b'\r\n', b'\n'))
    experiment.read_checkpoint(path)
    manifest_path.write_bytes(original.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n'))
    experiment.read_checkpoint(path)
    changed = json.loads(original)
    changed['seed'] += 1
    manifest_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='protocol'):
        experiment.read_checkpoint(path)


def test_partial_checkpoints_cannot_be_comparison_results(tiny_protocol, tmp_path):
    config, _, _ = tiny_protocol
    config['training']['epochs'] = 1
    path = experiment.train_baseline(config, tmp_path / 'train')
    checkpoint = experiment.read_checkpoint(path)
    checkpoint['training_complete'] = False
    partial = tmp_path / 'partial.pt'
    experiment.save_checkpoint(partial, checkpoint)
    with pytest.raises(ValueError, match='completed'):
        experiment.compare_checkpoints([partial, path], tmp_path / 'invalid', 'cpu')


def test_real_loader_rejects_subsampling_before_loading():
    config = load_config('configs/zenodo_graph_ppo_comparison.yaml')
    config['dataset']['limit_per_dataset'] = 1
    with pytest.raises(ValueError, match='subsampling'):
        experiment.load_fixed_splits(config)


def test_comparison_configs_have_identical_budgets():
    graph = load_config('configs/zenodo_graph_ppo_comparison.yaml')
    multi = load_config('configs/zenodo_tier_mappo_comparison.yaml')
    for key in ('dataset', 'resources', 'environment', 'model', 'training', 'evaluation'):
        assert graph[key] == multi[key]
