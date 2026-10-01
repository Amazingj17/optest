from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from cpn_hrl_dag.datasets.split import SplitManifest
from cpn_hrl_dag.evaluation import Evaluator, reporting
from cpn_hrl_dag.experiments import graph_baselines
from cpn_hrl_dag.policies.heuristics import HEFTPolicy
from cpn_hrl_dag.scenario.types import ComputeNode, Dependency, Scenario, Task
from cpn_hrl_dag.utils.config import config_hash
from cpn_hrl_dag.utils.progress import ProgressLog


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pipeline = script('run_server_comparison')
main_trainer = script('train_main_comparison')
comparison = script('compare_main_models')
plotter = script('plot_server_training')


def scenario(name, base=None):
    return Scenario(name, 'unit', [Task('a', 2.), Task('b', 3.)], [Dependency('a', 'b', 1.)],
                    [ComputeNode('edge', 'edge', 1.), ComputeNode('cloud', 'cloud', 2.)],
                    np.array([[1e9, 10.], [10., 1e9]]), metadata={'original_graph_id': base or name})


def test_config_plan_is_full_coverage_and_slots_respect_allocation(tmp_path):
    configs = pipeline.build_configs(tmp_path)
    main = configs['residual_hrl']
    assert main['training']['high_train_episodes'] == 864
    assert main['training']['joint_train_episodes'] == 1728
    assert main['curriculum']['stages'] == []
    for method in ['residual_hrl', 'graph_ppo', 'tier_mappo']:
        assert configs[method]['evaluation']['proxy_scenarios'] == 0
        assert configs[method]['evaluation']['evaluate_initial']
        comparison.validate_protocol(configs[method], configs['graph_ppo'])
    assert pipeline.validate_slots(['1'], 'cuda', {}) == ['1']
    assert pipeline.validate_slots(['1', '2'], 'cuda', {'CUDA_VISIBLE_DEVICES': '1,2'}) == ['1', '2']
    assert pipeline.validate_slots([], 'cpu', {}) == ['cpu']
    for slots in [['1', '1'], ['x'], []]:
        with pytest.raises(ValueError):
            pipeline.validate_slots(slots, 'cuda', {})
    with pytest.raises(ValueError, match='outside'):
        pipeline.validate_slots(['0'], 'cuda', {'CUDA_VISIBLE_DEVICES': '1'})


def test_progress_is_incremental_and_marks_failure(tmp_path):
    with pytest.raises(RuntimeError, match='fixture'):
        with ProgressLog(tmp_path, 'unit', 7) as progress:
            progress.emit('episode_complete', episode=1)
            assert len((tmp_path / 'events.jsonl').read_text().splitlines()) == 2
            assert json.loads((tmp_path / 'status.json').read_text())['episode'] == 1
            raise RuntimeError('fixture')
    assert json.loads((tmp_path / 'status.json').read_text())['status'] == 'failed'
    assert not (tmp_path / 'status.json.tmp').exists()


def test_evaluation_progress_does_not_change_metrics():
    scenes = [scenario('first'), scenario('second')]
    events = []
    records, summary = Evaluator(progress_callback=lambda *args: events.append(args)).evaluate(HEFTPolicy(), scenes)
    assert [event[0] for event in events] == [0, 1, 2]
    assert all(event[1] == 2 for event in events)
    assert summary['valid_schedule_rate'] == 1.0
    assert all(record.ratio == pytest.approx(1.) for record in records)


def test_small_three_model_training_and_training_plots(tmp_path, monkeypatch):
    configs = pipeline.build_configs(tmp_path, epochs=2, device='cpu', threads=1)
    manifest = SplitManifest(('unit:train-a', 'unit:train-b'), tuple(f'unit:val-{index}' for index in range(54)), ('unit:test',), 7)
    manifest_path = tmp_path / 'grapheonrl_mixed_iid_seed7.json'
    manifest.write(manifest_path)
    splits = dict(train=[scenario('train-a'), scenario('train-b')],
                  validation=[scenario(f'fixture:{index}:{resource}', f'val-{index}') for index in range(54) for resource in ['homogeneous', 'heterogeneous']],
                  test=[scenario('test')])
    monkeypatch.setattr(main_trainer, 'load_fixed_splits', lambda config: (splits, manifest))
    monkeypatch.setattr(graph_baselines, 'load_fixed_splits', lambda config: (splits, manifest))
    monkeypatch.setattr(reporting, '_write_plots', lambda *args: None)
    for method in ['residual_hrl', 'graph_ppo', 'tier_mappo']:
        config = configs[method]
        config['dataset']['split_manifest'] = str(manifest_path)
        config['evaluation']['bootstrap_samples'] = 0
        config['training']['ppo'].update(update_epochs=1, batch_size=4)
    main = configs['residual_hrl']
    main['training'].update(high_train_episodes=2, joint_train_episodes=2)
    main['model']['high']['hidden_dim'] = 8
    main['model']['low'].update(hidden_dim=8, heads=2)
    main_path = main_trainer.train_main(main)
    checkpoint, facts = comparison.inspect_main_training(main_path)
    assert facts['training_episodes'] == 4 and facts['training_transitions'] == 8
    assert facts['training_unique_scenarios'] == 2
    assert checkpoint['config_hash'] == config_hash(main)
    log = pd.read_csv(main_path.parent / 'train_log.csv')
    assert log.groupby('epoch').scenario_id.nunique().tolist() == [2, 2]
    # `high_train` is the legacy label of the explicit `high_only_eft` phase:
    # high PPO sampling with a fixed minimum-EFT low level.  The log records
    # both the phase name and the modules that were actually updated.
    assert log.phase.tolist() == ['high_only_eft', 'high_only_eft', 'joint', 'joint']
    assert log.phase_semantics.tolist() == log.phase.tolist()
    assert log.trainable_modules.tolist() == ['high', 'high', 'high+low', 'high+low']
    assert log.high_optimizer_updates.tolist() == [1, 2, 3, 4]
    assert log.low_optimizer_updates.tolist() == [0, 0, 1, 2]
    history = json.loads((main_path.parent / 'validation_history.json').read_text())
    assert [row['step'] for row in history] == [0, 2, 4]
    assert json.loads((main_path.parent / 'status.json').read_text())['status'] == 'complete'
    bad_checkpoint = torch.load(main_path, map_location='cpu', weights_only=True)
    bad_checkpoint['global_step'] = (int(bad_checkpoint['global_step']) + 1) % 5
    torch.save(bad_checkpoint, main_path.parent / 'wrong.pt')
    with pytest.raises(ValueError, match='selected best'):
        comparison.inspect_main_training(main_path.parent / 'wrong.pt')
    graph_paths = []
    for method in ['graph_ppo', 'tier_mappo']:
        config = configs[method]
        config['model'].update(hidden_dim=8, heads=2)
        path = graph_baselines.train_baseline(config, config['output_dir'])
        graph_paths.append(path)
        assert graph_baselines.read_checkpoint(path)['training_complete']
        history = json.loads((path.parent / 'validation_history.json').read_text())
        assert [row['epoch'] for row in history] == [0, 1, 2]
    figures = plotter.plot_training(main_path.parent, [path.parent for path in graph_paths], tmp_path / 'figures')
    assert (figures / 'training_comparison.png').stat().st_size > 1000
    assert len(pd.read_csv(figures / 'validation_points.csv')) == 9
    monkeypatch.setattr(comparison, 'load_fixed_splits', lambda config: (splits, manifest))
    search = configs['search']
    search['dataset']['split_manifest'] = str(manifest_path)
    search_path = tmp_path / 'search.yaml'
    search_path.write_text(yaml.safe_dump(search))
    compared = comparison.compare(main_path, graph_paths, search_path, tmp_path / 'comparison')
    assert len(pd.read_csv(compared / 'per_scene_all.csv')) == 648
    assert json.loads((compared / 'status.json').read_text())['matched_scene_coverage']
    report_plotter = script('plot_main_model_comparison')
    report = report_plotter.plot_main_comparison(compared)
    text = (report / 'RESULTS.md').read_text(encoding='utf-8')
    assert '最佳 step=100' not in text
    assert '历史主模型训练 150' not in text
    assert '任务决策数' in text
    with pytest.raises(FileExistsError):
        main_trainer.train_main(main)
    log.loc[0, 'episode'] = 2
    log.to_csv(main_path.parent / 'train_log.csv', index=False)
    with pytest.raises(ValueError, match='budget'):
        comparison.inspect_main_training(main_path)


def test_subprocess_queue_logs_assigns_slots_and_stops_on_failure(tmp_path):
    success = tmp_path / 'success'
    with ProgressLog(success, 'queue', 1) as progress:
        jobs = [dict(name=f'job_{index}', command=[sys.executable, '-u', '-c',
                'import os; print("device="+os.environ["CUDA_VISIBLE_DEVICES"], flush=True)']) for index in range(3)]
        pipeline.run_jobs(jobs, ['1', '2'], dict(os.environ), success, progress)
    assert 'device=1' in (success / 'job_0.log').read_text()
    assert 'device=2' in (success / 'job_1.log').read_text()
    failure = tmp_path / 'failure'
    with pytest.raises(RuntimeError, match='exited with 3'):
        with ProgressLog(failure, 'queue', 1) as progress:
            pipeline.run_jobs([dict(name='bad', command=[sys.executable, '-c', 'raise SystemExit(3)']),
                               dict(name='not_started', command=[sys.executable, '-c', 'print(1)'])],
                              ['cpu'], dict(os.environ), failure, progress)
    assert not (failure / 'not_started.log').exists()
    assert json.loads((failure / 'status.json').read_text())['status'] == 'failed'


def test_legacy_validation_regressions_are_not_hidden(tmp_path):
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump({'evaluation': {'proxy_scenarios': 0}}))
    pd.DataFrame([
        dict(step=0, phase='initial', scenario_id=None, num_tasks=0, reward=0, proxy_validation_mean_ratio=1., validation_mean_ratio=1.),
        dict(step=1, phase='joint', scenario_id='train-a', num_tasks=2, reward=-1., proxy_validation_mean_ratio=.9, validation_mean_ratio=.9),
        dict(step=2, phase='joint', scenario_id='train-a', num_tasks=2, reward=-1., proxy_validation_mean_ratio=1.2, validation_mean_ratio=np.nan),
    ]).to_csv(tmp_path / 'train_log.csv', index=False)
    _, validation = plotter.read_curves(tmp_path, main=True)
    assert validation.mean_ratio.tolist() == [1., .9, 1.2]
    assert validation.decisions.tolist() == [0, 2, 4]


def test_legacy_protocol_signature_survives_crlf_transfer(tmp_path):
    config = pipeline.build_configs(tmp_path)['graph_ppo']
    path = tmp_path / 'manifest.json'
    path.write_bytes(b'{"seed": 7}\r\n')
    config['dataset']['split_manifest'] = str(path)
    checkpoint = dict(format='graph-baselines-v1', method='graph_ppo', config=config,
                      config_hash=config_hash(config), protocol_signature=graph_baselines.protocol_signature(config, 'raw'))
    selected = tmp_path / 'best.pt'
    torch.save(checkpoint, selected)
    path.write_bytes(b'{"seed": 7}\n')
    assert graph_baselines.read_checkpoint(selected)['method'] == 'graph_ppo'
    path.write_bytes(b'{"seed": 8}\n')
    with pytest.raises(ValueError, match='protocol'):
        graph_baselines.read_checkpoint(selected)


@pytest.mark.parametrize('failure', [None, 'train_graph_ppo', 'evaluate_six_methods', 'plot_training'])
def test_pipeline_orders_stages_and_propagates_failure(tmp_path, monkeypatch, failure):
    output = tmp_path / 'pipeline'
    monkeypatch.setattr(sys, 'argv', ['run_server_comparison.py', '--output', str(output), '--device', 'cpu'])
    monkeypatch.setattr(pipeline, 'load_fixed_splits', lambda config: ({'train': range(864), 'validation': range(108)}, None))
    monkeypatch.setattr(pipeline, 'inspect_main_training', lambda path: ({}, {}))
    monkeypatch.setattr(pipeline, 'read_checkpoint', lambda path: {'training_complete': True, 'completed_epochs': 3})
    calls = []

    def execute(jobs, slots, environment, destination, progress):
        for job in jobs:
            calls.append(job['name'])
            if job['name'] == failure:
                raise RuntimeError('injected failure')
            if job['name'] == 'plot_comparison':
                folder = output / 'comparison'
                folder.mkdir()
                (folder / 'status.json').write_text(json.dumps({'status': 'complete', 'num_scenarios': 108}))

    monkeypatch.setattr(pipeline, 'run_jobs', execute)
    if failure:
        with pytest.raises(RuntimeError, match='injected'):
            pipeline.main()
    else:
        pipeline.main()
    expected = ['train_residual_hrl', 'train_graph_ppo', 'train_tier_mappo',
                'evaluate_six_methods', 'plot_comparison', 'plot_training']
    assert calls == (expected if failure is None else expected[:expected.index(failure) + 1])
    status = json.loads((output / 'status.json').read_text())
    assert status['status'] == ('complete' if failure is None else 'failed')
    assert not status['test_evaluated']
    assert (output / 'configs/residual_hrl.yaml').is_file()


def test_dry_run_creates_no_outputs(tmp_path, monkeypatch):
    output = tmp_path / 'dry_run'
    monkeypatch.setattr(sys, 'argv', ['run_server_comparison.py', '--output', str(output), '--device', 'cpu', '--dry-run'])
    pipeline.main()
    assert not output.exists()
