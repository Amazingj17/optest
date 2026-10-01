"""Fixed-split training and paired full-validation comparison of graph baselines."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import yaml

from cpn_hrl_dag.algorithms.graph_baselines import GraphBaselineTrainer
from cpn_hrl_dag.datasets.loading import default_registry, load_scenarios
from cpn_hrl_dag.datasets.resource_realizations import materialize_resources
from cpn_hrl_dag.datasets.split import SplitManifest, SplitManager
from cpn_hrl_dag.env.dag_env import CloudEdgeEndDAGEnv
from cpn_hrl_dag.evaluation import Evaluator, summarize, write_report
from cpn_hrl_dag.models.graph_baselines import GraphBaselineActorCritic
from cpn_hrl_dag.policies.graph_baselines import GraphBaselinePolicy
from cpn_hrl_dag.policies.heuristics import HEFTPolicy
from cpn_hrl_dag.utils.config import config_hash
from cpn_hrl_dag.utils.seed import seed_everything
from cpn_hrl_dag.utils.progress import ProgressLog, atomic_json, console_evaluation_progress


def select_device(value: str) -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu') if value == 'auto' else torch.device(value)


def protocol_signature(config, manifest_encoding='canonical') -> str:
    manifest = Path(config['dataset']['split_manifest'])
    raw = manifest.read_bytes()
    if manifest_encoding == 'canonical':
        raw = json.dumps(json.loads(raw), sort_keys=True, separators=(',', ':')).encode('utf-8')
    elif manifest_encoding == 'lf':
        raw = raw.replace(b'\r\n', b'\n')
    elif manifest_encoding == 'crlf':
        raw = raw.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
    elif manifest_encoding != 'raw':
        raise ValueError('unknown manifest encoding')
    payload = dict(dataset=config['dataset'], resources=config['resources'],
                   environment=config['environment'], resource_seed=config['experiment']['seed'],
                   manifest_sha256=hashlib.sha256(raw).hexdigest())
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()


def load_fixed_splits(config):
    dataset = config['dataset']
    if dataset.get('limit_per_dataset') is not None:
        raise ValueError('comparison experiments forbid dataset subsampling')
    if list(dataset.get('archive_task_counts', [])) != [50, 100, 300]:
        raise ValueError('comparison experiments require task scales 50, 100 and 300')
    if Path(dataset['split_manifest']).name != 'grapheonrl_mixed_iid_seed7.json':
        raise ValueError('comparison experiments require the fixed seed7 manifest')
    if config['resources'].get('mode') != 'dataset':
        raise ValueError('comparison experiments require dataset resources')
    if config.get('environment') != {'normalize_observations': True}:
        raise ValueError('comparison uses normalized observations and the unchanged default environment')
    scenarios = load_scenarios(default_registry(dataset.get('grapheonrl_system_configs'),
                                               dataset['archive_task_counts']), dataset['roots'])
    manifest = SplitManifest.read(dataset['split_manifest'])
    splits = materialize_resources(SplitManager.apply(scenarios, manifest), config['resources'],
                                  manifest, int(config['experiment']['seed']))
    for split, count, dags in [('train', 864, 432), ('validation', 108, 54), ('test', 108, 54)]:
        if len(splits[split]) != count or len({item.metadata['original_graph_id'] for item in splits[split]}) != dags:
            raise ValueError(f'fixed split mismatch: {split} must have {count} scenarios from {dags} DAGs')
    return splits, manifest


def model_dimensions(observation):
    return dict(task_dim=observation['task_features'].shape[-1],
                resource_dim=observation['resource_features'].shape[-1],
                pair_dim=observation['pair_node_features'].shape[-1],
                task_edge_dim=observation['task_edge_features'].shape[-1],
                resource_edge_dim=observation['resource_edge_features'].shape[-1])


def make_model(config, observation, device):
    return GraphBaselineActorCritic(**model_dimensions(observation), method=config['baseline'],
                                   **config['model']).to(device)


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def read_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get('format') != 'graph-baselines-v1':
        raise ValueError('not a graph baseline checkpoint')
    if checkpoint.get('method') != checkpoint['config'].get('baseline'):
        raise ValueError('checkpoint method/config mismatch')
    if checkpoint['method'] not in GraphBaselineActorCritic.methods:
        raise ValueError('unsupported checkpoint method')
    signatures = {protocol_signature(checkpoint['config'], encoding) for encoding in ['canonical', 'raw', 'lf', 'crlf']}
    if checkpoint['protocol_signature'] not in signatures:
        raise ValueError('dataset protocol or manifest changed since checkpoint creation')
    if checkpoint.get('config_hash') != config_hash(checkpoint['config']):
        raise ValueError('checkpoint configuration hash mismatch')
    return checkpoint


def restore_model(checkpoint, device='cpu'):
    model = GraphBaselineActorCritic(**checkpoint['model_dimensions'], method=checkpoint['method'],
                                    **checkpoint['config']['model']).to(device)
    model.load_state_dict(checkpoint['model_state'])
    return model.eval()


def prepare_output(output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'refusing to overwrite a nonempty experiment directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    return output


def evaluate_model(model, scenarios, config, device, output, training_seconds=0.0, progress=None):
    policy = GraphBaselinePolicy(model, device)
    callback = progress.evaluation(Path(output).name) if progress else console_evaluation_progress(model.method)
    records, _ = Evaluator(config['environment'], callback).evaluate(policy, scenarios)
    if not all(record.valid_schedule and np.isfinite(record.ratio) for record in records):
        raise RuntimeError('invalid validation schedules or metrics')
    report = summarize(records, model=model.method, split='validation', seed=int(config['training']['seed']),
                       config_hash=config_hash(config), training_time_seconds=training_seconds,
                       bootstrap_samples=int(config['evaluation'].get('bootstrap_samples', 1000)))
    report.update(parameter_count=sum(parameter.numel() for parameter in model.parameters()),
                  method_variant='project_adaptation', heft_safety_fallback=False,
                  p95_inference_time_ms=float(np.quantile([record.inference_time_ms for record in records], 0.95)))
    write_report(output, records, report)
    with (Path(output) / 'tier_placements.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['scenario_id', 'end', 'edge', 'cloud'])
        writer.writeheader()
        writer.writerows(dict(scenario_id=record.scenario_id, **policy.placement_counts[record.scenario_id]) for record in records)
    groups = {}
    for record in records:
        kind = record.scenario_id.rsplit(':', 1)[-1]
        kind = kind if kind in {'homogeneous', 'heterogeneous'} else 'unspecified'
        groups.setdefault(kind, []).append(record)
    with (Path(output) / 'ratio_by_resource_type.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['resource_type', 'num_scenarios', 'mean_ratio', 'valid_schedule_rate'])
        writer.writeheader()
        writer.writerows(dict(resource_type=kind, num_scenarios=len(values),
                              mean_ratio=float(np.mean([record.ratio for record in values])),
                              valid_schedule_rate=float(np.mean([record.valid_schedule for record in values])))
                         for kind, values in sorted(groups.items()))
    return records, report


def train_baseline(config, output_override=None):
    config = copy.deepcopy(config)
    method, seed = config['baseline'], int(config['training']['seed'])
    output = prepare_output(output_override or Path(config['output_dir']) / method / f'seed_{seed}')
    with ProgressLog(output, method, seed) as progress:
        return _train_baseline(config, output, progress)


def _train_baseline(config, output, progress):
    method, seed = config['baseline'], int(config['training']['seed'])
    if method not in GraphBaselineActorCritic.methods:
        raise ValueError('baseline must be graph_ppo or tier_mappo')
    epochs = int(config['training']['epochs'])
    if epochs < 1:
        raise ValueError('training requires at least one complete coverage epoch')
    torch.set_num_threads(int(config.get('torch_num_threads', 1)))
    seed_everything(seed, disable_cudnn=bool(config.get('device_options', {}).get('disable_cudnn', False)))
    device = select_device(config.get('device', 'auto'))
    progress.emit('loading_fixed_splits', device=str(device))
    splits, manifest = load_fixed_splits(config)
    (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    manifest.write(output / 'split_manifest.json')
    runtime = dict(device=str(device), torch_version=str(torch.__version__),
                   train_scenarios=len(splits['train']), validation_scenarios=len(splits['validation']),
                   test_evaluated=False, protocol_signature=protocol_signature(config))
    (output / 'runtime.json').write_text(json.dumps(runtime, indent=2) + '\n', encoding='utf-8')
    probe = CloudEdgeEndDAGEnv(**config['environment'])
    probe.reset(splits['train'][0])
    observation = probe.get_flat_observation()
    model = make_model(config, observation, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['training']['learning_rate']))
    trainer = GraphBaselineTrainer(model, optimizer, device,
                                   normalize_observations=config['environment']['normalize_observations'],
                                   **config['training']['ppo'])
    order_rng = np.random.default_rng(seed)
    best, training_seconds, transitions, episode = float('inf'), 0.0, 0, 0
    validation_history = []
    seen = set()

    def payload_for(epoch):
        return dict(format='graph-baselines-v1', method=method, config=config, config_hash=config_hash(config),
                    protocol_signature=runtime['protocol_signature'], model_dimensions=model_dimensions(observation),
                    model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                    selected_epoch=epoch, completed_epochs=epoch, episodes=episode, transitions=transitions,
                    best_validation_ratio=best, training_time_seconds=training_seconds, training_complete=epoch == epochs)

    if config.get('evaluation', {}).get('evaluate_initial', False):
        _, report = evaluate_model(model, splits['validation'], config, device, output / 'validation_initial', progress=progress)
        best = float(report['mean_ratio'])
        validation_history.append(dict(epoch=0, mean_ratio=best, best_mean_ratio=best))
        save_checkpoint(output / 'best.pt', payload_for(0))
        save_checkpoint(output / 'latest.pt', payload_for(0))
        atomic_json(output / 'validation_history.json', validation_history)
        progress.emit('validation_complete', epoch=0, mean_ratio=best, best_ratio=best)
    with (output / 'train_log.csv').open('w', newline='', encoding='utf-8') as handle:
        fields = ['epoch', 'episode', 'scenario_id', 'transitions', 'reward', 'loss', 'approx_kl', 'makespan', 'ratio',
                  'total_transitions', 'coverage', 'num_tasks', 'episode_seconds', 'learning_rate']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        handle.flush()
        for epoch in range(1, epochs + 1):
            for index in order_rng.permutation(len(splits['train'])):
                scenario = splits['train'][int(index)]
                progress.emit('episode_start', episode=episode + 1, total_episodes=epochs * len(splits['train']),
                              epoch=epoch, scenario_id=scenario.scenario_id, num_tasks=scenario.num_tasks)
                start = perf_counter()
                metrics = trainer.episode(scenario)
                duration = perf_counter() - start
                if not all(np.isfinite(value) for value in metrics.values()):
                    raise RuntimeError('non-finite training metrics')
                training_seconds += duration
                episode += 1
                transitions += metrics['transitions']
                seen.add(scenario.scenario_id)
                row = dict(epoch=epoch, episode=episode, scenario_id=scenario.scenario_id, **metrics,
                           total_transitions=transitions, coverage=len(seen), num_tasks=scenario.num_tasks,
                           episode_seconds=duration, learning_rate=optimizer.param_groups[0]['lr'])
                writer.writerow(row)
                handle.flush()
                progress.emit('episode_complete', total_episodes=epochs * len(splits['train']), **row)
            _, report = evaluate_model(model, splits['validation'], config, device,
                                       output / f'validation_epoch_{epoch:03d}', training_seconds, progress)
            ratio = float(report['mean_ratio'])
            improved = ratio < best
            best = min(best, ratio)
            validation_history.append(dict(epoch=epoch, mean_ratio=ratio, best_mean_ratio=best))
            payload = payload_for(epoch)
            save_checkpoint(output / 'latest.pt', payload)
            if improved:
                save_checkpoint(output / 'best.pt', payload)
            atomic_json(output / 'validation_history.json', validation_history)
            progress.emit('validation_complete', epoch=epoch, mean_ratio=ratio, best_ratio=best)
    selected = read_checkpoint(output / 'best.pt')
    selected.update(training_complete=True, completed_epochs=epochs, episodes=episode,
                    transitions=transitions, training_time_seconds=training_seconds)
    save_checkpoint(output / 'best.pt', selected)
    evaluate_model(restore_model(selected, device), splits['validation'], config, device,
                   output / 'validation_best', training_seconds, progress)
    progress.emit('complete', status='complete', episodes=episode, transitions=transitions,
                  coverage=len(seen), selected_epoch=selected['selected_epoch'], mean_ratio=best)
    return output / 'best.pt'


def compare_checkpoints(paths, output, device_name='auto'):
    if len(paths) < 2:
        raise ValueError('comparison needs both graph_ppo and tier_mappo checkpoints')
    checkpoints = [read_checkpoint(path) for path in paths]
    first = checkpoints[0]
    keys = set()
    for checkpoint in checkpoints:
        key = (checkpoint['method'], int(checkpoint['config']['training']['seed']))
        if key in keys:
            raise ValueError('duplicate method and seed in comparison')
        keys.add(key)
        if not checkpoint.get('training_complete'):
            raise ValueError('comparison requires completed training runs, not initialized or partial models')
        if checkpoint['protocol_signature'] != first['protocol_signature']:
            raise ValueError('cannot compare different dataset protocols')
        settings = {name: value for name, value in checkpoint['config']['training'].items() if name != 'seed'}
        reference = {name: value for name, value in first['config']['training'].items() if name != 'seed'}
        if settings != reference or checkpoint['config']['model'] != first['config']['model']:
            raise ValueError('comparison requires matching training budgets and backbone settings')
        if checkpoint['transitions'] != first['transitions']:
            raise ValueError('comparison checkpoints have unequal training transition budgets')
    seeds = sorted({seed for method, seed in keys})
    if any((method, seed) not in keys for seed in seeds for method in GraphBaselineActorCritic.methods):
        raise ValueError('each seed needs both comparison methods')
    config = first['config']
    torch.set_num_threads(int(config.get('torch_num_threads', 1)))
    device = select_device(device_name)
    output = prepare_output(output)
    splits, manifest = load_fixed_splits(config)
    manifest.write(output / 'split_manifest.json')
    heft_records, _ = Evaluator(config['environment']).evaluate(HEFTPolicy(), splits['validation'])
    heft_report = summarize(heft_records, model='heft', split='validation', seed=2026, config_hash=config_hash(config))
    write_report(output / 'heft', heft_records, heft_report)
    records_by_method, reports = {}, []
    for path, checkpoint in zip(paths, checkpoints):
        method, seed = checkpoint['method'], int(checkpoint['config']['training']['seed'])
        seed_everything(seed)
        records, report = evaluate_model(restore_model(checkpoint, device), splits['validation'],
                                         checkpoint['config'], device, output / f'{method}_seed_{seed}',
                                         checkpoint['training_time_seconds'])
        records_by_method[method, seed] = {record.scenario_id: record for record in records}
        reports.append(dict(method=method, seed=seed, checkpoint=str(Path(path)), mean_ratio=report['mean_ratio'],
                            max_ratio=report['max_ratio'], valid_schedule_rate=report['valid_schedule_rate'],
                            mean_inference_time_ms=report['mean_inference_time_ms'],
                            p95_inference_time_ms=report['p95_inference_time_ms'],
                            parameter_count=report['parameter_count'], selected_epoch=checkpoint['selected_epoch']))
    paired = []
    for seed in seeds:
        graph, multi = records_by_method['graph_ppo', seed], records_by_method['tier_mappo', seed]
        if graph.keys() != multi.keys():
            raise ValueError('comparison scenario IDs differ')
        for scenario_id, record in graph.items():
            other = multi[scenario_id]
            if abs(record.heft_makespan - other.heft_makespan) > 1e-9:
                raise ValueError('comparison HEFT denominators differ')
            paired.append(dict(seed=seed, scenario_id=scenario_id, base_dag_id=record.base_dag_id,
                               graph_ppo_ratio=record.ratio, tier_mappo_ratio=other.ratio,
                               delta_mappo_minus_ppo=other.ratio - record.ratio))
    for filename, rows in [('comparison.csv', reports), ('paired_per_scene.csv', paired)]:
        with (output / filename).open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    aggregate = {}
    for method in GraphBaselineActorCritic.methods:
        values = [report['mean_ratio'] for report in reports if report['method'] == method]
        aggregate[method] = dict(seeds=len(values), mean_validation_ratio=float(np.mean(values)),
                                 std_across_seeds=float(np.std(values)), per_seed_means=values)
    result = dict(split='validation', num_scenarios=len(splits['validation']), protocol_signature=first['protocol_signature'],
                  test_evaluated=False, pure_policies=True, aggregate=aggregate,
                  evaluation_device=str(device), torch_version=str(torch.__version__),
                  torch_num_threads=torch.get_num_threads(),
                  mappo_better=sum(row['delta_mappo_minus_ppo'] < -1e-9 for row in paired),
                  mappo_worse=sum(row['delta_mappo_minus_ppo'] > 1e-9 for row in paired),
                  tied=sum(abs(row['delta_mappo_minus_ppo']) <= 1e-9 for row in paired),
                  pair_count=len(paired), comparison_scope='seed-scenario pairs; not independent DAG samples')
    result['paired_by_seed'] = {}
    for seed in seeds:
        grouped = {}
        values = [row for row in paired if row['seed'] == seed]
        for row in values:
            grouped.setdefault(row['base_dag_id'], []).append(row['delta_mappo_minus_ppo'])
        group_sums = np.asarray([sum(deltas) for deltas in grouped.values()])
        group_counts = np.asarray([len(deltas) for deltas in grouped.values()])
        sampled = np.random.default_rng(seed).integers(0, len(grouped), size=(1000, len(grouped)))
        bootstrap = group_sums[sampled].sum(1) / group_counts[sampled].sum(1)
        result['paired_by_seed'][str(seed)] = dict(
            mean_delta=float(np.mean([row['delta_mappo_minus_ppo'] for row in values])),
            base_dags=len(grouped), ci95=np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            bootstrap_unit='base_dag', bootstrap_samples=1000,
        )
    (output / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result
