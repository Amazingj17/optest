"""Stage-one server preflight: verify a run is launchable WITHOUT training.

Checks, in order, and stops at the first failure:

1. the run configuration parses and its phase schedule is well formed;
2. every output directory is either absent or empty (nothing is ever overwritten);
3. the fork source checkpoint loads, its embedded config hash matches, and the
   fork configuration is compatible with it (model, scheduling environment, reward, seed);
4. the fixed data split loads and still has 864 / 108 / 108 scenarios from 432 / 54 / 54 DAGs;
5. a short probe episode confirms the environment, both masks and the phase's
   frozen/active module wiring actually run on this host;
6. the planned budget and the exact training command are printed.

It writes only an optional JSON report. The optional probe updates disposable
in-memory agents for one episode; it launches no formal training or checkpoint writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from cpn_hrl_dag.algorithms.hierarchical_trainer import HierarchicalTrainer  # noqa: E402
from cpn_hrl_dag.algorithms.phases import resolve_phase_schedule, summarize_phase_budget  # noqa: E402
from cpn_hrl_dag.experiments.graph_baselines import load_fixed_splits  # noqa: E402
from cpn_hrl_dag.utils.config import config_hash, load_config  # noqa: E402
from cpn_hrl_dag.utils.seed import seed_everything  # noqa: E402
from train import make_agents  # noqa: E402
from train_main_comparison import behavioural_environment  # noqa: E402


def _check_output_dir(path: Path, force: bool = False):
    if force:
        raise ValueError('destructive cleanup is not supported by preflight; select a new output directory')
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f'REFUSING: output directory is not empty: {path}; pick a new directory')
    return f'{path} (will be created)'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--fork-from', help='Fork source checkpoint; omit for training from scratch')
    parser.add_argument('--report', help='Optional path for a JSON preflight report')
    parser.add_argument('--skip-probe', action='store_true', help='Skip the single-episode environment probe')
    args = parser.parse_args()

    report: dict = {}
    config = load_config(args.config)
    if config.get('fork', {}).get('require_checkpoint') and not args.fork_from:
        raise SystemExit('REFUSING: this fork configuration requires --fork-from')
    print(f'[1/6] config        : {args.config}')
    schedule = resolve_phase_schedule(config)
    budget = summarize_phase_budget(schedule)
    report['config'] = str(Path(args.config).resolve())
    report['config_hash'] = config_hash(config)
    report['phases'] = [dict(name=item['name'], episodes=item['episodes'], frozen_high_mode=item['frozen_high_mode'],
                             high_action=item['policy'].high, low_action=item['policy'].low,
                             trainable=list(item['policy'].resolved_trainable), frozen=list(item['policy'].frozen))
                        for item in schedule]
    report['budget'] = budget
    for item in report['phases']:
        print(f"        phase {item['name']:<22} episodes={item['episodes']:<6} "
              f"high={item['high_action']:<18} low={item['low_action']:<14} trainable={item['trainable'] or ['none']}")
    print(f"        budget: episodes={budget['episodes']} optimizer_updates={budget['optimizer_updates']}")
    if int(config.get('evaluation', {}).get('proxy_scenarios', 0)) != 0:
        raise SystemExit('REFUSING: evaluation.proxy_scenarios must be 0 for the fixed 108-scenario validation')
    if float(config['training']['ppo'].get('lr_end_factor', 0.2)) <= 0.0:
        raise SystemExit('REFUSING: ppo.lr_end_factor must be positive')

    print('[2/6] output dirs')
    # Deliberately not resolved: a server-side absolute path (e.g. /data/...) must
    # be reported and checked exactly as written, not rewritten by this host.
    output = Path(config['output_dir'])
    report['output_dir'] = _check_output_dir(output)
    print(f'        {report["output_dir"]}')

    print('[3/6] dataset protocol')
    splits, manifest = load_fixed_splits(config)
    for split, expected_scenarios, expected_dags in (('train', 864, 432), ('validation', 108, 54), ('test', 108, 54)):
        actual_dags = len({item.metadata['original_graph_id'] for item in splits[split]})
        if len(splits[split]) != expected_scenarios or actual_dags != expected_dags:
            raise SystemExit(f'REFUSING: split {split} is {len(splits[split])}/{actual_dags}, expected '
                             f'{expected_scenarios}/{expected_dags}')
        print(f'        {split:<11} {len(splits[split])} scenarios / {actual_dags} base DAGs  OK')
    report['splits'] = {name: dict(scenarios=len(value), base_dags=len({item.metadata['original_graph_id'] for item in value}))
                        for name, value in splits.items() if name in ('train', 'validation', 'test')}
    report['manifest'] = str(Path(config['dataset']['split_manifest']).resolve())

    print('[4/6] fork source')
    if args.fork_from:
        import torch

        source = Path(args.fork_from).resolve()
        if not source.is_file():
            raise SystemExit(f'REFUSING: fork source checkpoint not found: {source}')
        checkpoint = torch.load(source, map_location='cpu', weights_only=False)
        if checkpoint.get('config_hash') != config_hash(checkpoint['config']):
            raise SystemExit('REFUSING: fork source checkpoint config hash mismatch')
        source_config = checkpoint['config']
        expected_step = config.get('fork', {}).get('expected_source_step')
        if expected_step is not None and int(checkpoint['global_step']) != int(expected_step):
            raise SystemExit(f'REFUSING: source must be step {expected_step}')
        if source_config['dataset'] != config['dataset'] or source_config.get('resources') != config.get('resources'):
            raise SystemExit('REFUSING: source dataset/resources configuration differs')
        if source_config['model'] != config['model']:
            raise SystemExit('REFUSING: fork source model configuration differs from this config')
        if behavioural_environment(source_config) != behavioural_environment(config):
            raise SystemExit(f'REFUSING: scheduling environment differs: '
                             f'{behavioural_environment(source_config)} != {behavioural_environment(config)}')
        if source_config.get('reward') != config.get('reward'):
            raise SystemExit('REFUSING: reward configuration differs from the fork source')
        if int(source_config['experiment']['seed']) != int(config['experiment']['seed']):
            raise SystemExit('REFUSING: fork source seed differs from this config seed')
        scheduler_restore = str(config.get('fork', {}).get('scheduler_restore', 'fresh'))
        if scheduler_restore not in ('fresh', 'restore'):
            raise SystemExit('REFUSING: fork.scheduler_restore must be "fresh" or "restore"')
        report['fork'] = dict(checkpoint=str(source),
                              checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                              source_global_step=int(checkpoint['global_step']),
                              source_best_validation_ratio=checkpoint.get('best_validation_ratio'),
                              source_config_hash=checkpoint['config_hash'],
                              scheduler_restore=scheduler_restore)
        print(f'        checkpoint       : {source}')
        print(f'        sha256           : {report["fork"]["checkpoint_sha256"]}')
        print(f'        source global_step={report["fork"]["source_global_step"]} '
              f'best_ratio={report["fork"]["source_best_validation_ratio"]}')
        print(f'        scheduler policy : {scheduler_restore}')
        print('        model/env/reward/seed compatibility: OK')
    else:
        report['fork'] = None
        print('        none (training from scratch)')

    if args.skip_probe:
        print('[5/6] probe          : skipped by --skip-probe')
    else:
        print('[5/6] probe episode')
        import torch

        device = torch.device(config.get('device', 'cpu'))
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise SystemExit('REFUSING: config requests cuda but torch.cuda.is_available() is False')
        seed_everything(int(config['experiment']['seed']), disable_cudnn=bool(config.get('device_options', {}).get('disable_cudnn', False)))
        high, low = make_agents(splits['train'][0], config, device)
        if args.fork_from:
            from train_main_comparison import _restore_for_training
            _restore_for_training(config, high, low, args.fork_from, device)
        trainer = HierarchicalTrainer(high, low, device, None, config.get('reward', {}),
                                      bool(config['model']['high'].get('include_heft_features', True)),
                                      bool(config['environment'].get('normalize_observations', False)))
        probe = splits['train'][0]
        metrics = trainer._episode(probe, schedule[0]['policy'])
        row = {key: value for key, value in metrics.items() if not key.startswith('diag_')}
        numeric = [value for value in row.values() if isinstance(value, (int, float)) and not isinstance(value, bool)]
        if not all(value == value and abs(value) != float('inf') for value in numeric):
            raise SystemExit('REFUSING: probe episode produced a non-finite metric')
        report['probe'] = dict(scenario_id=probe.scenario_id, num_tasks=probe.num_tasks, device=str(device),
                               **{key: (None if value is None else float(value)) for key, value in row.items()
                                  if isinstance(value, (int, float, type(None))) and not isinstance(value, bool)})
        print(f"        device={device} scenario={probe.scenario_id} tasks={probe.num_tasks}")
        print(f"        phase={row['phase']} decisions={row['decisions']} makespan_ratio={row['makespan_ratio']:.6f} "
              f"valid={row['valid_schedule']}")
        print(f"        high_loss={row['high_loss']} low_loss={row['low_loss']}  (None = frozen, as configured)")

    print('[6/6] training command')
    command = f'python scripts/train_main_comparison.py --config {Path(args.config).resolve()}'
    if args.fork_from:
        command += f' --fork-from {Path(args.fork_from).resolve()}'
    report['command'] = command
    print(f'        {command}')
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x', encoding='utf-8') as handle:
            handle.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        print(f'        report written to {target}')
    print('PREFLIGHT OK - no formal training launched; probe (unless skipped) updated in-memory models only')


if __name__ == '__main__':
    main()
