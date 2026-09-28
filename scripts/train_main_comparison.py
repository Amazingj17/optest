"""Full-coverage HRL training on the frozen validation protocol, without proxy selection."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import math
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from train import make_agents
from cpn_hrl_dag.algorithms.hierarchical_trainer import HierarchicalTrainer
from cpn_hrl_dag.algorithms.phases import resolve_phase_schedule, summarize_phase_budget
from cpn_hrl_dag.evaluation import DiagnosticsRecorder, Evaluator, HEFTCache, summarize, write_report, write_training_curve
from cpn_hrl_dag.experiments.graph_baselines import load_fixed_splits, prepare_output
from cpn_hrl_dag.policies.hrl import CPNHRLDAGPolicy
from cpn_hrl_dag.utils.config import config_hash, load_config
from cpn_hrl_dag.utils.progress import ProgressLog, atomic_json
from cpn_hrl_dag.utils.runtime import runtime_metadata
from cpn_hrl_dag.utils.seed import seed_everything

TRAIN_FIELDS = ['global_step', 'episode', 'epoch', 'phase', 'phase_semantics', 'trainable_modules', 'scenario_id',
                'num_tasks', 'transitions', 'coverage', 'decisions', 'reward', 'final_makespan', 'heft_makespan',
                'makespan_ratio', 'high_loss', 'low_loss', 'high_kl', 'low_kl', 'high_lr', 'low_lr',
                'high_optimizer_updates', 'low_optimizer_updates', 'episode_seconds']


def _trainer_kwargs(config):
    """Environment/model options shared by training and validation."""
    return dict(include_heft_features=bool(config.get('model', {}).get('high', {}).get('include_heft_features', True)),
                normalize_observations=bool(config.get('environment', {}).get('normalize_observations', False)))


# Environment keys that actually change scheduling behaviour.  Other keys in the
# mapping are inert for the fork comparison: the formal run's saved config kept
# only `normalize_observations` because the rest matched the defaults, so a fork
# config written from the base template must still be allowed to fork from it.
_BEHAVIOURAL_ENVIRONMENT_KEYS = ('ready_semantics', 'insertion_scheduling', 'normalize_observations')


def behavioural_environment(config):
    """Normalized scheduling-relevant view of an environment configuration."""
    environment = dict(config.get('environment', {}))
    return {
        'ready_semantics': environment.get('ready_semantics', 'list_schedule'),
        'insertion_scheduling': environment.get('insertion_scheduling', True),
        'normalize_observations': bool(environment.get('normalize_observations', False)),
    }


def _restore_for_training(config, high, low, checkpoint, device):
    """Load the checkpoint a new experiment forks from.

    A fork is *checkpoint initialization*, not exact resume: model (and, unless
    the fork config says otherwise, optimizer) state is restored deliberately,
    while the RNG stream, the scenario permutation cursor and the coverage
    counters restart from the declared seed.  Scheduler state is only restored
    when ``fork.scheduler_restore`` is ``"restore"``; the default is ``"fresh"``
    because a scheduler restored from an experiment with a different total step
    budget would otherwise carry an implicit, undocumented schedule.  Both
    choices are persisted in checkpoint metadata and ``fork_source.json``.
    """
    fork = dict(config.get('fork', {}))
    scheduler_restore = str(fork.get('scheduler_restore', 'fresh'))
    if scheduler_restore not in ('fresh', 'restore'):
        raise ValueError('fork.scheduler_restore must be "fresh" or "restore"')
    optimizer_restore = str(fork.get('optimizer_restore', 'restore'))
    if optimizer_restore not in ('restore', 'reset'):
        raise ValueError('fork.optimizer_restore must be "restore" or "reset"')
    if optimizer_restore == 'reset' and scheduler_restore == 'restore':
        raise ValueError('reset optimizer requires a fresh scheduler')
    restored = torch.load(checkpoint, map_location=device, weights_only=False)
    source = restored['config']
    if restored.get('config_hash') != config_hash(source):
        raise ValueError('fork source config hash mismatch')
    expected_step = fork.get('expected_source_step')
    if expected_step is not None and int(restored['global_step']) != int(expected_step):
        raise ValueError(f'fork source must be step {expected_step}, got {restored["global_step"]}')
    if source['model'] != config['model']:
        raise ValueError('fork source model configuration differs from the fork configuration')
    if behavioural_environment(source) != behavioural_environment(config):
        raise ValueError(f"fork source scheduling environment differs from the fork configuration: "
                         f"{behavioural_environment(source)} != {behavioural_environment(config)}")
    if source.get('reward') != config.get('reward'):
        raise ValueError('fork source reward configuration differs from the fork configuration')
    if int(source['experiment']['seed']) != int(config['experiment']['seed']):
        raise ValueError('fork source seed differs from the fork configuration seed')
    if source['dataset'] != config['dataset'] or source.get('resources') != config.get('resources'):
        raise ValueError('fork source dataset/resources configuration differs')
    high.model.load_state_dict(restored['high_model'])
    low.model.load_state_dict(restored['low_model'])
    for name, agent in [('high', high), ('low', low)]:
        if optimizer_restore == 'restore':
            agent.optimizer.load_state_dict(restored[f'{name}_optimizer'])
        if scheduler_restore == 'restore':
            state = restored.get(f'{name}_lr_scheduler')
            if state is None or agent.lr_scheduler is None:
                raise ValueError('requested scheduler restore but scheduler state is missing')
            agent.lr_scheduler.load_state_dict(state)
    if scheduler_restore == 'fresh':
        # Loading Adam also restores its decayed learning rate. Reset both lr
        # fields explicitly before constructing the new branch's scheduler.
        for name, agent in [('high', high), ('low', low)]:
            for group in agent.optimizer.param_groups:
                group['lr'] = float(config['model'][name]['learning_rate'])
                group['initial_lr'] = group['lr']
            agent.rebuild_lr_scheduler()
    return restored


def train_main(config, fork_checkpoint=None):
    config = copy.deepcopy(config)
    if config.get('fork', {}).get('require_checkpoint') and fork_checkpoint is None:
        raise ValueError('this fork configuration requires --fork-from')
    if int(config.get('evaluation', {}).get('proxy_scenarios', 0)) != 0:
        raise ValueError('formal HRL training forbids proxy validation')
    if not config.get('evaluation', {}).get('deterministic', True):
        raise ValueError('formal HRL validation must be deterministic')
    output = prepare_output(config['output_dir'])
    seed = int(config['experiment']['seed'])
    torch.set_num_threads(int(config.get('torch_num_threads', 4)))
    seed_everything(seed, disable_cudnn=bool(config.get('device_options', {}).get('disable_cudnn', True)))
    with ProgressLog(output, 'residual_hrl', seed) as progress:
        progress.emit('loading_fixed_splits', fork_checkpoint=str(fork_checkpoint) if fork_checkpoint else None)
        splits, manifest = load_fixed_splits(config)
        count = len(splits['train'])
        schedule = resolve_phase_schedule(config, episodes=count * int(config['training'].get('coverage_epochs', 0) or 0))
        total = sum(item['episodes'] for item in schedule)
        if total < 1:
            raise ValueError('phase schedule must contain at least one episode')
        budget = summarize_phase_budget(schedule)
        diagnostics_enabled = bool(config.get('training', {}).get('diagnostics', False))
        config['runtime'] = runtime_metadata()
        (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        atomic_json(output / 'reproducibility.json', config['runtime'])
        atomic_json(output / 'phase_plan.json', dict(
            budget=budget,
            phases=[dict(name=item['name'], episodes=item['episodes'], frozen_high_mode=item['frozen_high_mode'],
                         high_action=item['policy'].high, low_action=item['policy'].low,
                         trainable=list(item['policy'].resolved_trainable), frozen=list(item['policy'].frozen)) for item in schedule],
            configured_budgets={name: int(config['training'].get(field, 0)) for name, field in
                                [('low_pretrain', 'low_pretrain_episodes'), ('high_only_eft', 'high_train_episodes'), ('joint', 'joint_train_episodes')]},
            order='epoch-wise permutation seeded with experiment.seed; identical for both fork branches',
            fork_resume='checkpoint initialization; see fork_source.json for optimizer/scheduler policy; not exact resume',
            exact_resume_supported=False,
            diagnostics_enabled=diagnostics_enabled))
        manifest.write(output / 'split_manifest.json')
        device = torch.device(config['device'])
        high, low = make_agents(splits['train'][0], config, device)
        source_facts = None
        resume_offset = 0
        if fork_checkpoint is not None:
            restored = _restore_for_training(config, high, low, fork_checkpoint, device)
            resume_offset = int(restored['global_step'])
            source_facts = dict(checkpoint=str(fork_checkpoint),
                                checkpoint_sha256=hashlib.sha256(Path(fork_checkpoint).read_bytes()).hexdigest(),
                                source_global_step=resume_offset,
                                source_best_validation_ratio=(None if not math.isfinite(float(restored.get('best_validation_ratio', float('nan')))) else float(restored['best_validation_ratio'])),
                                source_training_state=dict(restored.get('training_state', {})),
                                source_seed=int(restored['config']['experiment']['seed']),
                                optimizer_restore=config.get('fork', {}).get('optimizer_restore', 'restore'),
                                scheduler_restore=config.get('fork', {}).get('scheduler_restore', 'fresh'),
                                fresh_scheduler_start_lr='configured model learning_rate',
                                initialization='checkpoint_initialization_not_exact_resume')
            atomic_json(output / 'fork_source.json', source_facts)
        trainer = HierarchicalTrainer(high, low, device, HEFTCache(output / 'heft_cache'), config.get('reward', {}), **_trainer_kwargs(config))
        recorder = DiagnosticsRecorder(output) if diagnostics_enabled else None
        if recorder is not None:
            trainer.diagnostics = recorder.capture
        history, validation_history, seen = [], [], set()
        episode, transitions, training_seconds = 0, 0, 0.0
        optimizer_updates = {'high': 0, 'low': 0}
        ppo_update_calls = {'high': 0, 'low': 0}
        scheduler_steps = {'high': 0, 'low': 0}
        best = float('inf')

        def validate(label):
            policy = CPNHRLDAGPolicy(high.model, low.model, device)
            records, summary = Evaluator(config['environment'], progress.evaluation(label)).evaluate(policy, splits['validation'])
            if summary['valid_schedule_rate'] != 1.0 or not np.isfinite(summary['mean_ratio']):
                raise RuntimeError('invalid validation schedules or metrics')
            return records, float(summary['mean_ratio'])

        def save_state(path, current_episode, current_best):
            HierarchicalTrainer.save(path, high, low, config, resume_offset + current_episode, current_best,
                                     training_state=dict(
                                         phase_schedule=[dict(name=item['name'], episodes=item['episodes'], frozen_high_mode=item['frozen_high_mode']) for item in schedule],
                                         configured_coverage_epochs=config['training'].get('coverage_epochs'),
                                         episodes_completed=current_episode, resume_offset=resume_offset,
                                         optimizer_updates=dict(optimizer_updates), ppo_update_calls=dict(ppo_update_calls),
                                         optimizer_update_unit='optimizer.step minibatches', scheduler_steps=dict(scheduler_steps),
                                         scenario_order='epoch_permutation_seed_%d' % seed, exact_resume_supported=False,
                                         fork_source=source_facts))

        try:
            # A latest checkpoint carries the run's historical best score, not
            # necessarily its own score. Re-evaluate the actual starting weights
            # and save them as best so a regressing branch retains its baseline.
            _, best = validate('fork_initial' if fork_checkpoint else 'initial')
            validation_history.append(dict(step=resume_offset, epoch=0, mean_ratio=best, best_mean_ratio=best))
            history.append(dict(global_step=resume_offset, episode=0,
                                phase='fork_initial' if fork_checkpoint else 'initial', validation_mean_ratio=best))
            save_state(output / 'best.pt', 0, best)
            save_state(output / 'latest.pt', 0, best)
            atomic_json(output / 'validation_history.json', validation_history)
            progress.emit('validation_complete', step=resume_offset, mean_ratio=best, best_ratio=best)
            episode_policies = [item['policy'] for item in schedule for _ in range(item['episodes'])]
            order_seed = np.random.default_rng(seed)
            epoch, index_in_epoch, permutation = 0, 0, None
            with (output / 'train_log.csv').open('x', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=TRAIN_FIELDS)
                writer.writeheader()
                handle.flush()
                while episode < total:
                    if permutation is None or index_in_epoch >= count:
                        epoch += 1
                        permutation = order_seed.permutation(count)
                        index_in_epoch = 0
                    scenario = splits['train'][int(permutation[index_in_epoch])]
                    index_in_epoch += 1
                    policy = episode_policies[episode]
                    phase = policy.name
                    progress.emit('episode_start', episode=episode + 1, total_episodes=total, epoch=epoch,
                                  phase=phase, scenario_id=scenario.scenario_id, num_tasks=scenario.num_tasks)
                    started = perf_counter()
                    metrics = trainer._episode(scenario, policy)
                    duration = perf_counter() - started
                    if not all(np.isfinite(value) for value in metrics.values() if isinstance(value, (int, float)) and not isinstance(value, bool)):
                        raise RuntimeError('non-finite training metrics')
                    episode += 1
                    transitions += scenario.num_tasks
                    seen.add(scenario.scenario_id)
                    training_seconds += duration
                    for level in policy.resolved_trainable:
                        optimizer_updates[level] += int(metrics[f'{level}_update_batches'])
                        ppo_update_calls[level] += 1
                        scheduler_steps[level] += int(getattr(trainer, level).lr_scheduler is not None)
                    row = dict(metrics, global_step=resume_offset + episode, episode=episode, epoch=epoch,
                               trainable_modules='+'.join(policy.resolved_trainable) or 'none',
                               scenario_id=scenario.scenario_id, num_tasks=scenario.num_tasks,
                               transitions=transitions, coverage=len(seen),
                               high_optimizer_updates=optimizer_updates['high'], low_optimizer_updates=optimizer_updates['low'],
                               episode_seconds=duration)
                    writer.writerow({name: row.get(name) for name in TRAIN_FIELDS})
                    handle.flush()
                    history.append(row)
                    if recorder is not None:
                        recorder.flush_captured(dict(episode=episode, global_step=resume_offset + episode, epoch=epoch,
                                                     scenario_id=scenario.scenario_id, num_tasks=scenario.num_tasks,
                                                     transitions=transitions, coverage=len(seen)))
                    save_state(output / 'latest.pt', episode, best)
                    progress.emit('episode_complete', total_episodes=total, episode=episode, epoch=epoch, phase=phase,
                                  scenario_id=scenario.scenario_id, num_tasks=scenario.num_tasks, reward=row['reward'],
                                  makespan_ratio=row['makespan_ratio'], decisions=row['decisions'],
                                  high_loss=row['high_loss'], low_loss=row['low_loss'], coverage=len(seen),
                                  transitions=transitions, episode_seconds=duration)
                    complete_epoch = index_in_epoch >= count or episode >= total
                    if complete_epoch:
                        _, ratio = validate(f'epoch_{epoch}')
                        if ratio < best:
                            best = ratio
                            save_state(output / 'best.pt', episode, best)
                        save_state(output / 'latest.pt', episode, best)
                        validation_history.append(dict(step=resume_offset + episode, epoch=epoch, mean_ratio=ratio, best_mean_ratio=best))
                        atomic_json(output / 'validation_history.json', validation_history)
                        progress.emit('validation_complete', step=resume_offset + episode, epoch=epoch, mean_ratio=ratio, best_ratio=best)
                atomic_json(output / 'budget.json', dict(planned=budget, planned_episodes=total,
                                                         executed=dict(episodes=episode, transitions=transitions,
                                                                       unique_scenarios=len(seen),
                                                                       coverage_epochs=episode / count,
                                                                       complete_coverage_epochs=episode // count,
                                                                       optimizer_updates=optimizer_updates,
                                                                       optimizer_update_unit='optimizer.step minibatches',
                                                                       ppo_update_calls=ppo_update_calls,
                                                                       scheduler_steps=scheduler_steps),
                                                         resume_offset=resume_offset, fork_source=source_facts))
        finally:
            if recorder is not None:
                recorder.close()
        save_state(output / 'latest.pt', episode, best)
        if not (output / 'best.pt').is_file():
            raise RuntimeError('initial best checkpoint was not saved')
        selected = trainer.load(output / 'best.pt', high, low, device)
        records, _ = validate('selected_best')
        report = summarize(records, model='CPN-HRL-DAG', split='validation', seed=seed,
                           config_hash=config_hash(config), training_time_seconds=training_seconds,
                           bootstrap_samples=int(config['evaluation'].get('bootstrap_samples', 1000)))
        report.update(training_episodes=episode, transitions=transitions, coverage=len(seen),
                      selected_step=int(selected['global_step']), training_time_scope='episode collection and PPO updates',
                      optimizer_updates_high=optimizer_updates['high'], optimizer_updates_low=optimizer_updates['low'],
                      optimizer_update_unit='optimizer.step minibatches', ppo_update_calls=ppo_update_calls,
                      scheduler_steps_high=scheduler_steps['high'], scheduler_steps_low=scheduler_steps['low'],
                      diagnostics_enabled=diagnostics_enabled, fork_source=source_facts,
                      matched_scenario_coverage='episode_budget_identical_across_fork_branches')
        write_report(output, records, report)
        write_training_curve(output, history)
        with (output / 'validation_log.csv').open('x', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=['step', 'validation_mean_ratio'])
            writer.writeheader()
            writer.writerows(dict(step=row['step'], validation_mean_ratio=row['mean_ratio']) for row in validation_history if row.get('step') is not None)
        progress.emit('complete', status='complete', episodes=episode, transitions=transitions, coverage=len(seen),
                      mean_ratio=report['mean_ratio'], selected_step=selected['global_step'],
                      optimizer_updates=optimizer_updates, scheduler_steps=scheduler_steps)
    return output / 'best.pt'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--fork-from', help='Source HRL checkpoint used to initialize a new experiment')
    arguments = parser.parse_args()
    print(train_main(load_config(arguments.config), arguments.fork_from), flush=True)
