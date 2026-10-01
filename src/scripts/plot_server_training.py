"""Plot actual incremental training logs and every recorded validation point."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def read_curves(root, main=False):
    root = Path(root)
    config = yaml.safe_load((root / 'config.yaml').read_text(encoding='utf-8'))
    if main and int(config.get('evaluation', {}).get('proxy_scenarios', 0)) != 0:
        raise ValueError('training comparison requires full validation, not proxy history')
    log = pd.read_csv(root / 'train_log.csv')
    trained = log[log.phase.isin(['low_pretrain', 'high_train', 'high_only_eft', 'joint'])].copy() if main else log.copy()
    if main:
        # The main trainer writes `global_step` plus a per-experiment `episode`
        # index; historical runs only wrote `step`.  For a forked branch
        # `global_step` continues the source run's count, which is exactly what
        # makes it a comparable axis across the two fork branches.
        step_column = 'global_step' if 'global_step' in trained.columns else 'step'
    else:
        step_column = 'episode'
    if trained[step_column].duplicated().any() or trained.empty:
        raise ValueError('training log must have unique, nonempty steps')
    trained = trained.sort_values(step_column)
    if set(trained[step_column]) != set(range(1, len(trained) + 1)):
        raise ValueError('training log has missing steps')
    if 'num_tasks' not in trained:
        if main:
            audit = json.loads((root / 'dataset_audit.json').read_text(encoding='utf-8'))
            mapping = {row['scenario_id']: row['num_tasks'] for row in audit['scenarios']}
            trained['num_tasks'] = trained.scenario_id.map(mapping)
        else:
            trained['num_tasks'] = trained.transitions
    if trained.num_tasks.isna().any() or (trained.num_tasks <= 0).any():
        raise ValueError('missing task counts for training history')
    trained['decisions'] = trained.num_tasks.cumsum()
    trained['coverage'] = (~trained.scenario_id.duplicated()).cumsum()
    if (root / 'validation_history.json').is_file():
        validation = pd.DataFrame(json.loads((root / 'validation_history.json').read_text(encoding='utf-8')))
        if 'step' not in validation:
            ends = trained.groupby('epoch')[step_column].max().to_dict()
            ends[0] = 0
            validation['step'] = validation.epoch.map(ends)
    else:
        ratio = log['proxy_validation_mean_ratio'].combine_first(log['validation_mean_ratio']) if 'proxy_validation_mean_ratio' in log else log['validation_mean_ratio']
        validation = pd.DataFrame({'step': log.step, 'mean_ratio': ratio}).dropna()
    decisions = dict(zip(trained[step_column], trained.decisions))
    decisions[0] = 0
    validation['decisions'] = validation.step.map(decisions)
    if validation.empty or validation.decisions.isna().any() or not np.isfinite(validation.mean_ratio).all():
        raise ValueError('invalid validation history')
    return trained, validation.sort_values('step')


def plot_training(main_run, graph_runs, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'refusing to overwrite {output}')
    output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), layout='constrained')
    metadata, validations = [], []
    runs = [('Residual HRL', main_run, True)]
    methods = set()
    for root in graph_runs:
        config = yaml.safe_load((Path(root) / 'config.yaml').read_text(encoding='utf-8'))
        method = config['baseline']
        if method in methods or method not in ['graph_ppo', 'tier_mappo']:
            raise ValueError('requires one graph_ppo and one tier_mappo run')
        methods.add(method)
        runs.append((method, root, False))
    if methods != {'graph_ppo', 'tier_mappo'}:
        raise ValueError('both graph baseline histories are required')
    for label, root, main in runs:
        trained, validation = read_curves(root, main)
        axes[0, 0].plot(validation.decisions, validation.mean_ratio, marker='o', ms=3, label=label)
        axes[0, 1].plot(validation.decisions, np.minimum.accumulate(validation.mean_ratio), label=label)
        axes[1, 0].plot(trained.decisions, trained.reward.rolling(50, min_periods=1).mean(), label=label)
        axes[1, 1].plot(trained.decisions, trained.coverage / 864 * 100, label=label)
        validations.append(validation.assign(method=label))
        metadata.append(dict(method=label, run=str(root), episodes=len(trained), decisions=int(trained.decisions.iloc[-1]),
                             unique_scenarios=int(trained.coverage.iloc[-1]), best_ratio=float(validation.mean_ratio.min())))
    for axis, title, ylabel in [
        (axes[0, 0], 'Every full-validation checkpoint', 'Mean ratio'),
        (axes[0, 1], 'Best validation so far', 'Mean ratio'),
        (axes[1, 0], 'Training reward (50-episode moving mean)', 'Normalized episode reward'),
        (axes[1, 1], 'Training scenario coverage', 'Unique scenarios / 864 (%)'),
    ]:
        axis.set(title=title, xlabel='Cumulative task decisions', ylabel=ylabel)
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    for axis in axes[0]:
        axis.axhline(1.0, ls='--', color='gray')
    figure.suptitle('Actual training histories | fixed full validation | budgets and phase designs may differ')
    for suffix in ['png', 'svg']:
        figure.savefig(output / f'training_comparison.{suffix}', dpi=170)
    plt.close(figure)
    pd.concat(validations).to_csv(output / 'validation_points.csv', index=False)
    pd.DataFrame(metadata).to_csv(output / 'training_budgets.csv', index=False)
    (output / 'README.md').write_text('# Training histories\n\nAll recorded full-validation points, including regressions, are shown.\n'
        'The x-axis counts task decisions, not optimizer steps or GPU compute. Phase designs and budgets may differ.\n'
        'Coverage uses the fixed 864-scene training split. Only three neural policies are trained.\n'
        '![Training](training_comparison.png)\n', encoding='utf-8')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-run', required=True)
    parser.add_argument('--graph-runs', nargs=2, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(plot_training(args.main_run, args.graph_runs, args.output))
