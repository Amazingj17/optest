"""Plot completed, paired validation runs without evaluating test scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ('graph_ppo', 'tier_mappo')
LABELS = {'graph_ppo': 'Graph PPO', 'tier_mappo': 'Tier MAPPO'}
COLORS = {'graph_ppo': '#2878b5', 'tier_mappo': '#d46a28'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_figure(figure, output, name):
    figure.savefig(output / f'{name}.png', dpi=170, facecolor='white')
    figure.savefig(output / f'{name}.svg', facecolor='white')
    plt.close(figure)


def plot_comparison(comparison_dir, output=None):
    source = Path(comparison_dir).resolve()
    output = Path(output).resolve() if output else source / 'figures'
    metadata = read_json(source / 'comparison.json')
    if metadata['split'] != 'validation' or metadata['test_evaluated']:
        raise ValueError('plots require validation-only comparison results')
    if metadata['num_scenarios'] != 108:
        raise ValueError('plots require all 108 fixed validation scenarios')
    reports = pd.read_csv(source / 'comparison.csv')
    paired = pd.read_csv(source / 'paired_per_scene.csv')
    seeds = sorted(reports.seed.unique())
    if reports.duplicated(['method', 'seed']).any():
        raise ValueError('duplicate method/seed results')
    if paired.duplicated(['seed', 'scenario_id']).any():
        raise ValueError('duplicate paired scenario results')
    if len(paired) != 108 * len(seeds):
        raise ValueError('incomplete paired validation results')
    scenes, placements, histories, training, training_summaries = [], [], [], [], []
    for seed in seeds:
        if set(reports.loc[reports.seed == seed, 'method']) != set(METHODS):
            raise ValueError('each seed requires both methods')
        pairs = paired.loc[paired.seed == seed]
        if len(pairs) != 108 or pairs.base_dag_id.nunique() != 54:
            raise ValueError('each seed requires 108 scenarios from 54 DAGs')
        for method in METHODS:
            report = reports.loc[(reports.seed == seed) & (reports.method == method)].iloc[0]
            folder = source / f'{method}_seed_{seed}'
            scene = pd.read_csv(folder / 'per_scene.csv')
            if len(scene) != 108 or scene.scenario_id.duplicated().any() or set(scene.scenario_id) != set(pairs.scenario_id):
                raise ValueError('per-scene and paired records differ')
            aligned = pairs.set_index('scenario_id').loc[scene.scenario_id, f'{method}_ratio'].to_numpy()
            if not np.allclose(scene.ratio, aligned, rtol=0, atol=1e-9):
                raise ValueError('paired ratios do not match per-scene records')
            if not np.isfinite(scene.ratio).all() or not np.isclose(scene.ratio.mean(), report.mean_ratio):
                raise ValueError('invalid or inconsistent ratios')
            scene['resource_type'] = scene.scenario_id.str.rsplit(':', n=1).str[-1]
            if not set(scene.resource_type).issubset({'homogeneous', 'heterogeneous'}):
                raise ValueError('unknown resource type')
            scene['method'], scene['seed'] = method, seed
            scenes.append(scene)
            placement = pd.read_csv(folder / 'tier_placements.csv')
            placement = scene[['scenario_id', 'num_tasks', 'resource_type']].merge(placement, on='scenario_id', validate='one_to_one')
            if len(placement) != 108 or not (placement[list(('end', 'edge', 'cloud'))].sum(axis=1) == placement.num_tasks).all():
                raise ValueError('placement totals must equal executed task counts')
            placement['method'], placement['seed'] = method, seed
            placements.append(placement)
            checkpoint = Path(report.checkpoint)
            if not checkpoint.is_absolute():
                checkpoint = Path(__file__).resolve().parents[1] / checkpoint
            run = checkpoint.parent
            history = pd.DataFrame(read_json(run / 'validation_history.json'))
            history['method'], history['seed'] = method, seed
            histories.append(history)
            log = pd.read_csv(run / 'train_log.csv')
            log['method'], log['seed'] = method, seed
            training.append(log)
            summary = read_json(folder / 'summary.json')
            training_summaries.append(dict(method=method, seed=seed, epochs=int(log.epoch.max()),
                                           episodes=len(log), transitions=int(log.transitions.sum()),
                                           training_seconds=summary['training_time_seconds']))
    if not np.allclose(paired.delta_mappo_minus_ppo, paired.tier_mappo_ratio - paired.graph_ppo_ratio, rtol=0, atol=1e-9):
        raise ValueError('paired delta sign or values are inconsistent')
    scenes, placements = pd.concat(scenes), pd.concat(placements)
    histories, training = pd.concat(histories), pd.concat(training)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titleweight': 'bold'})
    scope = f'Fixed validation: 108 scenarios / 54 base DAGs | {len(seeds)} seed(s) | no test evaluation'
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), layout='constrained')
    figure.suptitle('Pure learned scheduling policies vs HEFT\n' + scope, fontsize=15)
    means = [reports.loc[reports.method == method, 'mean_ratio'].mean() for method in METHODS]
    axes[0, 0].bar(['HEFT'] + [LABELS[method] for method in METHODS], [1.0] + means,
                   color=['#9299a1'] + [COLORS[method] for method in METHODS])
    axes[0, 0].axhline(1, color='#555555', ls='--', lw=1)
    for position, value in enumerate([1.0] + means):
        axes[0, 0].text(position, value, f'{value:.4f}', ha='center', va='bottom')
    axes[0, 0].set(title='A  Mean makespan / HEFT', ylabel='Ratio (lower is better)', ylim=(0, max(1, *means) * 1.2))
    for seed in seeds:
        points = paired.loc[paired.seed == seed]
        axes[0, 1].scatter(points.graph_ppo_ratio, points.tier_mappo_ratio, s=20, alpha=.55, label=f'Seed {seed}')
    low = min(1, paired.graph_ppo_ratio.min(), paired.tier_mappo_ratio.min()) * .95
    high = max(1, paired.graph_ppo_ratio.max(), paired.tier_mappo_ratio.max()) * 1.05
    axes[0, 1].plot([low, high], [low, high], color='#666666', ls='--', lw=1)
    axes[0, 1].axhline(1, color='#bbbbbb', lw=.8)
    axes[0, 1].axvline(1, color='#bbbbbb', lw=.8)
    axes[0, 1].set(title='B  Paired scenarios: below line favors MAPPO', xlabel='Graph PPO ratio', ylabel='Tier MAPPO ratio', xlim=(low, high), ylim=(low, high))
    axes[0, 1].legend(fontsize=8)
    axes[0, 2].hist(paired.delta_mappo_minus_ppo, bins=22, color='#6d83a0', edgecolor='white')
    axes[0, 2].axvline(0, color='#333333', ls='--')
    axes[0, 2].set(title='C  Paired difference', xlabel='MAPPO - PPO ratio (negative favors MAPPO)', ylabel='Seed-scenario pairs')
    axes[0, 2].text(.98, .95, f'MAPPO wins / loses / ties\n{metadata["mappo_better"]} / {metadata["mappo_worse"]} / {metadata["tied"]}',
                    transform=axes[0, 2].transAxes, ha='right', va='top', bbox={'facecolor': 'white', 'alpha': .85, 'edgecolor': 'none'})
    for axis, field, title in [(axes[1, 0], 'num_tasks', 'D  Task-size groups'), (axes[1, 1], 'resource_type', 'E  Resource groups')]:
        categories = sorted(scenes[field].unique())
        positions = np.arange(len(categories))
        for index, method in enumerate(METHODS):
            subset = scenes.loc[scenes.method == method]
            values = subset.groupby(field).ratio.mean().reindex(categories)
            axis.bar(positions + (index - .5) * .36, values, width=.36, label=LABELS[method], color=COLORS[method])
        counts = scenes.loc[(scenes.method == METHODS[0]) & (scenes.seed == seeds[0])].groupby(field).size()
        axis.set_xticks(positions, [f'{value}\nn={counts[value]}/seed' for value in categories])
        axis.axhline(1, color='#555555', ls='--', lw=1)
        axis.set(title=title, ylabel='Mean ratio (lower is better)')
        axis.legend(fontsize=8)
    positions = np.arange(2)
    for index, field in enumerate(['mean_inference_time_ms', 'p95_inference_time_ms']):
        values = [reports.loc[reports.method == method, field].mean() / 1000 for method in METHODS]
        axes[1, 2].bar(positions + (index - .5) * .36, values, width=.36, color=['#697d91', '#bcc8d3'][index], label=['Mean', 'P95'][index])
    axes[1, 2].set_xticks(positions, [LABELS[method] for method in METHODS])
    axes[1, 2].set(title=f'F  Full-schedule latency ({metadata["evaluation_device"]})', ylabel='Seconds / scenario (lower is better)')
    axes[1, 2].legend()
    save_figure(figure, output, 'comparison_dashboard')

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    figure.suptitle('Training and checkpoint selection\n' + scope)
    for method in METHODS:
        for seed in seeds:
            history = histories.loc[(histories.method == method) & (histories.seed == seed)]
            axes[0].plot(history.epoch, history.mean_ratio, marker='o', color=COLORS[method], label=f'{LABELS[method]} / {seed}')
            axes[0].plot(history.epoch, history.best_mean_ratio, ls=':', color=COLORS[method], alpha=.65)
            log = training.loc[(training.method == method) & (training.seed == seed)]
            smooth = log.ratio.rolling(100, min_periods=1).mean()
            axes[1].plot(log.episode, smooth, color=COLORS[method], label=f'{LABELS[method]} / {seed}')
    axes[0].axhline(1, color='#555555', ls='--', lw=1)
    axes[0].set(title='Full 108-scene validation; dotted = best so far', xlabel='Epoch', ylabel='Mean validation ratio')
    axes[0].set_xticks(sorted(histories.epoch.unique()))
    axes[1].set(title='Stochastic training (100-episode rolling mean)', xlabel='Training episode', ylabel='Training ratio (not validation)')
    for axis in axes:
        axis.legend(fontsize=8)
        axis.grid(alpha=.15)
    save_figure(figure, output, 'learning_curves')

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    figure.suptitle('Executed task placement shares (task-weighted, not proposals)\n' + scope)
    placement_rows = []
    for axis, resource_type in zip(axes, ['homogeneous', 'heterogeneous']):
        grouped = placements.loc[placements.resource_type == resource_type].groupby('method')[['end', 'edge', 'cloud']].sum().reindex(METHODS)
        shares = grouped.div(grouped.sum(axis=1), axis=0) * 100
        bottom = np.zeros(len(METHODS))
        for tier, color in [('end', '#78a88b'), ('edge', '#d5a14c'), ('cloud', '#628bbc')]:
            axis.bar([LABELS[method] for method in METHODS], shares[tier], bottom=bottom, label=tier.title(), color=color)
            for index, value in enumerate(shares[tier]):
                if value >= 5:
                    axis.text(index, bottom[index] + value / 2, f'{value:.1f}%', ha='center', va='center')
            bottom += shares[tier].to_numpy()
        axis.set(title=resource_type.title(), ylabel='Executed tasks (%)', ylim=(0, 110))
        axis.legend(loc='upper right', ncol=3, fontsize=8)
        for method in METHODS:
            placement_rows.append(dict(method=method, resource_type=resource_type, **{f'{tier}_percent': shares.loc[method, tier] for tier in ('end', 'edge', 'cloud')}))
    save_figure(figure, output, 'tier_placements')
    pd.DataFrame(placement_rows).to_csv(output / 'placement_shares.csv', index=False)
    scenes.groupby(['method', 'num_tasks', 'resource_type']).ratio.agg(['count', 'mean', 'std']).to_csv(output / 'grouped_ratios.csv')
    lines = ['# Graph PPO / Tier MAPPO 对比实验结果', '', scope, '',
             '正式比较仅使用固定 validation；未执行 test。比值越低越好，HEFT 基准为 1。', '',
             'Ratios are makespan / same-scenario HEFT; lower is better. HEFT = 1.', '',
             '| Method | Mean ratio | Maximum ratio | Valid | Mean latency (s) | P95 latency (s) | Parameters | Selected epoch |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in reports.itertuples():
        lines.append(f'| {LABELS[row.method]} / {row.seed} | {row.mean_ratio:.6f} | {row.max_ratio:.6f} | {row.valid_schedule_rate:.1%} | {row.mean_inference_time_ms / 1000:.4f} | {row.p95_inference_time_ms / 1000:.4f} | {row.parameter_count} | {row.selected_epoch} |')
    lines += ['', '## 训练预算与耗时', '',
              '| 方法 / seed | Epoch | 训练场景次数 | 调度步数 | 采样与更新耗时（小时） |',
              '| --- | ---: | ---: | ---: | ---: |']
    for entry in training_summaries:
        lines.append(f'| {LABELS[entry["method"]]} / {entry["seed"]} | {entry["epochs"]} | {entry["episodes"]} | {entry["transitions"]} | {entry["training_seconds"] / 3600:.3f} |')
    lines += ['', '训练耗时不包含数据加载和周期性 validation；推理时间包含策略初始化和完整调度回放。', '']
    lines += ['', '## Paired difference (MAPPO minus PPO)', '']
    for seed, paired_summary in metadata['paired_by_seed'].items():
        low_ci, high_ci = paired_summary['ci95']
        lines.append(f'- Seed {seed}: mean delta {paired_summary["mean_delta"]:.6f}; base-DAG grouped bootstrap 95% CI [{low_ci:.6f}, {high_ci:.6f}]. Negative favors MAPPO.')
    lines += ['', '## Interpretation limits', '',
              '- Validation selects the checkpoints and is not independent test evidence.',
              '- A single seed cannot establish training-seed robustness; multiple seeds remain descriptive here.',
              '- The methods share the backbone configuration and transition budget, not parameter count or compute cost.',
              '- Latency includes policy setup and full environment replay; it is not neural forward-pass latency.',
              '- Placement shares count executed tasks, so larger DAGs contribute more weight.',
              '- These are pure learned policies without HEFT safety fallback; ratios above one are possible.', '',
              '![Comparison](comparison_dashboard.png)', '', '![Learning curves](learning_curves.png)', '', '![Placement](tier_placements.png)', '']
    (output / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', required=True)
    parser.add_argument('--output')
    args = parser.parse_args()
    print(plot_comparison(args.comparison, args.output))


if __name__ == '__main__':
    main()
