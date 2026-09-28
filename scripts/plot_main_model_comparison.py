"""Plot and report the completed main-model/full-validation comparison."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ['heft', 'graph_ppo', 'tier_mappo', 'residual_hrl', 'hrl_safe', 'search_blocks']
LABELS = ['HEFT', 'Graph PPO', 'Tier MAPPO', 'Main: Residual HRL', 'Main: HRL-safe', 'Search: beam + blocks']
COLORS = ['#9b9b9b', '#2878b5', '#779fc3', '#c17738', '#9b5aa1', '#43876b']


def save_figure(figure, output, name):
    for suffix in ['png', 'svg']:
        figure.savefig(output / f'{name}.{suffix}', dpi=170, facecolor='white')
    plt.close(figure)


def plot_main_comparison(source):
    source = Path(source).resolve()
    status = json.loads((source / 'status.json').read_text(encoding='utf-8'))
    if status['status'] not in ['evaluated', 'complete'] or status['split'] != 'validation' or status['test_evaluated']:
        raise ValueError('requires a completed validation-only evaluation')
    summary = pd.read_csv(source / 'comparison.csv')
    frame = pd.read_csv(source / 'per_scene_all.csv')
    pairs = json.loads((source / 'paired_statistics.json').read_text(encoding='utf-8'))
    if set(summary.model) != set(METHODS) or summary.model.duplicated().any():
        raise ValueError('requires all six distinct methods')
    if frame.duplicated(['method', 'scenario_id']).any() or set(frame.method) != set(METHODS):
        raise ValueError('invalid per-scene method records')
    reference = frame.loc[frame.method == 'heft'].set_index('scenario_id').sort_index()
    if len(reference) != 108 or reference.base_dag_id.nunique() != 54:
        raise ValueError('requires fixed 108 scenarios / 54 DAGs')
    summary = summary.set_index('model').loc[METHODS]
    for method in METHODS:
        subset = frame.loc[frame.method == method].set_index('scenario_id').sort_index()
        if not subset.index.equals(reference.index) or not subset.base_dag_id.equals(reference.base_dag_id):
            raise ValueError('scenario identity mismatch')
        if not np.allclose(subset.heft_makespan, reference.heft_makespan, rtol=0, atol=1e-9):
            raise ValueError('HEFT denominator mismatch')
        if not np.isclose(subset.ratio.mean(), summary.loc[method, 'mean_ratio']):
            raise ValueError('summary ratio mismatch')
    output = source / 'figures'
    output.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titleweight': 'bold'})
    scope = f'Fixed validation: 108 scenarios / 54 DAGs | seed {status.get("seed", 2026)} | CPU | no test'
    budget_note = 'Scene coverage matched; compute budgets and architectures differ' if status.get('matched_scene_coverage') else 'Training budgets may differ; inspect the recorded budgets'
    figure, axes = plt.subplots(2, 2, figsize=(16, 10), layout='constrained')
    figure.suptitle('Main models and graph baselines\n' + scope + '\n' + budget_note + '; search costs extra computation', fontsize=14)
    positions = np.arange(len(METHODS))
    axes[0, 0].barh(positions, summary.mean_ratio, color=COLORS)
    axes[0, 0].set_yticks(positions, LABELS)
    axes[0, 0].invert_yaxis()
    axes[0, 0].axvline(1, ls='--', color='#555555')
    axes[0, 0].set(title='A  Overall scheduling quality', xlabel='Mean makespan / HEFT (lower is better)', xlim=(0, summary.mean_ratio.max() * 1.15))
    for position, value in enumerate(summary.mean_ratio):
        axes[0, 0].text(value + .01, position, f'{value:.4f}', va='center')
    axes[0, 1].barh(positions - .17, summary.mean_inference_time_ms / 1000, height=.34, color='#657a91', label='Mean')
    axes[0, 1].barh(positions + .17, summary.p95_inference_time_ms / 1000, height=.34, color='#b5c1cd', label='P95')
    axes[0, 1].set_yticks(positions, LABELS)
    axes[0, 1].invert_yaxis()
    axes[0, 1].set(title='B  Full-schedule latency including planning', xlabel='Seconds / scenario (lower is better)')
    axes[0, 1].legend(loc='upper right')
    frame['resource_type'] = frame.scenario_id.str.rsplit(':', n=1).str[-1]
    if set(frame.resource_type) != {'homogeneous', 'heterogeneous'}:
        raise ValueError('unexpected resource groups')
    for axis, field, title in [(axes[1, 0], 'num_tasks', 'C  Task-size groups'),
                               (axes[1, 1], 'resource_type', 'D  Resource groups')]:
        categories = sorted(frame[field].unique())
        positions = np.arange(len(categories))
        for index, method in enumerate(METHODS):
            values = frame.loc[frame.method == method].groupby(field).ratio.mean().reindex(categories)
            axis.bar(positions + (index - 2.5) * .12, values, width=.12, color=COLORS[index], label=LABELS[index])
        axis.set_xticks(positions, categories)
        axis.axhline(1, color='#555555', ls='--', lw=1)
        group_max = frame.groupby(['method', field]).ratio.mean().max()
        axis.set(title=title, ylabel='Mean ratio (lower is better)', ylim=(0, max(1, group_max) * 1.30))
        axis.legend(fontsize=8, ncol=2, loc='upper left')
    save_figure(figure, output, 'main_comparison_dashboard')

    figure, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    figure.suptitle('Paired main-model differences: candidate minus graph baseline\n54 base-DAG groups; bootstrap 95% CI; validation-selected checkpoints')
    for axis, baseline in zip(axes, ['graph_ppo', 'tier_mappo']):
        rows = [row for row in pairs if row['reference'] == baseline]
        expected = ['residual_hrl', 'hrl_safe', 'search_blocks']
        if sorted(row['candidate'] for row in rows) != sorted(expected):
            raise ValueError('missing or duplicate paired main-model statistics')
        for position, row in enumerate(rows):
            mean, bounds = row['mean_delta'], row['ci95']
            axis.hlines(position, bounds[0], bounds[1], color=COLORS[METHODS.index(row['candidate'])], linewidth=3)
            axis.scatter(mean, position, color=COLORS[METHODS.index(row['candidate'])], s=55)
        axis.set_yticks(range(len(rows)), [LABELS[METHODS.index(row['candidate'])] for row in rows])
        axis.invert_yaxis()
        axis.axvline(0, ls='--', color='#666666')
        axis.set(title='Reference: ' + LABELS[METHODS.index(baseline)], xlabel='Delta ratio (negative favors main candidate)')
    save_figure(figure, output, 'main_paired_differences')

    placements = []
    for method in METHODS:
        counts = pd.read_csv(source / method / 'tier_placements.csv')
        scene = frame.loc[frame.method == method, ['scenario_id', 'num_tasks', 'resource_type']]
        joined = scene.merge(counts, on='scenario_id', validate='one_to_one')
        if len(joined) != 108 or not (joined[['end', 'edge', 'cloud']].sum(axis=1) == joined.num_tasks).all():
            raise ValueError('placement totals do not match task counts')
        joined['method'] = method
        placements.append(joined)
    placements = pd.concat(placements)
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), layout='constrained')
    figure.suptitle('Executed placements (task-weighted, not proposals)\n' + scope)
    share_rows = []
    for axis, resource in zip(axes, ['homogeneous', 'heterogeneous']):
        counts = placements.loc[placements.resource_type == resource].groupby('method')[['end', 'edge', 'cloud']].sum().loc[METHODS]
        shares = counts.div(counts.sum(axis=1), axis=0) * 100
        left = np.zeros(len(METHODS))
        for tier, color in [('end', '#78a88b'), ('edge', '#d5a14c'), ('cloud', '#628bbc')]:
            axis.barh(np.arange(len(METHODS)), shares[tier], left=left, label=tier.title(), color=color)
            for position, value in enumerate(shares[tier]):
                if value >= 5:
                    axis.text(left[position] + value / 2, position, f'{value:.1f}%', ha='center', va='center')
            left += shares[tier].to_numpy()
        axis.set_yticks(np.arange(len(METHODS)), LABELS)
        axis.invert_yaxis()
        axis.set(title=resource.title(), xlabel='Executed tasks (%)', xlim=(0, 100))
        axis.legend(ncol=3, loc='lower left', bbox_to_anchor=(0, -.2), fontsize=8)
        for method in METHODS:
            share_rows.append(dict(method=method, resource_type=resource, **shares.loc[method].to_dict()))
    save_figure(figure, output, 'main_tier_placements')
    pd.DataFrame(share_rows).to_csv(output / 'placement_shares.csv', index=False)
    frame.groupby(['method', 'num_tasks', 'resource_type']).ratio.agg(['count', 'mean']).to_csv(output / 'grouped_ratios.csv')
    lines = ['# 主模型与图学习基线统一比较', '', scope, '',
             '本评估阶段复用上游训练的 checkpoint，在同一 CPU 上完成固定全量 validation；评估阶段不训练、不执行 test。', '',
             '## 总体结果', '',
             '| 方法 | 类别 | Mean ratio | Max ratio | 合法率 | 平均秒/场景 | P95秒/场景 | 参数量 | 训练 episode |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for method, label in zip(METHODS, LABELS):
        row = summary.loc[method]
        lines.append(f'| {label} | {row.category} | {row.mean_ratio:.6f} | {row.max_ratio:.6f} | {row.valid_schedule_rate:.1%} | {row.mean_inference_time_ms / 1000:.4f} | {row.p95_inference_time_ms / 1000:.4f} | {row.parameter_count} | {row.training_episodes} |')
    lines += ['', '## 实际训练预算与选模', '',
              '| 方法 | Episodes | 任务决策数 | 选中 checkpoint | 训练秒数 | 计时范围 |',
              '| --- | ---: | ---: | --- | ---: | --- |']
    for method in ['residual_hrl', 'graph_ppo', 'tier_mappo']:
        row = summary.loc[method]
        lines.append(f'| {method} | {row.training_episodes} | {row.get("training_transitions", "未记录")} | {row.get("selected_checkpoint", "未记录")} {row.get("checkpoint_unit", "")} | {row.get("training_time_seconds", "未记录")} | {row.get("training_time_scope", "未记录")} |')
    lines += ['', f'场景覆盖匹配标记：{status.get("matched_scene_coverage", False)}；不表示等参数量、等更新次数或等算力。', '',
              '## 配对差值：主模型候选减去基线', '',
              '| 参考基线 | 主模型候选 | 平均差值 | 基础 DAG bootstrap 95% CI | 候选胜/负/平 |',
              '| --- | --- | ---: | --- | --- |']
    for row in pairs:
        lines.append(f'| {row["reference"]} | {row["candidate"]} | {row["mean_delta"]:.6f} | [{row["ci95"][0]:.6f}, {row["ci95"][1]:.6f}] | {row["candidate_wins"]}/{row["candidate_losses"]}/{row["ties"]} |')
    lines += ['', '负差值表示主模型候选更好；按 54 个基础 DAG 成组重采样 1000 次，每个 DAG 的同构/异构实现一起采样。', '',
              '## 解释边界', '',
              '- 纯学习比较：Residual HRL、Graph PPO、Tier MAPPO；安全组合/搜索另列，不能把其收益全部归功于神经网络。',
              f'- HRL-safe 使用 checkpoint 对应 portfolio 配置（扰动候选数：{status.get("portfolio", {}).get("perturbation_candidates", "未记录")}）加主模型；独立搜索不包含学习候选。',
              '- 训练量和最佳 checkpoint 见实际预算表，不把历史预算写成新实验事实；复用模型可能采用不同训练阶段和验证频率。',
              f'- checkpoint 由 validation 选择。主模型训练配置的 proxy_scenarios={status.get("main_proxy_scenarios", "未记录")}；0 表示全量。此次统一评估仅使用完整 108 场景。',
              '- 所有学习结果只对应一个 seed，不保证跨 seed 稳健；这里的分组置信区间不是独立 test 泛化证据。',
              '- 时间含策略初始化、候选规划和完整环境回放，不是神经网络单次前向时间；本次没有多轮交错计时。',
              '- 同构场景只有 cloud 节点；任务分配比例按任务数量加权，不等于每个场景等权。', '',
              '![总体比较](main_comparison_dashboard.png)', '',
              '![配对差值](main_paired_differences.png)', '',
              '![实际执行分配](main_tier_placements.png)', '']
    (output / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    generated_at = datetime.now(timezone.utc).isoformat()
    status.update(status='complete', stage='plots_and_report_complete', report=str(output / 'RESULTS.md'),
                  evaluation_completed_at=status.get('evaluation_completed_at', status.get('updated_at')),
                  plots_generated_at=generated_at, updated_at=generated_at)
    (source / 'status.json').write_text(json.dumps(status, indent=2) + '\n', encoding='utf-8')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', required=True)
    arguments = parser.parse_args()
    print(plot_main_comparison(arguments.comparison))
