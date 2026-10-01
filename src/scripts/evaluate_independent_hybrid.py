"""Evaluate independent HEFT/search/HRL selection on the fixed 108 validation scenes."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np
import pandas as pd
import torch
import yaml

from compare_main_models import restore_hrl, validate_protocol, paired_statistics
from cpn_hrl_dag.evaluation import Evaluator, summarize, write_report
from cpn_hrl_dag.experiments.graph_baselines import load_fixed_splits, prepare_output, protocol_signature
from cpn_hrl_dag.policies.hybrid import IndependentHybridPolicy
from cpn_hrl_dag.utils.config import config_hash, load_config
from cpn_hrl_dag.utils.progress import ProgressLog, atomic_json
from cpn_hrl_dag.utils.runtime import runtime_metadata
from cpn_hrl_dag.utils.seed import seed_everything


def attribution(rows):
    """Use ratio tolerance for meaningful wins, not the exact floating-point tie rule."""
    tolerance = 1e-9
    no_hrl = np.array([min(r['heft_ratio'], r['search_blocks_ratio']) for r in rows])
    no_search = np.array([min(r['heft_ratio'], r['residual_hrl_ratio']) for r in rows])
    combined = np.array([r['hybrid_ratio'] for r in rows])
    hrl_gain = no_hrl - combined
    search_gain = no_search - combined
    if np.any(hrl_gain < -tolerance) or np.any(search_gain < -tolerance):
        raise AssertionError('hybrid is worse than one of its completed candidate subsets')
    return dict(ratio_tolerance=tolerance, selected_source_counts=dict(Counter(r['selected_source'] for r in rows)),
                hrl_strict_improvements=int(np.sum(hrl_gain > tolerance)),
                hrl_mean_ratio_reduction=float(hrl_gain.mean()),
                search_strict_improvements=int(np.sum(search_gain > tolerance)),
                search_mean_ratio_reduction=float(search_gain.mean()),
                hrl_matches_best=int(sum(abs(r['residual_hrl_ratio']-r['hybrid_ratio']) <= tolerance for r in rows)),
                hrl_tied_best=int(sum(abs(r['residual_hrl_ratio']-r['hybrid_ratio']) <= tolerance
                                      and min(r['heft_ratio'],r['search_blocks_ratio']) <= r['hybrid_ratio']+tolerance
                                      for r in rows)),
                no_hrl_mean_ratio=float(no_hrl.mean()), no_search_mean_ratio=float(no_search.mean()),
                note='Leave-one-branch-out makespans reuse independently completed candidates; '
                     'no-search is HEFT+HRL, not the existing HRL-safe portfolio.')


def write_overview(output, summaries, contribution, paired):
    lines = ['# HRL＋独立搜索＋HEFT 组合验证', '',
             '同一 checkpoint、搜索配置、seed 和原 108 场景；不训练，不评估 test。', '',
             '| 方法 | mean_ratio | 相对 HEFT 的平均 ratio 降幅 | 平均推理秒 | 合法率 |',
             '| --- | ---: | ---: | ---: | ---: |']
    for name, report in summaries.items():
        lines.append(f"| {name} | {report['mean_ratio']:.9f} | {(1-report['mean_ratio'])*100:.3f}% | "
                     f"{report['mean_inference_time_ms']/1000:.4f} | {report['valid_schedule_rate']:.1%} |")
    lines += ['', f"HRL 严格改善：{contribution['hrl_strict_improvements']}/108 场景；"
              f"相对移除 HRL 的平均 ratio 降低 {contribution['hrl_mean_ratio_reduction']:.9f}。",
              f"搜索严格改善：{contribution['search_strict_improvements']}/108 场景；"
              f"相对移除搜索的平均 ratio 降低 {contribution['search_mean_ratio_reduction']:.9f}。",
              '', '选中来源（精确最小值；完全相同时优先 HEFT、搜索、HRL）：',
              json.dumps(contribution['selected_source_counts'], ensure_ascii=False), '',
              '## 解释边界', '',
              '- 组合结果属于学习与搜索的系统结果，不能全部归因于 HRL。',
              '- 三个候选独立生成，组合逐场景不劣于任何完成且合法的候选。',
              '- 分支耗时在本次组合运行内测量，范围与 Evaluator 一致：policy.reset＋完整调度执行。',
              '- 组合耗时包含三路候选、候选环境初始化、校验和最终重放；未使用并行耗时或缓存历史结果。',
              '- 分支计时是进程内测量，不是隔离进程的性能基准；不可直接与其他机器历史耗时比较。',
              '- 去分支消融基于独立候选重新择优，未声称测得这些子集的独立端到端耗时。',
              '- validation 参与原模型选模；bootstrap 按基础 DAG 分组，不代表跨 seed 稳定性。',
              '- 国产操作系统实测仍需另行提供，本结果不构成系统合规证明。', '',
              '## 配对差值（组合减去对照）', '']
    for item in paired:
        lines.append(f"- {item['reference']}：均值 {item['mean_delta']:.9f}，95% CI {item['ci95']}，"
                     f"改善/退化/持平 {item['candidate_wins']}/{item['candidate_losses']}/{item['ties']}。")
    if contribution['hrl_strict_improvements'] == 0:
        lines += ['', '**本次未观察到 HRL 相对纯搜索的实质增量；当前主要性能收益来自搜索。**']
    (output / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def evaluate(config, output_override=None):
    if config.get('split', 'validation') != 'validation':
        raise ValueError('this entry evaluates only the fixed validation split')
    torch.set_num_threads(1)
    checkpoint_path = Path(config['checkpoint'])
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    reference = checkpoint['config']
    if checkpoint.get('config_hash') != config_hash(reference):
        raise ValueError('checkpoint config hash mismatch')
    search_config = load_config(config['search_config'])
    validate_protocol(search_config, reference)
    seed = int(reference['experiment']['seed'])
    seed_everything(seed, disable_cudnn=True)
    output = prepare_output(output_override or config['output_dir'])
    with ProgressLog(output, 'independent_hybrid', seed) as progress:
        (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        (output / 'search_config.yaml').write_text(yaml.safe_dump(search_config, sort_keys=False), encoding='utf-8')
        (output / 'model_config.yaml').write_text(yaml.safe_dump(reference, sort_keys=False), encoding='utf-8')
        provenance = dict(checkpoint=str(checkpoint_path), checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                          checkpoint_step=int(checkpoint['global_step']), checkpoint_config_hash=checkpoint['config_hash'],
                          search_config_hash=config_hash(search_config), protocol_signature=protocol_signature(reference),
                          seed=seed, runtime=runtime_metadata(), split='validation', retrained=False, test_evaluated=False,
                          independent_search=True, evaluation_threads=1,
                          branch_timing='in-process policy.reset plus full rollout, excludes environment reset',
                          hybrid_timing='all candidates, their environment setup/validation, selection and final replay')
        atomic_json(output / 'provenance.json', provenance)
        progress.emit('loading_fixed_splits')
        splits, manifest = load_fixed_splits(reference)
        manifest.write(output / 'split_manifest.json')
        scenes = splits['validation']
        if len(scenes) != 108 or len({s.metadata['original_graph_id'] for s in scenes}) != 54:
            raise ValueError('requires exactly 108 scenes / 54 base DAGs')
        policy = IndependentHybridPolicy(restore_hrl(checkpoint), search_config=search_config['search'], seed=seed)
        all_records = {name: [] for name in ('heft', 'residual_hrl', 'search_blocks', 'hybrid')}
        audits = []
        with (output / 'candidate_audit.jsonl').open('x', encoding='utf-8') as audit_file, \
             (output / 'candidate_schedules.jsonl').open('x', encoding='utf-8') as plans_file:
            for index, scenario in enumerate(scenes, 1):
                progress.emit('scene_start', completed=index-1, total=len(scenes), scenario_id=scenario.scenario_id)
                records, _ = Evaluator({'normalize_observations': True}).evaluate(policy, [scenario])
                combined = replace(records[0], policy='hybrid')
                if not combined.valid_schedule or not np.isclose(combined.makespan, policy.selected_makespan, rtol=0, atol=1e-9):
                    raise RuntimeError('hybrid replay disagrees with selected candidate')
                all_records['hybrid'].append(combined)
                audit = dict(scenario_id=scenario.scenario_id, base_dag_id=combined.base_dag_id,
                             num_tasks=scenario.num_tasks, resource_type=scenario.scenario_id.rsplit(':', 1)[-1],
                             heft_makespan=combined.heft_makespan, hybrid_ratio=combined.ratio,
                             hybrid_makespan=combined.makespan, hybrid_inference_time_ms=combined.inference_time_ms,
                             selected_source=policy.selected_candidate, search_internal_source=policy.search_selected_candidate,
                             hybrid_reset_time_ms=policy.reset_time_ms)
                for candidate in policy.candidates:
                    result = candidate.result
                    if combined.makespan > result.makespan + 1e-9:
                        raise AssertionError('combination lost a completed candidate')
                    record = replace(combined, policy=candidate.name, makespan=result.makespan,
                                     ratio=result.makespan/combined.heft_makespan, inference_time_ms=candidate.inference_time_ms)
                    all_records[candidate.name].append(record)
                    audit.update({f'{candidate.name}_ratio': record.ratio, f'{candidate.name}_makespan': record.makespan,
                                  f'{candidate.name}_inference_time_ms': candidate.inference_time_ms,
                                  f'{candidate.name}_wall_time_ms': candidate.wall_time_ms})
                    plans_file.write(json.dumps(dict(scenario_id=scenario.scenario_id, candidate=candidate.name,
                                                    makespan=result.makespan, decisions=candidate.decisions))+'\n')
                audits.append(audit)
                audit_file.write(json.dumps(audit, allow_nan=False)+'\n')
                audit_file.flush()
                plans_file.flush()
                progress.emit('scene_complete', completed=index, total=len(scenes), scenario_id=scenario.scenario_id,
                              ratio=combined.ratio, selected_source=policy.selected_candidate)
        frames, summaries = [], {}
        for name, records in all_records.items():
            report = summarize(records, model=name, split='validation', seed=seed,
                               config_hash=config_hash(config), bootstrap_samples=1000)
            report.update(category={'heft':'reference','residual_hrl':'pure_learning','search_blocks':'search_only',
                                    'hybrid':'learning_plus_independent_search'}[name],
                          retrained=False, evaluation_device='cpu', timing_scope=provenance['hybrid_timing' if name=='hybrid' else 'branch_timing'])
            # summarize uses this field name in the shared reporting module.
            report['mean_inference_time_ms'] = float(np.mean([r.inference_time_ms for r in records]))
            report['p95_inference_time_ms'] = float(np.quantile([r.inference_time_ms for r in records], .95))
            write_report(output/name, records, report)
            summaries[name] = report
            frame = pd.DataFrame([asdict(r) for r in records]); frame['method'] = name
            frames.append(frame)
        frame = pd.concat(frames, ignore_index=True)
        frame.to_csv(output/'per_scene_all.csv', index=False)
        pd.DataFrame(audits).to_csv(output/'candidate_audit.csv', index=False)
        pd.DataFrame(summaries.values()).to_csv(output/'comparison.csv', index=False)
        contribution = attribution(audits)
        paired = [paired_statistics(frame, name, 'hybrid', seed=seed) for name in ('heft','residual_hrl','search_blocks')]
        atomic_json(output/'attribution.json', contribution)
        atomic_json(output/'paired_statistics.json', paired)
        write_overview(output, summaries, contribution, paired)
        progress.emit('complete', status='complete', num_scenarios=108, num_base_dags=54,
                      hybrid_mean_ratio=summaries['hybrid']['mean_ratio'],
                      hrl_strict_improvements=contribution['hrl_strict_improvements'])
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/hrl_independent_hybrid_seed2026.yaml')
    parser.add_argument('--output', help='New result directory (existing nonempty directory is refused)')
    args = parser.parse_args()
    print(evaluate(load_config(args.config), args.output), flush=True)
