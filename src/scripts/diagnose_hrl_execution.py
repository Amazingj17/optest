"""Read-only stage-one diagnostic: compare execution modes of one HRL checkpoint.

This is *not* formal validation.  It runs on a fixed, pre-declared set of
training-source scenarios (one per task scale x resource type) and reports how
the same checkpoint behaves under three explicit execution behaviours:

* ``greedy``        - deterministic high task and deterministic low node
* ``repeat_sample`` - sampled high task and sampled low node, repeated K times
* ``fixed_low_eft`` - sampled high task, low level fixed at minimum EFT

Additional ``greedy_high_sample_low`` and ``greedy_high_fixed_eft`` modes hold
the high-level execution rule deterministic. Mode contrasts diagnose execution
behaviour, not the cause of a training regression: actions and visited states
can differ even with the same seed. Only ``greedy`` matches formal inference.
These samples never substitute for full 108-scenario validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from cpn_hrl_dag.algorithms.hierarchical_trainer import HierarchicalTrainer  # noqa: E402
from cpn_hrl_dag.evaluation import Evaluator  # noqa: E402
from cpn_hrl_dag.experiments.graph_baselines import load_fixed_splits  # noqa: E402
from cpn_hrl_dag.policies.heuristics import HEFTPolicy  # noqa: E402
from cpn_hrl_dag.policies.hrl import EXECUTION_MODES, HRLExecutionModePolicy  # noqa: E402
from cpn_hrl_dag.utils.config import config_hash, load_config  # noqa: E402
from cpn_hrl_dag.utils.progress import atomic_json, console_evaluation_progress  # noqa: E402
from cpn_hrl_dag.utils.seed import seed_everything  # noqa: E402

DEFAULT_MODES = EXECUTION_MODES


def resource_type(scenario) -> str:
    """Resource realization label already encoded in the scenario identity."""
    suffix = scenario.scenario_id.rsplit(':', 1)[-1]
    return suffix if suffix in {'homogeneous', 'heterogeneous'} else 'unspecified'


def select_diagnostic_samples(scenarios, per_group=1, seed=2026):
    """Deterministically pick training-source samples by task scale and resource type.

    Selection is fixed by ``seed`` and by sorted scenario identity, so it cannot
    be re-rolled to favour a particular checkpoint.  Samples come only from the
    training split; the 108-scenario validation set is never touched.
    """
    values = sorted(scenarios, key=lambda item: (item.num_tasks, resource_type(item), item.scenario_id))
    groups: dict[tuple[int, str], list] = {}
    for scenario in values:
        groups.setdefault((int(scenario.num_tasks), resource_type(scenario)), []).append(scenario)
    rng = np.random.default_rng(int(seed))
    selected = []
    for key in sorted(groups):
        members = groups[key]
        count = min(int(per_group), len(members))
        indices = rng.choice(len(members), size=count, replace=False)
        selected.extend(members[int(index)] for index in sorted(indices))
    return selected


def evaluate_mode(config, policy, scenarios, resource_types, repeats, seed, label):
    """Run one execution mode.

    The seed is set once per mode so repeated runs of a *sampled* mode draw
    different samples and the reported spread is real stochastic variance; for
    the deterministic modes the repeats are identical by construction and the
    spread is 0.0, which is exactly the expected result.
    """
    records = []
    seed_everything(seed, disable_cudnn=bool(config.get('device_options', {}).get('disable_cudnn', True)))
    for repeat in range(repeats):
        run_records, _ = Evaluator(config['environment'], console_evaluation_progress(f'{label}#{repeat + 1}')).evaluate(policy, scenarios)
        for record in run_records:
            records.append(dict(mode=label, repeat=repeat, scenario_id=record.scenario_id,
                                base_dag_id=record.base_dag_id, num_tasks=record.num_tasks,
                                num_nodes=record.num_nodes, resource_type=resource_types[record.scenario_id],
                                makespan=record.makespan, heft_makespan=record.heft_makespan, ratio=record.ratio,
                                valid_schedule=record.valid_schedule, inference_time_ms=record.inference_time_ms))
    ratios = np.asarray([row['ratio'] for row in records], dtype=np.float64)
    per_repeat = [float(np.mean([row['ratio'] for row in records if row['repeat'] == repeat])) for repeat in range(repeats)]
    return dict(mode=label, repeat_count=repeats, runs=len(records), samples=len(scenarios),
                mean_ratio=float(ratios.mean()), std_ratio=float(ratios.std()),
                min_ratio=float(ratios.min()), max_ratio=float(ratios.max()),
                valid_schedule_rate=float(np.mean([row['valid_schedule'] for row in records])),
                mean_inference_time_ms=float(np.mean([row['inference_time_ms'] for row in records])),
                per_repeat_mean_ratio=per_repeat,
                repeat_spread=float(max(per_repeat) - min(per_repeat))), records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='HRL best.pt/latest.pt to inspect (read-only)')
    parser.add_argument('--config', help='Config providing the fixed dataset protocol; defaults to the checkpoint config')
    parser.add_argument('--output', required=True)
    parser.add_argument('--modes', nargs='+', default=list(DEFAULT_MODES))
    parser.add_argument('--repeats', type=int, default=3, help='Repeats for sampled modes')
    parser.add_argument('--per-group', type=int, default=1, help='Samples per (task scale, resource type) group')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    unknown = sorted(set(args.modes) - set(EXECUTION_MODES))
    if unknown:
        raise ValueError(f'unknown execution modes: {unknown}')
    if args.repeats < 1 or args.per_group < 1 or args.threads < 1:
        raise ValueError('repeats and per-group must be positive')

    torch.set_num_threads(args.threads)
    from train import make_agents

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if checkpoint.get('config_hash') != config_hash(checkpoint['config']):
        raise ValueError('checkpoint config hash mismatch')
    config = load_config(args.config) if args.config else checkpoint['config']
    if args.config and config_hash(config) != checkpoint.get('config_hash'):
        raise ValueError('--config does not match the checkpoint configuration')
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f'refusing to overwrite an existing diagnostic directory: {output}')

    splits, _ = load_fixed_splits(config)
    samples = select_diagnostic_samples(splits['train'], per_group=args.per_group, seed=args.seed)
    resource_types = {scenario.scenario_id: resource_type(scenario) for scenario in samples}
    plan = dict(checkpoint=str(checkpoint_path), checkpoint_global_step=int(checkpoint['global_step']),
                checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(), threads=args.threads,
                checkpoint_config_hash=checkpoint['config_hash'], seed=args.seed,
                modes=list(args.modes), repeats=args.repeats, per_group=args.per_group,
                samples=[dict(scenario_id=item.scenario_id, num_tasks=item.num_tasks,
                              resource_type=resource_type(item)) for item in samples],
                split='train_diagnostic', formal_validation=False, formal_validation_scenarios=108,
                note='training-source diagnostic samples; never a substitute for the fixed 108-scenario validation')
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return

    device = torch.device(args.device)
    high, low = make_agents(splits['train'][0], config, device)
    HierarchicalTrainer.load(checkpoint_path, high, low, device)
    seed_everything(args.seed, disable_cudnn=bool(config.get('device_options', {}).get('disable_cudnn', True)))
    heft_records, heft_summary = Evaluator(config['environment']).evaluate(HEFTPolicy(), samples)
    reference = {'heft': {'mean_ratio': float(heft_summary['mean_ratio']),
                          'heft_self_ratio_check': float(np.mean([record.ratio for record in heft_records]))}}
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / 'diagnostic_plan.json', plan)
    summaries, all_records = {}, []
    for mode in args.modes:
        repeats = 1 if mode in {'greedy','greedy_high_fixed_eft'} else int(args.repeats)
        policy = HRLExecutionModePolicy(high.model, low.model, device, mode=mode)
        summary, records = evaluate_mode(config, policy, samples, resource_types, repeats, args.seed, mode)
        summaries[mode] = summary
        all_records.extend(records)
    with (output / 'diagnostic_per_run.jsonl').open('x', encoding='utf-8') as handle:
        for row in all_records:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        handle.flush()
    report = dict(plan=plan, reference=reference, modes=summaries,
                  interpretation='Execution-mode contrasts on training samples, not causal proof of training regression. '
                  'Policies may visit different states; matching a seed does not pair their action trajectories.',
                  joint_sampling_gap_vs_greedy=None, low_policy_mode_gap_under_sampled_high=None,
                  low_sampling_gap_under_greedy_high=None)
    if 'greedy' in summaries and 'repeat_sample' in summaries:
        report['joint_sampling_gap_vs_greedy'] = float(summaries['repeat_sample']['mean_ratio'] - summaries['greedy']['mean_ratio'])
    if 'repeat_sample' in summaries and 'fixed_low_eft' in summaries:
        report['low_policy_mode_gap_under_sampled_high'] = float(summaries['repeat_sample']['mean_ratio'] - summaries['fixed_low_eft']['mean_ratio'])
    if 'greedy_high_sample_low' in summaries and 'greedy' in summaries:
        report['low_sampling_gap_under_greedy_high'] = float(summaries['greedy_high_sample_low']['mean_ratio'] - summaries['greedy']['mean_ratio'])
    atomic_json(output / 'diagnostic_summary.json', report)
    print(json.dumps({'modes': {name: {'mean_ratio': value['mean_ratio'], 'std_ratio': value['std_ratio'],
                                       'repeat_spread': value['repeat_spread']} for name, value in summaries.items()},
                      'heft': reference['heft']}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
