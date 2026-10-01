"""Paired reference-reuse timing with exact candidate trajectory verification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import torch
from compare_main_models import restore_hrl
from cpn_hrl_dag.demo import (bundle_file, evaluate_scene, load_scenario, save_json, verify_bundle)
from cpn_hrl_dag.policies.hybrid import IndependentHybridPolicy
from cpn_hrl_dag.utils.seed import seed_everything


def benchmark(bundle, output, repeats=3, scene_ids=None):
    if repeats < 2:
        raise ValueError('at least two repetitions are required')
    bundle, output = Path(bundle), Path(output)
    if output.exists():
        raise FileExistsError(output)
    manifest = verify_bundle(bundle)
    config = json.loads((bundle / 'config.json').read_text(encoding='utf-8'))
    payloads = json.loads(bundle_file(bundle, config['scenarios']).read_text(encoding='utf-8'))
    if scene_ids:
        unknown = set(scene_ids) - {s['scenario_id'] for s in payloads}
        if unknown:
            raise ValueError(f'unknown scenes: {unknown}')
        payloads = [s for s in payloads if s['scenario_id'] in scene_ids]
    checkpoint = torch.load(bundle_file(bundle, config['checkpoint']), map_location='cpu', weights_only=False)
    torch.set_num_threads(1)
    rows = []
    for payload in payloads:
        scene = load_scenario(payload)
        policies = {flag: IndependentHybridPolicy(restore_hrl(checkpoint), search_config=config['search'],
                                                  seed=config['seed'], reuse_heft_reference=flag)
                    for flag in (False, True)}
        # Exclude one warmup per mode; alternate timing order in measured pairs.
        for flag in (False, True):
            seed_everything(config['seed'], disable_cudnn=True)
            evaluate_scene(policies[flag], scene)
        for repeat in range(repeats):
            pair = {}
            for flag in ((False, True) if repeat % 2 == 0 else (True, False)):
                seed_everything(config['seed'], disable_cudnn=True)
                pair[flag] = evaluate_scene(policies[flag], scene)
            before, after = pair[False], pair[True]
            for a, b in zip(before['candidates'], after['candidates']):
                if a['decisions'] != b['decisions'] or a['entries'] != b['entries'] or a['makespan'] != b['makespan']:
                    raise AssertionError('reference reuse changed a candidate trajectory')
            if before['selected_source'] != after['selected_source']:
                raise AssertionError('reference reuse changed selection')
            rows.append(dict(scenario_id=scene.scenario_id, repeat=repeat,
                             baseline_ms=before['record']['inference_time_ms'],
                             reused_ms=after['record']['inference_time_ms'], exact_trajectories=True))
            print(f'{scene.scenario_id} pair {repeat+1}/{repeats} verified', flush=True)
    before = statistics.median(row['baseline_ms'] for row in rows)
    after = statistics.median(row['reused_ms'] for row in rows)
    speedups = [row['baseline_ms'] / row['reused_ms'] for row in rows]
    output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(rows=rows, median_paired_speedup=statistics.median(speedups),
                  baseline_median_ms=before, reused_median_ms=after,
                  exact_trajectories=True, scenes=len(payloads), repeats=repeats,
                  platform=platform.platform(), python=platform.python_version(), torch=str(torch.__version__),
                  threads=1, checkpoint_sha256=manifest['files'][config['checkpoint']],
                  timing='hybrid inference includes all candidate setup, generation, validation and replay',
                  note='Local warm-process alternating paired timings; not a cross-hardware performance claim')
    save_json(output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--scene', action='append')
    args = parser.parse_args()
    benchmark(args.bundle, args.output, args.repeats, args.scene)
