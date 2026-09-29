"""Portable, auditable inputs and results for the independent hybrid demo."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .evaluation import Evaluator
from .scenario.types import Scenario, Task, Dependency, ComputeNode
from .scheduling.simulator import ScheduleSimulator
from .scheduling.execution_model import execution_model_for_scenario


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


def scenario_payload(scenario):
    # Infinite diagonal bandwidth is represented as a string in strict JSON.
    def matrix(value):
        return [[float(x) if np.isfinite(x) else str(float(x)) for x in row]
                for row in value] if value is not None else None
    return dict(scenario_id=scenario.scenario_id, dataset_source=scenario.dataset_source,
                tasks=[asdict(x) for x in scenario.tasks],
                dependencies=[asdict(x) for x in scenario.dependencies],
                compute_nodes=[asdict(x) for x in scenario.compute_nodes],
                bandwidth_matrix=matrix(scenario.bandwidth_matrix),
                latency_matrix=matrix(scenario.latency_matrix), metadata=dict(scenario.metadata))


def load_scenario(payload):
    return Scenario(scenario_id=payload['scenario_id'], dataset_source=payload['dataset_source'],
                    tasks=[Task(**x) for x in payload['tasks']],
                    dependencies=[Dependency(**x) for x in payload['dependencies']],
                    compute_nodes=[ComputeNode(**x) for x in payload['compute_nodes']],
                    bandwidth_matrix=payload['bandwidth_matrix'],
                    latency_matrix=payload.get('latency_matrix'), metadata=payload['metadata'])


def bundle_file(bundle, relative):
    root = Path(bundle).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('bundle path escapes its directory')
    return path


def verify_bundle(bundle):
    bundle = Path(bundle)
    manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    for relative, expected in manifest['files'].items():
        path = bundle_file(bundle, relative)
        if sha256(path) != expected:
            raise ValueError(f'bundle SHA256 mismatch: {relative}')
    return manifest


def audit_candidate(scenario, candidate):
    """Replay every decision independently of the candidate's environment."""
    simulator = ScheduleSimulator(scenario, execution_model_for_scenario(scenario.dataset_source))
    for task, node in candidate.decisions:
        simulator.schedule(task, node)
    actual = simulator.result()
    if len(actual.entries) != scenario.num_tasks:
        raise ValueError('incomplete candidate')
    if not np.isclose(actual.makespan, candidate.result.makespan, rtol=0, atol=1e-9):
        raise ValueError('candidate replay makespan mismatch')
    if actual.entries != candidate.result.entries:
        raise ValueError('candidate replay timeline mismatch')
    return actual


def evaluate_scene(policy, scenario):
    wall_start = perf_counter()
    records, _ = Evaluator({'normalize_observations': True}).evaluate(policy, [scenario])
    evaluation_wall_ms = (perf_counter() - wall_start) * 1000
    record = records[0]
    if not record.valid_schedule or not np.isclose(record.makespan, policy.selected_makespan,
                                                   rtol=0, atol=1e-9):
        raise ValueError('hybrid replay differs from selected candidate')
    if record.makespan > record.heft_makespan + 1e-9:
        raise ValueError('hybrid lost its HEFT guarantee')
    branches = []
    audit_start = perf_counter()
    for candidate in policy.candidates:
        result = audit_candidate(scenario, candidate)
        if record.makespan > result.makespan + 1e-9:
            raise ValueError('hybrid lost a completed candidate')
        branches.append(dict(name=candidate.name, makespan=result.makespan,
                             ratio=result.makespan / record.heft_makespan,
                             inference_ms=candidate.inference_time_ms,
                             wall_ms=candidate.wall_time_ms, valid=True,
                             communication_time=result.total_communication_time,
                             entries=[asdict(x) for x in result.entries],
                             decisions=candidate.decisions))
    return dict(scenario=scenario_payload(scenario), record=asdict(record), candidates=branches,
                selected_source=policy.selected_candidate,
                search_internal_source=policy.search_selected_candidate,
                evaluation_wall_ms=evaluation_wall_ms,
                independent_audit_ms=(perf_counter() - audit_start) * 1000,
                candidate_selection_and_replay_ms=max(0., record.inference_time_ms -
                                                      sum(c.wall_time_ms for c in policy.candidates)),
                all_candidates_replayed=True)


def summarize_demo(scenes):
    if not scenes:
        raise ValueError('no demo scenes')
    methods = {}
    for name in ('heft', 'search_blocks', 'residual_hrl', 'hybrid'):
        ratios, times = [], []
        for item in scenes:
            if name == 'hybrid':
                ratios.append(item['record']['ratio'])
                times.append(item['record']['inference_time_ms'])
            else:
                branch = next(c for c in item['candidates'] if c['name'] == name)
                ratios.append(branch['ratio'])
                times.append(branch['inference_ms'])
        methods[name] = dict(mean_ratio=float(np.mean(ratios)),
                             mean_inference_ms=float(np.mean(times)),
                             p95_inference_ms=float(np.quantile(times, .95)))
    return dict(num_scenarios=len(scenes), methods=methods,
                valid_schedule_rate=float(np.mean([s['record']['valid_schedule'] for s in scenes])),
                scope='fixed demonstration subset; not the full 108-scene validation score')


def render_html(template, output, data):
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False)
    # Prevent scenario strings from closing the inline script element.
    encoded = encoded.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    encoded = encoded.replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    page = Path(template).read_text(encoding='utf-8')
    if page.count('__DEMO_DATA__') != 1:
        raise ValueError('template must contain one data placeholder')
    Path(output).write_text(page.replace('__DEMO_DATA__', encoded), encoding='utf-8')
