"""Audit target feasibility on the entire fixed 108-scenario validation split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy

from cpn_hrl_dag.datasets.loading import default_registry, load_scenarios
from cpn_hrl_dag.datasets.resource_realizations import materialize_resources
from cpn_hrl_dag.datasets.split import SplitManifest, SplitManager
from cpn_hrl_dag.evaluation.lower_bounds import fractional_load_bound, optimistic_path_bound
from cpn_hrl_dag.scheduling.execution_model import execution_model_for_scenario
from cpn_hrl_dag.scheduling.simulator import ScheduleSimulator
from cpn_hrl_dag.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='outputs/validation_target_080_audit')
    args = parser.parse_args()
    config_path = Path('configs/zenodo_heft_safe_search_beam3_r3_blocks_2026.yaml')
    config = load_config(config_path)
    assert config['experiment']['seed'] == 2026
    dataset = config['dataset']
    manifest_path = Path(dataset['split_manifest'])
    assert manifest_path.as_posix() == 'data/manifests/grapheonrl_mixed_iid_seed7.json'
    manifest = SplitManifest.read(manifest_path)
    scenarios = load_scenarios(default_registry(dataset.get('grapheonrl_system_configs'),
                                               dataset.get('archive_task_counts')), dataset['roots'])
    validation = materialize_resources(SplitManager.apply(scenarios, manifest),
                                      config['resources'], manifest, config['experiment']['seed'])['validation']
    current_path = Path(config['output_dir']) / 'eval_search_validation/per_scene.csv'
    with current_path.open(encoding='utf-8') as handle:
        current = {row['scenario_id']: row for row in csv.DictReader(handle)}
    assert len(validation) == len(current) == 108
    assert {scenario.scenario_id for scenario in validation} == current.keys()
    assert len({scenario.metadata['original_graph_id'] for scenario in validation}) == 54
    records = []
    certificates = []
    for index, scenario in enumerate(validation, start=1):
        simulator = ScheduleSimulator(scenario, execution_model_for_scenario(scenario.dataset_source))
        path_bound = optimistic_path_bound(simulator)
        load_bound, weights = fractional_load_bound(simulator)
        assert np.isfinite(weights).all() and min(weights) >= 0.0 and sum(weights) <= 1.0
        reference = current[scenario.scenario_id]
        heft = float(reference['heft_makespan'])
        incumbent = float(reference['makespan'])
        bound = max(path_bound, load_bound) * (1.0 - 1e-9)
        assert np.isfinite(bound) and 0.0 < bound <= incumbent + 1e-7 * max(1.0, incumbent)
        record = dict(scenario_id=scenario.scenario_id, base_dag_id=scenario.metadata['original_graph_id'],
                      resource_type=scenario.scenario_id.rsplit(':', 1)[-1], num_tasks=scenario.num_tasks,
                      current_ratio=incumbent/heft, path_bound_ratio=path_bound/heft,
                      load_bound_ratio=load_bound/heft, lower_bound_ratio=bound/heft,
                      optimistic_remaining_ratio_gain=max(0.0, (incumbent-bound)/heft))
        records.append(record)
        certificates.append(dict(scenario_id=scenario.scenario_id, node_weights=weights))
        print(f"[{index:03d}/108] {scenario.scenario_id} current={incumbent/heft:.6f} bound={bound/heft:.6f}", flush=True)

    def summarize(values):
        return dict(count=len(values), current_mean=float(np.mean([row['current_ratio'] for row in values])),
                    path_bound_mean=float(np.mean([row['path_bound_ratio'] for row in values])),
                    load_bound_mean=float(np.mean([row['load_bound_ratio'] for row in values])),
                    lower_bound_mean=float(np.mean([row['lower_bound_ratio'] for row in values])),
                    optimistic_remaining_mean_gain=float(np.mean([row['optimistic_remaining_ratio_gain'] for row in values])))

    report = dict(split='validation', seed=config['experiment']['seed'], target=0.8, scipy_version=scipy.__version__,
                  config_path=str(config_path), manifest_path=str(manifest_path),
                  manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                  reference_sha256=hashlib.sha256(current_path.read_bytes()).hexdigest(), **summarize(records))
    report['by_resource_type'] = {kind: summarize([row for row in records if row['resource_type'] == kind])
                                  for kind in ('homogeneous', 'heterogeneous')}
    report['by_task_count'] = {size: summarize([row for row in records if row['num_tasks'] == size])
                             for size in (50, 100, 300)}
    report['target_excluded_by_bound'] = report['lower_bound_mean'] > report['target']
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'per_scene_bounds.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    (output / 'load_dual_certificates.json').write_text(json.dumps(certificates, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
