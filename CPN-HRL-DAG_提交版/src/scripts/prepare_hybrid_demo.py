"""Package a deterministic validation subset and an existing trained checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import torch
import yaml

from cpn_hrl_dag.demo import save_json, scenario_payload, sha256
from cpn_hrl_dag.experiments.graph_baselines import load_fixed_splits
from cpn_hrl_dag.utils.config import config_hash


def prepare(checkpoint_path, search_path, output, graph_ppo_path, tier_mappo_path):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'refusing to overwrite nonempty bundle: {output}')
    checkpoint_path, search_path = Path(checkpoint_path), Path(search_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    reference = checkpoint['config']
    if checkpoint.get('config_hash') != config_hash(reference):
        raise ValueError('checkpoint configuration hash mismatch')
    search = yaml.safe_load(search_path.read_text(encoding='utf-8'))
    if search['experiment']['seed'] != reference['experiment']['seed']:
        raise ValueError('search and checkpoint seed differ')
    print('Loading fixed validation scenarios...', flush=True)
    splits, _ = load_fixed_splits(reference)
    scenes = []
    for size in (50, 100, 300):
        for variant in ('heterogeneous', 'homogeneous'):
            candidates = [s for s in splits['validation'] if s.num_tasks == size
                          and s.scenario_id.endswith(':' + variant)]
            # Selection never inspects makespan or the identity of the winner.
            scenes.append(min(candidates, key=lambda s: s.scenario_id))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'assets').mkdir()
    shutil.copy2(checkpoint_path, output / 'assets/hrl_best.pt')
    baseline_paths = {'graph_ppo': Path(graph_ppo_path), 'tier_mappo': Path(tier_mappo_path)}
    for name, path in baseline_paths.items():
        shutil.copy2(path, output / f'assets/{name}_best.pt')
    shutil.copy2(ROOT / 'scripts/assets/hybrid_demo.html', output / 'assets/template.html')
    save_json(output / 'assets/scenarios.json', [scenario_payload(s) for s in scenes])
    save_json(output / 'config.json', dict(seed=int(reference['experiment']['seed']),
                                         checkpoint='assets/hrl_best.pt', scenarios='assets/scenarios.json',
                                         template='assets/template.html', search=search['search'],
                                         baseline_checkpoints={name: f'assets/{name}_best.pt' for name in baseline_paths}))
    files = ['assets/hrl_best.pt', 'assets/scenarios.json', 'assets/template.html', 'config.json']
    files.extend(f'assets/{name}_best.pt' for name in baseline_paths)
    save_json(output / 'manifest.json', dict(format='cpn-hybrid-demo-v1',
              files={p: sha256(output / p) for p in files}, provenance=dict(
                  checkpoint_source=checkpoint_path.as_posix(), checkpoint_step=int(checkpoint['global_step']),
                  baseline_checkpoint_sources={name: path.as_posix() for name, path in baseline_paths.items()},
                  search_config_sha256=sha256(search_path),
                  split_manifest_sha256=sha256(reference['dataset']['split_manifest']),
                  dataset_url='https://zenodo.org/records/18927122',
                  dataset_attribution='GrapheonRL benchmark, Aasish Kumar Sharma; STG-derived scheduling instances',
                  dataset_license='CC-BY-4.0',
                  transformation='Six canonical scenarios serialized from the existing dataset adapter; source metadata retained',
                  selection='First lexicographic validation scenario per 50/100/300 task count and heterogeneous/homogeneous group',
                  selected_scenarios=[s.scenario_id for s in scenes],
                  source_split='validation', formal_validation_scene_count=108,
                  validation_used_for_model_selection=True)))
    print(f'Prepared {len(scenes)} fixed demo scenes: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--search-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--graph-ppo-checkpoint', type=Path, required=True)
    parser.add_argument('--tier-mappo-checkpoint', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.checkpoint, args.search_config, args.output, args.graph_ppo_checkpoint, args.tier_mappo_checkpoint)
