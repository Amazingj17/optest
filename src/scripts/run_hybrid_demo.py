"""Run the offline search + HRL demonstration with bundled scenarios and weights."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import torch

from compare_main_models import restore_hrl
from cpn_hrl_dag.demo import (bundle_file, evaluate_scene, load_scenario, render_html,
                              save_json, summarize_demo, verify_bundle)
from cpn_hrl_dag.policies.hybrid import IndependentHybridPolicy
from cpn_hrl_dag.utils.seed import seed_everything


def run(bundle, output, scene_ids=None):
    started = perf_counter()
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'refusing to overwrite nonempty output: {output}')
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / 'status.json', dict(status='running'))
    try:
        manifest = verify_bundle(bundle)
        config = json.loads((bundle / 'config.json').read_text(encoding='utf-8'))
        required = {config['checkpoint'], config['scenarios'], config['template'], 'config.json'}
        if not required <= set(manifest['files']):
            raise ValueError('required demo asset missing from hash manifest')
        payloads = json.loads(bundle_file(bundle, config['scenarios']).read_text(encoding='utf-8'))
        if scene_ids:
            unknown = set(scene_ids) - {s['scenario_id'] for s in payloads}
            if unknown:
                raise ValueError(f'unknown scenario IDs: {sorted(unknown)}')
            payloads = [s for s in payloads if s['scenario_id'] in scene_ids]
        if not payloads:
            raise ValueError('no scenes selected')
        torch.set_num_threads(1)
        seed_everything(config['seed'], disable_cudnn=True)
        # These are locally prepared, hash-verified project checkpoints.
        checkpoint = torch.load(bundle_file(bundle, config['checkpoint']), map_location='cpu', weights_only=False)
        if int(checkpoint['config']['experiment']['seed']) != config['seed']:
            raise ValueError('checkpoint and demo seed differ')
        policy = IndependentHybridPolicy(restore_hrl(checkpoint), search_config=config['search'], seed=config['seed'])
        results = []
        with (output / 'progress.jsonl').open('x', encoding='utf-8') as log:
            for index, payload in enumerate(payloads, 1):
                scenario = load_scenario(payload)
                print(f'[{index}/{len(payloads)}] {scenario.scenario_id}', flush=True)
                result = evaluate_scene(policy, scenario)
                results.append(result)
                event = dict(completed=index, total=len(payloads), scenario_id=scenario.scenario_id,
                             ratio=result['record']['ratio'], selected_source=result['selected_source'],
                             inference_ms=result['record']['inference_time_ms'])
                log.write(json.dumps(event) + '\n')
                log.flush()
                print(f"  ratio={event['ratio']:.6f}; selected={event['selected_source']}; "
                      f"{event['inference_ms'] / 1000:.3f}s; validated", flush=True)
        summary = summarize_demo(results)
        provenance = dict(created_utc=datetime.now(timezone.utc).isoformat(), seed=config['seed'],
                          checkpoint_sha256=manifest['files'][config['checkpoint']],
                          checkpoint_step=int(checkpoint['global_step']),
                          python=platform.python_version(), torch=str(torch.__version__),
                          platform=platform.platform(), threads=1, device='cpu',
                          source=manifest['provenance'], retrained=False, test_evaluated=False,
                          inference_timing='candidate setup, generation, validation, selection and final replay',
                          evaluation_wall_timing='also includes outer environment setup and HEFT reference',
                          audit_timing='additional independent simulator replay; separate from inference',
                          branch_timing='in-process policy reset and rollout; excludes branch environment setup',
                          elapsed_before_export_s=perf_counter() - started)
        data = dict(summary=summary, provenance=provenance, scenes=results)
        save_json(output / 'results.json', data)
        save_json(output / 'summary.json', summary)
        save_json(output / 'provenance.json', provenance)
        with (output / 'metrics.csv').open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.DictWriter(handle, fieldnames=['scenario_id', 'num_tasks', 'method', 'makespan',
                                                        'ratio', 'inference_ms', 'valid'])
            writer.writeheader()
            for item in results:
                rows = item['candidates'] + [dict(name='hybrid', makespan=item['record']['makespan'],
                         ratio=item['record']['ratio'], inference_ms=item['record']['inference_time_ms'], valid=True)]
                for row in rows:
                    writer.writerow(dict(scenario_id=item['scenario']['scenario_id'],
                                         num_tasks=len(item['scenario']['tasks']), method=row['name'],
                                         **{key: row[key] for key in ('makespan', 'ratio', 'inference_ms', 'valid')}))
        render_html(bundle_file(bundle, config['template']), output / 'index.html', data)
        save_json(output / 'status.json', dict(status='complete', num_scenarios=len(results),
                                               elapsed_s=perf_counter() - started))
        print(f"Open: {output / 'index.html'}", flush=True)
        return data
    except BaseException as error:
        save_json(output / 'status.json', dict(status='failed', error=str(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=ROOT.parent / 'demo')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--scene', action='append', help='Exact scenario ID; repeat to select multiple scenes')
    args = parser.parse_args()
    run(args.bundle, args.output or args.bundle / 'runs' / datetime.now().strftime('%Y%m%d_%H%M%S_%f'), args.scene)
