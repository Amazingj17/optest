"""Run matched training, full validation, and plots sequentially with durable logs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
METHODS = ('graph_ppo', 'tier_mappo')


def verify_completed_run(run, method):
    from cpn_hrl_dag.experiments.graph_baselines import read_checkpoint

    checkpoint = read_checkpoint(run / 'best.pt')
    if checkpoint['method'] != method or not checkpoint.get('training_complete'):
        raise ValueError(f'{method}: refusing to reuse incomplete training')
    if checkpoint['completed_epochs'] != checkpoint['config']['training']['epochs']:
        raise ValueError(f'{method}: training budget is incomplete')
    summary = json.loads((run / 'validation_best' / 'summary.json').read_text(encoding='utf-8'))
    if summary['split'] != 'validation' or summary['num_scenarios'] != 108:
        raise ValueError(f'{method}: missing full best-checkpoint validation')
    return run / 'best.pt'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--training-root', default='outputs/graph_baselines')
    parser.add_argument('--reuse-graph-ppo', action='store_true')
    parser.add_argument('--wait-for-graph-pid', type=int)
    parser.add_argument('--evaluation-device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--training-device', default='auto', choices=['auto', 'cpu', 'cuda'])
    args = parser.parse_args()
    if args.wait_for_graph_pid is not None and (os.name != 'nt' or not args.reuse_graph_ppo or args.wait_for_graph_pid <= 0):
        parser.error('--wait-for-graph-pid requires Windows, a positive PID, and --reuse-graph-ppo')
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / 'src'))
    output, training_root = Path(args.output).resolve(), Path(args.training_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / 'pipeline_status.json'
    if status_path.exists():
        raise FileExistsError(f'refusing to overwrite pipeline status: {status_path}')
    runs = {method: training_root / method / 'seed_2026' for method in METHODS}
    for method, run in runs.items():
        if run.exists() and any(run.iterdir()) and not (method == 'graph_ppo' and args.reuse_graph_ppo):
            raise FileExistsError(f'refusing to overwrite training results: {run}')
    if (output / 'comparison').exists():
        raise FileExistsError('comparison output already exists')
    environment = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONIOENCODING='utf-8', MPLBACKEND='Agg')
    status = dict(status='running', stage='starting', started_at=datetime.now(timezone.utc).isoformat(),
                  pipeline_pid=os.getpid(), test_evaluated=False, validation_scenarios=108,
                  seed=2026, epochs_per_method=3, training_runs={method: str(run) for method, run in runs.items()},
                  evaluation_device=args.evaluation_device, training_device=args.training_device)

    def update(**values):
        status.update(values, updated_at=datetime.now(timezone.utc).isoformat())
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temporary.replace(status_path)
        print(f'{status["updated_at"]} {status["status"]}: {status["stage"]}', flush=True)

    def execute(stage, command):
        with (output / f'{stage}.log').open('x', encoding='utf-8') as handle:
            process = subprocess.Popen([sys.executable, '-u', *command], cwd=ROOT, env=environment,
                                       stdout=handle, stderr=subprocess.STDOUT)
            update(stage=stage, child_pid=process.pid, command=command)
            return_code = process.wait()
        if return_code:
            raise RuntimeError(f'{stage} failed with exit code {return_code}; see {output / (stage + ".log")}')

    update()
    try:
        if args.wait_for_graph_pid:
            update(stage='waiting_for_graph_ppo', child_pid=args.wait_for_graph_pid)
            subprocess.run(['powershell.exe', '-NoProfile', '-Command',
                            f'Wait-Process -Id {args.wait_for_graph_pid} -ErrorAction SilentlyContinue'], check=False)
        checkpoints = []
        for method in METHODS:
            if method != 'graph_ppo' or not args.reuse_graph_ppo:
                execute(f'train_{method}', ['scripts/train_graph_baselines.py', '--config',
                        f'configs/zenodo_{method}_comparison.yaml', '--output', str(runs[method]),
                        '--device', args.training_device])
            checkpoints.append(str(verify_completed_run(runs[method], method)))
        execute('evaluate', ['scripts/evaluate_graph_baselines.py', '--checkpoints', *checkpoints,
                            '--output', str(output / 'comparison'), '--device', args.evaluation_device])
        execute('plot', ['scripts/plot_graph_baseline_comparison.py', '--comparison', str(output / 'comparison')])
        update(status='complete', stage='complete', child_pid=None,
               report=str(output / 'comparison' / 'figures' / 'RESULTS.md'))
    except Exception as error:
        update(status='failed', error=str(error), child_pid=None)
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
