"""Train three neural models on explicitly assigned GPUs, then evaluate six policies."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from threading import Thread
from time import sleep

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from compare_main_models import inspect_main_training, validate_protocol
from cpn_hrl_dag.experiments.graph_baselines import load_fixed_splits, prepare_output, read_checkpoint
from cpn_hrl_dag.utils.config import load_config
from cpn_hrl_dag.utils.progress import ProgressLog, atomic_json
from cpn_hrl_dag.utils.runtime import runtime_metadata


def build_configs(output, epochs=3, seed=2026, device='cuda', threads=4):
    if epochs < 1 or seed < 0 or threads < 1:
        raise ValueError('epochs/threads must be positive and seed non-negative')
    configs = {}
    main = load_config(ROOT / 'configs/zenodo_heft_safe_fast_2026.yaml')
    main['training'].update(coverage_epochs=epochs, low_pretrain_episodes=0,
                           high_train_episodes=864 if epochs > 1 else 0,
                           joint_train_episodes=864 * (epochs - 1 if epochs > 1 else 1),
                           evaluation_interval=864, early_stop_patience=epochs + 1)
    main['curriculum'] = {'stages': []}
    main['bc']['enabled'] = False
    main['dataset'].pop('sampling', None)
    main['environment'] = {'normalize_observations': True}
    configs['residual_hrl'] = main
    for method in ['graph_ppo', 'tier_mappo']:
        config = load_config(ROOT / f'configs/zenodo_{method}_comparison.yaml')
        config['training'].update(epochs=epochs, seed=seed)
        configs[method] = config
    for method, config in configs.items():
        config['experiment'] = {'name': f'{method}_server_seed{seed}', 'seed': seed}
        config['device'] = device
        config['torch_num_threads'] = threads
        config['output_dir'] = str(Path(output) / 'training' / method)
        config['evaluation'].update(evaluate_initial=True, proxy_scenarios=0)
    search = load_config(ROOT / 'configs/zenodo_heft_safe_search_beam3_r3_blocks_2026.yaml')
    search['experiment']['seed'] = seed
    configs['search'] = search
    return configs


def validate_slots(gpus, device, environment):
    if device == 'cpu':
        return ['cpu']
    if not gpus or len(set(gpus)) != len(gpus) or any(not item.isdigit() for item in gpus):
        raise ValueError('provide distinct numeric GPU IDs, for example --gpus 1 or --gpus 0 1 2')
    inherited = environment.get('CUDA_VISIBLE_DEVICES')
    if inherited is not None and not set(gpus) <= set(inherited.split(',')):
        raise ValueError('requested GPUs are outside inherited CUDA_VISIBLE_DEVICES; respect scheduler allocation')
    return list(gpus)


def run_jobs(jobs, slots, environment, output, progress):
    pending = list(jobs)
    available = list(slots)
    running = []
    reader_errors = []

    def stream(process, handle, name):
        try:
            for line in process.stdout:
                handle.write(line)
                handle.flush()
                print(f'[{name}] {line}', end='', flush=True)
        except Exception as error:
            reader_errors.append(error)
        finally:
            process.stdout.close()
            handle.close()

    try:
        while pending or running:
            while pending and available:
                job = pending.pop(0)
                slot = available.pop(0)
                child_environment = dict(environment)
                if slot != 'inherit':
                    child_environment['CUDA_VISIBLE_DEVICES'] = '' if slot == 'cpu' else slot
                handle = (Path(output) / f'{job["name"]}.log').open('x', encoding='utf-8')
                try:
                    process = subprocess.Popen(job['command'], cwd=ROOT, env=child_environment,
                                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                               text=True, encoding='utf-8', errors='replace', bufsize=1)
                except BaseException:
                    handle.close()
                    raise
                reader = Thread(target=stream, args=(process, handle, job['name']), daemon=True)
                reader.start()
                running.append((job, slot, process, reader))
                progress.emit('job_started', job=job['name'], gpu=slot, child_pid=process.pid,
                              command=job['command'])
            for item in list(running):
                job, slot, process, reader = item
                code = process.poll()
                if code is None:
                    continue
                reader.join()
                running.remove(item)
                available.append(slot)
                if code:
                    raise RuntimeError(f'{job["name"]} exited with {code}; see {Path(output) / (job["name"] + ".log")}')
                progress.emit('job_complete', job=job['name'], gpu=slot, child_pid=process.pid)
            if reader_errors:
                raise RuntimeError(f'cannot persist child logs: {reader_errors[0]}')
            if pending or running:
                sleep(.2)
    finally:
        for _, _, process, _ in running:
            if process.poll() is None:
                process.terminate()
        for _, _, process, reader in running:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            reader.join(timeout=10)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--output', required=True)
    result.add_argument('--gpus', nargs='+', default=['1'], help='Only GPUs you are authorized to use; defaults to 1')
    result.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    result.add_argument('--epochs', type=int, default=3)
    result.add_argument('--seed', type=int, default=2026)
    result.add_argument('--threads', type=int, default=4)
    result.add_argument('--reuse-main', help='Completed best.pt plus train_log.csv, summary.json and config in its directory')
    result.add_argument('--dry-run', action='store_true', help='Print plan without creating files, loading data or launching training')
    return result


def main():
    args = parser().parse_args()
    output = Path(args.output).resolve()
    reused = Path(args.reuse_main).resolve() if args.reuse_main else None
    os.chdir(ROOT)
    slots = validate_slots(args.gpus, args.device, os.environ)
    configs = build_configs(output, args.epochs, args.seed, args.device, args.threads)
    reference = configs['graph_ppo']
    if not Path(reference['dataset']['split_manifest']).is_file():
        raise FileNotFoundError('the frozen manifest must already exist')
    for config in configs.values():
        validate_protocol(config, reference)
    if reused:
        checkpoint, facts = inspect_main_training(reused)
        if int(checkpoint['config']['experiment']['seed']) != args.seed:
            raise ValueError('reused main training seed must match --seed')
        validate_protocol(checkpoint['config'], reference)
    else:
        facts = None
    jobs = []
    for method in ['residual_hrl', 'graph_ppo', 'tier_mappo']:
        if method == 'residual_hrl' and reused:
            continue
        script = 'scripts/train_main_comparison.py' if method == 'residual_hrl' else 'scripts/train_graph_baselines.py'
        command = [sys.executable, '-u', script, '--config', str(output / 'configs' / f'{method}.yaml')]
        if method != 'residual_hrl':
            command += ['--output', configs[method]['output_dir']]
        jobs.append(dict(name=f'train_{method}', command=command))
    plan = dict(output=str(output), slots=slots, seed=args.seed, coverage_epochs=args.epochs,
                new_episodes_per_model=864 * args.epochs, validation_scenarios=108, test_evaluated=False,
                reused_main=str(reused) if reused else None, reused_main_facts=facts, jobs=jobs,
                evaluation_device='cpu', evaluation_threads=1)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'refusing to overwrite {output}')
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return
    output = prepare_output(output)
    environment = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONIOENCODING='utf-8',
                       PYTHONUNBUFFERED='1', MPLBACKEND='Agg', OMP_NUM_THREADS=str(args.threads),
                       MKL_NUM_THREADS=str(args.threads), CUBLAS_WORKSPACE_CONFIG=':4096:8')

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'received signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    with ProgressLog(output, 'server_pipeline', args.seed) as progress:
        atomic_json(output / 'run_plan.json', plan)
        atomic_json(output / 'environment.json', runtime_metadata())
        progress.emit('preflight', slots=slots, validation_scenarios=108)
        if args.device == 'cuda':
            probe_environment = dict(environment, CUDA_VISIBLE_DEVICES=','.join(slots))
            probe = subprocess.run([sys.executable, '-c',
                f'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count()=={len(slots)}; '
                'print([(index, torch.cuda.get_device_name(index)) for index in range(torch.cuda.device_count())])'],
                env=probe_environment, check=True, capture_output=True, text=True)
            progress.emit('gpu_preflight', devices=probe.stdout.strip())
        splits, _ = load_fixed_splits(reference)
        progress.emit('dataset_preflight', train=len(splits['train']), validation=len(splits['validation']))
        del splits
        (output / 'configs').mkdir()
        for method, config in configs.items():
            (output / 'configs' / f'{method}.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        run_jobs(jobs, slots, environment, output, progress)
        main_path = reused or Path(configs['residual_hrl']['output_dir']) / 'best.pt'
        inspect_main_training(main_path)
        graph_paths = [Path(configs[method]['output_dir']) / 'best.pt' for method in ['graph_ppo', 'tier_mappo']]
        for path in graph_paths:
            checkpoint = read_checkpoint(path)
            if not checkpoint['training_complete'] or checkpoint['completed_epochs'] != args.epochs:
                raise ValueError(f'incomplete training: {path}')
        comparison = output / 'comparison'
        commands = [
            dict(name='evaluate_six_methods', command=[sys.executable, '-u', 'scripts/compare_main_models.py',
                 '--main-checkpoint', str(main_path), '--graph-checkpoints', *map(str, graph_paths),
                 '--search-config', str(output / 'configs/search.yaml'), '--output', str(comparison)]),
            dict(name='plot_comparison', command=[sys.executable, '-u', 'scripts/plot_main_model_comparison.py', '--comparison', str(comparison)]),
            dict(name='plot_training', command=[sys.executable, '-u', 'scripts/plot_server_training.py',
                 '--main-run', str(main_path.parent), '--graph-runs', *[str(path.parent) for path in graph_paths],
                 '--output', str(output / 'figures')]),
        ]
        evaluation_environment = dict(environment, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        for job in commands:
            run_jobs([job], ['cpu'], evaluation_environment, output, progress)
        result = json.loads((comparison / 'status.json').read_text(encoding='utf-8'))
        if result['status'] != 'complete' or result['num_scenarios'] != 108:
            raise RuntimeError('comparison did not complete all validation scenarios')
        progress.emit('complete', status='complete', report=str(comparison / 'figures/RESULTS.md'),
                      training_figures=str(output / 'figures'), reused_main=bool(reused))


if __name__ == '__main__':
    main()
