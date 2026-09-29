"""Auditable openEuler CPU acceptance: tests, real training and checkpoint replay.

Run from a prepared checkout with the raw dataset. No installation, downloading,
test-split evaluation or changes to the original model checkpoints are performed.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def platform_facts():
    release = platform.freedesktop_os_release() if platform.system() == 'Linux' else {}
    return dict(os_release=release, system=platform.system(), kernel=platform.release(),
                architecture=platform.machine(), python=sys.version, executable=sys.executable,
                container_marker=Path('/.dockerenv').exists() or Path('/run/.containerenv').exists(),
                packages=dict(sorted((dist.metadata['Name'], dist.version)
                                     for dist in importlib.metadata.distributions() if dist.metadata['Name'])))


def require_target(facts, execution_kind):
    if facts['system'] != 'Linux' or facts['os_release'].get('ID', '').lower() != 'openeuler':
        raise RuntimeError('Acceptance requires an actual openEuler userspace; no host OS override is supported.')
    if sys.version_info < (3, 10):
        raise RuntimeError('Python >= 3.10 is required.')
    if facts['container_marker'] and execution_kind != 'container':
        raise RuntimeError('Detected container: use --execution-kind container, not vm/native.')


def source_hashes():
    files = [ROOT / 'pyproject.toml', ROOT / '.dockerignore']
    for folder in ('src', 'scripts', 'configs', 'tests', 'deploy', 'data/manifests'):
        files.extend(p for p in (ROOT / folder).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.pyo'))
    return {str(p.relative_to(ROOT)).replace('\\', '/'): sha256(p) for p in sorted(files) if p.is_file()}


def verify_report(directory):
    directory = Path(directory)
    report = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    with (directory / 'per_scene.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if report['num_scenarios'] != 108 or report['num_base_dags'] != 54 or len(rows) != 108:
        raise RuntimeError('Acceptance requires all 108 validation scenarios / 54 base DAGs.')
    if len({row['scenario_id'] for row in rows}) != 108 or len({row['base_dag_id'] for row in rows}) != 54:
        raise RuntimeError('Duplicate scenarios or incorrect base-DAG coverage.')
    if report['valid_schedule_rate'] != 1.0 or any(row['valid_schedule'] != 'True' for row in rows):
        raise RuntimeError('Invalid schedule in acceptance output.')
    ratios = [float(row['ratio']) for row in rows]
    if not all(math.isfinite(ratio) and ratio > 0 for ratio in ratios):
        raise RuntimeError('Non-finite or non-positive ratio.')
    for row in rows:
        makespan, heft = float(row['makespan']), float(row['heft_makespan'])
        if not (math.isfinite(makespan) and math.isfinite(heft) and makespan > 0 and heft > 0):
            raise RuntimeError('Invalid makespan.')
        if not math.isclose(float(row['ratio']), makespan / heft, rel_tol=1e-10):
            raise RuntimeError('Ratio differs from paired makespans.')
    if not math.isclose(report['mean_ratio'], sum(ratios) / len(ratios), rel_tol=1e-10):
        raise RuntimeError('Summary differs from per-scene records.')
    return report


def compare_replay(left, right):
    def records(directory):
        with (Path(directory) / 'per_scene.csv').open(encoding='utf-8', newline='') as stream:
            return {row['scenario_id']: row for row in csv.DictReader(stream)}
    before, after = records(left), records(right)
    if before.keys() != after.keys():
        raise RuntimeError('Checkpoint replay scenario identities differ.')
    for key in before:
        for field in ('makespan', 'heft_makespan'):
            if not math.isclose(float(before[key][field]), float(after[key][field]), rel_tol=1e-10, abs_tol=1e-10):
                raise RuntimeError(f'Checkpoint replay differs for {key}: {field}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--execution-kind', required=True, choices=['container', 'vm', 'native'])
    parser.add_argument('--frozen-config', type=Path)
    parser.add_argument('--frozen-checkpoint', type=Path)
    args = parser.parse_args(argv)
    if bool(args.frozen_config) != bool(args.frozen_checkpoint):
        parser.error('--frozen-config and --frozen-checkpoint must be supplied together')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite existing evidence.
    (output / 'logs').mkdir()
    status = dict(status='running', started_at=now(), execution_kind=args.execution_kind,
                  scope='CPU platform acceptance; two-episode training is not a performance result',
                  test_evaluated=False, stages=[])
    status_path = output / 'acceptance.json'
    save_json(status_path, status)
    environment = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), MPLBACKEND='Agg',
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       PYTHONUNBUFFERED='1', PYTHONUTF8='1', PYTHONHASHSEED='2026')

    def run(name, command):
        stage = dict(name=name, command=[str(x) for x in command], started_at=now(), status='running')
        status['stages'].append(stage)
        save_json(status_path, status)
        print(f'[{now()}] {name}', flush=True)
        with (output / 'logs' / f'{name}.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(stage['command'], cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end='', flush=True)
                code = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                stage.update(status='interrupted', finished_at=now())
                save_json(status_path, status)
                raise
        stage.update(returncode=code, status='passed' if code == 0 else 'failed', finished_at=now())
        save_json(status_path, status)
        if code:
            raise RuntimeError(f'{name} failed with exit code {code}; see logs/{name}.log')

    try:
        facts = platform_facts()
        facts['declared_execution_kind'] = args.execution_kind
        save_json(output / 'environment.json', facts)
        require_target(facts, args.execution_kind)
        (output / 'os-release.txt').write_text(Path('/etc/os-release').read_text(), encoding='utf-8')
        save_json(output / 'source_sha256.json', source_hashes())
        (output / 'requirements-resolved.txt').write_text(''.join(
            f'{name}=={version}\n' for name, version in facts['packages'].items()
            if name.lower() != 'cpn-hrl-dag'), encoding='utf-8')
        run('dependency_check', [sys.executable, '-m', 'pip', 'check'])
        run('dataset_check', [sys.executable, 'scripts/prepare_data.py'])
        data_root = ROOT / 'data/raw/zenodo-18927122-derived'
        archives = [data_root / 'system_configs.tar.xz']
        for size in (50, 100, 300):
            archives.extend(sorted(data_root.glob(f'rnc{size}_*_json.tar.xz')))
        if len(archives) != 7 or not all(path.is_file() for path in archives):
            raise RuntimeError('Expected system archive and homo/hetero archives for scales 50/100/300.')
        save_json(output / 'data_sha256.json', {p.name: sha256(p) for p in archives})
        run('pytest', [sys.executable, '-m', 'pytest', '-q', '--junitxml', str(output / 'pytest.xml')])

        import yaml
        config = yaml.safe_load((ROOT / 'configs/openeuler_cpu_smoke.yaml').read_text(encoding='utf-8'))
        config['output_dir'] = str(output / 'smoke_training')
        runtime_config = output / 'smoke_config.yaml'
        runtime_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        run('short_training', [sys.executable, 'scripts/train_main_comparison.py', '--config', runtime_config])
        smoke_report = verify_report(output / 'smoke_training')
        if smoke_report['training_episodes'] != 2 or min(smoke_report['optimizer_updates_high'], smoke_report['optimizer_updates_low']) < 1:
            raise RuntimeError('Short training did not update both model levels.')
        status['smoke_training'] = smoke_report

        def evaluate(config, checkpoint, target, name, policy='hrl'):
            config = dict(config, output_dir=str(target), device='cpu', torch_num_threads=1)
            path = output / f'{name}_config.yaml'
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
            command = [sys.executable, 'scripts/evaluate.py', '--config', path, '--policy', policy, '--split', 'validation']
            if checkpoint:
                command.extend(['--checkpoint', str(checkpoint)])
            run(name, command)
            report_dir = target / f'eval_{policy}_validation'
            report = verify_report(report_dir)
            status[name] = report
            return report_dir

        replay = evaluate(config, output / 'smoke_training/best.pt', output / 'smoke_replay', 'checkpoint_reload')
        compare_replay(output / 'smoke_training', replay)
        status['checkpoint_reload_equal'] = True
        evaluate(config, None, output / 'heft', 'heft_validation', policy='heft')
        if not math.isclose(status['heft_validation']['mean_ratio'], 1.0, abs_tol=1e-12):
            raise RuntimeError('HEFT self-ratio must be one.')
        if args.frozen_checkpoint:
            checkpoint = args.frozen_checkpoint.resolve()
            frozen_config = yaml.safe_load(args.frozen_config.read_text(encoding='utf-8'))
            if frozen_config['dataset'] != config['dataset'] or frozen_config['resources'] != config['resources'] or frozen_config['environment'] != config['environment']:
                raise RuntimeError('Frozen configuration must use the same fixed data/resources/environment.')
            status['frozen_checkpoint'] = dict(path=str(checkpoint), sha256=sha256(checkpoint), config_sha256=sha256(args.frozen_config))
            evaluate(frozen_config, checkpoint, output / 'frozen', 'frozen_validation')
            if sha256(checkpoint) != status['frozen_checkpoint']['sha256']:
                raise RuntimeError('Frozen checkpoint changed during evaluation.')
        status.update(status='passed', finished_at=now())
        save_json(status_path, status)
        print(f'Platform acceptance passed: {status_path}', flush=True)
    except BaseException as exc:
        status.update(status='failed', finished_at=now(), error=f'{type(exc).__name__}: {exc}')
        save_json(status_path, status)
        (output / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
