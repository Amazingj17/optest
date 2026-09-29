from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/verify_platform.py'
SPEC = importlib.util.spec_from_file_location('platform_acceptance', SCRIPT)
acceptance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(acceptance)


def test_target_detection_does_not_accept_ubuntu_or_windows():
    for facts in [dict(system='Windows', os_release={}, container_marker=False),
                  dict(system='Linux', os_release={'ID': 'ubuntu'}, container_marker=True)]:
        with pytest.raises(RuntimeError, match='actual openEuler'):
            acceptance.require_target(facts, 'container')


def test_container_cannot_be_reported_as_native_or_vm():
    facts = dict(system='Linux', os_release={'ID': 'openEuler'}, container_marker=True)
    acceptance.require_target(facts, 'container')
    for kind in ('native', 'vm'):
        with pytest.raises(RuntimeError, match='Detected container'):
            acceptance.require_target(facts, kind)


def report_fixture(path):
    path.mkdir()
    rows = [dict(scenario_id=f'scene-{i}', base_dag_id=f'dag-{i // 2}', makespan='2.0',
                 heft_makespan='2.0', ratio='1.0', valid_schedule='True') for i in range(108)]
    with (path / 'per_scene.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    acceptance.save_json(path / 'summary.json', dict(num_scenarios=108, num_base_dags=54,
                                                   valid_schedule_rate=1.0, mean_ratio=1.0))
    return path


def test_report_checks_actual_rows_and_replay(tmp_path):
    first, second = report_fixture(tmp_path / 'first'), report_fixture(tmp_path / 'second')
    assert acceptance.verify_report(first)['mean_ratio'] == 1.0
    acceptance.compare_replay(first, second)
    path = second / 'per_scene.csv'
    path.write_text(path.read_text().replace('2.0,2.0,1.0', '3.0,2.0,1.5', 1))
    with pytest.raises(RuntimeError, match='replay differs'):
        acceptance.compare_replay(first, second)
    with pytest.raises(RuntimeError, match='Summary differs'):
        acceptance.verify_report(second)


def test_failure_recorded_and_existing_evidence_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(acceptance, 'platform_facts', lambda: dict(
        system='Windows', os_release={}, container_marker=False))
    target = tmp_path / 'rejected'
    with pytest.raises(RuntimeError, match='actual openEuler'):
        acceptance.main(['--output', str(target), '--execution-kind', 'container'])
    status = json.loads((target / 'acceptance.json').read_text())
    assert status['status'] == 'failed'
    assert status['test_evaluated'] is False
    original = (target / 'acceptance.json').read_bytes()
    with pytest.raises(FileExistsError):
        acceptance.main(['--output', str(target), '--execution-kind', 'container'])
    assert (target / 'acceptance.json').read_bytes() == original
