from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'run_graph_baseline_comparison.py'
SPEC = importlib.util.spec_from_file_location('baseline_pipeline', SCRIPT)
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


@pytest.mark.parametrize('fail_stage', [None, 'train_graph_ppo', 'train_tier_mappo', 'evaluate', 'plot'])
def test_pipeline_sequences_stages_and_stops_on_failure(tmp_path, monkeypatch, fail_stage):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline, 'ROOT', tmp_path)
    output = tmp_path / 'run'
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), '--output', str(output), '--training-root', str(tmp_path / 'models'),
                                    '--training-device', 'cpu', '--evaluation-device', 'cpu'])
    monkeypatch.setattr(pipeline, 'verify_completed_run', lambda run, method: run / 'best.pt')
    calls = []

    class Process:
        pid = 1234

        def __init__(self, command, **kwargs):
            self.stage = Path(kwargs['stdout'].name).stem
            self.command = command
            calls.append(self)

        def wait(self):
            return 1 if self.stage == fail_stage else 0

    monkeypatch.setattr(pipeline.subprocess, 'Popen', Process)
    if fail_stage:
        with pytest.raises(RuntimeError, match='failed with exit code'):
            pipeline.main()
    else:
        pipeline.main()
    status = json.loads((output / 'pipeline_status.json').read_text())
    stages = ['train_graph_ppo', 'train_tier_mappo', 'evaluate', 'plot']
    expected = stages if not fail_stage else stages[:stages.index(fail_stage) + 1]
    assert [call.stage for call in calls] == expected
    assert status['status'] == ('failed' if fail_stage else 'complete')
    assert status['test_evaluated'] is False
    assert status['training_device'] == status['evaluation_device'] == 'cpu'
    for call in calls:
        if call.stage.startswith('train_'):
            assert call.command[-2:] == ['--device', 'cpu']
        assert '--split' not in call.command
        assert '--limit' not in call.command
    with pytest.raises(FileExistsError, match='overwrite pipeline status'):
        pipeline.main()


def test_verify_completed_run_rejects_partial_checkpoint(tmp_path, monkeypatch):
    import cpn_hrl_dag.experiments.graph_baselines as experiment

    monkeypatch.setattr(experiment, 'read_checkpoint', lambda path: {'method': 'graph_ppo', 'training_complete': False})
    with pytest.raises(ValueError, match='incomplete training'):
        pipeline.verify_completed_run(tmp_path, 'graph_ppo')
