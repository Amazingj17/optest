from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from cpn_hrl_dag.demo import (audit_candidate, evaluate_scene, load_scenario, render_html,
                              save_json, scenario_payload, sha256, summarize_demo, verify_bundle)
from cpn_hrl_dag.policies.heuristics import HEFTPolicy
from cpn_hrl_dag.policies.hybrid import IndependentHybridPolicy
from cpn_hrl_dag.scenario.types import Scenario, Task, Dependency, ComputeNode


def fixture_scene():
    return Scenario('demo:tiny:heterogeneous', 'unit',
                    [Task('a', 4), Task('b', 6), Task('c', 3)],
                    [Dependency('a', 'c', 2), Dependency('b', 'c', 3)],
                    [ComputeNode('edge', 'edge', 1), ComputeNode('cloud', 'cloud', 2)],
                    np.array([[float('inf'), 3], [3, float('inf')]]),
                    metadata={'original_graph_id': 'tiny'})


def policy(reuse=True):
    return IndependentHybridPolicy(HEFTPolicy(), search_config={'perturbation_candidates': 2},
                                   reuse_heft_reference=reuse)


def test_serialized_scenario_preserves_schedule_and_strict_json():
    scene = fixture_scene()
    encoded = json.dumps(scenario_payload(scene), allow_nan=False)
    restored = load_scenario(json.loads(encoded))
    left, right = evaluate_scene(policy(), scene), evaluate_scene(policy(), restored)
    for a, b in zip(left['candidates'], right['candidates']):
        assert a['entries'] == b['entries']
        assert a['makespan'] == b['makespan']
    assert left['record']['ratio'] == right['record']['ratio']
    assert summarize_demo([right])['valid_schedule_rate'] == 1


def test_reference_reuse_avoids_two_baselines_without_changing_candidates(monkeypatch):
    from cpn_hrl_dag.baselines.heft import HEFTScheduler
    original = HEFTScheduler.schedule
    calls = []
    def count(self, scenario):
        calls.append(scenario.scenario_id)
        return original(self, scenario)
    monkeypatch.setattr(HEFTScheduler, 'schedule', count)
    a = evaluate_scene(policy(False), fixture_scene())
    uncached = len(calls)
    calls.clear()
    b = evaluate_scene(policy(True), fixture_scene())
    assert uncached - len(calls) == 2
    assert a['selected_source'] == b['selected_source']
    for old, new in zip(a['candidates'], b['candidates']):
        assert old['entries'] == new['entries']
        assert old['decisions'] == new['decisions']


def test_audit_rejects_tampered_candidate():
    scene, candidate_policy = fixture_scene(), policy()
    candidate_policy.reset(scene)
    candidate = candidate_policy.candidates[0]
    with pytest.raises(ValueError, match='makespan mismatch'):
        audit_candidate(scene, replace(candidate, result=replace(candidate.result, makespan=12345)))
    with pytest.raises((RuntimeError, ValueError)):
        audit_candidate(scene, replace(candidate, decisions=candidate.decisions[:-1]))


def test_audit_uses_dataset_device_speeds():
    scene = Scenario('device-speeds', 'grapheonrl', [Task('gpu-task', 12, device_requirement='gpu')], [],
                     [ComputeNode('accelerator', 'cloud', 1, metadata={
                         'features': ['gpu'], 'device_speeds': {'cpu': 1, 'gpu': 6}})],
                     [[1e9]], metadata={'original_graph_id': 'device-speeds'})
    result = evaluate_scene(policy(), scene)
    assert result['record']['makespan'] == 2
    assert result['all_candidates_replayed']


def test_bundle_tampering_and_path_escape_are_rejected(tmp_path):
    asset = tmp_path / 'asset.json'
    asset.write_text('{}')
    save_json(tmp_path / 'manifest.json', {'files': {'asset.json': sha256(asset)}})
    verify_bundle(tmp_path)
    asset.write_text('{"modified":true}')
    with pytest.raises(ValueError, match='SHA256'):
        verify_bundle(tmp_path)
    save_json(tmp_path / 'manifest.json', {'files': {'../outside': 'x'}})
    with pytest.raises(ValueError, match='escapes'):
        verify_bundle(tmp_path)


def test_html_data_cannot_escape_script_element(tmp_path):
    template = tmp_path / 'template.html'
    template.write_text('<script type="application/json">__DEMO_DATA__</script>')
    output = tmp_path / 'index.html'
    render_html(template, output, {'id': '</script><script>alert(1)</script>'})
    text = output.read_text()
    assert text.count('</script>') == 1
    payload = text.split('>', 1)[1].rsplit('</script>', 1)[0]
    assert json.loads(payload)['id'] == '</script><script>alert(1)</script>'


def test_demo_entry_protects_existing_output_and_records_failure(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from run_hybrid_demo import run
    output = tmp_path / 'output'
    output.mkdir()
    (output / 'keep').write_text('keep')
    with pytest.raises(FileExistsError):
        run(tmp_path / 'missing', output)
    assert (output / 'keep').read_text() == 'keep'
    fresh = tmp_path / 'fresh'
    with pytest.raises(FileNotFoundError):
        run(tmp_path / 'missing', fresh)
    assert json.loads((fresh / 'status.json').read_text(encoding='utf-8'))['status'] == 'failed'
