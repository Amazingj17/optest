from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from cpn_hrl_dag.evaluation import Evaluator
from cpn_hrl_dag.policies.base import SchedulerPolicy
from cpn_hrl_dag.policies.hybrid import IndependentHybridPolicy
from cpn_hrl_dag.policies.heuristics import HEFTPolicy
from cpn_hrl_dag.policies.portfolio import HEFTSafePortfolioPolicy
from cpn_hrl_dag.scenario.types import Scenario, Task, ComputeNode


def scene():
    # LPT/HEFT makespan 7; placing the two 3-unit jobs together yields 6.
    return Scenario('hybrid-fixture', 'unit', [Task(str(i), w) for i, w in enumerate([3,3,2,2,2])],
                    [], [ComputeNode('a','cloud',1.), ComputeNode('b','edge',1.)],
                    np.full((2,2), 100.), metadata={'original_graph_id':'hybrid-fixture'})


class ScriptedPolicy(SchedulerPolicy):
    name = 'scripted'
    def __init__(self, nodes=(0,0,1,1,1)):
        self.nodes = nodes
    def reset(self, scenario):
        self.cursor = 0
    def select_task(self, observation, ready_mask, deterministic=True):
        return self.cursor
    def select_node(self, observation, task_id, node_mask, deterministic=True):
        result = self.nodes[self.cursor]
        self.cursor += 1
        return result


def test_learned_candidate_can_strictly_improve_and_replay_its_schedule():
    policy = IndependentHybridPolicy(ScriptedPolicy(), search_config={'perturbation_candidates':0})
    records, _ = Evaluator({'normalize_observations':True}).evaluate(policy, [scene()])
    assert records[0].valid_schedule
    assert records[0].makespan == 6
    assert records[0].ratio == pytest.approx(6/7)
    assert policy.selected_candidate == 'residual_hrl'
    assert len(policy.planned_decisions) == 5
    assert records[0].makespan == min(c.result.makespan for c in policy.candidates)
    assert records[0].inference_time_ms >= sum(c.inference_time_ms for c in policy.candidates)


def test_independent_search_matches_standalone_even_when_hrl_changes():
    config = dict(perturbation_candidates=8, task_top_k=2, node_top_k=2,
                  local_search_rounds=2, local_search_beam_width=3, local_search_critical_tasks=5,
                  local_search_block_rounds=1)
    standalone = HEFTSafePortfolioPolicy(seed=2026, normalize_observations=True, **config)
    direct, _ = Evaluator({'normalize_observations':True}).evaluate(standalone, [scene()])
    for learned in [ScriptedPolicy(), ScriptedPolicy((0,0,0,0,0))]:
        combined = IndependentHybridPolicy(learned, search_config=config)
        records, _ = Evaluator({'normalize_observations':True}).evaluate(combined, [scene()])
        search = next(c for c in combined.candidates if c.name=='search_blocks')
        assert search.result.makespan == direct[0].makespan
        assert search.decisions == standalone.planned_decisions
        assert combined.search_policy.learned_policy is None
        assert records[0].makespan <= direct[0].makespan
        assert records[0].ratio <= 1


def test_exact_ties_prefer_heft_and_do_not_count_as_learning_gain():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    from evaluate_independent_hybrid import attribution
    policy = IndependentHybridPolicy(HEFTPolicy(), search_config={'perturbation_candidates':0})
    records, _ = Evaluator({'normalize_observations':True}).evaluate(policy, [scene()])
    assert policy.selected_candidate == 'heft'
    result = attribution([dict(heft_ratio=1.,search_blocks_ratio=1.,residual_hrl_ratio=1.,
                               hybrid_ratio=records[0].ratio,selected_source='heft')])
    assert result['hrl_strict_improvements'] == 0
    assert result['hrl_tied_best'] == 1
    with pytest.raises(AssertionError):
        attribution([dict(heft_ratio=1.,search_blocks_ratio=.9,residual_hrl_ratio=1.,
                          hybrid_ratio=1.,selected_source='heft')])


def test_invalid_candidate_is_not_silently_hidden_and_clears_previous_plan():
    learned = ScriptedPolicy()
    policy = IndependentHybridPolicy(learned, search_config={'perturbation_candidates':0})
    policy.reset(scene())
    learned.nodes = (999,)*5
    with pytest.raises(ValueError):
        policy.reset(scene())
    assert not policy.candidates
    with pytest.raises(RuntimeError):
        _ = policy.planned_decisions


def test_replay_checks_task_ready_and_node_feasibility():
    policy = IndependentHybridPolicy(ScriptedPolicy(), search_config={'perturbation_candidates':0})
    policy.reset(scene())
    with pytest.raises(ValueError, match='ready'):
        policy.select_task({}, np.zeros(5,dtype=bool))
    with pytest.raises(ValueError, match='inconsistent'):
        policy.select_node({}, 4, np.ones(2,dtype=bool))
    with pytest.raises(ValueError, match='infeasible'):
        policy.select_node({}, 0, np.zeros(2,dtype=bool))


def test_search_cannot_be_configured_with_a_learned_branch():
    with pytest.raises(ValueError, match='reserved'):
        IndependentHybridPolicy(HEFTPolicy(), search_config={'learned_policy':HEFTPolicy()})


def test_evaluation_entry_full_protocol_artifacts_and_write_protection(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    import evaluate_independent_hybrid as entry
    from cpn_hrl_dag.datasets.split import SplitManifest
    from cpn_hrl_dag.evaluation import reporting
    from cpn_hrl_dag.utils.config import load_config
    config = load_config('configs/hrl_independent_hybrid_seed2026.yaml')
    scenes = [replace(scene(), scenario_id=f'fixture:{dag}:{resource}',
                      metadata={'original_graph_id':str(dag)}) for dag in range(54) for resource in ('homogeneous','heterogeneous')]
    manifest = SplitManifest(('unit:t',),tuple(f'unit:{i}' for i in range(54)),(),7)
    monkeypatch.setattr(entry, 'load_fixed_splits', lambda _: ({'validation':scenes}, manifest))
    monkeypatch.setattr(entry, 'restore_hrl', lambda _: ScriptedPolicy())
    # Test the reporting/entry pipeline on small fixtures; no formal DAGs are evaluated here.
    real_hybrid = IndependentHybridPolicy
    monkeypatch.setattr(entry, 'IndependentHybridPolicy', lambda learned, **kwargs:
                        real_hybrid(learned, search_config={'perturbation_candidates':0}))
    monkeypatch.setattr(reporting, '_write_plots', lambda *args:None)
    output = entry.evaluate(config, tmp_path/'run')
    status = json.loads((output/'status.json').read_text())
    assert status['status']=='complete' and status['num_scenarios']==108
    assert status['hrl_strict_improvements']==108
    assert status['test_evaluated'] is False
    assert len((output/'candidate_schedules.jsonl').read_text().splitlines())==324
    assert len((output/'per_scene_all.csv').read_text().splitlines())==433
    assert (output/'RESULTS.md').is_file()
    assert json.loads((output/'attribution.json').read_text())['search_strict_improvements']==0
    with pytest.raises(FileExistsError):
        entry.evaluate(config, output)
    with pytest.raises(ValueError, match='only'):
        entry.evaluate(dict(config,split='test'),tmp_path/'forbidden')
