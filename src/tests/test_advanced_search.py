from __future__ import annotations

import numpy as np
import pytest
from dataclasses import replace

from cpn_hrl_dag.baselines.advanced import CPOPScheduler, PEFTScheduler, ScaledHEFTScheduler
from cpn_hrl_dag.evaluation.evaluator import Evaluator
from cpn_hrl_dag.policies.portfolio import HEFTSafePortfolioPolicy, PortfolioCandidate
from cpn_hrl_dag.scenario.types import ComputeNode, Dependency, Scenario, Task
from cpn_hrl_dag.scheduling.simulator import ScheduleSimulator


def _two_task_scenario() -> Scenario:
    return Scenario(
        scenario_id="advanced-two-task",
        dataset_source="unit",
        tasks=[Task("a", 4.0), Task("b", 2.0)],
        dependencies=[Dependency("a", "b", 10.0)],
        compute_nodes=[
            ComputeNode("slow", "edge", 1.0),
            ComputeNode("fast", "cloud", 2.0),
        ],
        bandwidth_matrix=np.asarray([[1e9, 10.0], [10.0, 1e9]]),
        metadata={"original_graph_id": "advanced-two-task"},
    )


def test_peft_optimistic_cost_table_and_schedule() -> None:
    scenario = _two_task_scenario()
    result, analysis = PEFTScheduler().schedule(scenario)
    np.testing.assert_allclose(analysis.optimistic_cost_table, [[2.0, 1.0], [0.0, 0.0]])
    np.testing.assert_allclose(analysis.priority, [1.5, 0.0])
    assert analysis.task_order == (0, 1)
    by_task = {entry.task_position: entry for entry in result.entries}
    assert by_task[0].node_position == 1
    assert by_task[1].node_position == 1
    assert result.makespan == 3.0


def test_cpop_pins_the_averaged_critical_path_to_one_processor() -> None:
    scenario = Scenario(
        scenario_id="cpop-manual",
        dataset_source="unit",
        tasks=[Task("a", 10.0), Task("b", 10.0), Task("c", 2.0)],
        dependencies=[Dependency("a", "b", 10.0), Dependency("a", "c", 0.0)],
        compute_nodes=[
            ComputeNode("slow", "edge", 1.0),
            ComputeNode("fast", "cloud", 2.0),
        ],
        bandwidth_matrix=np.asarray([[1e9, 10.0], [10.0, 1e9]]),
        metadata={"original_graph_id": "cpop-manual"},
    )
    result, analysis = CPOPScheduler().schedule(scenario)
    assert analysis.critical_path == (0, 1)
    assert analysis.critical_node == 1
    by_task = {entry.task_position: entry for entry in result.entries}
    assert by_task[0].node_position == by_task[1].node_position == 1
    assert result.makespan == 10.0


def test_scaled_heft_is_deterministic_and_legal() -> None:
    first, _ = ScaledHEFTScheduler(communication_scale=0.0).schedule(_two_task_scenario())
    second, _ = ScaledHEFTScheduler(communication_scale=0.0).schedule(_two_task_scenario())
    assert first == second


def test_advanced_search_is_heft_safe_and_auditable() -> None:
    policy = HEFTSafePortfolioPolicy(
        perturbation_candidates=0,
        include_peft=True,
        peft_lookahead_weights=(0.5, 1.0, 2.0),
        include_cpop=True,
        heft_communication_scales=(0.0, 2.0),
        node_discrepancy_candidates=1,
        local_search_rounds=1,
        local_search_critical_tasks=2,
        local_search_node_alternatives=1,
        seed=19,
    )
    first, first_summary = Evaluator().evaluate(policy, [_two_task_scenario()])
    first_names = tuple(candidate.name for candidate in policy.candidates)
    second, second_summary = Evaluator().evaluate(policy, [_two_task_scenario()])
    assert first[0].valid_schedule
    assert first_summary["mean_ratio"] <= 1.0
    assert second_summary["mean_ratio"] == first_summary["mean_ratio"]
    assert first[0].makespan == second[0].makespan
    assert first_names[0] == "heft"
    assert "cpop" in first_names
    assert "peft_w1" in first_names
    assert any(name.startswith("lds_node_") for name in first_names)
    assert any(name.startswith("local_r00_") for name in first_names)
    assert policy.name == "heft_safe_lns"


def test_critical_chain_task_order_moves_remain_precedence_legal() -> None:
    scenario = Scenario(
        scenario_id="task-order-lns",
        dataset_source="unit",
        tasks=[Task("a", 2.0), Task("b", 6.0), Task("c", 5.0), Task("d", 1.0)],
        dependencies=[
            Dependency("a", "b", 1.0),
            Dependency("a", "c", 1.0),
            Dependency("b", "d", 1.0),
            Dependency("c", "d", 1.0),
        ],
        compute_nodes=[
            ComputeNode("edge", "edge", 1.0),
            ComputeNode("cloud", "cloud", 2.0),
        ],
        bandwidth_matrix=np.asarray([[1e9, 5.0], [5.0, 1e9]]),
        metadata={"original_graph_id": "task-order-lns"},
    )
    policy = HEFTSafePortfolioPolicy(
        perturbation_candidates=0,
        local_search_rounds=1,
        local_search_critical_tasks=4,
        local_search_task_alternatives=1,
    )
    records, summary = Evaluator().evaluate(policy, [scenario])
    assert records[0].valid_schedule
    assert summary["mean_ratio"] <= 1.0
    assert any(candidate.name.startswith("local_task_r00_") for candidate in policy.candidates)


def test_beam_search_is_deterministic_valid_and_deduplicated() -> None:
    policy = HEFTSafePortfolioPolicy(
        perturbation_candidates=4,
        local_search_rounds=3,
        local_search_critical_tasks=2,
        local_search_task_alternatives=1,
        local_search_beam_width=3,
        seed=19,
    )
    first, summary = Evaluator().evaluate(policy, [_two_task_scenario()])
    candidates = policy.candidates
    decisions = policy.planned_decisions
    second, _ = Evaluator().evaluate(policy, [_two_task_scenario()])
    assert first[0].valid_schedule and second[0].valid_schedule
    assert summary["mean_ratio"] <= 1.0
    assert policy.candidates == candidates
    assert policy.planned_decisions == decisions
    seen = set()
    for candidate in candidates:
        key = policy._schedule_key(candidate)
        if candidate.name.startswith("beam_"):
            assert key not in seen
        seen.add(key)


def test_beam_search_explores_after_non_improving_round(monkeypatch) -> None:
    scenario = _two_task_scenario()
    result, _ = ScaledHEFTScheduler().schedule(scenario)
    initial = PortfolioCandidate("initial", replace(result, makespan=3.0))
    plateau = PortfolioCandidate("plateau", replace(result, makespan=4.0))
    improved = PortfolioCandidate("improved", replace(result, makespan=2.0))
    policy = HEFTSafePortfolioPolicy(local_search_rounds=3, local_search_beam_width=3)
    expanded = []

    def moves(scenario, incumbent, round_index, execution, communication):
        expanded.append(incumbent.result.makespan)
        return {3.0: [plateau, plateau], 4.0: [improved], 2.0: []}[incumbent.result.makespan]

    monkeypatch.setattr(policy, "_schedule_key", lambda candidate: candidate.result.makespan)
    monkeypatch.setattr(policy, "_critical_path_node_moves", moves)
    monkeypatch.setattr(policy, "_critical_path_task_moves", lambda *args: [])
    candidates = [initial]
    policy._beam_search(scenario, candidates, np.zeros(2), None, None)
    assert expanded == [3.0, 4.0, 2.0]
    assert len(candidates) == 3
    assert min(candidate.result.makespan for candidate in candidates) == 2.0


def test_beam_width_one_preserves_default_search() -> None:
    options = dict(perturbation_candidates=2, local_search_rounds=2, local_search_critical_tasks=2)
    default = HEFTSafePortfolioPolicy(**options)
    explicit = HEFTSafePortfolioPolicy(**options, local_search_beam_width=1)
    default.reset(_two_task_scenario())
    explicit.reset(_two_task_scenario())
    assert default.candidates == explicit.candidates
    assert default.planned_decisions == explicit.planned_decisions


@pytest.mark.parametrize("width", [0, -1])
def test_invalid_beam_width_is_rejected(width) -> None:
    with pytest.raises(ValueError, match="beam_width"):
        HEFTSafePortfolioPolicy(local_search_beam_width=width)


def _prefix_cache_scenario() -> Scenario:
    return Scenario(
        scenario_id="prefix-cache-branches",
        dataset_source="unit",
        tasks=[Task(f"task-{index}", float(index % 5 + 2)) for index in range(12)],
        dependencies=[Dependency(f"task-{index}", f"task-{index + 3}", 2.0) for index in range(9)],
        compute_nodes=[ComputeNode("edge", "edge", 1.0), ComputeNode("cloud", "cloud", 2.0)],
        bandwidth_matrix=np.asarray([[1e9, 5.0], [5.0, 1e9]]),
        metadata={"original_graph_id": "prefix-cache-branches"},
    )


@pytest.mark.parametrize("width", [1, 3])
def test_prefix_cache_preserves_every_candidate_and_reduces_replay(width, monkeypatch) -> None:
    scenario = _prefix_cache_scenario()
    options = dict(
        perturbation_candidates=4,
        local_search_rounds=3,
        local_search_critical_tasks=6,
        local_search_node_alternatives=2,
        local_search_task_alternatives=2,
        local_search_beam_width=width,
        seed=19,
    )
    schedule = ScheduleSimulator.schedule
    calls = []

    def counted_schedule(simulator, task, node):
        calls.append((task, node))
        return schedule(simulator, task, node)

    monkeypatch.setattr(ScheduleSimulator, "schedule", counted_schedule)
    uncached = HEFTSafePortfolioPolicy(**options, local_search_prefix_cache=False)
    uncached.reset(scenario)
    uncached_count = len(calls)
    calls.clear()
    cached = HEFTSafePortfolioPolicy(**options, local_search_prefix_cache=True)
    cached.reset(scenario)
    assert cached.candidates == uncached.candidates
    assert cached.planned_decisions == uncached.planned_decisions
    assert cached.selected_candidate == uncached.selected_candidate
    assert len(calls) < uncached_count
    assert not cached._prefix_states
    assert cached._prefix_incumbent is None
    cached.reset(_two_task_scenario())
    uncached.reset(_two_task_scenario())
    assert cached.candidates == uncached.candidates


def test_prefix_snapshots_are_isolated_and_support_out_of_order_depths() -> None:
    scenario = _prefix_cache_scenario()
    incumbent, _ = ScaledHEFTScheduler().schedule(scenario)
    policy = HEFTSafePortfolioPolicy()
    empty = ScheduleSimulator(scenario)
    for depth in (8, 3, 9, 0, 3):
        snapshot = policy._prefix_template(incumbent, depth, empty)
        expected = empty.clone()
        by_task = {entry.task_position: entry.node_position for entry in incumbent.entries}
        for task in incumbent.decision_order[:depth]:
            expected.schedule(task, by_task[task])
        assert snapshot.entries == expected.entries
        assert snapshot.decision_order == expected.decision_order
        assert snapshot.ready_task_positions() == expected.ready_task_positions()
        branch = snapshot.clone()
        task = incumbent.decision_order[depth]
        branch.schedule(task, by_task[task])
        assert len(snapshot.entries) == depth
    assert not empty.entries


@pytest.mark.parametrize("width", [1, 3])
@pytest.mark.parametrize("prefix_cache", [False, True])
def test_result_cache_preserves_candidates_and_skips_simulation(width, prefix_cache, monkeypatch) -> None:
    options = dict(
        perturbation_candidates=4,
        local_search_rounds=3,
        local_search_critical_tasks=6,
        local_search_node_alternatives=2,
        local_search_task_alternatives=2,
        local_search_beam_width=width,
        local_search_prefix_cache=prefix_cache,
        seed=19,
    )
    schedule = ScheduleSimulator.schedule
    calls = []

    def counted_schedule(simulator, task, node):
        calls.append((task, node))
        return schedule(simulator, task, node)

    monkeypatch.setattr(ScheduleSimulator, "schedule", counted_schedule)
    baseline = HEFTSafePortfolioPolicy(**options, local_search_result_cache=False)
    cached = HEFTSafePortfolioPolicy(**options, local_search_result_cache=True)
    baseline.reset(_prefix_cache_scenario())
    baseline_count = len(calls)
    calls.clear()
    cached.reset(_prefix_cache_scenario())
    assert cached.candidates == baseline.candidates
    assert cached.planned_decisions == baseline.planned_decisions
    assert cached.selected_candidate == baseline.selected_candidate
    if width > 1:
        assert cached.replay_cache_hits > 0
    assert baseline.replay_cache_hits == 0
    assert baseline_count - len(calls) == cached.replay_cache_skipped_tasks
    assert not cached._replay_results
    cached.reset(_two_task_scenario())
    baseline.reset(_two_task_scenario())
    assert cached.candidates == baseline.candidates
    assert cached.planned_decisions == baseline.planned_decisions
    assert not cached._replay_results


def test_result_cache_requires_exact_order_and_node_assignments() -> None:
    scenario = _prefix_cache_scenario()
    result, _ = ScaledHEFTScheduler().schedule(scenario)
    policy = HEFTSafePortfolioPolicy(local_search_result_cache=True)
    policy._remember_replay(result)
    key = policy._schedule_key(PortfolioCandidate("original", result))
    assert policy._cached_replay(key, 0) is result
    changed_node = ((key[0][0], 1 - key[0][1]),) + key[1:]
    changed_order = (key[1], key[0]) + key[2:]
    assert policy._cached_replay(changed_node, 0) is None
    assert policy._cached_replay(changed_order, 0) is None


@pytest.mark.parametrize("kind", ["node", "task"])
def test_duplicate_preserve_move_skips_entire_suffix(kind, monkeypatch) -> None:
    scenario = _prefix_cache_scenario()
    result, analysis = ScaledHEFTScheduler().schedule(scenario)
    template = ScheduleSimulator(scenario)
    policy = HEFTSafePortfolioPolicy(local_search_result_cache=True)

    def replay():
        if kind == "node":
            return policy._replay_with_node_move(
                scenario, result, result.decision_order[0], 0, False,
                template.execution_model, template.communication_model, template,
            )
        return policy._replay_with_task_move(
            scenario, result, 0, 0, False, analysis.upward_rank,
            template.execution_model, template.communication_model, template,
        )

    first = replay()
    assert first is not None

    def unexpected_schedule(*args):
        raise AssertionError("duplicate candidate must not schedule any tasks")

    monkeypatch.setattr(ScheduleSimulator, "schedule", unexpected_schedule)
    assert replay() is first
    assert policy.replay_cache_hits == 1
    assert policy.replay_cache_skipped_tasks == scenario.num_tasks


def test_joint_block_move_crosses_single_task_barrier() -> None:
    scenario = replace(_two_task_scenario(), dependencies=[Dependency("a", "b", 100.0)])
    simulator = ScheduleSimulator(scenario)
    simulator.schedule(0, 0)
    simulator.schedule(1, 0)
    incumbent = simulator.result()
    empty = ScheduleSimulator(scenario)
    policy = HEFTSafePortfolioPolicy(local_search_block_rounds=2)
    for task in (0, 1):
        single = policy._replay_with_node_move(
            scenario, incumbent, task, 0, False, empty.execution_model, empty.communication_model, empty,
        )
        assert single.makespan > incumbent.makespan
    moves = policy._critical_block_moves(
        scenario, PortfolioCandidate("slow", incumbent), 0, empty.execution_model, empty.communication_model,
    )
    assert moves
    assert min(candidate.result.makespan for candidate in moves) == 3.0
    assert incumbent.makespan == 6.0
    for candidate in moves:
        assert all(entry.node_position == 1 for entry in candidate.result.entries)


def test_block_moves_skip_incompatible_joint_placements() -> None:
    scenario = replace(
        _two_task_scenario(),
        tasks=[Task("a", 4.0, cpu_requirement=1.0), Task("b", 2.0, memory_requirement=1.0)],
        compute_nodes=[
            ComputeNode("cpu", "edge", 1.0, cpu_capacity=1.0, memory_capacity=0.0),
            ComputeNode("memory", "cloud", 2.0, cpu_capacity=0.0, memory_capacity=1.0),
        ],
    )
    simulator = ScheduleSimulator(scenario)
    simulator.schedule(0, 0)
    simulator.schedule(1, 1)
    policy = HEFTSafePortfolioPolicy(local_search_block_rounds=2)
    assert policy._critical_block_moves(
        scenario, PortfolioCandidate("forced", simulator.result()), 0,
        simulator.execution_model, simulator.communication_model,
    ) == []


@pytest.mark.parametrize("width", [1, 3])
@pytest.mark.parametrize("result_cache", [False, True])
def test_block_refinement_preserves_baseline_and_is_cache_equivalent(width, result_cache) -> None:
    scenario = _prefix_cache_scenario()
    options = dict(
        perturbation_candidates=4, local_search_rounds=2, local_search_critical_tasks=4,
        local_search_task_alternatives=1, local_search_beam_width=width,
        local_search_result_cache=result_cache, seed=19,
    )
    baseline = HEFTSafePortfolioPolicy(**options)
    baseline.reset(scenario)
    policy = HEFTSafePortfolioPolicy(**options, local_search_block_rounds=2)
    records, summary = Evaluator().evaluate(policy, [scenario])
    assert records[0].valid_schedule
    assert summary["mean_ratio"] <= 1.0
    assert policy.selected_makespan <= baseline.selected_makespan
    assert policy.candidates[:len(baseline.candidates)] == baseline.candidates
    assert any(candidate.name.startswith("block_") for candidate in policy.candidates)
    assert sum(candidate.name.startswith("block_") for candidate in policy.candidates) <= 32
    first = policy.candidates
    policy.reset(scenario)
    assert policy.candidates == first
    uncached = HEFTSafePortfolioPolicy(**options, local_search_block_rounds=2, local_search_prefix_cache=False)
    uncached.reset(scenario)
    assert uncached.candidates == first


@pytest.mark.parametrize("greedy_repair", [False, True])
def test_block_replay_forces_all_three_tasks_and_preserves_order(greedy_repair) -> None:
    scenario = _prefix_cache_scenario()
    simulator = ScheduleSimulator(scenario)
    for task in scenario.cache.topological_order:
        simulator.schedule(task, 0)
    incumbent = simulator.result()
    policy = HEFTSafePortfolioPolicy()
    result = policy._replay_with_block_move(incumbent, (0, 3, 6), 1, greedy_repair, ScheduleSimulator(scenario))
    assert result.decision_order == incumbent.decision_order
    entries = {entry.task_position: entry for entry in result.entries}
    assert all(entries[task].node_position == 1 for task in (0, 3, 6))
    assert len(entries) == scenario.num_tasks


@pytest.mark.parametrize("options", [
    {"local_search_block_rounds": -1},
    {"local_search_blocks": 0},
    {"local_search_block_max_size": 1},
    {"local_search_block_node_alternatives": 0},
])
def test_invalid_block_options_are_rejected(options) -> None:
    with pytest.raises(ValueError, match="block"):
        HEFTSafePortfolioPolicy(**options)
