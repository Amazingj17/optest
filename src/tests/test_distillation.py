from __future__ import annotations

import numpy as np
import torch

from cpn_hrl_dag.algorithms.dagger import (
    CompletionRegretDAgger,
    completion_regret_targets,
)
from cpn_hrl_dag.algorithms.distillation import DAGPairDistiller, SearchTeacherCache
from cpn_hrl_dag.env.dag_env import CloudEdgeEndDAGEnv
from cpn_hrl_dag.models.dag_pair import DAGPairGraphActorCritic
from cpn_hrl_dag.policies.dag_pair import DAGPairGraphPolicy
from cpn_hrl_dag.policies.portfolio import HEFTSafePortfolioPolicy
from cpn_hrl_dag.scenario.types import ComputeNode, Dependency, Scenario, Task


def _scenario(name: str, workload: float) -> Scenario:
    return Scenario(
        scenario_id=name,
        dataset_source="unit",
        tasks=[Task("a", workload), Task("b", 2.0), Task("c", 1.0)],
        dependencies=[Dependency("a", "b", 1.0), Dependency("a", "c", 0.0)],
        compute_nodes=[ComputeNode("slow", "edge", 1.0), ComputeNode("fast", "cloud", 2.0)],
        bandwidth_matrix=np.asarray([[1e9, 10.0], [10.0, 1e9]]),
        metadata={"original_graph_id": name},
    )


def _model(scenario: Scenario) -> DAGPairGraphActorCritic:
    env = CloudEdgeEndDAGEnv(normalize_observations=True)
    env.reset(scenario)
    observation = env.get_flat_observation()
    return DAGPairGraphActorCritic(
        observation["task_features"].shape[-1],
        observation["resource_features"].shape[-1],
        observation["pair_node_features"].shape[-1],
        observation["task_edge_features"].shape[-1],
        observation["resource_edge_features"].shape[-1],
        hidden_dim=16,
        heads=4,
        task_layers=1,
        resource_layers=1,
    )


def test_search_teacher_cache_round_trip(tmp_path) -> None:
    scenario = _scenario("teacher", 4.0)
    cache = SearchTeacherCache(tmp_path, "unit-search")
    policy = HEFTSafePortfolioPolicy(
        perturbation_candidates=2,
        include_peft=True,
        peft_lookahead_weights=(0.25,),
    )
    generated = cache.get_or_compute(scenario, policy)
    reloaded = cache.get_or_compute(scenario, policy)
    assert generated == reloaded
    assert len(generated.decisions) == scenario.num_tasks
    assert generated.ratio <= 1.0
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_joint_graph_distillation_smoke_is_finite_and_legal(tmp_path) -> None:
    scenarios = [_scenario("teacher-a", 4.0), _scenario("teacher-b", 5.0)]
    teacher_policy = HEFTSafePortfolioPolicy(perturbation_candidates=2)
    cache = SearchTeacherCache(tmp_path, "distillation-smoke")
    teachers = {
        scenario.scenario_id: cache.get_or_compute(scenario, teacher_policy)
        for scenario in scenarios
    }
    model = _model(scenarios[0])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    result = DAGPairDistiller(
        model,
        optimizer,
        torch.device("cpu"),
        states_per_scenario=2,
    ).fit(scenarios, teachers, epochs=1, seed=3)
    assert result.scenarios == 2
    assert result.decisions == 4
    assert np.isfinite(result.actor_loss)
    assert np.isfinite(result.value_loss)
    assert 0.0 <= result.action_accuracy <= 1.0

    policy = DAGPairGraphPolicy(model, normalize_observations=True)
    from cpn_hrl_dag.evaluation.evaluator import Evaluator

    records, summary = Evaluator({"normalize_observations": True}).evaluate(policy, scenarios)
    assert all(record.valid_schedule for record in records)
    assert summary["valid_schedule_rate"] == 1.0


def test_completion_regret_oracle_is_deterministic_and_normalized() -> None:
    scenario = _scenario("regret-oracle", 8.0)
    env = CloudEdgeEndDAGEnv(normalize_observations=True)
    env.reset(scenario)
    assert env.simulator is not None
    assert env.upward_rank is not None
    first = completion_regret_targets(
        env.simulator,
        env.upward_rank,
        (0, 1),
        heft_makespan=env.heft_makespan,
        temperature=0.05,
    )
    second = completion_regret_targets(
        env.simulator,
        env.upward_rank,
        (0, 1),
        heft_makespan=env.heft_makespan,
        temperature=0.05,
    )
    assert first == second
    assert first.best_action in first.actions
    assert min(first.normalized_regrets) == 0.0
    assert np.isclose(sum(first.probabilities), 1.0)
    assert all(value >= 0.0 for value in first.normalized_regrets)


def test_completion_regret_dagger_visits_only_legal_states() -> None:
    scenarios = [_scenario("dagger-a", 4.0), _scenario("dagger-b", 6.0)]
    model = _model(scenarios[0])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    result = CompletionRegretDAgger(
        model,
        optimizer,
        torch.device("cpu"),
        states_per_scenario=2,
        max_candidate_actions=2,
        heuristic_task_candidates=1,
        heuristic_node_candidates=2,
        student_candidates=1,
    ).fit(scenarios, epochs=1, seed=7, teacher_beta=0.5)
    assert result.scenarios == 2
    assert result.decisions == 4
    assert result.completion_evaluations >= result.decisions
    assert np.isfinite(result.actor_loss)
    assert np.isfinite(result.mean_student_regret)
    assert 0.0 <= result.action_accuracy <= 1.0
    assert 0.0 <= result.rollout_teacher_rate <= 1.0

    from cpn_hrl_dag.evaluation.evaluator import Evaluator

    records, summary = Evaluator({"normalize_observations": True}).evaluate(
        DAGPairGraphPolicy(model, normalize_observations=True), scenarios
    )
    assert all(record.valid_schedule for record in records)
    assert summary["valid_schedule_rate"] == 1.0

    safe_records, safe_summary = Evaluator({"normalize_observations": True}).evaluate(
        HEFTSafePortfolioPolicy(
            perturbation_candidates=0,
            learned_policy=DAGPairGraphPolicy(model, normalize_observations=True),
            normalize_observations=True,
        ),
        scenarios,
    )
    assert all(record.valid_schedule for record in safe_records)
    assert safe_summary["mean_ratio"] <= 1.0
