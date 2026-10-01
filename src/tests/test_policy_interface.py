from __future__ import annotations

import numpy as np
import cpn_hrl_dag.evaluation.evaluator as evaluator_module

from cpn_hrl_dag.evaluation.evaluator import Evaluator
from cpn_hrl_dag.policies.heuristics import GreedyEFTPolicy, HEFTPolicy, RandomPolicy
from cpn_hrl_dag.policies.portfolio import HEFTSafePortfolioPolicy
from cpn_hrl_dag.policies.flat import FlatPPOPolicy
from cpn_hrl_dag.models.flat import FlatTaskNodeActorCritic
from cpn_hrl_dag.models.high_lstm import HighLevelLSTMActorCritic
from cpn_hrl_dag.models.low_gat import LowLevelGATActorCritic
from cpn_hrl_dag.policies.hrl import CPNHRLDAGPolicy
from cpn_hrl_dag.env.dag_env import CloudEdgeEndDAGEnv
from cpn_hrl_dag.scenario.types import ComputeNode, Dependency, Scenario, Task


def scenario() -> Scenario:
    return Scenario("policy", "unit", [Task("a", 4), Task("b", 2)], [Dependency("a", "b", 1)], [ComputeNode("e", "edge", 1), ComputeNode("c", "cloud", 2)], np.array([[1e9, 10.0], [10.0, 1e9]]), metadata={"original_graph_id": "policy"})


def test_all_baseline_policies_are_legal_and_heft_self_ratio_is_one() -> None:
    evaluator = Evaluator()
    for policy in (RandomPolicy(2), GreedyEFTPolicy(), HEFTPolicy()):
        records, summary = evaluator.evaluate(policy, [scenario()])
        assert records[0].valid_schedule
        assert summary["valid_schedule_rate"] == 1.0
    _, summary = evaluator.evaluate(HEFTPolicy(), [scenario()])
    assert summary["mean_ratio"] == 1.0


def test_flat_ppo_policy_uses_legal_joint_actions() -> None:
    env = CloudEdgeEndDAGEnv()
    env.reset(scenario())
    flat = env.get_flat_observation()
    model = FlatTaskNodeActorCritic(flat["task_features"].shape[1], flat["resource_features"].shape[1], flat["pair_node_features"].shape[-1], 8)
    records, summary = Evaluator().evaluate(FlatPPOPolicy(model), [scenario()])
    assert records[0].valid_schedule and summary["valid_schedule_rate"] == 1.0


def test_heft_safe_portfolio_is_deterministic_legal_and_never_worse_than_heft() -> None:
    candidate = HEFTSafePortfolioPolicy(
        perturbation_candidates=8,
        task_top_k=2,
        node_top_k=2,
        seed=17,
        learned_policy=RandomPolicy(5),
    )
    first, first_summary = Evaluator().evaluate(candidate, [scenario()])
    second, second_summary = Evaluator().evaluate(candidate, [scenario()])
    assert first[0].valid_schedule
    assert first_summary["mean_ratio"] <= 1.0
    assert second_summary["mean_ratio"] == first_summary["mean_ratio"]
    assert candidate.candidates[0].name == "heft"
    assert candidate.name == "cpn_hrl_dag_heft_safe"


def test_zero_perturbation_portfolio_exactly_replays_heft() -> None:
    policy = HEFTSafePortfolioPolicy(perturbation_candidates=0)
    _, summary = Evaluator().evaluate(
        policy,
        [scenario()],
    )
    assert summary["mean_ratio"] == 1.0
    assert policy.name == "heft_safe_portfolio"


def test_zero_initialized_residual_hrl_exactly_replays_heft() -> None:
    env = CloudEdgeEndDAGEnv(normalize_observations=True)
    high_observation, _ = env.reset(scenario())
    task = int(np.flatnonzero(env.get_ready_mask())[0])
    env.select_task(task)
    low_observation = env.get_low_observation(task)
    high = HighLevelLSTMActorCritic(
        high_observation["task_features"].shape[1],
        high_observation["resource_features"].shape[1],
        8,
        heuristic_residual=True,
    )
    low = LowLevelGATActorCritic(
        low_observation["node_features"].shape[1],
        high_observation["task_features"].shape[1],
        8,
        2,
        heuristic_residual=True,
    )
    _, summary = Evaluator({"normalize_observations": True}).evaluate(
        CPNHRLDAGPolicy(high, low),
        [scenario()],
    )
    assert summary["mean_ratio"] == 1.0


def test_zero_residual_uses_float64_heft_tie_break() -> None:
    precision_scenario = Scenario(
        "precision",
        "unit",
        [Task("only", 1.0)],
        [],
        [ComputeNode("slower", "edge", 1.0), ComputeNode("faster", "cloud", 1.0 + 1e-9)],
        np.array([[1e9, 10.0], [10.0, 1e9]]),
        metadata={"original_graph_id": "precision"},
    )
    env = CloudEdgeEndDAGEnv(normalize_observations=True)
    high_observation, _ = env.reset(precision_scenario)
    task = int(np.flatnonzero(env.get_ready_mask())[0])
    env.select_task(task)
    low_observation = env.get_low_observation(task)
    assert low_observation["node_features"][0, 5] == low_observation["node_features"][1, 5]
    assert low_observation["heuristic_eft"][1] < low_observation["heuristic_eft"][0]
    high = HighLevelLSTMActorCritic(
        high_observation["task_features"].shape[1],
        high_observation["resource_features"].shape[1],
        8,
        heuristic_residual=True,
    )
    low = LowLevelGATActorCritic(
        low_observation["node_features"].shape[1],
        high_observation["task_features"].shape[1],
        8,
        2,
        heuristic_residual=True,
    )
    _, summary = Evaluator({"normalize_observations": True}).evaluate(
        CPNHRLDAGPolicy(high, low),
        [precision_scenario],
    )
    assert summary["mean_ratio"] == 1.0
    _, heft_summary = Evaluator({"normalize_observations": True}).evaluate(
        HEFTPolicy(),
        [precision_scenario],
    )
    assert heft_summary["mean_ratio"] == 1.0


def test_evaluator_inference_timer_includes_policy_reset(monkeypatch) -> None:
    events: list[str] = []

    class TracingHEFT(HEFTPolicy):
        def reset(self, value: Scenario) -> None:
            events.append("reset")
            super().reset(value)

    def clock() -> float:
        events.append("clock")
        return float(len(events))

    monkeypatch.setattr(evaluator_module, "perf_counter", clock)
    records, _ = Evaluator().evaluate(TracingHEFT(), [scenario()])
    assert events[:2] == ["clock", "reset"]
    assert records[0].inference_time_ms == 2000.0
