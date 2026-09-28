from __future__ import annotations

import numpy as np

from cpn_hrl_dag.env.dag_env import CloudEdgeEndDAGEnv
from cpn_hrl_dag.evaluation.evaluator import Evaluator
from cpn_hrl_dag.models.dag_pair import DAGPairGraphActorCritic
from cpn_hrl_dag.policies.dag_pair import DAGPairGraphPolicy, graph_dynamic_tensors, graph_static_tensors
from cpn_hrl_dag.scenario.types import ComputeNode, Dependency, Scenario, Task


def _scenario(extra_task: bool = False) -> Scenario:
    tasks = [Task("root", 3.0), Task("child", 2.0)]
    dependencies = [Dependency("root", "child", 0.0)]
    if extra_task:
        tasks.append(Task("other", 1.0))
        dependencies.append(Dependency("root", "other", 2.0))
    return Scenario(
        scenario_id=f"graph-pair-{len(tasks)}",
        dataset_source="unit",
        tasks=tasks,
        dependencies=dependencies,
        compute_nodes=[
            ComputeNode("edge", "edge", 1.0),
            ComputeNode("cloud", "cloud", 2.0),
            ComputeNode("end", "end", 0.5),
        ],
        bandwidth_matrix=np.asarray(
            [[1e9, 10.0, 4.0], [10.0, 1e9, 6.0], [4.0, 6.0, 1e9]]
        ),
        metadata={"original_graph_id": f"graph-pair-{len(tasks)}"},
    )


def _model_and_observation() -> tuple[DAGPairGraphActorCritic, dict]:
    env = CloudEdgeEndDAGEnv(normalize_observations=True)
    env.reset(_scenario())
    observation = env.get_flat_observation()
    model = DAGPairGraphActorCritic(
        observation["task_features"].shape[-1],
        observation["resource_features"].shape[-1],
        observation["pair_node_features"].shape[-1],
        observation["task_edge_features"].shape[-1],
        observation["resource_edge_features"].shape[-1],
        hidden_dim=16,
        heads=4,
        task_layers=2,
        resource_layers=1,
    )
    return model, observation


def test_flat_observation_preserves_zero_data_dependency_and_graph_shapes() -> None:
    _, observation = _model_and_observation()
    assert observation["task_adjacency"].shape == (2, 2)
    assert observation["task_adjacency"][0, 1]
    assert observation["task_edge_features"][0, 1, 0] == 0.0
    assert observation["resource_adjacency"].shape == (3, 3)
    assert observation["resource_edge_features"].shape == (3, 3, 2)
    assert observation["pair_mask"].shape == (2, 3)
    assert observation["pair_mask"].any()


def test_graph_pair_model_shapes_are_finite() -> None:
    model, observation = _model_and_observation()
    device = next(model.parameters()).device
    static = model.encode_static(*graph_static_tensors(observation, device))
    logits, value = model.score_dynamic(static, *graph_dynamic_tensors(observation, device))
    assert logits.shape == (1, 6)
    assert value.shape == (1,)
    assert np.isfinite(logits.detach().numpy()).all()
    assert np.isfinite(value.detach().numpy()).all()


def test_graph_pair_policy_is_legal_for_variable_task_counts() -> None:
    model, _ = _model_and_observation()
    policy = DAGPairGraphPolicy(model, normalize_observations=True)
    records, summary = Evaluator({"normalize_observations": True}).evaluate(
        policy,
        [_scenario(), _scenario(extra_task=True)],
    )
    assert len(records) == 2
    assert all(record.valid_schedule for record in records)
    assert summary["valid_schedule_rate"] == 1.0
    assert summary["mean_ratio"] == 1.0
