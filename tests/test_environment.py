from __future__ import annotations

import numpy as np
import pytest

from cpn_hrl_dag.env.dag_env import CloudEdgeEndDAGEnv
from cpn_hrl_dag.scenario.types import ComputeNode, Dependency, Scenario, Task
from cpn_hrl_dag.scheduling.simulator import InvalidSchedulingAction


def scenario() -> Scenario:
    return Scenario(
        scenario_id="env", dataset_source="unit",
        tasks=[Task("a", 4.0), Task("b", 2.0), Task("c", 2.0, device_requirement="gpu")],
        dependencies=[Dependency("a", "b", 2.0)],
        compute_nodes=[ComputeNode("cpu", "edge", 1.0, gpu_capacity=0.0), ComputeNode("gpu", "cloud", 2.0, gpu_capacity=1.0, metadata={"features": ["gpu"]})],
        bandwidth_matrix=np.array([[1e9, 2.0], [2.0, 1e9]]),
    )


def test_environment_ready_and_node_masks_and_reward() -> None:
    env = CloudEdgeEndDAGEnv()
    observation, info = env.reset(scenario())
    assert observation["ready_mask"].tolist() == [True, False, True]
    assert info["valid_schedule"] is False
    assert env.get_node_mask(2).tolist() == [False, True]
    env.select_task(0)
    _, reward, terminated, truncated, _ = env.step(0, 1)
    assert reward < 0.0 and not terminated and not truncated
    assert env.get_ready_mask().tolist() == [False, True, True]
    low = env.get_low_observation(1)
    assert low["node_features"].shape == (2, 8)
    assert low["node_features"][0, 5] > 0.0


def test_environment_rejects_illegal_hierarchical_action() -> None:
    env = CloudEdgeEndDAGEnv()
    env.reset(scenario())
    env.select_task(0)
    with pytest.raises(InvalidSchedulingAction, match="differs"):
        env.step(2, 1)
    with pytest.raises(InvalidSchedulingAction, match="not ready"):
        env.select_task(1)


def test_no_heft_feature_ablation_removes_all_rank_inputs() -> None:
    observation, _ = CloudEdgeEndDAGEnv(include_heft_features=False).reset(scenario())
    assert np.allclose(observation["task_features"][:, 14], 0.0)
    assert np.allclose(observation["task_features"][:, 15], 0.0)


def test_normalized_observations_are_finite_and_time_scaled() -> None:
    raw_env = CloudEdgeEndDAGEnv()
    raw_high, _ = raw_env.reset(scenario())
    normalized_env = CloudEdgeEndDAGEnv(normalize_observations=True)
    normalized_high, _ = normalized_env.reset(scenario())
    assert np.isfinite(normalized_high["task_features"]).all()
    assert np.isfinite(normalized_high["resource_features"]).all()
    np.testing.assert_allclose(
        normalized_high["task_features"][:, 9:12],
        raw_high["task_features"][:, 9:12] / normalized_env.heft_makespan,
    )
    task = int(np.flatnonzero(normalized_env.get_ready_mask())[0])
    normalized_env.select_task(task)
    low = normalized_env.get_low_observation(task)
    assert np.isfinite(low["node_features"]).all()
    assert np.isfinite(low["edge_features"]).all()


def test_makespan_delta_reward_telescopes_exactly_without_shaping() -> None:
    env = CloudEdgeEndDAGEnv()
    observation, _ = env.reset(scenario())
    total_reward = 0.0
    while not env.simulator.done:
        task = int(np.flatnonzero(env.get_ready_mask())[0])
        env.select_task(task)
        low = env.get_low_observation(task)
        node = int(min(np.flatnonzero(env.get_node_mask(task)), key=lambda index: (float(low["node_features"][index, 5]), int(index))))
        observation, reward, _, _, _ = env.step(task, node)
        total_reward += reward
    assert total_reward == pytest.approx(-env.simulator.result().makespan / env.heft_makespan)
