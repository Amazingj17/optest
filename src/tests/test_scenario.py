from __future__ import annotations

import numpy as np
import pytest

from cpn_hrl_dag.scenario.resources import (
    ClosedRange,
    CloudEdgeEndResourceGenerator,
    ResourceConfig,
    ResourceTierConfig,
)
from cpn_hrl_dag.scenario.types import ComputeNode, Dependency, Scenario, ScenarioValidationError, Task
from cpn_hrl_dag.scheduling.communication_model import MatrixCommunicationModel
from cpn_hrl_dag.scheduling.execution_model import HeterogeneousDeviceExecutionModel, SimpleSpeedExecutionModel


def build_scenario() -> Scenario:
    return Scenario(
        scenario_id="unit-chain",
        dataset_source="unit",
        tasks=[Task("a", 8.0), Task("b", 4.0), Task("c", 12.0)],
        dependencies=[Dependency("a", "b", 10.0), Dependency("a", "c", 4.0)],
        compute_nodes=[ComputeNode("end-0", "end", 2.0), ComputeNode("cloud-0", "cloud", 8.0)],
        bandwidth_matrix=np.array([[1e9, 10.0], [10.0, 1e9]]),
        latency_matrix=np.array([[0.0, 0.1], [0.1, 0.0]]),
        metadata={"original_graph_id": "unit-chain"},
    )


def test_scenario_preprocessing_is_deterministic() -> None:
    scenario = build_scenario()
    assert scenario.cache.topological_order == (0, 1, 2)
    assert scenario.cache.predecessors == ((), (0,), (0,))
    assert scenario.cache.successors == ((1, 2), (), ())
    np.testing.assert_array_equal(scenario.cache.in_degree, [0, 1, 1])
    np.testing.assert_array_equal(scenario.cache.topological_level, [0, 1, 1])
    assert scenario.cache.edge_data_matrix[0, 1] == 10.0
    assert not scenario.cache.edge_data_matrix.flags.writeable


def test_scenario_rejects_cycle() -> None:
    with pytest.raises(ScenarioValidationError, match="cycle"):
        Scenario(
            scenario_id="cycle",
            dataset_source="unit",
            tasks=[Task("a", 1.0), Task("b", 1.0)],
            dependencies=[Dependency("a", "b", 1.0), Dependency("b", "a", 1.0)],
            compute_nodes=[ComputeNode("n", "edge", 1.0)],
            bandwidth_matrix=np.ones((1, 1)),
        )


def test_execution_and_communication_models() -> None:
    scenario = build_scenario()
    assert SimpleSpeedExecutionModel().duration(scenario.tasks[0], scenario.compute_nodes[1]) == 1.0
    model = MatrixCommunicationModel()
    assert model.duration(scenario, 0, 1, 0, 0) == 0.0
    assert model.duration(scenario, 0, 1, 0, 1) == pytest.approx(1.1)
    gpu_task = Task("gpu", 20.0, device_requirement="gpu")
    gpu_node = ComputeNode("g", "cloud", 1.0, metadata={"device_speeds": {"CPU": 1.0, "GPU": 10.0}})
    assert HeterogeneousDeviceExecutionModel().duration(gpu_task, gpu_node) == 2.0


def test_resource_generation_is_seed_reproducible() -> None:
    resource_config = ResourceConfig(
        config_id="unit-resource-config",
        tiers={
            "cloud": ResourceTierConfig(ClosedRange(1, 1), ClosedRange(8.0, 8.0)),
            "edge": ResourceTierConfig(ClosedRange(2, 2), ClosedRange(2.0, 4.0)),
        },
        bandwidth={"cloud_cloud": ClosedRange(100.0, 100.0), "cloud_edge": ClosedRange(10.0, 20.0), "edge_edge": ClosedRange(5.0, 5.0)},
        latency={"cloud_cloud": ClosedRange(0.0, 0.0), "cloud_edge": ClosedRange(0.01, 0.02), "edge_edge": ClosedRange(0.0, 0.0)},
    )
    generator = CloudEdgeEndResourceGenerator()
    first, second = generator.generate(resource_config, 7), generator.generate(resource_config, 7)
    assert first.nodes == second.nodes
    np.testing.assert_array_equal(first.bandwidth_matrix, second.bandwidth_matrix)
    np.testing.assert_array_equal(first.latency_matrix, second.latency_matrix)
