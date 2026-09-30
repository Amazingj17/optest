from __future__ import annotations

from pathlib import Path

import pytest

from cpn_hrl_dag.datasets.grapheonrl_adapter import GrapheonRLAdapter


DATA_ROOT = Path("data/raw/zenodo-18927122-derived")


def test_grapheonrl_discovery_and_heterogeneous_mapping() -> None:
    adapter = GrapheonRLAdapter(system_configs_root=DATA_ROOT)
    paths = tuple(adapter.discover(DATA_ROOT))
    source = next(path for path in paths if getattr(path, "member_name", "") == "rnc50_hetero/rand0000_hetero.json")
    assert len(paths) >= 1080
    scenario = adapter.load(source)
    assert scenario.scenario_id == "grapheonrl:rnc50:rand0000:heterogeneous"
    assert scenario.num_tasks == 50
    assert scenario.num_nodes == 8
    assert scenario.tasks[0].id == "T1"
    assert scenario.tasks[0].device_requirement == "gpu"
    assert scenario.tasks[0].memory_requirement == 28672.0
    assert scenario.compute_nodes[scenario.node_position("iot_1")].node_type == "end"
    assert scenario.compute_nodes[scenario.node_position("edge_1")].node_type == "edge"
    assert scenario.compute_nodes[scenario.node_position("cloud_1")].node_type == "cloud"
    assert scenario.compute_nodes[scenario.node_position("grete_p3_gpu")].gpu_capacity == 1.0
    t1, t5 = scenario.task_position("T1"), scenario.task_position("T5")
    assert scenario.cache.edge_data_matrix[t1, t5] == scenario.tasks[t1].metadata["raw_task"]["data"]
    assert scenario.bandwidth_matrix[scenario.node_position("iot_1"), scenario.node_position("edge_1")] == 100.0


def test_grapheonrl_homogeneous_300_task_variant() -> None:
    adapter = GrapheonRLAdapter(system_configs_root=DATA_ROOT)
    source = next(path for path in adapter.discover(DATA_ROOT) if getattr(path, "member_name", "") == "rnc300_homo/rand0000_homo.json")
    scenario = adapter.load(source)
    assert scenario.num_tasks == 300
    assert scenario.num_nodes == 3
    assert {task.device_requirement for task in scenario.tasks} == {"cpu"}
    assert {node.node_type for node in scenario.compute_nodes} == {"cloud"}
    assert scenario.metadata["original_graph_id"] == "rnc300:rand0000"
