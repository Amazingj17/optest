from __future__ import annotations

import numpy as np
import pytest

from cpn_hrl_dag.datasets.split import SplitLeakageError, SplitManager, SplitManifest, scenario_base_key
from cpn_hrl_dag.scenario.types import ComputeNode, Scenario, Task


def scenario(base_id: str, realization: int) -> Scenario:
    return Scenario(
        scenario_id=f"{base_id}-r{realization}",
        dataset_source="unit",
        tasks=[Task("task", 1.0)],
        dependencies=[],
        compute_nodes=[ComputeNode("node", "edge", 1.0)],
        bandwidth_matrix=np.ones((1, 1)),
        metadata={"original_graph_id": base_id, "resource_generation_seed": realization},
    )


def test_same_base_dag_resource_realizations_stay_together(tmp_path) -> None:
    scenarios = [scenario("dag-a", 1), scenario("dag-a", 2), scenario("dag-b", 1), scenario("dag-c", 1), scenario("dag-d", 1)]
    manifest = SplitManager.create([scenario_base_key(item) for item in scenarios], seed=42, train_fraction=0.5, validation_fraction=0.25)
    grouped = SplitManager.apply(scenarios, manifest)
    split_names = {manifest.split_for(scenario_base_key(item)) for item in scenarios if item.metadata["original_graph_id"] == "dag-a"}
    assert split_names == {next(name for name, values in grouped.items() if any(item.metadata["original_graph_id"] == "dag-a" for item in values))}
    path = tmp_path / "split_manifest.json"
    manifest.write(path)
    assert SplitManifest.read(path) == manifest


def test_manifest_rejects_base_dag_leakage() -> None:
    with pytest.raises(SplitLeakageError, match="leakage"):
        SplitManifest(("unit:dag-a",), ("unit:dag-a",), ("unit:dag-b",), seed=1)


def test_apply_detects_manifest_scenario_mismatch() -> None:
    manifest = SplitManifest(("unit:dag-a",), ("unit:dag-b",), ("unit:dag-c",), seed=1)
    with pytest.raises(KeyError, match="absent"):
        SplitManager.apply([scenario("dag-other", 1)], manifest)
