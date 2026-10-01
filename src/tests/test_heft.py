from __future__ import annotations

import numpy as np
import pytest

from cpn_hrl_dag.baselines.heft import HEFTScheduler
from cpn_hrl_dag.scenario.types import ComputeNode, Dependency, Scenario, Task


def test_heft_upward_rank_order_eft_and_makespan() -> None:
    scenario = Scenario(
        scenario_id="heft-manual", dataset_source="unit",
        tasks=[Task("a", 10.0), Task("b", 10.0), Task("c", 2.0)],
        dependencies=[Dependency("a", "b", 10.0), Dependency("a", "c", 0.0)],
        compute_nodes=[ComputeNode("slow", "edge", 1.0), ComputeNode("fast", "cloud", 2.0)],
        bandwidth_matrix=np.array([[1e9, 10.0], [10.0, 1e9]]),
    )
    result, analysis = HEFTScheduler().schedule(scenario)
    np.testing.assert_allclose(analysis.average_execution_cost, [7.5, 7.5, 1.5])
    assert analysis.average_communication_cost[(0, 1)] == pytest.approx(1.0)
    np.testing.assert_allclose(analysis.upward_rank, [16.0, 7.5, 1.5])
    assert analysis.task_order == (0, 1, 2)
    by_task = {entry.task_position: entry for entry in result.entries}
    assert by_task[0].node_position == 1
    assert by_task[1].node_position == 1
    assert by_task[1].start == 5.0
    assert by_task[2].node_position == 0  # c uses the idle slow node instead of waiting for fast
    assert result.makespan == 10.0


def test_heft_rejects_permanently_infeasible_task() -> None:
    scenario = Scenario(
        scenario_id="infeasible", dataset_source="unit",
        tasks=[Task("gpu", 1.0, device_requirement="gpu")], dependencies=[],
        compute_nodes=[ComputeNode("cpu", "edge", 1.0, gpu_capacity=0.0)], bandwidth_matrix=np.ones((1, 1)),
    )
    with pytest.raises(ValueError, match="no permanently feasible node"):
        HEFTScheduler().schedule(scenario)
