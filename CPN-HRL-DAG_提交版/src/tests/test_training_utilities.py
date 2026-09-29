from __future__ import annotations

import numpy as np
import torch

from cpn_hrl_dag.datasets.sampling import CurriculumStage, DatasetBalancedSampler
from cpn_hrl_dag.datasets.protocols import resource_ood_split, scale_generalization_split
from cpn_hrl_dag.datasets.split import SplitManager
from cpn_hrl_dag.scenario.resources import resource_config_from_mapping
from cpn_hrl_dag.datasets.resource_realizations import materialize_resources
from cpn_hrl_dag.evaluation.heft_cache import HEFTCache
from cpn_hrl_dag.scenario.types import ComputeNode, Scenario, Task
from cpn_hrl_dag.scheduling.communication_model import MatrixCommunicationModel
from cpn_hrl_dag.scheduling.execution_model import SimpleSpeedExecutionModel
from cpn_hrl_dag.algorithms.bc import BehaviorCloner
from cpn_hrl_dag.env.dag_env import CloudEdgeEndDAGEnv
from cpn_hrl_dag.models.high_lstm import HighLevelLSTMActorCritic
from cpn_hrl_dag.models.low_gat import LowLevelGATActorCritic
from cpn_hrl_dag.algorithms.hierarchical_trainer import HierarchicalTrainer
from cpn_hrl_dag.utils.config import config_hash


def _scenario(identifier: str, source: str, tasks: int = 2) -> Scenario:
    return Scenario(
        scenario_id=identifier,
        dataset_source=source,
        tasks=[Task(f"t{index}", 1.0) for index in range(tasks)],
        dependencies=[],
        compute_nodes=[ComputeNode("n", "edge", 1.0)],
        bandwidth_matrix=np.ones((1, 1)),
        metadata={"original_graph_id": identifier},
    )


def test_heft_cache_is_stable_and_semantics_aware(tmp_path) -> None:
    scenario = _scenario("cache", "unit")
    cache = HEFTCache(tmp_path)
    execution, communication = SimpleSpeedExecutionModel(), MatrixCommunicationModel()
    first = cache.get_or_compute(scenario, execution, communication)
    second = cache.get_or_compute(scenario, execution, communication)
    assert first == second == 2.0
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_dataset_balanced_sampler_respects_curriculum() -> None:
    small = _scenario("small", "a", 2)
    large = _scenario("large", "b", 5)
    sampler = DatasetBalancedSampler([small, large], {"a": 1.0, "b": 1.0}, seed=3)
    assert sampler.sample(CurriculumStage(2, 2, 1)).scenario_id == "small"
    assert sampler.sample(CurriculumStage(5, 5, 1)).scenario_id == "large"


def test_scale_protocol_keeps_base_dags_isolated() -> None:
    scenarios = [_scenario("a", "grapheonrl", 2), _scenario("b", "grapheonrl", 3), _scenario("c", "grapheonrl", 6)]
    scale = scale_generalization_split(scenarios, max_train_tasks=2, min_test_tasks=6)
    assert [item.scenario_id for item in scale["train"]] == ["a"]
    assert [item.scenario_id for item in scale["test"]] == ["c"]


def test_behavior_cloner_runs_on_a_legal_heft_demo() -> None:
    scenario = _scenario("bc", "unit")
    env = CloudEdgeEndDAGEnv(); high, _ = env.reset(scenario); env.select_task(0); low = env.get_low_observation(0)
    high_model = HighLevelLSTMActorCritic(high['task_features'].shape[1], high['resource_features'].shape[1], 8)
    low_model = LowLevelGATActorCritic(low['node_features'].shape[1], high['task_features'].shape[1], 8, 2)
    result = BehaviorCloner(high_model, low_model, torch.optim.Adam(high_model.parameters()), torch.optim.Adam(low_model.parameters()), torch.device('cpu')).fit([scenario])
    assert result.decisions == scenario.num_tasks


def test_resource_ood_realizes_disjoint_topologies_and_preserves_provenance() -> None:
    scenarios = [_scenario("a", "unit"), _scenario("b", "unit"), _scenario("c", "unit")]
    manifest = SplitManager.create([f"unit:{item.scenario_id}" for item in scenarios], seed=2)
    raw = {"config_id": "tiny", "cloud": {"count": [1, 1], "speed": [2, 2]}, "bandwidth": {"cloud_cloud": [5, 5]}, "latency": {"cloud_cloud": [0, 0]}}
    grouped = resource_ood_split(scenarios, manifest, resource_config_from_mapping(raw), resource_config_from_mapping(raw), [3], [4])
    assert all(":resource:tiny:" in item.scenario_id for values in grouped.values() for item in values)
    assert {item.metadata["original_graph_id"] for item in grouped["train"]}.isdisjoint({item.metadata["original_graph_id"] for item in grouped["validation"]})


def test_generated_resources_are_identical_for_train_and_evaluation_builders() -> None:
    scenarios = [_scenario("a", "unit"), _scenario("b", "unit"), _scenario("c", "unit")]
    manifest = SplitManager.create([f"unit:{item.scenario_id}" for item in scenarios], seed=2)
    native = SplitManager.apply(scenarios, manifest)
    settings = {"mode": "generated", "train_seeds": [1], "validation_seeds": [2], "test_seeds": [3], "config": {"cloud": {"count": [1, 1], "speed": [2, 2]}, "bandwidth": {"cloud_cloud": [5, 5]}}}
    first = materialize_resources(native, settings, manifest, 99)
    second = materialize_resources(native, settings, manifest, 99)
    assert [item.scenario_id for item in first["validation"]] == [item.scenario_id for item in second["validation"]]
    assert first["validation"][0].metadata["resource_generation_seed"] == 2
    assert "gpu" not in first["validation"][0].compute_nodes[0].metadata["features"]


def test_generated_gpu_capacity_advertises_device_affinity() -> None:
    config = resource_config_from_mapping({"cloud": {"count": [1, 1], "speed": [2, 2], "gpu_capacity": [1, 1]}, "bandwidth": {"cloud_cloud": [5, 5]}})
    from cpn_hrl_dag.scenario.resources import CloudEdgeEndResourceGenerator
    assert "gpu" in CloudEdgeEndResourceGenerator().generate(config, 1).nodes[0].metadata["features"]


def test_lstm_and_gat_ablations_produce_valid_variable_size_outputs() -> None:
    high = HighLevelLSTMActorCritic(3, 2, 8, use_lstm=False)
    low = LowLevelGATActorCritic(8, 3, 8, 2, use_gat=False)
    logits, value = high(torch.ones(1, 5, 3), torch.ones(1, 5, dtype=torch.bool), torch.ones(1, 2, 2))
    node_logits, node_value = low(torch.ones(1, 3, 8), torch.ones(1, 3), torch.ones(1, 3, 3, dtype=torch.bool), torch.ones(1, 3, 3, 2))
    assert logits.shape == (1, 5) and value.shape == (1,)
    assert node_logits.shape == (1, 3) and node_value.shape == (1,)


def test_zero_initialized_residual_actors_exactly_follow_rank_and_eft() -> None:
    high = HighLevelLSTMActorCritic(17, 5, 8, heuristic_residual=True, heuristic_weight=4.0)
    task_features = torch.zeros(1, 3, 17)
    task_features[0, :, 14] = torch.tensor([0.2, 0.9, 0.5])
    logits, _ = high(task_features, torch.ones(1, 3, dtype=torch.bool), torch.ones(1, 2, 5))
    torch.testing.assert_close(logits, 4.0 * task_features[..., 14])

    low = LowLevelGATActorCritic(8, 17, 8, 2, heuristic_residual=True, heuristic_weight=4.0)
    node_features = torch.ones(1, 3, 8)
    node_features[0, :, 5] = torch.tensor([0.7, 0.2, 0.4])
    node_logits, _ = low(node_features, task_features[:, 0], torch.ones(1, 3, 3, dtype=torch.bool), torch.ones(1, 3, 3, 2))
    torch.testing.assert_close(node_logits, -4.0 * node_features[..., 5])


def test_config_hash_is_shared_and_ignores_runtime_facts() -> None:
    base = {"experiment": {"seed": 7}, "training": {"episodes": 2}}
    with_runtime = {**base, "runtime": {"python": "different-host-version"}}
    assert config_hash(base) == config_hash(with_runtime)
    assert HierarchicalTrainer.config_hash(with_runtime) == config_hash(base)
