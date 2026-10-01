"""Named experiment protocols with explicit topology-isolation checks."""
from __future__ import annotations

from typing import Iterable

from .split import SplitManager, scenario_base_key
from cpn_hrl_dag.scenario.types import Scenario
from cpn_hrl_dag.scenario.resources import ResourceConfig, realize_scenario


def scenario_domain(scenario: Scenario) -> str:
    """Return audited source-domain metadata when it is available."""
    raw = scenario.metadata.get("original_metadata", {})
    if isinstance(raw, dict) and isinstance(raw.get("domain"), str):
        return raw["domain"]
    return scenario.dataset_source


def scale_generalization_split(scenarios: Iterable[Scenario], max_train_tasks: int, min_test_tasks: int) -> dict[str, list[Scenario]]:
    """Separate topology identities by task scale; validation is medium/held-out."""
    values = list(scenarios)
    train = [item for item in values if item.num_tasks <= max_train_tasks]
    test = [item for item in values if item.num_tasks >= min_test_tasks]
    train_keys, test_keys = {scenario_base_key(x) for x in train}, {scenario_base_key(x) for x in test}
    if train_keys & test_keys:
        raise ValueError("scale thresholds overlap base DAG identities")
    middle = [item for item in values if scenario_base_key(item) not in train_keys | test_keys]
    grouped = {"train": train, "validation": middle, "test": test}
    SplitManager.assert_no_leakage(grouped)
    return grouped


def resource_ood_split(
    scenarios: Iterable[Scenario], manifest: "SplitManifest", train_config: ResourceConfig,
    evaluation_config: ResourceConfig, train_seeds: Iterable[int], evaluation_seeds: Iterable[int],
) -> dict[str, list[Scenario]]:
    """Realize disjoint DAG splits under explicitly different resource profiles.

    The function intentionally separates topology first (via ``manifest``),
    then materializes seeds.  It consequently tests resource OOD without
    allowing a graph to move from training into validation or test.
    """
    grouped = SplitManager.apply(scenarios, manifest)
    result: dict[str, list[Scenario]] = {}
    for name, values in grouped.items():
        config = train_config if name == "train" else evaluation_config
        seeds = tuple(train_seeds if name == "train" else evaluation_seeds)
        if not seeds:
            raise ValueError("resource OOD seed lists must be non-empty")
        result[name] = [realize_scenario(item, config, int(seed)) for item in values for seed in seeds]
    SplitManager.assert_no_leakage(result)
    return result
