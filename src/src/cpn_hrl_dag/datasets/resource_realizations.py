"""Leakage-safe conversion from split base DAGs to resource realizations."""

from __future__ import annotations

from typing import Any, Mapping

from cpn_hrl_dag.datasets.split import SplitManifest
from cpn_hrl_dag.scenario.resources import realize_scenario, resource_config_from_mapping
from cpn_hrl_dag.scenario.types import Scenario


def materialize_resources(
    splits: Mapping[str, list[Scenario]], resources: Mapping[str, Any] | None,
    manifest: SplitManifest, default_seed: int,
) -> dict[str, list[Scenario]]:
    """Apply native/generated/hybrid resources identically in train and eval.

    Validation/test use fixed seed lists.  Scenario identifiers change, but
    ``original_graph_id`` does not, so callers can still assert topology split
    isolation on the materialized result.
    """
    result = {name: list(items) for name, items in splits.items()}
    settings = dict(resources or {})
    mode = str(settings.get("mode", "dataset"))
    if mode == "dataset":
        return result
    if mode not in {"generated", "hybrid"}:
        raise ValueError("resources.mode must be dataset, generated, or hybrid")
    if "config" not in settings:
        raise ValueError("generated/hybrid resources require resources.config")
    profile = resource_config_from_mapping(settings["config"])
    by_split = {
        "train": tuple(int(seed) for seed in settings.get("train_seeds", [default_seed])),
        "validation": tuple(int(seed) for seed in settings.get("validation_seeds", manifest.resource_seeds.get("validation", [default_seed]))),
        "test": tuple(int(seed) for seed in settings.get("test_seeds", manifest.resource_seeds.get("test", settings.get("validation_seeds", [default_seed])))),
    }
    for split_name, seeds in by_split.items():
        if not seeds:
            raise ValueError(f"resources needs at least one fixed {split_name} seed")
        generated = [realize_scenario(item, profile, seed) for item in result[split_name] for seed in seeds]
        result[split_name] = generated if mode == "generated" else result[split_name] + generated
    return result
