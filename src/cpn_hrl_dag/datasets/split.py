"""Leakage-safe train/validation/test manifests grouped by base DAG identity."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from cpn_hrl_dag.scenario.types import Scenario


class SplitLeakageError(ValueError):
    """Raised when one base DAG appears in more than one data split."""


def scenario_base_key(scenario: Scenario) -> str:
    """Return source-qualified identity used for all topology-leakage checks."""

    base_id = scenario.metadata.get("original_graph_id")
    if not isinstance(base_id, str) or not base_id:
        raise ValueError(f"scenario {scenario.scenario_id!r} has no metadata.original_graph_id")
    return f"{scenario.dataset_source}:{base_id}"


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """A frozen split contract over base DAG IDs, never scenario realizations."""

    train_base_dag_ids: tuple[str, ...]
    validation_base_dag_ids: tuple[str, ...]
    test_base_dag_ids: tuple[str, ...]
    seed: int
    protocol: str = "mixed_iid"
    resource_seeds: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for label, values in (
            ("train", self.train_base_dag_ids),
            ("validation", self.validation_base_dag_ids),
            ("test", self.test_base_dag_ids),
        ):
            if len(values) != len(set(values)):
                raise SplitLeakageError(f"{label} base DAG IDs contain duplicates")
        train, validation, test = map(set, (self.train_base_dag_ids, self.validation_base_dag_ids, self.test_base_dag_ids))
        overlaps = {
            "train_validation": train & validation,
            "train_test": train & test,
            "validation_test": validation & test,
        }
        found = {name: sorted(values) for name, values in overlaps.items() if values}
        if found:
            raise SplitLeakageError(f"base DAG leakage across splits: {found}")

    def split_for(self, base_key: str) -> str:
        if base_key in self.train_base_dag_ids:
            return "train"
        if base_key in self.validation_base_dag_ids:
            return "validation"
        if base_key in self.test_base_dag_ids:
            return "test"
        raise KeyError(f"base DAG is absent from split manifest: {base_key}")

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_json_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: str | Path) -> "SplitManifest":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        required = {"train_base_dag_ids", "validation_base_dag_ids", "test_base_dag_ids", "seed"}
        missing = required - set(raw)
        if missing:
            raise ValueError(f"split manifest missing keys: {sorted(missing)}")
        return cls(
            train_base_dag_ids=tuple(raw["train_base_dag_ids"]),
            validation_base_dag_ids=tuple(raw["validation_base_dag_ids"]),
            test_base_dag_ids=tuple(raw["test_base_dag_ids"]),
            seed=int(raw["seed"]),
            protocol=str(raw.get("protocol", "mixed_iid")),
            resource_seeds={key: tuple(value) for key, value in raw.get("resource_seeds", {}).items()},
            metadata=dict(raw.get("metadata", {})),
        )


class SplitManager:
    """Creates and applies manifests with topology grouping as a hard invariant."""

    @staticmethod
    def create(
        base_dag_ids: Iterable[str],
        *,
        seed: int,
        train_fraction: float = 0.8,
        validation_fraction: float = 0.1,
        protocol: str = "mixed_iid",
        resource_seeds: Mapping[str, Sequence[int]] | None = None,
    ) -> SplitManifest:
        unique_ids = sorted(set(base_dag_ids))
        if len(unique_ids) < 3:
            raise ValueError("at least three base DAG IDs are required for train/validation/test splitting")
        if not 0.0 < train_fraction < 1.0 or not 0.0 < validation_fraction < 1.0 or train_fraction + validation_fraction >= 1.0:
            raise ValueError("split fractions must be positive and leave a non-empty test fraction")
        rng = np.random.default_rng(seed)
        shuffled = list(np.asarray(unique_ids, dtype=object)[rng.permutation(len(unique_ids))])
        train_count = int(np.floor(len(shuffled) * train_fraction))
        validation_count = int(np.floor(len(shuffled) * validation_fraction))
        train_count = min(max(1, train_count), len(shuffled) - 2)
        validation_count = max(1, validation_count)
        if train_count + validation_count >= len(shuffled):
            validation_count = len(shuffled) - train_count - 1
        manifest = SplitManifest(
            train_base_dag_ids=tuple(sorted(shuffled[:train_count])),
            validation_base_dag_ids=tuple(sorted(shuffled[train_count : train_count + validation_count])),
            test_base_dag_ids=tuple(sorted(shuffled[train_count + validation_count :])),
            seed=seed,
            protocol=protocol,
            resource_seeds={key: tuple(int(seed) for seed in value) for key, value in (resource_seeds or {}).items()},
            metadata={"num_base_dags": len(unique_ids), "split_unit": "dataset_source:original_graph_id"},
        )
        return manifest

    @staticmethod
    def apply(scenarios: Iterable[Scenario], manifest: SplitManifest) -> dict[str, list[Scenario]]:
        grouped: dict[str, list[Scenario]] = {"train": [], "validation": [], "test": []}
        seen_scenarios: set[str] = set()
        for scenario in scenarios:
            if scenario.scenario_id in seen_scenarios:
                raise ValueError(f"duplicate scenario ID in split input: {scenario.scenario_id}")
            seen_scenarios.add(scenario.scenario_id)
            grouped[manifest.split_for(scenario_base_key(scenario))].append(scenario)
        for values in grouped.values():
            values.sort(key=lambda item: item.scenario_id)
        SplitManager.assert_no_leakage(grouped)
        return grouped

    @staticmethod
    def assert_no_leakage(grouped: Mapping[str, Iterable[Scenario]]) -> None:
        observed: dict[str, set[str]] = {}
        for split_name in ("train", "validation", "test"):
            scenarios = grouped.get(split_name, [])
            observed[split_name] = {scenario_base_key(scenario) for scenario in scenarios}
        overlaps = {
            "train_validation": observed["train"] & observed["validation"],
            "train_test": observed["train"] & observed["test"],
            "validation_test": observed["validation"] & observed["test"],
        }
        found = {name: sorted(values) for name, values in overlaps.items() if values}
        if found:
            raise SplitLeakageError(f"scenario grouping violates base-DAG isolation: {found}")
