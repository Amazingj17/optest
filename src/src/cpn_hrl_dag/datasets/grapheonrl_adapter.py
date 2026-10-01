"""Adapter for GrapheonRL's official STG-to-JSON Zenodo release."""

from __future__ import annotations

import json
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from cpn_hrl_dag.scenario.resources import CloudEdgeEndResourceGenerator, NodeTypeMapper, ResourceConfig
from cpn_hrl_dag.scenario.types import ComputeNode, Dependency, Scenario, ScenarioValidationError, Task

from .base import DatasetAdapter, DatasetSchemaError


@dataclass(frozen=True, slots=True)
class ArchiveWorkflow:
    """One JSON workflow member inside the immutable Zenodo release archive."""

    archive_path: Path
    member_name: str

    @property
    def display_path(self) -> str:
        return f"{self.archive_path.resolve()}!{self.member_name}"


class GrapheonRLAdapter(DatasetAdapter):
    """Convert official STG JSON and companion node configurations to Scenario.

    The upstream task JSON has no pairwise network. Its documented HEFT
    compatibility rule uses one node rate (maximum interface rate) and builds
    a pairwise link as the minimum rate at its two endpoints.
    """

    name = "grapheonrl"
    adapter_version = "grapheonrl-stg-json-v2-archive"
    archive_variants_per_family = 180

    def __init__(
        self,
        system_configs_root: str | Path | None = None,
        node_type_mapper: NodeTypeMapper | None = None,
        *,
        local_bandwidth: float = 1e9,
        archive_task_counts: Iterable[int] | None = None,
    ) -> None:
        self.system_configs_root = None if system_configs_root is None else Path(system_configs_root)
        self.node_type_mapper = node_type_mapper or NodeTypeMapper(
            exact={"IoT": "end", "Edge": "edge", "Cloud": "cloud", "HPC": "cloud"}, prefixes={}
        )
        self.local_bandwidth = float(local_bandwidth)
        self.archive_task_counts = None if archive_task_counts is None else frozenset(int(value) for value in archive_task_counts)
        if self.archive_task_counts is not None and (not self.archive_task_counts or min(self.archive_task_counts) <= 0):
            raise ValueError("archive_task_counts must contain positive task counts")
        if not np.isfinite(self.local_bandwidth) or self.local_bandwidth <= 0.0:
            raise ValueError("local_bandwidth must be finite and positive")

    def discover(self, root: str | Path) -> Iterable[Path | ArchiveWorkflow]:
        root_path = Path(root)
        archives = sorted(root_path.glob("*_json.tar.xz"), key=lambda item: item.name)
        members: list[ArchiveWorkflow] = []
        for archive in archives:
            standard = re.fullmatch(r"rnc(\d+)_(hetero|homo)_json\.tar\.xz", archive.name)
            if standard:
                task_count, variant = standard.groups()
                if self.archive_task_counts is not None and int(task_count) not in self.archive_task_counts:
                    continue
                folder = f"rnc{task_count}_{variant}"
                members.extend(
                    ArchiveWorkflow(archive, f"{folder}/rand{index:04d}_{variant}.json")
                    for index in range(self.archive_variants_per_family)
                )
                continue
            if self.archive_task_counts is not None:
                # The scale-filtered training view intentionally contains only
                # declared rnc<size> families; FPPPP remains available when
                # the adapter is instantiated without a scale filter.
                continue
            try:
                with tarfile.open(archive, mode="r:xz") as package:
                    members.extend(
                        ArchiveWorkflow(archive, member.name)
                        for member in package.getmembers()
                        if member.isfile() and member.name.endswith(".json")
                    )
            except (tarfile.TarError, OSError) as exc:
                raise DatasetSchemaError(f"cannot inspect GrapheonRL archive {archive}: {exc}") from exc
        if not members:
            workflows = root_path / "workflows" if (root_path / "workflows").is_dir() else root_path
            paths = sorted(
                (item for item in workflows.rglob("*.json") if item.name != "schema(simplified).json" and "system_configs" not in item.parts),
                key=lambda item: item.as_posix(),
            )
            if paths:
                return tuple(paths)
            raise FileNotFoundError(f"no GrapheonRL workflow JSON files or archives under {root_path}")
        def archive_order(item: ArchiveWorkflow) -> tuple[int, str, str]:
            matched = re.fullmatch(r"rnc(\d+)_(hetero|homo)_json\.tar\.xz", item.archive_path.name)
            return (int(matched.group(1)) if matched else -1, item.archive_path.name, item.member_name)
        return tuple(sorted(members, key=archive_order))

    def load(self, path: str | Path | ArchiveWorkflow, resource_config: ResourceConfig | str | Path | None = None) -> Scenario:
        archive_member = path if isinstance(path, ArchiveWorkflow) else None
        workflow_path = archive_member.archive_path if archive_member else Path(path)
        if archive_member is None and not workflow_path.is_file():
            raise FileNotFoundError(f"GrapheonRL workflow does not exist: {workflow_path}")
        source_label = archive_member.display_path if archive_member else str(workflow_path.resolve())
        raw = self._read_archive_json(archive_member, "workflow") if archive_member else self._read_json(workflow_path, "workflow")
        meta = self._require_mapping(raw.get("meta"), workflow_path, "meta")
        raw_tasks = self._require_mapping(raw.get("tasks"), workflow_path, "tasks")
        if not raw_tasks:
            raise DatasetSchemaError(f"{workflow_path}: tasks cannot be empty")
        tasks = tuple(self._parse_task(task_id, item, workflow_path) for task_id, item in raw_tasks.items())
        dependencies = self._parse_dependencies(raw_tasks, workflow_path)
        base_graph_id = self._base_graph_id(meta, workflow_path)
        generation_seed: int | None = None
        resource_config_id: str
        if isinstance(resource_config, ResourceConfig):
            generation_seed = self._resource_seed(base_graph_id, resource_config)
            generated = CloudEdgeEndResourceGenerator().generate(resource_config, generation_seed)
            nodes, bandwidth, latency = generated.nodes, generated.bandwidth_matrix, generated.latency_matrix
            resource_config_id = generated.config_id
        else:
            if resource_config is not None:
                config_path = Path(resource_config)
                nodes, bandwidth = self._load_nodes(config_path)
                resource_config_id = f"dataset:{config_path.name}"
            elif archive_member is not None:
                config_raw, config_label = self._load_archive_system_config(archive_member.archive_path.parent, meta, source_label)
                nodes, bandwidth = self._load_nodes_from_raw(config_raw, config_label)
                resource_config_id = f"dataset:{config_label}"
            else:
                config_path = self._infer_system_config(meta, workflow_path)
                nodes, bandwidth = self._load_nodes(config_path)
                resource_config_id = f"dataset:{config_path.name}"
            latency = None
        scenario = Scenario(
            scenario_id=f"grapheonrl:{base_graph_id}:{str(meta.get('variant', 'unknown'))}",
            dataset_source=self.name,
            tasks=tasks,
            dependencies=dependencies,
            compute_nodes=nodes,
            bandwidth_matrix=bandwidth,
            latency_matrix=latency,
            metadata={
                "dataset_source": self.name,
                "original_path": source_label,
                "original_graph_id": base_graph_id,
                "adapter_version": self.adapter_version,
                "resource_generation_seed": generation_seed,
                "resource_config_id": resource_config_id,
                "original_metadata": dict(meta),
                "variant": meta.get("variant"),
                "units": {
                    "workload": "source_duration_unit",
                    "memory": meta.get("unit_memory_required"),
                    "edge_data": meta.get("unit_data"),
                    "bandwidth": "MBps",
                },
            },
        )
        self.validate(scenario)
        return scenario

    def validate(self, scenario: Scenario) -> None:
        if scenario.dataset_source != self.name:
            raise DatasetSchemaError(f"expected grapheonrl scenario, got {scenario.dataset_source!r}")
        required = {"original_path", "original_graph_id", "adapter_version", "original_metadata", "variant"}
        missing = required - set(scenario.metadata)
        if missing:
            raise DatasetSchemaError(f"GrapheonRL scenario metadata is missing {sorted(missing)}")
        for task in scenario.tasks:
            if task.cpu_requirement is None or task.memory_requirement is None:
                raise DatasetSchemaError(f"GrapheonRL task {task.id!r} lost core or memory requirements")
        try:
            _ = scenario.cache.topological_order
        except ScenarioValidationError as exc:
            raise DatasetSchemaError(str(exc)) from exc

    def _parse_task(self, task_id: str, raw: Any, workflow_path: str | Path) -> Task:
        if not isinstance(task_id, str) or not task_id:
            raise DatasetSchemaError(f"{workflow_path}: task IDs must be non-empty strings")
        item = self._require_mapping(raw, workflow_path, f"tasks.{task_id}")
        features = item.get("features")
        if not isinstance(features, list) or not features or not all(isinstance(feature, str) for feature in features):
            raise DatasetSchemaError(f"{workflow_path}: tasks.{task_id}.features must be a non-empty string list")
        lower_features = tuple(sorted({feature.lower() for feature in features}))
        return Task(
            id=task_id,
            workload=self._number(item, "duration", workflow_path, positive=True),
            cpu_requirement=self._number(item, "cores", workflow_path, positive=True),
            memory_requirement=self._number(item, "memory_required", workflow_path, non_negative=True),
            device_requirement="gpu" if "gpu" in lower_features else "cpu",
            metadata={"features": lower_features, "tags": item.get("tags", []), "raw_task": dict(item)},
        )

    def _parse_dependencies(self, raw_tasks: Mapping[str, Any], workflow_path: str | Path) -> tuple[Dependency, ...]:
        dependencies: list[Dependency] = []
        for child_id, raw_task in raw_tasks.items():
            task = self._require_mapping(raw_task, workflow_path, f"tasks.{child_id}")
            raw_predecessors = task.get("dependencies")
            if not isinstance(raw_predecessors, list) or not all(isinstance(item, str) for item in raw_predecessors):
                raise DatasetSchemaError(f"{workflow_path}: tasks.{child_id}.dependencies must be a string list")
            for predecessor_id in raw_predecessors:
                if predecessor_id not in raw_tasks:
                    raise DatasetSchemaError(
                        f"{workflow_path}: task {child_id!r} references missing predecessor {predecessor_id!r}"
                    )
                predecessor = self._require_mapping(raw_tasks[predecessor_id], workflow_path, f"tasks.{predecessor_id}")
                dependencies.append(
                    Dependency(
                        src=predecessor_id,
                        dst=child_id,
                        data_size=self._number(predecessor, "data", workflow_path, non_negative=True),
                        metadata={"edge_data_policy": "predecessor_task_data", "source_task": predecessor_id},
                    )
                )
        return tuple(dependencies)

    def _infer_system_config(self, meta: Mapping[str, Any], workflow_path: Path) -> Path:
        source_path = meta.get("system_config")
        if not isinstance(source_path, str) or not source_path:
            raise DatasetSchemaError(f"{workflow_path}: meta.system_config must be a non-empty string")
        root = self.system_configs_root
        if root is None:
            for ancestor in workflow_path.parents:
                candidate = ancestor / "system_configs"
                if candidate.is_dir():
                    root = candidate
                    break
        if root is None:
            raise DatasetSchemaError("system_configs_root is required when no sibling system_configs directory exists")
        candidates = sorted(root.rglob(Path(source_path).name))
        if len(candidates) != 1:
            raise DatasetSchemaError(
                f"expected exactly one companion system config named {Path(source_path).name!r} under {root}, found {len(candidates)}"
            )
        return candidates[0]

    def _load_nodes(self, config_path: Path) -> tuple[tuple[ComputeNode, ...], np.ndarray]:
        raw = self._read_json(config_path, "system configuration")
        return self._load_nodes_from_raw(raw, config_path)

    def _load_nodes_from_raw(self, raw: Mapping[str, Any], config_path: str | Path) -> tuple[tuple[ComputeNode, ...], np.ndarray]:
        raw_nodes = self._require_mapping(raw.get("nodes"), config_path, "nodes")
        if not raw_nodes:
            raise DatasetSchemaError(f"{config_path}: nodes cannot be empty")
        nodes: list[ComputeNode] = []
        rates: list[float] = []
        for node_id, raw_node in raw_nodes.items():
            item = self._require_mapping(raw_node, config_path, f"nodes.{node_id}")
            tier = item.get("tier")
            if not isinstance(tier, str) or not tier:
                raise DatasetSchemaError(f"{config_path}: nodes.{node_id}.tier must be a non-empty string")
            try:
                node_type = self.node_type_mapper.map(tier)
            except KeyError as exc:
                raise DatasetSchemaError(f"cannot map GrapheonRL tier {tier!r}; configure NodeTypeMapper") from exc
            features = item.get("features")
            if not isinstance(features, list) or not all(isinstance(feature, str) for feature in features):
                raise DatasetSchemaError(f"{config_path}: nodes.{node_id}.features must be a string list")
            processing_speed = self._require_mapping(item.get("processing_speed"), config_path, f"nodes.{node_id}.processing_speed")
            normalized_speeds = {str(key).lower(): float(value) for key, value in processing_speed.items()}
            if not normalized_speeds or any(not np.isfinite(value) or value <= 0.0 for value in normalized_speeds.values()):
                raise DatasetSchemaError(f"{config_path}: nodes.{node_id}.processing_speed needs positive finite values")
            transfer = self._require_mapping(item.get("data_transfer_rate"), config_path, f"nodes.{node_id}.data_transfer_rate")
            transfer_rates = [float(value) for value in transfer.values()]
            if not transfer_rates or any(not np.isfinite(value) or value <= 0.0 for value in transfer_rates):
                raise DatasetSchemaError(f"{config_path}: nodes.{node_id}.data_transfer_rate needs positive finite values")
            lower_features = {feature.lower() for feature in features}
            nodes.append(
                ComputeNode(
                    id=node_id,
                    node_type=node_type,
                    compute_speed=max(normalized_speeds.values()),
                    cpu_capacity=self._number(item, "cores", config_path, positive=True),
                    gpu_capacity=1.0 if "gpu" in lower_features else 0.0,
                    memory_capacity=self._number(item, "memory", config_path, non_negative=True),
                    metadata={
                        "raw_node": dict(item),
                        "raw_tier": tier,
                        "features": sorted(lower_features),
                        "device_speeds": normalized_speeds,
                        "data_transfer_rate": dict(transfer),
                    },
                )
            )
            rates.append(max(transfer_rates))
        bandwidth = np.minimum.outer(np.asarray(rates, dtype=np.float64), np.asarray(rates, dtype=np.float64))
        np.fill_diagonal(bandwidth, self.local_bandwidth)
        return tuple(nodes), bandwidth

    @staticmethod
    def _base_graph_id(meta: Mapping[str, Any], workflow_path: Path) -> str:
        source = meta.get("source")
        if not isinstance(source, str) or not source:
            raise DatasetSchemaError(f"{workflow_path}: meta.source must be a non-empty string")
        source_path = Path(source)
        if len(source_path.parts) < 2:
            # The packaged FPPPP control graph is a single ``fpppp.stg`` file,
            # unlike the rnc<size>/randXXXX.stg families.
            return source_path.stem
        return f"rnc{source_path.parts[-2]}:{source_path.stem}"

    @staticmethod
    def _resource_seed(base_graph_id: str, config: ResourceConfig) -> int:
        import hashlib

        digest = hashlib.sha256(f"{base_graph_id}\0{config.config_id}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="little", signed=False)

    @staticmethod
    def _read_json(path: Path, label: str) -> Mapping[str, Any]:
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise DatasetSchemaError(f"invalid {label} JSON at {path}: {exc}") from exc
        if not isinstance(parsed, Mapping):
            raise DatasetSchemaError(f"{path}: {label} root must be an object")
        return parsed

    @staticmethod
    def _read_archive_json(source: ArchiveWorkflow, label: str) -> Mapping[str, Any]:
        try:
            with tarfile.open(source.archive_path, mode="r:xz") as package:
                member = package.getmember(source.member_name)
                extracted = package.extractfile(member)
                if extracted is None:
                    raise DatasetSchemaError(f"missing {label} member {source.display_path}")
                parsed = json.loads(extracted.read().decode("utf-8"))
        except (tarfile.TarError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DatasetSchemaError(f"invalid archived {label} JSON at {source.display_path}: {exc}") from exc
        if not isinstance(parsed, Mapping):
            raise DatasetSchemaError(f"{source.display_path}: {label} root must be an object")
        return parsed

    def _load_archive_system_config(self, archive_root: Path, meta: Mapping[str, Any], workflow_label: str) -> tuple[Mapping[str, Any], str]:
        source_path = meta.get("system_config")
        if not isinstance(source_path, str) or not source_path:
            raise DatasetSchemaError(f"{workflow_label}: meta.system_config must be a non-empty string")
        config_name = Path(source_path).name
        roots = [archive_root]
        if self.system_configs_root is not None and self.system_configs_root not in roots:
            roots.insert(0, self.system_configs_root)
        for root in roots:
            archive = root / "system_configs.tar.xz"
            if not archive.is_file():
                continue
            member_name = f"system_configs/{config_name}"
            try:
                raw = self._read_archive_json(ArchiveWorkflow(archive, member_name), "system configuration")
            except DatasetSchemaError:
                continue
            return raw, f"{archive.resolve()}!{member_name}"
        raise DatasetSchemaError(f"{workflow_label}: no packaged system config named {config_name!r}")

    @staticmethod
    def _require_mapping(value: Any, path: str | Path, label: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise DatasetSchemaError(f"{path}: {label} must be an object")
        return value

    @staticmethod
    def _number(mapping: Mapping[str, Any], key: str, path: str | Path, *, positive: bool = False, non_negative: bool = False) -> float:
        value = mapping.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise DatasetSchemaError(f"{path}: {key} must be a finite number")
        numeric = float(value)
        if positive and numeric <= 0.0:
            raise DatasetSchemaError(f"{path}: {key} must be positive")
        if non_negative and numeric < 0.0:
            raise DatasetSchemaError(f"{path}: {key} must be non-negative")
        return numeric
