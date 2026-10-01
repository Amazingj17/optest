"""Config-driven reproducible cloud-edge-end resource generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .types import ComputeNode


@dataclass(frozen=True, slots=True)
class ClosedRange:
    """Finite inclusive range sampled uniformly with a supplied generator."""

    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.minimum) or not np.isfinite(self.maximum) or self.minimum > self.maximum:
            raise ValueError("range endpoints must be finite and minimum <= maximum")

    def sample_float(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.minimum, self.maximum))

    def sample_int(self, rng: np.random.Generator) -> int:
        if int(self.minimum) != self.minimum or int(self.maximum) != self.maximum:
            raise ValueError("count ranges need integral endpoints")
        return int(rng.integers(int(self.minimum), int(self.maximum) + 1))


@dataclass(frozen=True, slots=True)
class ResourceTierConfig:
    """One configurable resource tier, such as ``cloud`` or ``edge``."""

    count: ClosedRange
    speed: ClosedRange
    cpu_capacity: ClosedRange | None = None
    gpu_capacity: ClosedRange | None = None
    memory_capacity: ClosedRange | None = None


@dataclass(frozen=True, slots=True)
class ResourceConfig:
    """Generator input. Pair keys are canonicalized, e.g. ``cloud_edge``."""

    tiers: Mapping[str, ResourceTierConfig]
    bandwidth: Mapping[str, ClosedRange]
    latency: Mapping[str, ClosedRange] | None = None
    config_id: str = "generated"


@dataclass(frozen=True, slots=True)
class GeneratedResources:
    """Generated resource realization that can be inserted into a Scenario."""

    nodes: tuple[ComputeNode, ...]
    bandwidth_matrix: np.ndarray
    latency_matrix: np.ndarray | None
    seed: int
    config_id: str


def _pair_key(left: str, right: str) -> str:
    return "_".join(sorted((left.lower(), right.lower())))


class CloudEdgeEndResourceGenerator:
    """Generates arbitrary pairwise resource matrices from explicit ranges."""

    def generate(self, config: ResourceConfig, seed: int) -> GeneratedResources:
        if not config.tiers:
            raise ValueError("resource config must define at least one tier")
        rng = np.random.default_rng(seed)
        nodes: list[ComputeNode] = []
        for tier_name in sorted(config.tiers):
            tier = config.tiers[tier_name]
            for index in range(tier.count.sample_int(rng)):
                gpu_capacity = None if tier.gpu_capacity is None else tier.gpu_capacity.sample_float(rng)
                # Keep generated node affinity semantics consistent with native
                # adapters: a positive GPU capacity advertises the `gpu` feature.
                features = ["cpu"] + (["gpu"] if gpu_capacity is not None and gpu_capacity > 0.0 else [])
                nodes.append(
                    ComputeNode(
                        id=f"{tier_name}-{index}",
                        node_type=tier_name,
                        compute_speed=tier.speed.sample_float(rng),
                        cpu_capacity=None if tier.cpu_capacity is None else tier.cpu_capacity.sample_float(rng),
                        gpu_capacity=gpu_capacity,
                        memory_capacity=None if tier.memory_capacity is None else tier.memory_capacity.sample_float(rng),
                        metadata={"resource_generation_seed": seed, "resource_config_id": config.config_id, "features": features},
                    )
                )
        if not nodes:
            raise ValueError("resource config generated zero nodes")
        bandwidth = np.empty((len(nodes), len(nodes)), dtype=np.float64)
        latency = None if config.latency is None else np.empty_like(bandwidth)
        for left_index, left in enumerate(nodes):
            for right_index, right in enumerate(nodes):
                pair = _pair_key(left.node_type, right.node_type)
                if pair not in config.bandwidth:
                    raise ValueError(f"missing bandwidth range for node-type pair {pair!r}")
                bandwidth[left_index, right_index] = config.bandwidth[pair].sample_float(rng)
                if bandwidth[left_index, right_index] <= 0.0:
                    raise ValueError(f"bandwidth range for {pair!r} must generate positive values")
                if latency is not None:
                    if pair not in config.latency:
                        raise ValueError(f"missing latency range for node-type pair {pair!r}")
                    latency[left_index, right_index] = config.latency[pair].sample_float(rng)
                    if latency[left_index, right_index] < 0.0:
                        raise ValueError(f"latency range for {pair!r} must generate non-negative values")
        bandwidth.setflags(write=False)
        if latency is not None:
            latency.setflags(write=False)
        return GeneratedResources(tuple(nodes), bandwidth, latency, seed, config.config_id)


def resource_config_from_mapping(raw: Mapping[str, Any]) -> ResourceConfig:
    """Parse the public YAML representation without hard-coded tier ranges.

    Each numerical range is written as ``{min: x, max: y}`` or ``[x, y]``.
    The caller supplies every tier-pair used by its resource profile, making a
    missing network assumption an explicit configuration error.
    """
    def interval(value: Any) -> ClosedRange:
        if isinstance(value, Mapping):
            return ClosedRange(float(value["min"]), float(value["max"]))
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return ClosedRange(float(value[0]), float(value[1]))
        raise ValueError("resource ranges require {min,max} or a two-value list")
    raw_tiers = raw.get("tiers") or {key: value for key, value in raw.items() if key in {"cloud", "edge", "end"}}
    tiers = {
        str(name): ResourceTierConfig(
            count=interval(spec["count"]), speed=interval(spec["speed"]),
            cpu_capacity=None if "cpu_capacity" not in spec else interval(spec["cpu_capacity"]),
            gpu_capacity=None if "gpu_capacity" not in spec else interval(spec["gpu_capacity"]),
            memory_capacity=None if "memory_capacity" not in spec else interval(spec["memory_capacity"]),
        ) for name, spec in raw_tiers.items()
    }
    bandwidth = {str(key): interval(value) for key, value in raw["bandwidth"].items()}
    latency_raw = raw.get("latency")
    latency = None if latency_raw is None else {str(key): interval(value) for key, value in latency_raw.items()}
    return ResourceConfig(tiers, bandwidth, latency, str(raw.get("config_id", "generated")))


def realize_scenario(scenario: "Scenario", config: ResourceConfig, seed: int) -> "Scenario":
    """Return a new resource realization while preserving immutable DAG provenance.

    This function never mutates raw data or the input scenario.  Its scenario
    identifier includes the config and seed, whereas splitting continues to use
    ``metadata.original_graph_id`` and therefore cannot leak a topology.
    """
    from .types import Scenario
    generated = CloudEdgeEndResourceGenerator().generate(config, seed)
    metadata = dict(scenario.metadata)
    metadata.update({"resource_generation_seed": seed, "resource_config_id": config.config_id})
    return Scenario(
        scenario_id=f"{scenario.scenario_id}:resource:{config.config_id}:{seed}",
        dataset_source=scenario.dataset_source, tasks=scenario.tasks, dependencies=scenario.dependencies,
        compute_nodes=generated.nodes, bandwidth_matrix=generated.bandwidth_matrix,
        latency_matrix=generated.latency_matrix, metadata=metadata,
    )


@dataclass(frozen=True, slots=True)
class NodeTypeMapper:
    """Maps exact raw labels or prefixes to standardized cloud/edge/end labels."""

    exact: Mapping[str, str]
    prefixes: Mapping[str, str]

    def map(self, raw_name: str, fallback: str | None = None) -> str:
        if raw_name in self.exact:
            return self.exact[raw_name]
        for prefix, node_type in sorted(self.prefixes.items(), key=lambda item: -len(item[0])):
            if raw_name.startswith(prefix):
                return node_type
        if fallback is not None:
            return fallback
        raise KeyError(f"no node-type mapping for raw resource label {raw_name!r}")
