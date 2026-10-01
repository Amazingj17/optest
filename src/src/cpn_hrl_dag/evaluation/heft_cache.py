"""Persistent, semantics-aware HEFT reference cache.

The cache is deliberately owned by evaluation rather than an adapter: the
same key is valid for rewards, baselines, and reports only when scenario data
and both timing models are identical.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from cpn_hrl_dag.baselines.heft import HEFTScheduler
from cpn_hrl_dag.scenario.types import Scenario
from cpn_hrl_dag.scheduling.communication_model import CommunicationModel
from cpn_hrl_dag.scheduling.execution_model import ExecutionTimeModel


class HEFTCache:
    """File-backed cache keyed by scenario content and simulation semantics."""

    cache_version = "heft-cache-v1-insertion-list-schedule"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def key(
        self,
        scenario: Scenario,
        execution_model: ExecutionTimeModel,
        communication_model: CommunicationModel,
    ) -> str:
        payload: dict[str, Any] = {
            "cache_version": self.cache_version,
            "scenario_id": scenario.scenario_id,
            "tasks": [(str(t.id), t.workload, t.cpu_requirement, t.gpu_requirement, t.memory_requirement, t.device_requirement) for t in scenario.tasks],
            "edges": [(str(e.src), str(e.dst), e.data_size) for e in scenario.dependencies],
            "nodes": [(str(n.id), n.node_type, n.compute_speed, n.cpu_capacity, n.gpu_capacity, n.memory_capacity, dict(n.metadata)) for n in scenario.compute_nodes],
            "bandwidth": scenario.bandwidth_matrix.tolist(),
            "latency": None if scenario.latency_matrix is None else scenario.latency_matrix.tolist(),
            "execution_model": execution_model.name,
            "communication_model": communication_model.name,
        }
        serialized = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def get_or_compute(
        self,
        scenario: Scenario,
        execution_model: ExecutionTimeModel,
        communication_model: CommunicationModel,
    ) -> float:
        cache_key = self.key(scenario, execution_model, communication_model)
        target = self.root / f"{cache_key}.json"
        if target.is_file():
            raw = json.loads(target.read_text(encoding="utf-8"))
            value = float(raw["heft_makespan"])
            if np.isfinite(value) and value > 0.0:
                return value
            raise ValueError(f"invalid HEFT cache entry: {target}")
        result, _ = HEFTScheduler(execution_model, communication_model).schedule(scenario)
        payload = {"key": cache_key, "scenario_id": scenario.scenario_id, "heft_makespan": result.makespan}
        target.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        return float(result.makespan)
