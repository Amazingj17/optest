"""Communication-time abstractions based on the unified pairwise matrix."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from cpn_hrl_dag.scenario.types import Scenario


class CommunicationModel(ABC):
    """Computes transfer time for one already-known dependency placement."""

    name: str

    @abstractmethod
    def duration(
        self, scenario: Scenario, source_task: int, target_task: int, source_node: int, target_node: int
    ) -> float:
        """Return transfer time; same-node transfer must always be exactly zero."""


class MatrixCommunicationModel(CommunicationModel):
    """Direct pairwise bandwidth with optional direct pairwise latency."""

    name = "matrix_bandwidth_latency"

    def duration(
        self, scenario: Scenario, source_task: int, target_task: int, source_node: int, target_node: int
    ) -> float:
        if source_node == target_node:
            return 0.0
        data_size = float(scenario.cache.edge_data_matrix[source_task, target_task])
        bandwidth = float(scenario.bandwidth_matrix[source_node, target_node])
        if not np.isfinite(bandwidth) or bandwidth <= 0.0:
            raise ValueError("cross-node bandwidth must be finite and positive")
        latency = 0.0 if scenario.latency_matrix is None else float(scenario.latency_matrix[source_node, target_node])
        return data_size / bandwidth + latency
