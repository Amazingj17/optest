"""Deterministic dataset-balanced and curriculum-aware scenario sampling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from cpn_hrl_dag.scenario.types import Scenario


@dataclass(frozen=True, slots=True)
class CurriculumStage:
    """Inclusive task-count range and number of episodes for one stage."""

    min_tasks: int
    max_tasks: int
    episodes: int


class DatasetBalancedSampler:
    """Samples source first, then scenario, so large sources do not dominate."""

    def __init__(self, scenarios: Iterable[Scenario], weights: Mapping[str, float], seed: int) -> None:
        self.by_source: dict[str, tuple[Scenario, ...]] = {}
        for scenario in scenarios:
            self.by_source.setdefault(scenario.dataset_source, []).append(scenario)
        self.by_source = {source: tuple(sorted(items, key=lambda item: item.scenario_id)) for source, items in self.by_source.items()}
        if not self.by_source:
            raise ValueError("sampler needs at least one scenario")
        sources = tuple(sorted(self.by_source))
        values = np.asarray([float(weights.get(source, 1.0)) for source in sources], dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0.0).any() or values.sum() <= 0.0:
            raise ValueError("dataset sampling weights must be non-negative and sum to a positive value")
        self.sources, self.probabilities, self.rng = sources, values / values.sum(), np.random.default_rng(seed)

    def sample(self, stage: CurriculumStage | None = None) -> Scenario:
        available = {source: values if stage is None else tuple(x for x in values if stage.min_tasks <= x.num_tasks <= stage.max_tasks) for source, values in self.by_source.items()}
        source_indices = [index for index, source in enumerate(self.sources) if available[source]]
        if not source_indices:
            raise ValueError("curriculum stage contains no training scenarios")
        p = self.probabilities[source_indices]; p = p / p.sum()
        source = self.sources[int(self.rng.choice(source_indices, p=p))]
        items = available[source]
        return items[int(self.rng.integers(len(items)))]
