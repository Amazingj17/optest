"""Dataset-registry-backed loading used by all command line entry points."""
from __future__ import annotations
from pathlib import Path
from typing import Iterable
from .registry import DatasetRegistry
from .grapheonrl_adapter import GrapheonRLAdapter
from cpn_hrl_dag.scenario.types import Scenario

def default_registry(
    grapheonrl_system_configs: str | Path | None = None,
    grapheonrl_archive_task_counts: Iterable[int] | None = (50, 100, 300),
) -> DatasetRegistry:
    registry=DatasetRegistry()
    registry.register('grapheonrl',GrapheonRLAdapter(grapheonrl_system_configs, archive_task_counts=grapheonrl_archive_task_counts))
    return registry

def load_scenarios(registry:DatasetRegistry, roots:dict[str,str|Path], *, limit_per_dataset:int|None=None)->list[Scenario]:
    """Load native source files with no source-specific branches outside registry."""
    scenarios=[]
    for name,root in roots.items():
        adapter=registry.get(name); paths=list(adapter.discover(root))
        if limit_per_dataset is not None: paths=paths[:limit_per_dataset]
        scenarios.extend(adapter.load(path) for path in paths)
    if not scenarios: raise ValueError('no scenarios loaded')
    return scenarios
