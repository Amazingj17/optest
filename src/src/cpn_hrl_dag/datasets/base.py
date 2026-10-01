"""Interfaces isolating raw benchmark formats from scheduling logic."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable

from cpn_hrl_dag.scenario.types import Scenario


class DatasetSchemaError(ValueError):
    """Raised for an inspected source file that violates its audited schema."""


class DatasetAdapter(ABC):
    """Converts one raw dataset format into the common :class:`Scenario`."""

    name: str
    adapter_version: str

    @abstractmethod
    def discover(self, root: str | Path) -> Iterable[Path]:
        """Return source scenario paths in deterministic order."""

    @abstractmethod
    def load(self, path: str | Path, resource_config: Any | None = None) -> Scenario:
        """Load one source path without mutating raw data."""

    @abstractmethod
    def validate(self, scenario: Scenario) -> None:
        """Validate adapter-specific provenance and common scenario constraints."""
