"""Single registry used by preparation, training, and evaluation entry points."""

from __future__ import annotations

from typing import Iterator

from .base import DatasetAdapter


class DatasetRegistry:
    """Named adapter registry with explicit duplicate-registration protection."""

    def __init__(self) -> None:
        self._adapters: dict[str, DatasetAdapter] = {}

    def register(self, name: str, adapter: DatasetAdapter) -> None:
        normalized = name.strip().lower()
        if not normalized:
            raise ValueError("dataset registry name must be non-empty")
        if normalized in self._adapters:
            raise ValueError(f"dataset adapter already registered: {normalized}")
        self._adapters[normalized] = adapter

    def get(self, name: str) -> DatasetAdapter:
        try:
            return self._adapters[name.strip().lower()]
        except KeyError as exc:
            raise KeyError(f"unknown dataset adapter: {name!r}") from exc

    def __iter__(self) -> Iterator[tuple[str, DatasetAdapter]]:
        yield from self._adapters.items()
