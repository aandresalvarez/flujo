from __future__ import annotations

from typing import Dict, Optional, Any
from packaging.version import Version, InvalidVersion

from .domain.dsl.pipeline import Pipeline


class PipelineRegistry:
    """Simple in-memory registry for pipelines keyed by name and version."""

    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, Pipeline[Any, Any]]] = {}

    def register(self, pipeline: Pipeline[Any, Any], name: str, version: str) -> None:
        """Register ``pipeline`` under ``name`` and ``version``."""
        try:
            Version(version)
        except InvalidVersion as e:  # pragma: no cover - defensive
            raise ValueError(f"Invalid version '{version}': {e}") from e
        versions = self._store.setdefault(name, {})
        versions[version] = pipeline

    def get(self, name: str, version: str) -> Optional[Pipeline[Any, Any]]:
        """Return the pipeline registered under ``name`` and ``version`` if present."""
        return self._store.get(name, {}).get(version)

    def get_latest(self, name: str) -> Optional[Pipeline[Any, Any]]:
        """Return the pipeline with the highest semantic version for ``name``."""
        versions = self._store.get(name)
        if not versions:
            return None
        latest_version = max(versions, key=lambda v: Version(v))
        return versions[latest_version]


__all__ = ["PipelineRegistry"]
