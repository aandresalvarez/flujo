from __future__ import annotations

from typing import Any, Dict, Optional

from packaging.version import Version, InvalidVersion

from ..domain.dsl.pipeline import Pipeline


class PipelineRegistry:
    """Simple in-memory registry for pipeline objects."""

    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, Pipeline[Any, Any]]] = {}

    def register(self, pipeline: Pipeline[Any, Any], name: str, version: str) -> None:
        """Register ``pipeline`` under ``name`` and ``version``."""
        try:
            Version(version)
        except InvalidVersion as e:
            raise ValueError(f"Invalid version: {version}") from e
        versions = self._store.setdefault(name, {})
        versions[version] = pipeline

    def get(self, name: str, version: str) -> Optional[Pipeline[Any, Any]]:
        """Return the pipeline registered for ``name`` and ``version`` if present."""
        versions = self._store.get(name)
        if not versions:
            return None
        return versions.get(version)

    def get_latest_version(self, name: str) -> Optional[str]:
        """Return the latest registered version for ``name``."""
        versions = self._store.get(name)
        if not versions:
            return None
        return max(versions.keys(), key=Version)

    def get_latest(self, name: str) -> Optional[Pipeline[Any, Any]]:
        """Return the latest registered pipeline for ``name`` if any."""
        ver = self.get_latest_version(name)
        if ver is None:
            return None
        return self._store[name][ver]


__all__ = ["PipelineRegistry"]
