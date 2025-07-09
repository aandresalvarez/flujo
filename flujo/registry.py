from __future__ import annotations

import uuid
from typing import Any, Callable, Dict, Optional

from packaging.version import Version, InvalidVersion

from .domain.dsl.pipeline import Pipeline


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


class CallableRegistry:
    """Registry for storing and retrieving callable functions by unique string ID.

    This registry allows callable functions to be serialized and deserialized
    by storing them with unique identifiers that can be referenced in the IR.
    """

    def __init__(self) -> None:
        self._store: Dict[str, Callable[..., Any]] = {}
        self._reverse_store: Dict[int, str] = {}  # id(func) -> ref_id for deduplication

    def register(self, func: Callable[..., Any]) -> str:
        """Register a callable function and return its unique reference ID.

        If the function is already registered, returns the existing reference ID.
        """
        func_id = id(func)

        # Check if already registered
        if func_id in self._reverse_store:
            return self._reverse_store[func_id]

        # Generate new reference ID
        ref_id = str(uuid.uuid4())

        # Store the function
        self._store[ref_id] = func
        self._reverse_store[func_id] = ref_id

        return ref_id

    def get(self, ref_id: str) -> Callable[..., Any]:
        """Retrieve a callable function by its reference ID."""
        if ref_id not in self._store:
            raise KeyError(f"Callable with reference ID '{ref_id}' not found in registry")
        return self._store[ref_id]

    def clear(self) -> None:
        """Clear all registered callables."""
        self._store.clear()
        self._reverse_store.clear()

    def __len__(self) -> int:
        """Return the number of registered callables."""
        return len(self._store)


__all__ = ["PipelineRegistry", "CallableRegistry"]
