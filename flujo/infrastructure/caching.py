"""Compatibility caching utilities for robustness tests.

Provides a synchronous InMemoryLRUCache wrapper that maps to the core _LRUCache
implementation exposed from flujo.infra.caching.
"""

from __future__ import annotations

import warnings

from flujo.infra.caching import InMemoryLRUCache

warnings.warn(
    "flujo.infrastructure.caching is deprecated; use flujo.infra.caching.InMemoryLRUCache instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["InMemoryLRUCache"]
