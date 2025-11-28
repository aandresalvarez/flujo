"""Compatibility caching utilities for robustness tests.

Provides a synchronous InMemoryLRUCache wrapper that maps to the core _LRUCache
implementation in flujo.application.core.default_components.
"""

from __future__ import annotations

from flujo.application.core.default_components import _LRUCache as InMemoryLRUCache

__all__ = ["InMemoryLRUCache"]
