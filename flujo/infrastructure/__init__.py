from __future__ import annotations

import warnings

from flujo.infra.caching import InMemoryLRUCache

warnings.warn(
    "flujo.infrastructure is deprecated; use flujo.infra instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["InMemoryLRUCache"]
