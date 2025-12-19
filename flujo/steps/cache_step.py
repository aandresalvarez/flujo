from __future__ import annotations

import warnings

from flujo.domain.dsl import cache_step as _cache_step

warnings.warn(
    "flujo.steps.cache_step is deprecated; use flujo.domain.dsl.cache_step instead.",
    DeprecationWarning,
    stacklevel=2,
)

for _name in dir(_cache_step):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_cache_step, _name)

__all__ = getattr(
    _cache_step, "__all__", [name for name in dir(_cache_step) if not name.startswith("_")]
)
