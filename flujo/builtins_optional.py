from __future__ import annotations

import warnings

from .builtins import optional as _optional

warnings.warn(
    "flujo.builtins_optional is deprecated; use flujo.builtins.optional instead.",
    DeprecationWarning,
    stacklevel=2,
)

for _name in dir(_optional):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_optional, _name)

__all__ = getattr(
    _optional, "__all__", [name for name in dir(_optional) if not name.startswith("_")]
)
