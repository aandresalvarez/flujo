from __future__ import annotations

import warnings

from .builtins import architect as _architect

warnings.warn(
    "flujo.builtins_architect is deprecated; use flujo.builtins.architect instead.",
    DeprecationWarning,
    stacklevel=2,
)

for _name in dir(_architect):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_architect, _name)

__all__ = getattr(
    _architect, "__all__", [name for name in dir(_architect) if not name.startswith("_")]
)
