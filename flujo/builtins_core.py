from __future__ import annotations

import warnings

from .builtins import core as _core

warnings.warn(
    "flujo.builtins_core is deprecated; use flujo.builtins.core instead.",
    DeprecationWarning,
    stacklevel=2,
)

for _name in dir(_core):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_core, _name)

__all__ = getattr(_core, "__all__", [name for name in dir(_core) if not name.startswith("_")])
