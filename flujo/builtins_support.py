from __future__ import annotations

import warnings

from .builtins import support as _support

warnings.warn(
    "flujo.builtins_support is deprecated; use flujo.builtins.support instead.",
    DeprecationWarning,
    stacklevel=2,
)

for _name in dir(_support):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_support, _name)

__all__ = getattr(_support, "__all__", [name for name in dir(_support) if not name.startswith("_")])
