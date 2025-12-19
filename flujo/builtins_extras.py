from __future__ import annotations

import warnings

from .builtins import extras as _extras

warnings.warn(
    "flujo.builtins_extras is deprecated; use flujo.builtins.extras instead.",
    DeprecationWarning,
    stacklevel=2,
)

for _name in dir(_extras):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_extras, _name)

__all__ = getattr(_extras, "__all__", [name for name in dir(_extras) if not name.startswith("_")])
