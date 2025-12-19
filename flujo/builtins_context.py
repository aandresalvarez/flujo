from __future__ import annotations

import warnings

from .builtins import context as _context

warnings.warn(
    "flujo.builtins_context is deprecated; use flujo.builtins.context instead.",
    DeprecationWarning,
    stacklevel=2,
)

for _name in dir(_context):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_context, _name)

__all__ = getattr(_context, "__all__", [name for name in dir(_context) if not name.startswith("_")])
