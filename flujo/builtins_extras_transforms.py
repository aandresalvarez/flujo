from __future__ import annotations

import warnings

from .builtins import extras_transforms as _extras_transforms

warnings.warn(
    "flujo.builtins_extras_transforms is deprecated; use flujo.builtins.extras_transforms instead.",
    DeprecationWarning,
    stacklevel=2,
)

for _name in dir(_extras_transforms):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_extras_transforms, _name)

__all__ = getattr(
    _extras_transforms,
    "__all__",
    [name for name in dir(_extras_transforms) if not name.startswith("_")],
)
