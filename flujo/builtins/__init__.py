from __future__ import annotations

# Facade module that composes builtin registrations split across helper modules.
# Imports are intentionally wildcarded to preserve public surface and side-effect
# registrations on import (skills registry population).

from ..agents.wrapper import make_agent_async  # noqa: F401
from . import core as _core
from . import extras as _extras
from .core import register_core_builtins  # noqa: F401,F403
from .architect import *  # noqa: F401,F403
from .support import *  # noqa: F401,F403
from .extras import _DDGSAsync, _DDGS_CLASS  # noqa: F401
from .extras import *  # noqa: F401,F403
from .optional import register_optional_builtins  # noqa: F401,F403
from .context import register_context_builtins  # noqa: F401,F403


def _register_builtins() -> None:
    """Register all builtins (core + extras) in a deterministic, idempotent way."""
    _core._register_builtins()
    try:
        _extras._register_builtins()
    except Exception:
        # Extras rely on optional deps; ignore failures to keep core usable.
        pass


# Re-export for backward compatibility with tests that import _register_builtins directly.
__all__ = [name for name in globals().keys() if not name.startswith("_")]
__all__.append("_register_builtins")
__all__.extend(["_DDGSAsync", "_DDGS_CLASS"])
