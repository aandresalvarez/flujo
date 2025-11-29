from __future__ import annotations

# Facade module that composes builtin registrations split across helper modules.
# Imports are intentionally wildcarded to preserve public surface and side-effect
# registrations on import (skills registry population).

from .agents.wrapper import make_agent_async  # noqa: F401
from .builtins_core import _register_builtins, register_core_builtins  # noqa: F401,F403
from .builtins_architect import *  # noqa: F401,F403
from .builtins_support import *  # noqa: F401,F403
from .builtins_extras import _DDGSAsync, _DDGS_CLASS  # noqa: F401
from .builtins_extras import *  # noqa: F401,F403
from .builtins_optional import register_optional_builtins  # noqa: F401,F403
from .builtins_context import register_context_builtins  # noqa: F401,F403

# Re-export for backward compatibility with tests that import _register_builtins directly.
__all__ = [name for name in globals().keys() if not name.startswith("_")]
__all__.append("_register_builtins")
__all__.extend(["_DDGSAsync", "_DDGS_CLASS"])
