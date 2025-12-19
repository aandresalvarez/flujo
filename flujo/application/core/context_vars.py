"""Compatibility shim for `context_vars` (moved to core/context/)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flujo.application.core.context.context_vars import *  # noqa: F401,F403
    from flujo.application.core.context.context_vars import _CACHE_OVERRIDE  # noqa: F401

from importlib import import_module
import sys as _sys

_module = import_module("flujo.application.core.context.context_vars")
_sys.modules[__name__] = _module
