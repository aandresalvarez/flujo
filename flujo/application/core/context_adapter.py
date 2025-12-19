"""Compatibility shim for `context_adapter` (moved to core/context/)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flujo.application.core.context.context_adapter import *  # noqa: F401,F403

from importlib import import_module
import sys as _sys

_module = import_module("flujo.application.core.context.context_adapter")
_sys.modules[__name__] = _module
