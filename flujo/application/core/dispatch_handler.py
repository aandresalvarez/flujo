"""Compatibility shim for `dispatch_handler` (moved to core/execution/)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flujo.application.core.execution.dispatch_handler import *  # noqa: F401,F403

from importlib import import_module
import sys as _sys

_module = import_module("flujo.application.core.execution.dispatch_handler")
_sys.modules[__name__] = _module
