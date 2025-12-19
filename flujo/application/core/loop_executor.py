"""Compatibility shim for `loop_executor` (moved to core/execution/)."""

from __future__ import annotations

from importlib import import_module
import sys as _sys

_module = import_module("flujo.application.core.execution.loop_executor")
_sys.modules[__name__] = _module
