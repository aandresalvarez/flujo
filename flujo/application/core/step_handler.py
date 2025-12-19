"""Compatibility shim for `step_handler` (moved to core/execution/)."""

from __future__ import annotations

from importlib import import_module
import sys as _sys

_module = import_module("flujo.application.core.execution.step_handler")
_sys.modules[__name__] = _module
