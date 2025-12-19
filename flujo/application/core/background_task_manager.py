"""Compatibility shim for `background_task_manager` (moved to core/runtime/)."""

from __future__ import annotations

from importlib import import_module
import sys as _sys

_module = import_module("flujo.application.core.runtime.background_task_manager")
_sys.modules[__name__] = _module
