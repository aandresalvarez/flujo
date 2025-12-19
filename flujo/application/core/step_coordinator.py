"""Compatibility shim for `step_coordinator` (moved to core/orchestration/)."""

from __future__ import annotations

from importlib import import_module
import sys as _sys

_module = import_module("flujo.application.core.orchestration.step_coordinator")
_sys.modules[__name__] = _module
