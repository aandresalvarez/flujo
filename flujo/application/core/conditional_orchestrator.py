"""Compatibility shim for `conditional_orchestrator` (moved to core/orchestration/)."""

from __future__ import annotations

from importlib import import_module
import sys as _sys

_module = import_module("flujo.application.core.orchestration.conditional_orchestrator")
_sys.modules[__name__] = _module
