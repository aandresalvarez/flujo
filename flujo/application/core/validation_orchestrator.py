"""Compatibility shim for `validation_orchestrator` (moved to core/orchestration/)."""

from __future__ import annotations

from importlib import import_module
import sys as _sys

_module = import_module("flujo.application.core.orchestration.validation_orchestrator")
_sys.modules[__name__] = _module
