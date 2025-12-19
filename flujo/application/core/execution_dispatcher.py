"""Compatibility shim for `execution_dispatcher` (moved to core/execution/)."""

from __future__ import annotations

from importlib import import_module
import sys as _sys

_module = import_module("flujo.application.core.execution.execution_dispatcher")
_sys.modules[__name__] = _module
