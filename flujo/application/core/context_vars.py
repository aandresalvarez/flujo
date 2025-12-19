"""Compatibility shim for `context_vars` (moved to core/context/)."""

from __future__ import annotations

from importlib import import_module
import sys as _sys

_module = import_module("flujo.application.core.context.context_vars")
_sys.modules[__name__] = _module
