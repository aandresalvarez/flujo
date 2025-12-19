"""Compatibility shim for `error_utils` (moved to core/support/)."""

from __future__ import annotations

from importlib import import_module
import sys as _sys

_module = import_module("flujo.application.core.support.error_utils")
_sys.modules[__name__] = _module
