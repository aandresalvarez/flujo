"""Compatibility shim for `failure_builder` (moved to core/support/)."""

from __future__ import annotations

from importlib import import_module
import sys as _sys

_module = import_module("flujo.application.core.support.failure_builder")
_sys.modules[__name__] = _module
